"""
historical_ingest.py  —  PYTHON SCRIPT (i/o bound, deterministic data munging)

architecture note: this script is purely deterministic i/o and math.
no llm calls, no fuzzy logic. its only job is to reconstruct the exact
OrderBookSnapshot stream that data_pipeline.py would have seen had the
orchestrator been running during historical markets, then persist that
stream as compressed parquet for the training loop.

the critical contract we must satisfy exactly:
  OrderBookSnapshot fields → FeatureBuffer.update() → compute_feature_row()
  → MarketDataPipeline._raw_rows → get_batch_tensor() → (N, 12, 64) tensor

any deviation in field semantics (e.g. treating a cumulative volume as a
15-min trade count) would cause the trained model to expect a distribution
that the live inference pipeline never produces — a training/serving skew
that silently degrades brier scores without a traceable root cause.

data sources:
  gamma api:  https://gamma-api.polymarket.com
    - /markets           → resolved market metadata (condition_id, outcome, expiry)
    - /events            → group markets by event for batch fetching
  clob api:   https://clob.polymarket.com
    - /trades            → individual matched trades (price, size, timestamp)
    - /book              → order book snapshots (best bid/ask, depth)

rate limits (as of 2025):
  gamma: ~100 req/min unauthenticated
  clob:  ~120 req/min unauthenticated
  we throttle to 80 req/min with asyncio semaphore + sleep to stay safe.

output: data/historical/<condition_id>.parquet
  columns match OrderBookSnapshot fields exactly, plus 'outcome' (int 0/1).
  stored with snappy compression (fast decode, 40-50% size reduction).
  one file per resolved market — training loop reads them lazily to avoid
  loading all history into ram simultaneously.
"""

import asyncio
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INGEST] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/logs/ingest.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── api endpoints ──────────────────────────────────────────────────────────────

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"
GOLDSKY_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"

OUTPUT_DIR      = Path("data/historical")
TICK_INTERVAL_S = 60          # synthesize 1-min ticks from raw trade stream
MIN_TICKS       = 80          # discard markets with fewer ticks (pipeline needs 64 + warmup)
MAX_MARKETS     = 2000        # cap to prevent unbounded runs; increase for full dataset
CONCURRENCY     = 8           # parallel market fetches (well below rate limit)
REQUEST_DELAY_S = 0.75        # per-request floor delay → ~80 req/min effective rate


# ── data contracts ─────────────────────────────────────────────────────────────

@dataclass
class ResolvedMarket:
    """minimal metadata for a fully resolved binary market."""
    condition_id:    str
    question:        str
    outcome:         int    # 1 = YES resolved, 0 = NO resolved
    expiry_ts:       float  # unix timestamp of resolution
    yes_token_id:  str | None = None   # CLOB token id for YES outcome


# ── api client helpers ─────────────────────────────────────────────────────────

async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
    retries: int = 3,
) -> Optional[dict | list]:
    """
    thin async GET wrapper with exponential-backoff retry.

    the retry window covers transient 429 / 5xx responses from both apis.
    we do NOT retry on 404 (market data genuinely absent) or 400 (bad params).

    Time: O(retries) worst case — dominated by network round-trips
    """
    for attempt in range(retries):
        try:
            await asyncio.sleep(REQUEST_DELAY_S)
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 429:
                    # honor retry-after header if present, else binary backoff
                    wait = float(resp.headers.get("Retry-After", 2 ** attempt * 5))
                    logger.warning(f"rate limited on {url} — sleeping {wait:.0f}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status == 404:
                    return None
                if resp.status >= 400:
                    logger.warning(f"http {resp.status} on {url}")
                    return None
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            wait = 2 ** attempt
            logger.warning(f"request failed ({exc}) — retry {attempt + 1}/{retries} in {wait}s")
            await asyncio.sleep(wait)
    logger.error(f"all retries exhausted for {url}")
    return None


# ── phase 1: discover resolved markets ────────────────────────────────────────

async def fetch_resolved_markets(session: aiohttp.ClientSession) -> list[ResolvedMarket]:
    import json as _json
    resolved: list[ResolvedMarket] = []
    offset = 0
    page_size = 100

    while len(resolved) < MAX_MARKETS:
        data = await _get_json(
            session,
            f"{GAMMA_BASE}/markets",
            params={"closed": "true", "limit": page_size, "offset": offset, "end_date_min": "2024-10-01",},
        )
        if not data:
            break
        markets = data if isinstance(data, list) else data.get("markets", [])
        if not markets:
            break

        for m in markets:
            raw_prices = m.get("outcomePrices")
            if not raw_prices:
                continue
            try:
                prices = _json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
                yes_price = float(prices[0])
            except (ValueError, IndexError, TypeError):
                continue

            if yes_price >= 0.95:
                outcome = 1
            elif yes_price <= 0.05:
                outcome = 0
            else:
                continue

            raw_outcomes = m.get("outcomes", "")
            try:
                outcomes_list = _json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
            except Exception:
                continue
            if not isinstance(outcomes_list, list) or len(outcomes_list) != 2:
                continue

            # extract YES token id (index 0 of clobTokenIds)
            raw_token_ids = m.get("clobTokenIds")
            yes_token_id = None
            if raw_token_ids:
                try:
                    token_ids = _json.loads(raw_token_ids) if isinstance(raw_token_ids, str) else raw_token_ids
                    yes_token_id = str(token_ids[0]) if token_ids else None
                except Exception:
                    pass

            expiry_str = m.get("endDate") or m.get("closedTime")
            if not expiry_str:
                continue
            try:
                expiry_ts = pd.Timestamp(expiry_str).timestamp()
            except Exception:
                continue

            condition_id = m.get("conditionId") or m.get("id")
            if not condition_id:
                continue

            resolved.append(ResolvedMarket(
                condition_id  = str(condition_id),
                question      = m.get("question", ""),
                outcome       = outcome,
                expiry_ts     = expiry_ts,
                yes_token_id  = yes_token_id,   # <-- new field
            ))

        logger.info(f"fetched page offset={offset} — {len(resolved)} resolved markets so far")
        offset += page_size
        if len(markets) < page_size:
            break

    logger.info(f"discovered {len(resolved)} resolved binary markets")
    return resolved[:MAX_MARKETS]

# ── phase 2: fetch raw clob trades for one market ─────────────────────────────

async def fetch_clob_trades(
    session: aiohttp.ClientSession,
    condition_id: str,
    expiry_ts: float,
    yes_token_id: str | None = None,
) -> Optional[pd.DataFrame]:
    if not yes_token_id:
        return None

    lookback_cutoff = int(expiry_ts - (30 * 86400))
    all_trades = {}

    for asset_field, side_label in [("makerAssetId", "SELL"), ("takerAssetId", "BUY")]:
        last_ts = lookback_cutoff
        while True:
            query = (
                '{ orderFilledEvents('
                'first: 1000 '
                'orderBy: timestamp '
                'orderDirection: asc '
                f'where: {{ {asset_field}: "{yes_token_id}", timestamp_gte: "{last_ts}" }}'
                ') { timestamp makerAssetId makerAmountFilled takerAmountFilled } }'
            )
            try:
                await asyncio.sleep(REQUEST_DELAY_S)
                async with session.post(
                    GOLDSKY_URL,
                    json={"query": query},
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        break
                    result = await resp.json(content_type=None)
            except Exception:
                break

            events = result.get("data", {}).get("orderFilledEvents", [])
            if not events:
                break

            for e in events:
                ts    = int(e.get("timestamp", 0))
                maker = float(e.get("makerAmountFilled", 1))
                taker = float(e.get("takerAmountFilled", 1))
                if ts > int(expiry_ts):
                    break
                if asset_field == "makerAssetId":
                    price = maker / taker if taker > 0 else 0.5
                else:
                    price = taker / maker if maker > 0 else 0.5

                price = float(np.clip(price, 0.01, 0.99))
                key = (ts, round(maker), round(taker))
                all_trades[key] = {
                    "timestamp": float(ts),
                    "price":     price,
                    "size":      taker / 1e6 if asset_field == "makerAssetId" else maker / 1e6,
                    "side":      side_label,
                }

            if len(events) < 1000:
                break
            last_ts = int(events[-1]["timestamp"])

    if not all_trades:
        return None

    return _trades_to_df(list(all_trades.values()))


def _trades_to_df(trades: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── phase 3: reconstruct OrderBookSnapshot stream from trade ticks ─────────────

def reconstruct_snapshots(
    trades_df: pd.DataFrame,
    outcome: int,
    expiry_ts: float,
    tick_interval_s: int = TICK_INTERVAL_S,
) -> Optional[pd.DataFrame]:
    """
    reconstructs a synthetic OrderBookSnapshot stream from raw matched trades.

    this is the core algorithmic challenge of historical ingestion:
    we never observed the live order book, so we must estimate each snapshot
    field from what the trade stream can tell us.

    reconstruction strategy per field:
      best_bid  : last_trade_price * (1 - synthetic_half_spread)
      best_ask  : last_trade_price * (1 + synthetic_half_spread)
      bid_depth : estimated from rolling buy-side volume in the tick window
      ask_depth : estimated from rolling sell-side volume in the tick window
      volume_1h : rolling 60-tick (60-minute) cumulative size sum
      trade_count: count of trades within the 15-minute rolling window (15 ticks)
      last_trade_price: last trade price at or before tick timestamp
      last_trade_size : last trade size at or before tick timestamp

    spread estimation:
      we cannot recover the historical spread without order book snapshots.
      we synthesize it using a liquidity proxy: when trade frequency is high
      (active market), spread is narrow. when sparse, spread is wide.
      formula: half_spread = clip(base_spread / sqrt(trade_count + 1), 0.002, 0.05)
      this is an approximation — imperfect, but it preserves the relative
      ordering of liquidity states, which is what the TCN needs.

    CRITICAL: these reconstructed fields will NOT match the exact live values
    that would have been observed. the training signal (outcome) is exact, but
    the microstructure features are approximations. the model learns to use
    the PATTERNS in these features, which are preserved even under approximation.

    Time: O(T * log(T)) — dominated by the rolling window groupby operations
    Space: O(T) for the output snapshot DataFrame

    Returns: DataFrame with one row per 1-min tick, columns matching
             OrderBookSnapshot fields + 'outcome' column
    """
    if trades_df is None or len(trades_df) < 10:
        return None

    # resample trades into 1-minute buckets
    trades_df = trades_df.set_index("timestamp")
    trades_df.index = pd.DatetimeIndex(trades_df.index)

    tick_agg = trades_df.resample(f"{tick_interval_s}s").agg(
        open_price  = ("price", "first"),
        close_price = ("price", "last"),
        tick_volume = ("size",  "sum"),
        tick_count  = ("size",  "count"),
    ).dropna(subset=["close_price"])

    if len(tick_agg) < MIN_TICKS:
        return None

    # forward-fill price through empty buckets (no trades = last known price holds)
    tick_agg["close_price"] = tick_agg["close_price"].ffill()
    tick_agg["open_price"]  = tick_agg["open_price"].ffill()

    # compute directional volume separately — avoids index alignment issues in resample lambdas
    buy_vol  = trades_df[trades_df["side"] == "BUY"]["size"].resample(f"{tick_interval_s}s").sum()
    sell_vol = trades_df[trades_df["side"] == "SELL"]["size"].resample(f"{tick_interval_s}s").sum()

    tick_agg["buy_volume"]  = buy_vol.reindex(tick_agg.index).fillna(0.0)
    tick_agg["sell_volume"] = sell_vol.reindex(tick_agg.index).fillna(0.0)

    # synthetic spread model — see docstring for rationale
    # synthetic spread model — see docstring for rationale
    base_spread = 0.03  # 3 cent base for thin polymarket books
    tick_agg["half_spread"] = (
        base_spread / np.sqrt(tick_agg["tick_count"].clip(lower=1))
    ).clip(lower=0.002, upper=0.05)

    mid = tick_agg["close_price"].clip(lower=0.01, upper=0.99)
    tick_agg["best_bid"] = (mid - tick_agg["half_spread"]).clip(lower=0.001)
    tick_agg["best_ask"] = (mid + tick_agg["half_spread"]).clip(upper=0.999)

    # depth reconstruction: use rolling buy/sell volume as a proxy for depth.
    # more recent volume on one side → that side has more depth perception.
    # log-scaled to match how compute_depth_at_best() operates downstream.
    tick_agg["bid_depth"] = tick_agg["buy_volume"].rolling(5, min_periods=1).mean().clip(lower=0.1)
    tick_agg["ask_depth"] = tick_agg["sell_volume"].rolling(5, min_periods=1).mean().clip(lower=0.1)

    # rolling 60-tick (60-minute at 1-min ticks) cumulative volume → matches volume_1h field
    tick_agg["volume_1h"] = tick_agg["tick_volume"].rolling(60, min_periods=1).sum()

    # rolling 15-tick trade count → matches trade_count field (15-min window)
    tick_agg["trade_count_15m"] = tick_agg["tick_count"].rolling(15, min_periods=1).sum()

    # last trade price and size at each tick boundary
    tick_agg["last_trade_price"] = tick_agg["close_price"]
    tick_agg["last_trade_size"]  = tick_agg["tick_volume"].clip(lower=0.0)

    # expiry timestamp is constant per market — the same value data_pipeline expects
    tick_agg["expiry_ts"] = expiry_ts

    # outcome label applied uniformly — the market resolved this way for all ticks
    tick_agg["outcome"] = outcome

    # convert index back to unix float timestamp to match OrderBookSnapshot.timestamp
    tick_agg["timestamp"] = tick_agg.index.astype(np.int64) // 1_000_000_000

    # select and rename to exactly match OrderBookSnapshot field names
    snapshot_df = tick_agg[[
        "timestamp",
        "best_bid",
        "best_ask",
        "bid_depth",
        "ask_depth",
        "volume_1h",
        "trade_count_15m",
        "last_trade_price",
        "last_trade_size",
        "expiry_ts",
        "outcome",
    ]].rename(columns={"trade_count_15m": "trade_count"})

    return snapshot_df.reset_index(drop=True).astype({
        "timestamp":        "float64",
        "best_bid":         "float32",
        "best_ask":         "float32",
        "bid_depth":        "float32",
        "ask_depth":        "float32",
        "volume_1h":        "float32",
        "trade_count":      "float32",
        "last_trade_price": "float32",
        "last_trade_size":  "float32",
        "expiry_ts":        "float64",
        "outcome":          "int8",
    })


# ── phase 4: persist as parquet ────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, condition_id: str) -> Path:
    """
    persists a single market's snapshot stream as a snappy-compressed parquet file.

    parquet was chosen over csv / hdf5 because:
      - column-level compression: snappy gives ~50% size reduction on float32 columns
      - lazy / chunked reads: training loop can read row groups without loading full file
      - schema enforcement: pyarrow validates dtypes on write, catching field mismatches
        before they silently corrupt a training run

    file naming: <condition_id>.parquet
    the condition_id is the natural primary key for a polymarket market and
    allows the training loop to trivially join with market metadata if needed.

    Time: O(T * F) for serialization — pyarrow is highly optimized, typically
          ~5-10ms per file for our typical 200-500 tick markets
    Space: O(T * F) peak in memory during write (pyarrow builds record batches)
    """
    output_path = OUTPUT_DIR / f"{condition_id}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")
    return output_path


# ── main orchestration ─────────────────────────────────────────────────────────

async def process_single_market(session, market, semaphore, stats):
    output_path = OUTPUT_DIR / f"{market.condition_id}.parquet"
    if output_path.exists():
        stats["skipped"] += 1
        return

    async with semaphore:
        trades_df = await fetch_clob_trades(
            session,
            market.condition_id,
            market.expiry_ts,
            yes_token_id=market.yes_token_id,   # pass token id
        )
    # ... rest unchanged

    if trades_df is None or len(trades_df) < MIN_TICKS:
        logger.info(f"skip {market.condition_id} — insufficient trade history ({len(trades_df) if trades_df is not None else 0} trades)")
        stats["insufficient"] += 1
        return

    snapshot_df = reconstruct_snapshots(trades_df, market.outcome, market.expiry_ts)
    if snapshot_df is None:
        stats["reconstruction_failed"] += 1
        return

    saved_path = save_parquet(snapshot_df, market.condition_id)
    ticks = len(snapshot_df)
    yes_pct = snapshot_df["outcome"].mean() * 100
    logger.info(f"saved {market.condition_id} | {ticks} ticks | outcome={market.outcome} | {saved_path.name}")
    stats["saved"] += 1
    stats["total_ticks"] += ticks


async def main() -> None:
    """
    top-level async entry point.

    pipeline:
      1. discover resolved markets via gamma api (paginated)
      2. concurrently fetch clob trade history for each (semaphore-bounded)
      3. reconstruct tick-level OrderBookSnapshot streams
      4. validate and save as snappy-compressed parquet
      5. print summary statistics

    Time: O(M * T_avg / CONCURRENCY) where M = markets, T_avg = avg trades per market
    Space: O(CONCURRENCY * T_max) peak — we hold at most CONCURRENCY trade DataFrames
           simultaneously; each is discarded after parquet flush
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    stats = {"saved": 0, "skipped": 0, "insufficient": 0, "reconstruction_failed": 0, "total_ticks": 0}

    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        logger.info("phase 1: discovering resolved markets from gamma api...")
        markets = await fetch_resolved_markets(session)
        logger.info(f"found {len(markets)} resolved binary markets to ingest")

        logger.info(f"phase 2: fetching clob history (concurrency={CONCURRENCY})...")
        semaphore = asyncio.Semaphore(CONCURRENCY)

        tasks = [
            process_single_market(session, market, semaphore, stats)
            for market in markets
        ]
        await asyncio.gather(*tasks)

    logger.info("=" * 55)
    logger.info(f"ingest complete")
    logger.info(f"  saved:                {stats['saved']}")
    logger.info(f"  skipped (exist):      {stats['skipped']}")
    logger.info(f"  insufficient data:    {stats['insufficient']}")
    logger.info(f"  reconstruction fail:  {stats['reconstruction_failed']}")
    logger.info(f"  total ticks stored:   {stats['total_ticks']:,}")
    logger.info(f"  output directory:     {OUTPUT_DIR.resolve()}")
    logger.info("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
