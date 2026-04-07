"""
scan_agent.py  —  PYTHON SCRIPT (i/o bound, deterministic market discovery)

architecture note: purely deterministic i/o and filtering. no llm calls,
no fuzzy logic. discovers active polymarket markets that pass qualification
filters and returns MarketSnapshot objects with real order book depth attached.

pipeline per scan cycle:
  gamma /markets (active, paginated)
    → price filter    [0.15, 0.85]        — exclude near-resolved edges
    → volume filter   > $5,000 24h        — ensure meaningful tape exists
    → expiry filter   > 24h from now      — avoid flash/settlement markets
    → category filter politics | crypto   — highest order-book pressure signature
    → clob /book?token_id=<yes_token_id>  (concurrent, semaphore-bounded)
        → depth filter > $1,000 within 5% of mid
        → BookSnapshot construction (top-3 bid/ask levels)
    → MarketSnapshot returned with .book populated

kalshi: KalshiScanner stub kept for interface compatibility with orchestrator.
        wire in after polymarket pipeline is validated end-to-end.

train/serve alignment:
  BookSnapshot fields feed into _build_ob_snapshot() in orchestrator.py,
  which constructs an OrderBookSnapshot for MarketDataPipeline.ingest().
  field semantics must match data_pipeline.py expectations exactly:
    best_bid / best_ask   →  spread_width, implied_probability
    bid_depth / ask_depth →  order_book_imbalance, depth_at_best
    volume_1h             →  compute_volume_log (log1p scaled downstream)
    last_trade_price      →  vwap_deviation numerator
    trade_count_15m       →  compute_trade_intensity (0.0 until /trades added)
    last_trade_size       →  FeatureBuffer.trade_size_history (0.0 until /trades added)

  volume_1h is approximated as volume_24h / 24 from gamma metadata.
  trade_count_15m and last_trade_size remain 0.0 — they degrade features
  [6] and [9] but do not corrupt other features. upgrade path: add a
  /trades fetch per market after /book if brier scores show those features
  are load-bearing in the compound skill audit.
"""

import asyncio
import json as _json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# ── api endpoints ──────────────────────────────────────────────────────────────

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# ── scan filter constants ──────────────────────────────────────────────────────

PRICE_MIN              = 0.15   # exclude near-NO edge — illiquid, TCN edge cases
PRICE_MAX              = 0.85   # exclude near-YES edge — near-resolved, no edge left
MIN_VOLUME_24H         = 5_000  # $5k minimum daily volume — ensures real tape exists
MIN_EXPIRY_HOURS       = 24     # skip markets resolving within 24h
DEPTH_FILTER_MIN_USD   = 1_000  # $1k depth within 5% of mid — confirms tradeable liquidity
BOOK_FETCH_CONCURRENCY = 10     # max concurrent /book requests — stays under clob rate limit
REQUEST_DELAY_S        = 0.3    # floor delay per gamma page request
GAMMA_PAGE_SIZE        = 100
MAX_MARKETS_PER_CYCLE  = 200

# categories with consistent order-book pressure signatures
# gamma uses lowercase tag strings — match exactly
ALLOWED_TAGS = {"politics", "crypto", "political", "cryptocurrency", "elections", "election"}


# ── data contracts ─────────────────────────────────────────────────────────────

@dataclass
class BookSnapshot:
    """
    real order book data from clob /book endpoint.

    best_bid / best_ask: top-of-book prices.
    bid_depth / ask_depth: sum of top-3 resting size levels in USD.
      top-3 avoids counting dead liquidity at $0.01 far off-market.
    volume_1h: approximated as volume_24h / 24 from gamma metadata.
    last_trade_price: yes_price midpoint proxy until /trades is added.
    trade_count_15m: unavailable from /book — 0.0 stub, degrades feature [6].
    last_trade_size: unavailable from /book — 0.0 stub, degrades feature [9].
    """
    best_bid:         float
    best_ask:         float
    bid_depth:        float
    ask_depth:        float
    volume_1h:        float = 0.0
    last_trade_price: float = 0.0
    last_trade_size:  float = 0.0
    trade_count_15m:  float = 0.0


@dataclass
class MarketSnapshot:
    """
    qualified market opportunity, ready for MarketDataPipeline.ingest().
    .book is None only if /book fetch failed — orchestrator falls back to
    the approximation path in _build_ob_snapshot() in that case.
    """
    market_id:    str
    platform:     str
    question:     str
    yes_price:    float
    expiry_ts:    float = 0.0
    yes_token_id: Optional[str] = None
    book:         Optional[BookSnapshot] = None


# ── http helper ────────────────────────────────────────────────────────────────

async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
    retries: int = 3,
) -> Optional[dict | list]:
    """
    thin async GET with exponential-backoff retry.
    mirrors historical_ingest._get_json exactly — same apis, same failure modes.
    does not retry on 404 (data genuinely absent) or 400 (bad params).
    """
    for attempt in range(retries):
        try:
            await asyncio.sleep(REQUEST_DELAY_S)
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 429:
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
            logger.warning(f"request failed ({exc}) — retry {attempt+1}/{retries} in {wait}s")
            await asyncio.sleep(wait)
    logger.error(f"all retries exhausted for {url}")
    return None


# ── qualification filters ──────────────────────────────────────────────────────

def _passes_price_filter(yes_price: float) -> bool:
    return PRICE_MIN <= yes_price <= PRICE_MAX


def _passes_volume_filter(volume_24h: float) -> bool:
    return volume_24h >= MIN_VOLUME_24H


def _passes_expiry_filter(expiry_ts: float) -> bool:
    return (expiry_ts - time.time()) / 3600.0 >= MIN_EXPIRY_HOURS


def _passes_category_filter(tags: list) -> bool:
    if not tags:
        return False
    return any(t.lower() in ALLOWED_TAGS for t in tags)


def _depth_within_pct(bids: list, asks: list, yes_price: float, pct: float = 0.05) -> float:
    """
    sums USD depth on both sides within `pct` of the mid price.
    used to confirm $1k of tradeable liquidity exists before including a market.
    levels outside the band are resting far-off orders that won't fill at market.
    """
    mid   = yes_price
    lo    = mid * (1 - pct)
    hi    = mid * (1 + pct)
    depth = 0.0
    for level in bids:
        try:
            p, s = float(level["price"]), float(level["size"])
            if p >= lo:
                depth += p * s
        except (KeyError, ValueError):
            continue
    for level in asks:
        try:
            p, s = float(level["price"]), float(level["size"])
            if p <= hi:
                depth += p * s
        except (KeyError, ValueError):
            continue
    return depth


# ── polymarket scanner ─────────────────────────────────────────────────────────

class PolymarketScanner:
    """
    two-phase scanner:
      phase 1 — gamma /markets: paginated fetch + qualification filters
      phase 2 — clob /book:     concurrent enrichment, semaphore-bounded

    gamma is the only source for: question text, condition_id, yes_token_id,
    volume_24h, expiry date, category tags.
    clob /book is the only source for: best_bid, best_ask, resting depth.
    """

    def __init__(self, session: aiohttp.ClientSession):
        self._session = session

    def _parse_market(self, m: dict) -> Optional[tuple["MarketSnapshot", float]]:
        """
        parses a single gamma /markets entry into (MarketSnapshot, volume_24h).
        mirrors historical_ingest._parse_market field-for-field to avoid
        introducing new parsing divergence between ingest and live paths.
        returns None on any parse failure.
        """
        # yes price from outcomePrices[0]
        try:
            raw_prices = m.get("outcomePrices")
            if not raw_prices:
                return None
            prices    = _json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            yes_price = float(prices[0])
        except (ValueError, IndexError, TypeError):
            return None

        # yes_token_id from clobTokenIds[0]
        raw_token_ids = m.get("clobTokenIds")
        yes_token_id  = None
        if raw_token_ids:
            try:
                token_ids    = _json.loads(raw_token_ids) if isinstance(raw_token_ids, str) else raw_token_ids
                yes_token_id = str(token_ids[0]) if token_ids else None
            except Exception:
                pass

        # expiry timestamp
        expiry_str = m.get("endDate") or m.get("endDateIso")
        if not expiry_str:
            return None
        try:
            expiry_ts = pd.Timestamp(expiry_str).timestamp()
        except Exception:
            return None

        condition_id = m.get("conditionId") or m.get("id")
        if not condition_id:
            return None

        volume_24h = float(m.get("volume24hr") or m.get("volume") or 0)

        snap = MarketSnapshot(
            market_id    = str(condition_id),
            platform     = "polymarket",
            question     = m.get("question", ""),
            yes_price    = yes_price,
            expiry_ts    = expiry_ts,
            yes_token_id = yes_token_id,
        )
        return snap, volume_24h

    async def fetch_qualified(self) -> List[tuple["MarketSnapshot", float]]:
        """
        paginates gamma /markets and applies all non-depth filters.
        returns list of (MarketSnapshot, volume_24h) pairs.
        """
        qualified: List[tuple[MarketSnapshot, float]] = []
        offset = 0
        pages_scanned = 0
        MAX_PAGES = 5  # circuit breaker

        while len(qualified) < MAX_MARKETS_PER_CYCLE and pages_scanned < MAX_PAGES:
            data = await _get_json(
                self._session,
                f"{GAMMA_BASE}/markets",
                params={
                    "active": "true", 
                    "closed": "false",
                    "limit": GAMMA_PAGE_SIZE, 
                    "offset": offset,
                    "order": "volume24hr",  # NO UNDERSCORE. this caused the 422.
                },
            )
            if not data:
                break

            markets = data if isinstance(data, list) else data.get("markets", [])
            if not markets:
                break

            for m in markets:
                result = self._parse_market(m)
                if result is None:
                    continue
                snap, vol24h = result

                if not _passes_price_filter(snap.yes_price):
                    continue
                if not _passes_volume_filter(vol24h):
                    continue
                if not _passes_expiry_filter(snap.expiry_ts):
                    continue

                # parse category tags — may be list of strings or list of dicts
                raw_tags = m.get("tags") or []
                if isinstance(raw_tags, str):
                    try:
                        raw_tags = _json.loads(raw_tags)
                    except Exception:
                        raw_tags = []
                tag_strings = []
                for t in raw_tags:
                    if isinstance(t, str):
                        tag_strings.append(t)
                    elif isinstance(t, dict):
                        tag_strings.append(t.get("label", "") or t.get("slug", ""))
                if not _passes_category_filter(tag_strings):
                    continue

                qualified.append((snap, vol24h))

            logger.debug(f"[SCAN] gamma offset={offset} — {len(qualified)} qualified")
            offset += GAMMA_PAGE_SIZE
            pages_scanned += 1
            if len(markets) < GAMMA_PAGE_SIZE:
                break

        logger.info(f"[SCAN] polymarket: {len(qualified)} passed pre-book filters")
        return qualified[:MAX_MARKETS_PER_CYCLE]

    async def _fetch_book_one(
        self,
        snap: MarketSnapshot,
        vol24h: float,
        semaphore: asyncio.Semaphore,
    ) -> Optional[BookSnapshot]:
        """
        fetches clob /book for one market and applies depth filter.
        returns None if /book fails or depth < DEPTH_FILTER_MIN_USD.
        semaphore must be shared across all concurrent calls.
        """
        if not snap.yes_token_id:
            return None

        async with semaphore:
            data = await _get_json(
                self._session,
                f"{CLOB_BASE}/book",
                params={"token_id": snap.yes_token_id},
            )

        if not data:
            return None

        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            return None

        # depth filter — must have $1k within 5% of mid to be tradeable
        depth = _depth_within_pct(bids, asks, snap.yes_price, pct=0.05)
        if depth < DEPTH_FILTER_MIN_USD:
            logger.debug(
                f"[BOOK] {snap.market_id} depth=${depth:.0f} "
                f"below ${DEPTH_FILTER_MIN_USD} — dropped"
            )
            return None

        try:
            best_bid  = float(bids[0]["price"])
            best_ask  = float(asks[0]["price"])
            bid_depth = sum(float(b["size"]) for b in bids[:3])
            ask_depth = sum(float(a["size"]) for a in asks[:3])
        except (KeyError, ValueError, IndexError) as exc:
            logger.warning(f"[BOOK] parse error {snap.market_id}: {exc}")
            return None

        return BookSnapshot(
            best_bid         = best_bid,
            best_ask         = best_ask,
            bid_depth        = bid_depth,
            ask_depth        = ask_depth,
            volume_1h        = vol24h / 24.0,   # hourly proxy from gamma 24h volume
            last_trade_price = snap.yes_price,  # midpoint proxy until /trades added
        )

    async def enrich_with_books(
        self,
        pairs: List[tuple["MarketSnapshot", float]],
    ) -> List[MarketSnapshot]:
        """
        fires _fetch_book_one() concurrently for all qualified snapshots.
        markets that fail depth filter or /book fetch are dropped entirely —
        a market without confirmed liquidity should not enter the pipeline.

        returns only the snapshots that passed both /book fetch and depth filter,
        with .book populated and ready for MarketDataPipeline.ingest().
        """
        semaphore = asyncio.Semaphore(BOOK_FETCH_CONCURRENCY)
        books = await asyncio.gather(
            *[self._fetch_book_one(s, v, semaphore) for s, v in pairs],
            return_exceptions=True,
        )

        enriched: List[MarketSnapshot] = []
        for (snap, _), book in zip(pairs, books):
            if isinstance(book, Exception):
                logger.warning(f"[BOOK] exception for {snap.market_id}: {book}")
                continue
            if book is None:
                continue
            snap.book = book
            enriched.append(snap)

        logger.info(
            f"[SCAN] polymarket: {len(enriched)}/{len(pairs)} "
            f"passed depth filter and book enrichment"
        )
        return enriched


# ── kalshi scanner (stub) ─────────────────────────────────────────────────────

class KalshiScanner:
    """
    interface stub. wire in after polymarket pipeline is validated.
    kalshi uses cent-denominated prices, requires api key auth on all
    endpoints, and has a different book structure from polymarket clob.
    """

    def __init__(self, session: aiohttp.ClientSession, api_key: str):
        self._session = session
        self._api_key = api_key

    async def fetch_markets(self) -> List[MarketSnapshot]:
        return []


# ── scan cycle entry point ─────────────────────────────────────────────────────

async def run_scan_cycle(
    poly_scanner: PolymarketScanner,
    kalshi_scanner: KalshiScanner,
) -> List[MarketSnapshot]:
    """
    single scan cycle called by orchestrator on SCAN_INTERVAL_SECONDS timer.

    runs gamma fetch and kalshi stub concurrently, then enriches polymarket
    snapshots with real /book depth. total latency = max(gamma_fetch, kalshi_stub)
    + book_enrichment_latency (bounded by BOOK_FETCH_CONCURRENCY semaphore).

    the [WAITING_FOR_WARMUP] state for new markets is handled downstream in
    _run_tcn_inference() — it returns (0.5, 0.1) until 64 ticks are buffered.
    scan_agent's job is only to surface qualified markets; warmup tracking
    lives in the per-market MarketDataPipeline instances in orchestrator.main().
    """
    poly_pairs, kalshi_markets = await asyncio.gather(
        poly_scanner.fetch_qualified(),
        kalshi_scanner.fetch_markets(),
    )

    poly_markets = await poly_scanner.enrich_with_books(poly_pairs)
    all_markets  = poly_markets + kalshi_markets

    logger.info(
        f"[SCAN] cycle complete — {len(all_markets)} qualified markets "
        f"({len(poly_markets)} polymarket, {len(kalshi_markets)} kalshi)"
    )
    return all_markets


# ── standalone smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [SCAN] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    async def _test():
        async with aiohttp.ClientSession() as session:
            poly   = PolymarketScanner(session)
            kalshi = KalshiScanner(session, os.environ.get("KALSHI_API_KEY", ""))
            markets = await run_scan_cycle(poly, kalshi)

            print(f"\n{'='*70}")
            print(f"total qualified markets: {len(markets)}")
            print(f"{'='*70}")
            for m in markets[:15]:
                if m.book:
                    book_str = (
                        f"bid={m.book.best_bid:.3f} ask={m.book.best_ask:.3f} "
                        f"b_depth={m.book.bid_depth:.0f} a_depth={m.book.ask_depth:.0f} "
                        f"vol1h=${m.book.volume_1h:.0f}"
                    )
                else:
                    book_str = "no book (fallback)"
                print(f"  p={m.yes_price:.3f} | {m.question[:50]:<50} | {book_str}")

    asyncio.run(_test())