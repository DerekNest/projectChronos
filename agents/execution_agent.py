"""
agents/execution_agent.py  —  PYTHON SCRIPT (async i/o, deterministic routing)

architecture note: no llm calls, no fuzzy logic. routes order requests to either
the paper engine (local simulation) or the live polymarket clob engine based on
the LIVE_MODE environment variable. the orchestrator never calls both — it calls
get_execution_engine() once per operation and gets back whichever engine is active.

paper mode:
  simulates fills with realistic slippage. fills are immediate (no resting order
  queue) at limit_price + a small synthetic spread cost. paper positions are tracked
  in-memory only — they vanish on restart, which is correct: paper trading is for
  pipeline validation, not persistence.

  slippage model: gaussian noise centered at 0 with sigma = PAPER_SLIPPAGE_SIGMA.
  clipped to +/-PAPER_MAX_SLIPPAGE so no fill deviates more than 1.5% from limit.

live mode (polymarket clob):
  posts limit orders via the clob REST api using L1 authentication (ecdsa signature
  over the order hash). polymarket's clob requires:
    - L1_ADDRESS env var: your externally-owned account (eoa) address
    - L1_PRIVATE_KEY env var: hex private key for signing order hashes
    - orders are signed off-chain and submitted as json — no on-chain tx for entry
    - settlement IS on-chain (gnosis chain), handled by polymarket, not us

  order lifecycle on clob:
    POST /order                    -> order accepted, returns order_id
    GET  /order/{order_id}         -> poll until status = MATCHED or CANCELLED
    MATCHED with size > 0          -> filled (partial or full)

  cancellation: if the order hasn't filled within LIVE_ORDER_TIMEOUT_S seconds
  we cancel it and return EXPIRED. limit-only — we never chase with market orders.

  NOTE: _sign_order() is a stub. wire in py-clob-client before live trading:
    pip install py-clob-client

env vars consumed:
  LIVE_MODE          "true" / "false" (default: "false" -> paper mode)
  L1_ADDRESS         eoa address for clob authentication (live only)
  L1_PRIVATE_KEY     hex private key for order signing (live only)
  CLOB_API_KEY       optional: api key if using clob key-based auth tier
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── mode flag ──────────────────────────────────────────────────────────────────

LIVE_MODE: bool = os.environ.get("LIVE_MODE", "false").lower() == "true"

# ── api endpoints ──────────────────────────────────────────────────────────────

CLOB_BASE = "https://clob.polymarket.com"

# ── execution parameters ───────────────────────────────────────────────────────

PAPER_SLIPPAGE_SIGMA  = 0.003   # gaussian sigma for synthetic fill slippage (~0.3%)
PAPER_MAX_SLIPPAGE    = 0.015   # hard clip: no paper fill deviates > 1.5% from limit
LIVE_ORDER_TIMEOUT_S  = 30      # seconds to wait for a clob fill before cancelling
LIVE_POLL_INTERVAL_S  = 2       # clob order status poll frequency
LIVE_MAX_RETRIES      = 3       # retries for transient api failures


# ── data contracts ─────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    FILLED   = "FILLED"
    PARTIAL  = "PARTIAL"    # partially filled — orchestrator treats as valid
    EXPIRED  = "EXPIRED"    # timed out or cancelled without fill
    REJECTED = "REJECTED"   # clob rejected the order (bad params, insufficient funds)
    ERROR    = "ERROR"      # unexpected exception during submission


@dataclass
class OrderRequest:
    """
    submitted by orchestrator to engine.submit_limit_order().
    size_usd is the dollar amount to deploy — engine converts to contract
    quantity using limit_price (contracts = size_usd / limit_price).
    """
    market_id:    str
    platform:     str
    side:         OrderSide
    size_usd:     float      # USD notional to deploy
    limit_price:  float      # price in [0, 1] for YES contract
    signal_price: float      # price at predict-skill signal time (slippage ref)


@dataclass
class OrderResult:
    """
    returned by engine.submit_limit_order() and engine.close_position().
    fill_size_usd may be < request.size_usd on partial fills.
    paper_mode flag lets the orchestrator tag log lines correctly.
    """
    order_id:       str
    status:         OrderStatus
    fill_price:     float = 0.0
    fill_size_usd:  float = 0.0
    paper_mode:     bool  = True
    error_msg:      str   = ""


@dataclass
class Position:
    """
    open position tracked in orchestrator.open_positions dict.
    entry_order holds the OrderResult from submit_limit_order().
    opened_at is unix timestamp — used to compute holding period.
    """
    position_id:    str
    market_id:      str
    platform:       str
    entry_order:    OrderResult
    entry_price:    float
    size_usd:       float
    predicted_prob: float
    opened_at:      float   # unix timestamp


# ── paper execution engine ─────────────────────────────────────────────────────

class PaperExecutionEngine:
    """
    simulates order fills for pipeline validation.
    all fills are immediate with synthetic gaussian slippage.
    no external calls, no state persistence.
    """

    async def submit_limit_order(self, req: OrderRequest) -> OrderResult:
        """
        simulates a limit order fill with synthetic slippage.
        always returns FILLED — paper mode never rejects or times out.
        buy slippage: price creeps up slightly (pays spread).
        sell slippage: price creeps down slightly (receives less).
        """
        await asyncio.sleep(0.01)   # yield to event loop

        import random
        noise     = random.gauss(0, PAPER_SLIPPAGE_SIGMA)
        noise     = max(-PAPER_MAX_SLIPPAGE, min(PAPER_MAX_SLIPPAGE, noise))
        direction = 1.0 if req.side == OrderSide.BUY else -1.0
        fill_price = max(0.01, min(0.99, req.limit_price + direction * abs(noise)))

        # contracts bought = usd / entry_limit; notional at fill = contracts * fill_price
        contracts     = req.size_usd / max(req.limit_price, 1e-6)
        fill_size_usd = contracts * fill_price

        order_id = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
        logger.info(
            f"[PAPER] filled {req.market_id} {req.side.value} "
            f"${fill_size_usd:.2f} @ {fill_price:.4f} "
            f"(slippage {fill_price - req.limit_price:+.4f})"
        )

        return OrderResult(
            order_id      = order_id,
            status        = OrderStatus.FILLED,
            fill_price    = fill_price,
            fill_size_usd = fill_size_usd,
            paper_mode    = True,
        )

    async def close_position(self, position: Position, exit_price: float) -> OrderResult:
        """
        simulates position close at the resolution price.
        binary markets resolve to ~1.0 (YES) or ~0.0 (NO) — no fill slippage
        needed here since resolution is deterministic, not a market order.
        """
        await asyncio.sleep(0.01)
        close_size = position.size_usd * (exit_price / max(position.entry_price, 1e-6))
        order_id   = f"PAPER-CLOSE-{uuid.uuid4().hex[:8].upper()}"

        logger.info(
            f"[PAPER] closed {position.market_id} @ {exit_price:.4f} "
            f"entry={position.entry_price:.4f} "
            f"pnl={(exit_price - position.entry_price) / max(position.entry_price, 1e-9):+.4f}"
        )

        return OrderResult(
            order_id      = order_id,
            status        = OrderStatus.FILLED,
            fill_price    = exit_price,
            fill_size_usd = close_size,
            paper_mode    = True,
        )


# ── live execution engine (polymarket clob) ────────────────────────────────────

class LivePolymarketEngine:
    """
    submits signed limit orders to the polymarket clob api.

    IMPORTANT: _sign_order() is currently a stub. the clob will reject any
    order submitted with the placeholder signature. wire in py-clob-client
    before going live:

      pip install py-clob-client

      from py_clob_client.client import ClobClient
      from py_clob_client.clob_types import OrderArgs
      client = ClobClient(host=CLOB_BASE, key=L1_PRIVATE_KEY, chain_id=137)
      signed_order = client.create_order(OrderArgs(
          token_id=market_id, price=limit_price, size=contracts, side="BUY"
      ))

    required env vars:
      L1_ADDRESS      your eoa address (0x...)
      L1_PRIVATE_KEY  hex private key (with or without 0x prefix)
    """

    def __init__(self, session: aiohttp.ClientSession):
        self._session  = session
        self._address  = os.environ.get("L1_ADDRESS", "")
        self._priv_key = os.environ.get("L1_PRIVATE_KEY", "")
        self._api_key  = os.environ.get("CLOB_API_KEY", "")

        if not self._address or not self._priv_key:
            logger.error(
                "[LIVE] L1_ADDRESS and L1_PRIVATE_KEY env vars are required for live mode."
            )

    def _sign_order(self, order_params: dict) -> str:
        """
        STUB — returns placeholder signature. wire in py-clob-client before live use.
        the clob will reject any order bearing this signature.
        """
        logger.warning(
            "[LIVE] _sign_order() is a stub. orders will be rejected until "
            "py-clob-client signing is wired in."
        )
        return "0x_STUB_SIGNATURE"

    def _build_order_payload(self, req: OrderRequest) -> dict:
        """
        constructs the json body for POST /order.
        amounts are in 1e6 units (usdc has 6 decimals on polygon/gnosis).
        """
        contracts    = req.size_usd / max(req.limit_price, 1e-6)
        maker_amount = int(req.size_usd * 1e6)
        taker_amount = int(contracts * 1e6)
        salt         = int(time.time() * 1000)
        expiry       = int(time.time()) + 300      # 5-minute order lifetime

        order_params = {
            "maker":        self._address,
            "taker":        "0x0000000000000000000000000000000000000000",
            "makerAmount":  str(maker_amount),
            "takerAmount":  str(taker_amount),
            "makerAssetId": "0",             # usdc on polymarket
            "takerAssetId": req.market_id,   # YES token id
            "side":         req.side.value,
            "salt":         str(salt),
            "expiry":       str(expiry),
        }
        return {**order_params, "signature": self._sign_order(order_params)}

    async def _post_json(self, url: str, payload: dict) -> Optional[dict]:
        headers = {
            "Content-Type": "application/json",
            "POLY_ADDRESS":  self._address,
        }
        if self._api_key:
            headers["POLY_API_KEY"] = self._api_key

        for attempt in range(LIVE_MAX_RETRIES):
            try:
                async with self._session.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(float(resp.headers.get("Retry-After", 2 ** attempt * 3)))
                        continue
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.warning(f"[LIVE] POST {url} -> {resp.status}: {body[:200]}")
                        return None
                    return await resp.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(f"[LIVE] POST error (attempt {attempt+1}): {exc}")
                await asyncio.sleep(2 ** attempt)
        return None

    async def _get_json(self, url: str) -> Optional[dict]:
        headers = {"POLY_ADDRESS": self._address}
        if self._api_key:
            headers["POLY_API_KEY"] = self._api_key
        try:
            async with self._session.get(
                url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status >= 400:
                    return None
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

    async def _poll_until_filled(self, order_id: str) -> tuple[OrderStatus, float, float]:
        """
        polls GET /order/{order_id} until matched, cancelled, or timeout.
        returns (status, fill_price, fill_size_usd).

        clob order status values:
          live      — resting on book
          matched   — fully matched
          cancelled — cancelled or expired
        """
        deadline = time.time() + LIVE_ORDER_TIMEOUT_S

        while time.time() < deadline:
            data = await self._get_json(f"{CLOB_BASE}/order/{order_id}")
            if data is None:
                await asyncio.sleep(LIVE_POLL_INTERVAL_S)
                continue

            status = data.get("status", "").lower()

            if status == "matched":
                # amounts in 1e6 units
                maker_filled = float(data.get("makerAmountFilled", 0)) / 1e6
                taker_filled = float(data.get("takerAmountFilled", 0)) / 1e6
                fill_price   = max(0.01, min(0.99, maker_filled / max(taker_filled, 1e-9)))
                maker_total  = float(data.get("makerAmount", maker_filled)) / 1e6
                is_partial   = maker_filled < maker_total * 0.999
                fill_status  = OrderStatus.PARTIAL if is_partial else OrderStatus.FILLED

                logger.info(
                    f"[LIVE] {order_id} {fill_status.value} — "
                    f"${maker_filled:.2f} @ {fill_price:.4f}"
                )
                return fill_status, fill_price, maker_filled

            elif status == "cancelled":
                logger.info(f"[LIVE] {order_id} cancelled by clob")
                return OrderStatus.EXPIRED, 0.0, 0.0

            await asyncio.sleep(LIVE_POLL_INTERVAL_S)

        # timeout — attempt cancellation
        logger.warning(f"[LIVE] {order_id} timed out after {LIVE_ORDER_TIMEOUT_S}s — cancelling")
        await self._post_json(f"{CLOB_BASE}/cancel", {"orderID": order_id})
        return OrderStatus.EXPIRED, 0.0, 0.0

    async def submit_limit_order(self, req: OrderRequest) -> OrderResult:
        order_id = f"LIVE-{uuid.uuid4().hex[:12].upper()}"
        logger.info(
            f"[LIVE] submitting {req.side.value} {req.market_id} "
            f"${req.size_usd:.2f} @ {req.limit_price:.4f}"
        )

        payload  = self._build_order_payload(req)
        response = await self._post_json(f"{CLOB_BASE}/order", payload)

        if response is None:
            return OrderResult(
                order_id=order_id, status=OrderStatus.REJECTED,
                paper_mode=False, error_msg="POST /order returned null",
            )

        clob_id = response.get("orderID") or response.get("id")
        if not clob_id:
            return OrderResult(
                order_id=order_id, status=OrderStatus.REJECTED,
                paper_mode=False,
                error_msg=f"no orderID in response: {str(response)[:200]}",
            )

        fill_status, fill_price, fill_size_usd = await self._poll_until_filled(clob_id)

        return OrderResult(
            order_id      = clob_id,
            status        = fill_status,
            fill_price    = fill_price,
            fill_size_usd = fill_size_usd,
            paper_mode    = False,
        )

    async def close_position(self, position: Position, exit_price: float) -> OrderResult:
        """
        submits a SELL limit order at exit_price for early exit.
        for resolved markets (exit_price ~= 0 or 1), polymarket handles
        redemption on-chain automatically — this call may time out, which is fine.
        """
        close_req = OrderRequest(
            market_id    = position.market_id,
            platform     = position.platform,
            side         = OrderSide.SELL,
            size_usd     = position.size_usd,
            limit_price  = exit_price,
            signal_price = exit_price,
        )
        return await self.submit_limit_order(close_req)


# ── factory ────────────────────────────────────────────────────────────────────

def get_execution_engine(
    platform: str,
    session: aiohttp.ClientSession,
) -> "PaperExecutionEngine | LivePolymarketEngine":
    """
    returns the correct engine. LIVE_MODE=False always returns paper — the only
    switch point, so there's no path to accidentally hit live apis in paper mode.
    unknown platforms fall back to paper regardless of LIVE_MODE.
    """
    if not LIVE_MODE:
        return PaperExecutionEngine()

    if platform == "polymarket":
        return LivePolymarketEngine(session)

    logger.warning(f"[EXEC] no live engine for '{platform}' — using paper")
    return PaperExecutionEngine()


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [EXEC] %(message)s",
    )

    async def run_tests():
        print("running execution_agent self-test (paper mode)...\n")

        engine = PaperExecutionEngine()

        # ── test 1: buy order fills ──
        req = OrderRequest(
            market_id="TEST-MKT-001", platform="polymarket",
            side=OrderSide.BUY, size_usd=100.0,
            limit_price=0.55, signal_price=0.55,
        )
        result = await engine.submit_limit_order(req)
        assert result.status == OrderStatus.FILLED
        assert result.paper_mode is True
        assert 0.01 <= result.fill_price <= 0.99
        assert abs(result.fill_size_usd - 100.0) / 100.0 < 0.05
        print(f"  ✓  BUY: ${result.fill_size_usd:.2f} @ {result.fill_price:.4f}")

        # ── test 2: sell order fills ──
        req_sell = OrderRequest(
            market_id="TEST-MKT-001", platform="polymarket",
            side=OrderSide.SELL, size_usd=100.0,
            limit_price=0.55, signal_price=0.55,
        )
        result_sell = await engine.submit_limit_order(req_sell)
        assert result_sell.status == OrderStatus.FILLED
        print(f"  ✓  SELL: ${result_sell.fill_size_usd:.2f} @ {result_sell.fill_price:.4f}")

        # ── test 3: close_position on YES win ──
        pos = Position(
            position_id="POS-TEST001", market_id="TEST-MKT-001",
            platform="polymarket", entry_order=result,
            entry_price=result.fill_price, size_usd=result.fill_size_usd,
            predicted_prob=0.65, opened_at=time.time(),
        )
        close_yes = await engine.close_position(pos, exit_price=1.0)
        assert close_yes.fill_price == 1.0
        pnl = (1.0 - pos.entry_price) / pos.entry_price
        print(f"  ✓  close YES win: pnl={pnl:+.4f}")

        # ── test 4: close_position on NO loss ──
        close_no = await engine.close_position(pos, exit_price=0.0)
        assert close_no.fill_price == 0.0
        pnl_no = (0.0 - pos.entry_price) / pos.entry_price
        print(f"  ✓  close NO loss: pnl={pnl_no:+.4f}")

        # ── test 5: factory returns paper when LIVE_MODE=False ──
        async with aiohttp.ClientSession() as session:
            e = get_execution_engine("polymarket", session)
            assert isinstance(e, PaperExecutionEngine)
            print(f"  ✓  get_execution_engine -> PaperExecutionEngine (LIVE_MODE={LIVE_MODE})")

        # ── test 6: 5 concurrent fills don't interfere ──
        async def _fill(i):
            r = OrderRequest(
                market_id=f"MKT-{i:03d}", platform="polymarket",
                side=OrderSide.BUY, size_usd=50.0,
                limit_price=0.30 + i * 0.10, signal_price=0.30 + i * 0.10,
            )
            return await engine.submit_limit_order(r)

        results = await asyncio.gather(*[_fill(i) for i in range(5)])
        assert all(r.status == OrderStatus.FILLED for r in results)
        prices = [r.fill_price for r in results]
        assert prices[0] < prices[-1], "fill ordering broken under concurrency"
        print(f"  ✓  5 concurrent fills: {[f'{p:.4f}' for p in prices]}")

        print("\nall execution_agent tests passed ✓")

    asyncio.run(run_tests())