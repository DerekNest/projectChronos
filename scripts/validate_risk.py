"""
scripts/validate_risk.py  —  PYTHON SCRIPT (deterministic, no llm)

architecture note: synchronous by design. called via run_in_executor() from
the orchestrator's async event loop to avoid blocking it. all logic here is
pure arithmetic — no i/o, no randomness, no external calls.

responsibilities:
  1. gate: reject trades that violate hard risk limits
  2. size: compute position size via fractional kelly criterion

the gate runs first. if any check fails the trade is rejected immediately
and kelly is never computed — there's no point sizing a trade we won't take.

kelly criterion for binary markets:
  f* = (b * p - q) / b
  where:
    b = decimal_odds - 1  (net profit per unit staked on a win)
    p = p_model           (our estimated probability of YES)
    q = 1 - p             (our estimated probability of NO)

  this is the fraction of bankroll that maximizes expected log-growth.
  we use KELLY_FRACTION * f* (default 0.5 = half-kelly) to account for
  model miscalibration. full kelly assumes a perfectly calibrated model;
  our brier score of 0.1132 is good but not perfect — half-kelly cuts
  ruin risk substantially at modest cost to long-run growth rate.

  position_size_usd = min(kelly_fraction * f* * bankroll, MAX_POSITION_USD)
  then capped again at MAX_POSITION_PCT * bankroll to prevent concentration.

gate checks (in order, fail-fast):
  1. concurrent position cap  — max open positions at once
  2. daily drawdown limit      — halt if daily loss exceeds threshold
  3. bankroll floor            — never bet if bankroll is below survival floor
  4. min edge guard            — second line of defense after predict skill
  5. fill slippage pre-check   — reject if fill_price drifted from signal_price
  6. kelly positive check      — reject if kelly fraction is ≤ 0 (negative edge)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── risk parameters ────────────────────────────────────────────────────────────
# these mirror the RISK config object that config/settings.py will expose.
# defined here as module-level constants so validate_risk.py is self-contained
# and testable without the full config tree.
# the orchestrator imports RISK from config.settings and passes risk_state
# which was constructed using those same values — no duplication risk.

KELLY_FRACTION       = 0.5     # half-kelly: conservative given imperfect calibration
MAX_POSITION_PCT     = 0.05    # never bet more than 5% of current bankroll on one trade
MAX_POSITION_USD     = 500.0   # hard USD cap regardless of bankroll size
MIN_POSITION_USD     = 5.0     # don't submit dust orders the clob will reject
MAX_CONCURRENT       = 5       # max open positions simultaneously
MAX_DAILY_DRAWDOWN   = 0.10    # halt trading if daily loss exceeds 10% of daily open bankroll
BANKROLL_FLOOR_USD   = 500.0   # survival floor: stop trading if bankroll drops below this
MIN_EDGE             = 0.04    # minimum edge (p_model - p_market) to approve a trade
MAX_SLIPPAGE_PRE     = 0.02    # reject if fill_price already deviated >2% from signal_price


# ── data contracts ─────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    """
    mutable state shared across all concurrent pipeline tasks.
    the orchestrator holds one instance and passes it to every process_market call.

    NOTE: concurrent_positions is incremented by the orchestrator after a fill
    and decremented on settlement. validate_risk reads it but never mutates it —
    mutation is the orchestrator's responsibility to keep the reference consistent.

    daily_start_bankroll is set once at session start (or midnight reset if we add
    a daily reset loop). it does not update intraday so the drawdown check is always
    relative to the day's opening equity.
    """
    bankroll:             float   # current equity in USD
    peak_bankroll:        float   # high-water mark (used for max drawdown tracking)
    daily_start_bankroll: float   # bankroll at start of trading day
    daily_api_cost_usd:   float   # accumulated llm api costs today (informational)
    concurrent_positions: int     # number of currently open positions


@dataclass
class TradeProposal:
    """
    all information validate_and_size needs to make a gate + sizing decision.
    constructed by the orchestrator immediately before calling validate_and_size.

    decimal_odds: 1 / p_market — e.g. p_market=0.40 → decimal_odds=2.50
    fill_price:   the price we expect to pay (pre-fill estimate = signal_price)
    signal_price: the price when the predict skill fired (used for slippage check)
    """
    market_id:    str
    platform:     str
    p_model:      float   # ensemble probability estimate
    p_market:     float   # current market implied probability (yes price)
    decimal_odds: float   # 1 / p_market
    fill_price:   float   # expected fill price
    signal_price: float   # price at signal generation time


@dataclass
class SizingDecision:
    """
    output of validate_and_size. the orchestrator checks .approved first;
    if True it uses .position_size_usd to construct the OrderRequest.
    """
    approved:           bool
    position_size_usd:  float = 0.0
    kelly_fraction:     float = 0.0   # raw kelly f* before half-kelly and caps (for logging)
    rejection_reason:   str   = ""


# ── gate checks ────────────────────────────────────────────────────────────────

def _check_concurrent_cap(state: RiskState) -> Optional[str]:
    if state.concurrent_positions >= MAX_CONCURRENT:
        return (
            f"concurrent position cap reached "
            f"({state.concurrent_positions}/{MAX_CONCURRENT})"
        )
    return None


def _check_daily_drawdown(state: RiskState) -> Optional[str]:
    if state.daily_start_bankroll <= 0:
        return "daily_start_bankroll is zero — cannot compute drawdown"
    daily_loss_pct = (state.daily_start_bankroll - state.bankroll) / state.daily_start_bankroll
    if daily_loss_pct >= MAX_DAILY_DRAWDOWN:
        return (
            f"daily drawdown limit breached: "
            f"{daily_loss_pct:.2%} loss vs {MAX_DAILY_DRAWDOWN:.0%} limit"
        )
    return None


def _check_bankroll_floor(state: RiskState) -> Optional[str]:
    if state.bankroll < BANKROLL_FLOOR_USD:
        return (
            f"bankroll ${state.bankroll:.2f} below survival floor "
            f"${BANKROLL_FLOOR_USD:.2f}"
        )
    return None


def _check_min_edge(proposal: TradeProposal) -> Optional[str]:
    edge = proposal.p_model - proposal.p_market
    if edge < MIN_EDGE:
        return (
            f"edge {edge:.4f} below minimum {MIN_EDGE:.4f} "
            f"(p_model={proposal.p_model:.4f}, p_market={proposal.p_market:.4f})"
        )
    return None


def _check_slippage(proposal: TradeProposal) -> Optional[str]:
    # guard against zero signal_price to avoid div-by-zero
    if proposal.signal_price <= 0:
        return "signal_price is zero or negative — cannot compute slippage"
    slippage = abs(proposal.fill_price - proposal.signal_price) / proposal.signal_price
    if slippage > MAX_SLIPPAGE_PRE:
        return (
            f"pre-fill slippage {slippage:.2%} exceeds limit {MAX_SLIPPAGE_PRE:.0%} "
            f"(fill={proposal.fill_price:.4f}, signal={proposal.signal_price:.4f})"
        )
    return None


# ── kelly sizer ────────────────────────────────────────────────────────────────

def _compute_kelly(proposal: TradeProposal, bankroll: float) -> SizingDecision:
    """
    computes fractional kelly position size.

    formula:  f* = (b * p - q) / b
      b = decimal_odds - 1
      p = p_model
      q = 1 - p_model

    degenerate cases:
      - b <= 0: decimal_odds <= 1 means p_market >= 1.0 — market already resolved,
        we should never reach here given the scan filter (price max 0.85), but guard anyway.
      - f* <= 0: negative expected value from kelly's perspective — reject.
        this is a secondary check; predict skill's edge gate should catch this first.
      - kelly_size < MIN_POSITION_USD: dust order, not worth submitting.
    """
    b = proposal.decimal_odds - 1.0
    p = proposal.p_model
    q = 1.0 - p

    if b <= 0:
        return SizingDecision(
            approved=False,
            rejection_reason=f"decimal_odds {proposal.decimal_odds:.4f} ≤ 1 — market at or above par"
        )

    f_star = (b * p - q) / b

    if f_star <= 0:
        return SizingDecision(
            approved=False,
            kelly_fraction=f_star,
            rejection_reason=f"kelly fraction {f_star:.4f} ≤ 0 — no positive edge"
        )

    # apply fractional kelly and dual caps
    kelly_size   = KELLY_FRACTION * f_star * bankroll
    pct_capped   = min(kelly_size, MAX_POSITION_PCT * bankroll)
    final_size   = min(pct_capped, MAX_POSITION_USD)

    if final_size < MIN_POSITION_USD:
        return SizingDecision(
            approved=False,
            kelly_fraction=f_star,
            rejection_reason=(
                f"sized position ${final_size:.2f} below minimum "
                f"${MIN_POSITION_USD:.2f} — dust order"
            )
        )

    return SizingDecision(
        approved=True,
        position_size_usd=round(final_size, 2),
        kelly_fraction=f_star,
    )


# ── public entry point ─────────────────────────────────────────────────────────

def validate_and_size(proposal: TradeProposal, state: RiskState) -> SizingDecision:
    """
    single entry point called by the orchestrator via run_in_executor.
    synchronous, pure arithmetic, no i/o.

    gate runs first in fail-fast order — checks are ordered from cheapest to
    most informative so early failures produce the most actionable log message.
    kelly is only computed if all gate checks pass.

    args:
        proposal: trade parameters from predict skill output
        state:    shared risk state (read-only here, mutated by orchestrator)

    returns:
        SizingDecision with approved=True and a position_size_usd, or
        approved=False with a rejection_reason describing which check failed.
    """
    mid = f"[{proposal.market_id[:16]}]"

    # ── gate ──────────────────────────────────────────────────────────────────
    for check_fn, args in [
        (_check_concurrent_cap,  (state,)),
        (_check_daily_drawdown,  (state,)),
        (_check_bankroll_floor,  (state,)),
        (_check_min_edge,        (proposal,)),
        (_check_slippage,        (proposal,)),
    ]:
        reason = check_fn(*args)
        if reason:
            logger.info(f"[RISK] {mid} REJECTED — {reason}")
            return SizingDecision(approved=False, rejection_reason=reason)

    # ── size ──────────────────────────────────────────────────────────────────
    decision = _compute_kelly(proposal, state.bankroll)

    if decision.approved:
        logger.info(
            f"[RISK] {mid} APPROVED — "
            f"kelly_f*={decision.kelly_fraction:.4f} "
            f"half_kelly_size=${decision.position_size_usd:.2f} "
            f"edge={proposal.p_model - proposal.p_market:.4f} "
            f"bankroll=${state.bankroll:.2f}"
        )
    else:
        logger.info(f"[RISK] {mid} REJECTED (kelly) — {decision.rejection_reason}")

    return decision


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("running validate_risk self-test...\n")

    base_state = RiskState(
        bankroll             = 10_000.0,
        peak_bankroll        = 10_000.0,
        daily_start_bankroll = 10_000.0,
        daily_api_cost_usd   = 0.0,
        concurrent_positions = 0,
    )

    def _proposal(p_model=0.60, p_market=0.45, fill=0.45, signal=0.45) -> TradeProposal:
        return TradeProposal(
            market_id    = "TEST-MARKET",
            platform     = "polymarket",
            p_model      = p_model,
            p_market     = p_market,
            decimal_odds = 1.0 / max(p_market, 1e-6),
            fill_price   = fill,
            signal_price = signal,
        )

    def _state(**kwargs) -> RiskState:
        import copy
        s = copy.copy(base_state)
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    # ── test 1: normal approval ──
    d = validate_and_size(_proposal(), base_state)
    assert d.approved, f"expected approval, got: {d.rejection_reason}"
    assert d.position_size_usd > 0
    # p=0.60, p_market=0.45, b=1/0.45-1≈1.222, f*=(1.222*0.60-0.40)/1.222≈0.272
    # half-kelly = 0.5*0.272*10000 = $1360 but capped at 5% = $500
    assert d.position_size_usd == 500.0, f"expected $500 (pct cap), got ${d.position_size_usd}"
    print(f"  ✓  normal approval: ${d.position_size_usd:.2f}, kelly_f*={d.kelly_fraction:.4f}")

    # ── test 2: concurrent cap ──
    d = validate_and_size(_proposal(), _state(concurrent_positions=5))
    assert not d.approved
    assert "concurrent" in d.rejection_reason
    print(f"  ✓  concurrent cap: rejected — {d.rejection_reason}")

    # ── test 3: daily drawdown ──
    d = validate_and_size(_proposal(), _state(bankroll=8_900.0))
    assert not d.approved
    assert "drawdown" in d.rejection_reason
    print(f"  ✓  daily drawdown: rejected — {d.rejection_reason}")

    # ── test 4: bankroll floor ──
    d = validate_and_size(_proposal(), _state(bankroll=499.0, daily_start_bankroll=499.0))
    assert not d.approved
    assert "floor" in d.rejection_reason
    print(f"  ✓  bankroll floor: rejected — {d.rejection_reason}")

    # ── test 5: min edge ──
    d = validate_and_size(_proposal(p_model=0.47, p_market=0.45), base_state)
    assert not d.approved
    assert "edge" in d.rejection_reason
    print(f"  ✓  min edge: rejected — {d.rejection_reason}")

    # ── test 6: slippage ──
    d = validate_and_size(_proposal(fill=0.47, signal=0.45), base_state)
    assert not d.approved
    assert "slippage" in d.rejection_reason
    print(f"  ✓  slippage: rejected — {d.rejection_reason}")

    # ── test 7: kelly negative edge (p_model below breakeven) ──
    d = validate_and_size(_proposal(p_model=0.35, p_market=0.45), base_state)
    assert not d.approved
    print(f"  ✓  negative kelly: rejected — {d.rejection_reason}")

    # ── test 8: small bankroll uses pct cap correctly ──
    d = validate_and_size(_proposal(), _state(bankroll=1_000.0, daily_start_bankroll=1_000.0))
    assert d.approved
    # 5% of $1000 = $50, which is above MIN_POSITION_USD and below MAX_POSITION_USD
    assert d.position_size_usd == 50.0, f"expected $50 (5% of $1000), got ${d.position_size_usd}"
    print(f"  ✓  pct cap on small bankroll: ${d.position_size_usd:.2f}")

    # ── test 9: thin edge, small kelly output above dust floor ──
    # edge = 0.045 (just above MIN_EDGE=0.04): p_model=0.495, p_market=0.45
    # b=1/0.45-1=1.222, f*=(1.222*0.495-0.505)/1.222≈0.091
    # half-kelly=0.5*0.091*600=$27.3 → above $5 floor, below $500 cap
    d = validate_and_size(
        _proposal(p_model=0.495, p_market=0.45),
        _state(bankroll=600.0, daily_start_bankroll=600.0),
    )
    assert d.approved
    print(f"  ✓  thin edge approved: ${d.position_size_usd:.2f}, kelly_f*={d.kelly_fraction:.4f}")

    print("\nall validate_risk tests passed ✓")