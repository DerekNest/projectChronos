"""
agents/compound_agent.py  —  PYTHON SCRIPT (deterministic i/o, no llm)

architecture note: the llm reasoning for post-mortem classification lives
in _call_compound_skill() in orchestrator.py. this module handles only the
deterministic parts: persisting trade records, reading them back, computing
aggregate metrics, and rendering the dashboard. no fuzzy logic here.

storage format:
  data/logs/trades.jsonl     — one json line per closed trade (append-only)
  data/logs/failures.md      — human-readable failure log (append-only markdown)

trades.jsonl is the source of truth for all metrics. it is append-only to
prevent accidental overwrites — if you need to correct a record, append a
corrected version with the same trade_id and load_trade_history() will
de-duplicate by trade_id keeping the last occurrence.

metrics computed by compute_metrics():
  win_rate      — fraction of trades with pnl_pct > 0
  brier_score   — mean((p_model - outcome)^2), calibration quality
  sharpe        — annualized sharpe on per-trade pnl (assumes ~5 trades/day)
  profit_factor — gross_wins / gross_losses, ignores sizing
  total_pnl_pct — sum of pnl_pct across all trades (position-weighted would
                  require tracking size_usd at close, which we have via position_size)
  total_trades  — raw count

thresholds from compound/SKILL.md:
  win_rate      ≥ 0.60   → review EARLY_ENTRY and MODEL_OVERCONFIDENT classes
  sharpe        ≥ 2.0    → review position sizing in validate_risk.py
  profit_factor ≥ 1.5    → review MIN_EDGE in settings.py
  brier_score   < 0.25   → trigger ensemble recalibration
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────

TRADES_LOG_PATH  = Path("data/logs/trades.jsonl")
FAILURE_LOG_PATH = Path("data/logs/failures.md")

# ── performance thresholds (from compound/SKILL.md) ───────────────────────────

TARGET_WIN_RATE      = 0.60
TARGET_SHARPE        = 2.0
TARGET_PROFIT_FACTOR = 1.5
TARGET_BRIER         = 0.25    # above this → recalibration needed
TRADES_PER_DAY_EST   = 5.0     # used to annualize sharpe (sqrt(252 * trades_per_day))
MIN_TRADES_FOR_STATS = 5       # don't print misleading stats with fewer than 5 trades


# ── data contracts ─────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """
    complete record of one closed trade. written to trades.jsonl on settlement.

    position_size: fraction of bankroll at entry (size_usd / bankroll).
      stored as a fraction rather than USD so metrics stay valid even after
      bankroll changes between trades.

    pnl_pct: (exit_price - entry_price) / entry_price for LONG_YES.
      this is the return on capital deployed, not the return on total bankroll.
      total_pnl requires multiplying by position_size.
    """
    trade_id:              str
    market_id:             str
    platform:              str
    entry_price:           float
    exit_price:            float
    position_size:         float   # fraction of bankroll at entry
    predicted_probability: float   # p_model at signal time
    actual_outcome:        int     # 1 = YES resolved, 0 = NO resolved
    entry_ts:              float   # unix timestamp
    exit_ts:               float   # unix timestamp
    pnl_pct:               float   # (exit - entry) / entry


@dataclass
class PerformanceMetrics:
    """aggregated metrics over all closed trades."""
    total_trades:   int
    win_rate:       float
    brier_score:    float
    sharpe:         float
    profit_factor:  float
    total_pnl_pct:  float   # sum of position-weighted pnl
    avg_pnl_pct:    float   # mean per-trade pnl_pct (unweighted)
    max_drawdown:   float   # max peak-to-trough on cumulative pnl curve


# ── i/o helpers ────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    TRADES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_trade(record: TradeRecord) -> None:
    """
    appends one trade record to trades.jsonl as a single json line.
    append-only: never overwrites. if trade_id already exists, the later
    record wins when load_trade_history() de-duplicates.

    called synchronously from handle_settlement() after position closes.
    """
    _ensure_dirs()
    line = json.dumps(asdict(record))
    with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.info(
        f"[COMPOUND] logged trade {record.trade_id} — "
        f"pnl={record.pnl_pct:+.4f} outcome={record.actual_outcome}"
    )


def log_failure(record: TradeRecord, failure_class: str, analysis: str) -> None:
    """
    appends a failure entry to failures.md in human-readable markdown format.
    the last ~60 lines of this file are fed back into _call_compound_skill()
    as context for pattern detection — format matters for llm readability.

    called only on losing trades (and ~50% of winning trades for calibration).
    """
    _ensure_dirs()
    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(record.exit_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    entry = (
        f"\n---\n"
        f"**{ts}** | `{record.trade_id}` | `{failure_class}`\n"
        f"- market: {record.market_id}\n"
        f"- entry={record.entry_price:.4f} exit={record.exit_price:.4f} "
        f"p_model={record.predicted_probability:.4f} outcome={record.actual_outcome} "
        f"pnl={record.pnl_pct:+.4f}\n"
        f"- analysis: {analysis}\n"
    )
    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
    logger.debug(f"[COMPOUND] failure logged: {failure_class} for {record.trade_id}")


def load_trade_history() -> list[TradeRecord]:
    """
    reads trades.jsonl and returns all records as TradeRecord objects.
    de-duplicates by trade_id, keeping the last occurrence (allows corrections
    by appending a fixed record with the same trade_id).

    returns an empty list if the file doesn't exist yet — safe to call before
    any trades have been logged.
    """
    if not TRADES_LOG_PATH.exists():
        return []

    seen: dict[str, TradeRecord] = {}
    bad_lines = 0

    with open(TRADES_LOG_PATH, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                record = TradeRecord(**d)
                seen[record.trade_id] = record   # last write wins
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                bad_lines += 1
                logger.warning(f"[COMPOUND] bad line {lineno} in trades.jsonl: {exc}")

    if bad_lines:
        logger.warning(f"[COMPOUND] {bad_lines} malformed lines skipped in trades.jsonl")

    # return in chronological order
    records = list(seen.values())
    records.sort(key=lambda r: r.entry_ts)
    return records


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(history: list[TradeRecord]) -> PerformanceMetrics:
    """
    computes all performance metrics over a list of closed trades.

    sharpe annualization:
      we don't have a time-indexed equity curve — we have per-trade returns.
      we treat each trade as one period and annualize by assuming TRADES_PER_DAY_EST
      trades/day × 252 trading days/year.
      sharpe = mean(pnl_pct) / std(pnl_pct) * sqrt(252 * TRADES_PER_DAY_EST)
      this is a rough approximation; a proper time-series sharpe would require
      mapping each trade to its holding period.

    profit factor:
      gross_wins / abs(gross_losses) on raw pnl_pct (not position-weighted).
      if there are no losing trades yet, returns inf (which the dashboard shows as ∞).

    max drawdown:
      peak-to-trough on the cumulative sum of position-weighted pnl.
      position_weighted_pnl = pnl_pct * position_size
      this approximates the actual bankroll drawdown assuming position_size is
      the fraction staked (which it is — set to size_usd / bankroll at entry).

    returns zero-valued metrics (not nan) for edge cases (empty history, single trade).
    this avoids crashing print_dashboard() when the bot has just started.
    """
    n = len(history)

    zero = PerformanceMetrics(
        total_trades=n, win_rate=0.0, brier_score=0.0,
        sharpe=0.0, profit_factor=0.0, total_pnl_pct=0.0,
        avg_pnl_pct=0.0, max_drawdown=0.0,
    )

    if n == 0:
        return zero

    pnl   = np.array([r.pnl_pct for r in history], dtype=np.float64)
    sizes = np.array([r.position_size for r in history], dtype=np.float64)
    probs = np.array([r.predicted_probability for r in history], dtype=np.float64)
    outs  = np.array([r.actual_outcome for r in history], dtype=np.float64)

    # win rate
    win_rate = float(np.mean(pnl > 0))

    # brier score
    brier = float(np.mean((probs - outs) ** 2))

    # sharpe (needs at least 2 trades for std dev)
    if n >= 2 and np.std(pnl) > 1e-9:
        annualization = np.sqrt(252.0 * TRADES_PER_DAY_EST)
        sharpe = float(np.mean(pnl) / np.std(pnl, ddof=1) * annualization)
    else:
        sharpe = 0.0

    # profit factor
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win  = float(np.sum(wins))  if len(wins)   > 0 else 0.0
    gross_loss = float(np.sum(losses)) if len(losses) > 0 else 0.0
    if gross_loss < -1e-9:
        profit_factor = gross_win / abs(gross_loss)
    elif gross_win > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # position-weighted pnl
    weighted_pnl = pnl * sizes
    total_pnl    = float(np.sum(weighted_pnl))
    avg_pnl      = float(np.mean(pnl))

    # max drawdown on cumulative weighted pnl curve
    cum_pnl   = np.cumsum(weighted_pnl)
    peak      = np.maximum.accumulate(cum_pnl)
    drawdowns = peak - cum_pnl
    max_dd    = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return PerformanceMetrics(
        total_trades  = n,
        win_rate      = win_rate,
        brier_score   = brier,
        sharpe        = sharpe,
        profit_factor = profit_factor,
        total_pnl_pct = total_pnl,
        avg_pnl_pct   = avg_pnl,
        max_drawdown  = max_dd,
    )


# ── dashboard ──────────────────────────────────────────────────────────────────

def _fmt_pf(pf: float) -> str:
    """format profit factor, replacing float inf with readable ∞."""
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def _status(value: float, target: float, higher_is_better: bool) -> str:
    """returns ✓ or ✗ depending on whether the metric hits its target."""
    ok = value >= target if higher_is_better else value < target
    return "✓" if ok else "✗"


def print_dashboard(metrics: PerformanceMetrics) -> None:
    """
    prints a compact performance dashboard to stdout after each settlement.
    thresholds from compound/SKILL.md — ✓ / ✗ flags highlight breaches.

    suppresses most stats if fewer than MIN_TRADES_FOR_STATS trades exist to
    avoid printing misleading sharpe / profit factor on 1-2 data points.
    """
    n = metrics.total_trades
    sep = "─" * 52

    lines = [
        "",
        sep,
        f"  projectChronos — performance dashboard ({n} trades)",
        sep,
    ]

    if n < MIN_TRADES_FOR_STATS:
        lines.append(f"  (need {MIN_TRADES_FOR_STATS} trades for reliable stats — {n} so far)")
    else:
        wr_s  = _status(metrics.win_rate,      TARGET_WIN_RATE,      higher_is_better=True)
        sh_s  = _status(metrics.sharpe,        TARGET_SHARPE,        higher_is_better=True)
        pf_s  = _status(metrics.profit_factor, TARGET_PROFIT_FACTOR, higher_is_better=True)
        br_s  = _status(metrics.brier_score,   TARGET_BRIER,         higher_is_better=False)

        lines += [
            f"  win rate      {metrics.win_rate:.1%}   target ≥{TARGET_WIN_RATE:.0%}   {wr_s}",
            f"  brier score   {metrics.brier_score:.4f}  target <{TARGET_BRIER:.2f}    {br_s}",
            f"  sharpe        {metrics.sharpe:.2f}    target ≥{TARGET_SHARPE:.1f}     {sh_s}",
            f"  profit factor {_fmt_pf(metrics.profit_factor):<6}  target ≥{TARGET_PROFIT_FACTOR:.1f}     {pf_s}",
        ]

        if metrics.brier_score >= TARGET_BRIER:
            lines.append(f"  ⚠  RECALIBRATION FLAG — brier {metrics.brier_score:.4f} ≥ {TARGET_BRIER}")

    lines += [
        f"  total pnl     {metrics.total_pnl_pct:+.4f} (position-weighted)",
        f"  avg pnl/trade {metrics.avg_pnl_pct:+.4f}",
        f"  max drawdown  {metrics.max_drawdown:.4f}",
        sep,
        "",
    ]

    output = "\n".join(lines)
    print(output)
    logger.info(
        f"[COMPOUND] dashboard — trades={n} "
        f"wr={metrics.win_rate:.2%} brier={metrics.brier_score:.4f} "
        f"sharpe={metrics.sharpe:.2f} pf={_fmt_pf(metrics.profit_factor)}"
    )


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil, time as _time

    # redirect logs to stdout for test visibility
    logging.basicConfig(level=logging.WARNING)

    # use a temp dir so we don't pollute real data/logs
    tmp = tempfile.mkdtemp()
    import agents.compound_agent as _mod   # re-import to patch paths
    _orig_trades  = _mod.TRADES_LOG_PATH
    _orig_failure = _mod.FAILURE_LOG_PATH
    _mod.TRADES_LOG_PATH  = Path(tmp) / "trades.jsonl"
    _mod.FAILURE_LOG_PATH = Path(tmp) / "failures.md"

    now = _time.time()

    def _rec(trade_id, pnl, outcome, p_model=0.65, size=0.03):
        return TradeRecord(
            trade_id=trade_id, market_id=f"MKT-{trade_id}",
            platform="polymarket",
            entry_price=0.45, exit_price=0.45 + pnl * 0.45,
            position_size=size, predicted_probability=p_model,
            actual_outcome=outcome, entry_ts=now, exit_ts=now + 3600,
            pnl_pct=pnl,
        )

    print("running compound_agent self-test...\n")

    # ── test 1: log_trade + load round-trip ──
    r1 = _rec("T001", +0.20, 1)
    r2 = _rec("T002", -0.15, 0)
    r3 = _rec("T003", +0.30, 1)
    for r in [r1, r2, r3]:
        log_trade(r)

    history = load_trade_history()
    assert len(history) == 3, f"expected 3 records, got {len(history)}"
    assert history[0].trade_id == "T001"
    print("  ✓  log_trade + load_trade_history round-trip (3 records)")

    # ── test 2: de-duplication (later record wins) ──
    r1_corrected = _rec("T001", +0.25, 1)   # corrected pnl
    log_trade(r1_corrected)
    history = load_trade_history()
    assert len(history) == 3, "de-dup failed: should still be 3 records"
    t001 = next(r for r in history if r.trade_id == "T001")
    assert t001.pnl_pct == 0.25, f"expected corrected pnl 0.25, got {t001.pnl_pct}"
    print("  ✓  de-duplication: later record with same trade_id wins")

    # ── test 3: log_failure writes readable markdown ──
    _mod._ensure_dirs()   # ensure patched tmp dir exists before writing
    _mod.log_failure(r2, "MODEL_OVERCONFIDENT", "p_model was 0.65 but market resolved NO.")
    content = _mod.FAILURE_LOG_PATH.read_text()
    assert "MODEL_OVERCONFIDENT" in content
    assert "T002" in content
    print("  ✓  log_failure writes markdown with correct failure class and trade_id")

    # ── test 4: compute_metrics on known values ──
    # history after de-dup: T001=+0.25, T002=-0.15, T003=+0.30
    m = compute_metrics(history)
    assert m.total_trades == 3
    assert abs(m.win_rate - 2/3) < 1e-9, f"expected win_rate=0.667, got {m.win_rate}"
    expected_brier = ((0.65-1)**2 + (0.65-0)**2 + (0.65-1)**2) / 3
    assert abs(m.brier_score - expected_brier) < 1e-6, f"brier mismatch: {m.brier_score}"
    # gross_win = 0.25+0.30=0.55, gross_loss=0.15, pf=0.55/0.15≈3.67
    assert abs(m.profit_factor - (0.55 / 0.15)) < 1e-6, f"pf mismatch: {m.profit_factor}"
    print(f"  ✓  compute_metrics: win_rate={m.win_rate:.3f} brier={m.brier_score:.4f} pf={m.profit_factor:.2f}")

    # ── test 5: empty history ──
    m_empty = compute_metrics([])
    assert m_empty.total_trades == 0
    assert m_empty.win_rate == 0.0
    print("  ✓  compute_metrics handles empty history without error")

    # ── test 6: print_dashboard renders without crash ──
    print_dashboard(m)
    print("  ✓  print_dashboard rendered (see output above)")

    # ── test 7: profit factor with no losses ──
    wins_only = [_rec(f"W{i}", +0.10, 1) for i in range(6)]
    m_wins = compute_metrics(wins_only)
    assert m_wins.profit_factor == float("inf")
    assert _fmt_pf(m_wins.profit_factor) == "∞"
    print("  ✓  profit_factor=∞ when no losing trades")

    # cleanup
    shutil.rmtree(tmp)
    _mod.TRADES_LOG_PATH  = _orig_trades
    _mod.FAILURE_LOG_PATH = _orig_failure

    print("\nall compound_agent tests passed ✓")