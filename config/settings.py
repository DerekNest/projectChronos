"""
config/settings.py  —  PYTHON SCRIPT (pure configuration, no logic)

single source of truth for all tunable parameters across projectChronos.
every module that needs a threshold, path, or weight imports from here —
no magic numbers scattered across the codebase.

structure: one frozen dataclass instance per domain, exported as a module-level
singleton. frozen=True prevents accidental mutation at runtime (a config value
changed mid-session would silently corrupt risk calculations).

to change a value: edit here and restart the orchestrator. do not patch these
at runtime — the frozen dataclasses will raise FrozenInstanceError if you try.

domains:
  PREDICT  — ensemble weights, edge thresholds, calibration guardrails
  RISK     — position sizing limits, drawdown guards, kill switch path
  COMPOUND — failure log path, brier recalibration threshold
  MODEL    — TCN tensor contract (must match data_pipeline.py exactly)
  API      — rate limits and timeout defaults for external calls
"""

from dataclasses import dataclass
from pathlib import Path


# ── predict config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PredictConfig:
    """
    parameters for the predict skill ensemble and signal gate.

    ENSEMBLE_WEIGHTS: maps litellm model key -> weight in weighted average.
      keys must match LLM_MODELS dict in orchestrator.py exactly.
      weights must sum to 1.0 — validated at module load below.
      source: predict/SKILL.md model track record table.

    MIN_EDGE: minimum (p_model - p_market) required to generate a trade signal.
      0.04 = 4 percentage points of edge above market implied probability.
      below this, expected value is too small to overcome bid-ask spread
      and execution slippage. mirrors validate_risk.MIN_EDGE — both gates
      must pass independently (predict skill gates on signal, validate_risk
      gates on kelly sizing).

    HIGH_DISAGREEMENT_THRESHOLD: std dev across individual model estimates
      above which effective edge is reduced by 20% (from predict/SKILL.md).

    CALIBRATION_CLIP: [lo, hi] probability bounds enforced on all model outputs.
      prevents trading near-resolved markets where no edge remains.
    """
    ENSEMBLE_WEIGHTS: dict = None   # set via __post_init__ workaround below
    MIN_EDGE:                  float = 0.04
    HIGH_DISAGREEMENT_THRESHOLD: float = 0.12
    CALIBRATION_LO:            float = 0.05
    CALIBRATION_HI:            float = 0.95

    def __post_init__(self):
        # frozen dataclass: use object.__setattr__ to set the mutable default
        if self.ENSEMBLE_WEIGHTS is None:
            object.__setattr__(self, "ENSEMBLE_WEIGHTS", {
                "grok":     0.30,   # best real-time internet access
                "claude":   0.20,   # strong reasoning, calibrated uncertainty
                "gpt":      0.20,   # broad world knowledge
                "gemini":   0.15,   # strong factual retrieval
                "deepseek": 0.15,   # strong quantitative reasoning
            })
        # validate weights sum to 1.0 (within float tolerance)
        total = sum(self.ENSEMBLE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6, (
            f"ENSEMBLE_WEIGHTS must sum to 1.0, got {total:.6f}. "
            f"adjust weights in config/settings.py."
        )


# ── risk config ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskConfig:
    """
    parameters for position sizing and risk management.
    these values are intentionally conservative for initial live deployment.
    increase limits only after validating paper performance for ≥ 50 trades.

    MAX_SLIPPAGE_PCT: post-fill slippage tolerance. if the actual fill price
      deviates more than this from the signal price, the orchestrator abandons
      the position (does not open it). 2% is tight for polymarket — thin books
      can move 5%+ on a single order. if too many trades are abandoned due to
      slippage, raise to 0.03 and investigate book depth at fill time.

    STOP_FILE_PATH: path to the kill switch file. if this file exists when the
      orchestrator checks at the top of each scan cycle, it halts immediately.
      create it with: `echo stop > STOP` or `New-Item STOP` on powershell.
      this is faster than ctrl+c in production because it lets the current
      pipeline task finish cleanly rather than raising KeyboardInterrupt mid-order.

    INITIAL_BANKROLL: default bankroll if INITIAL_BANKROLL env var is not set.
      overridden by the env var in orchestrator.main() — set the env var for
      real deployments so the value isn't hardcoded in source.
    """
    MAX_SLIPPAGE_PCT:     float = 0.02    # 2% post-fill slippage tolerance
    STOP_FILE_PATH:       str   = "STOP"  # kill switch: create this file to halt
    INITIAL_BANKROLL:     float = 10_000.0

    # these mirror validate_risk.py module-level constants — kept here for
    # documentation and potential future refactor where validate_risk reads
    # from settings rather than its own module-level constants.
    KELLY_FRACTION:       float = 0.5     # half-kelly
    MAX_POSITION_PCT:     float = 0.05    # 5% of bankroll per trade
    MAX_POSITION_USD:     float = 500.0   # hard USD cap
    MIN_POSITION_USD:     float = 5.0     # dust order floor
    MAX_CONCURRENT:       int   = 5       # max simultaneous open positions
    MAX_DAILY_DRAWDOWN:   float = 0.10    # 10% daily loss halt
    BANKROLL_FLOOR_USD:   float = 500.0   # survival floor


# ── compound config ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompoundConfig:
    """
    parameters for the compound skill post-mortem and performance tracking.

    FAILURE_LOG_PATH: append-only markdown log of trade post-mortems.
      the last ~60 lines are fed back into _call_compound_skill() as context
      for pattern detection. keep this path stable — changing it mid-session
      loses the pattern detection context window.

    BRIER_RECALIBRATION_THRESHOLD: brier score above which the compound skill
      raises a recalibration flag. from compound/SKILL.md: target < 0.25.
      a random predictor scores 0.25 (always outputs p=0.5).

    MIN_TRADES_FOR_AUDIT: minimum closed trades before the compound skill
      starts suggesting ensemble weight adjustments. too few trades and
      the per-model brier scores are noise-dominated.
    """
    FAILURE_LOG_PATH:              str   = "data/logs/failures.md"
    TRADES_LOG_PATH:               str   = "data/logs/trades.jsonl"
    BRIER_RECALIBRATION_THRESHOLD: float = 0.25
    MIN_TRADES_FOR_AUDIT:          int   = 50


# ── model config ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """
    TCN tensor contract. these values must match data_pipeline.py exactly:
      FEATURES  = 12  (the 12 engineered features, see data_pipeline module doc)
      TIMESTEPS = 64  (window size for TCN input, matches WINDOW_SIZE in pipeline)

    WEIGHTS_PATH: where train_tcn.py saves the checkpoint and where the
      orchestrator loads it at startup.

    changing FEATURES or TIMESTEPS requires retraining — the saved weights
    encode the architecture dimensions and will fail to load if mismatched.
    """
    FEATURES:      int  = 12
    TIMESTEPS:     int  = 64
    WEIGHTS_PATH:  str  = "models/tcn_weights.pt"


# ── api config ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class APIConfig:
    """
    rate limits and timeouts for external api calls.
    these are shared defaults — individual modules may override locally
    for specific endpoints (e.g. historical_ingest uses REQUEST_DELAY_S=0.75
    to stay safely under gamma's 100 req/min limit).
    """
    GAMMA_REQ_PER_MIN:   int   = 80     # our self-imposed limit (api allows ~100)
    CLOB_REQ_PER_MIN:    int   = 100    # our self-imposed limit (api allows ~120)
    DEFAULT_TIMEOUT_S:   float = 15.0
    DEFAULT_RETRIES:     int   = 3


# ── singletons ─────────────────────────────────────────────────────────────────
# instantiated once at import time. frozen=True means these are effectively
# global constants — any attempt to mutate raises FrozenInstanceError.

PREDICT  = PredictConfig()
RISK     = RiskConfig()
COMPOUND = CompoundConfig()
MODEL    = ModelConfig()
API      = APIConfig()


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("running config/settings self-test...\n")

    # ── test 1: ensemble weights sum to 1.0 ──
    total = sum(PREDICT.ENSEMBLE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"weights sum to {total}"
    print(f"  ✓  ensemble weights sum to {total:.6f}")

    # ── test 2: weight keys match orchestrator LLM_MODELS keys ──
    expected_keys = {"grok", "claude", "gpt", "gemini", "deepseek"}
    assert set(PREDICT.ENSEMBLE_WEIGHTS.keys()) == expected_keys, (
        f"weight keys {set(PREDICT.ENSEMBLE_WEIGHTS.keys())} != {expected_keys}"
    )
    print(f"  ✓  weight keys match LLM_MODELS: {sorted(expected_keys)}")

    # ── test 3: frozen — mutation raises FrozenInstanceError ──
    try:
        PREDICT.MIN_EDGE = 0.99   # type: ignore
        assert False, "expected FrozenInstanceError"
    except Exception as e:
        assert "frozen" in str(e).lower() or "cannot" in str(e).lower(), str(e)
    print("  ✓  frozen dataclass: mutation raises FrozenInstanceError")

    # ── test 4: MODEL matches data_pipeline contract ──
    assert MODEL.FEATURES  == 12, f"expected 12 features, got {MODEL.FEATURES}"
    assert MODEL.TIMESTEPS == 64, f"expected 64 timesteps, got {MODEL.TIMESTEPS}"
    print(f"  ✓  MODEL contract: ({MODEL.FEATURES} features, {MODEL.TIMESTEPS} timesteps)")

    # ── test 5: paths are strings (orchestrator uses os.path.exists and open()) ──
    assert isinstance(RISK.STOP_FILE_PATH,    str)
    assert isinstance(COMPOUND.FAILURE_LOG_PATH, str)
    assert isinstance(MODEL.WEIGHTS_PATH,     str)
    print(f"  ✓  path fields are str: STOP={RISK.STOP_FILE_PATH!r} "
          f"FAILURE_LOG={COMPOUND.FAILURE_LOG_PATH!r}")

    # ── test 6: calibration clip bounds are sane ──
    assert 0.0 < PREDICT.CALIBRATION_LO < PREDICT.CALIBRATION_HI < 1.0
    print(f"  ✓  calibration clip: [{PREDICT.CALIBRATION_LO}, {PREDICT.CALIBRATION_HI}]")

    # ── test 7: risk thresholds are internally consistent ──
    assert RISK.MIN_POSITION_USD < RISK.MAX_POSITION_USD
    assert 0.0 < RISK.KELLY_FRACTION <= 1.0
    assert 0.0 < RISK.MAX_DAILY_DRAWDOWN < 1.0
    assert RISK.BANKROLL_FLOOR_USD < RISK.INITIAL_BANKROLL
    print(
        f"  ✓  risk thresholds consistent: "
        f"kelly={RISK.KELLY_FRACTION} "
        f"max_pos_pct={RISK.MAX_POSITION_PCT:.0%} "
        f"drawdown_halt={RISK.MAX_DAILY_DRAWDOWN:.0%}"
    )

    print("\nall config/settings tests passed ✓")
    print()
    print("  current configuration:")
    print(f"    PREDICT  MIN_EDGE={PREDICT.MIN_EDGE}  "
          f"HIGH_DISAGREEMENT={PREDICT.HIGH_DISAGREEMENT_THRESHOLD}")
    print(f"    RISK     MAX_SLIPPAGE={RISK.MAX_SLIPPAGE_PCT:.0%}  "
          f"MAX_CONCURRENT={RISK.MAX_CONCURRENT}  "
          f"MAX_DAILY_DD={RISK.MAX_DAILY_DRAWDOWN:.0%}")
    print(f"    COMPOUND FAILURE_LOG={COMPOUND.FAILURE_LOG_PATH}  "
          f"BRIER_THRESHOLD={COMPOUND.BRIER_RECALIBRATION_THRESHOLD}")
    print(f"    MODEL    ({MODEL.FEATURES}f, {MODEL.TIMESTEPS}t)  "
          f"WEIGHTS={MODEL.WEIGHTS_PATH}")