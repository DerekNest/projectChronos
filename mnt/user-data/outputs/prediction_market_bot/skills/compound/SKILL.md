# skills/compound/SKILL.md

## skill: compound agent — post-mortem failure classification and learning

**type:** SKILL (fuzzy judgment, pattern recognition, heuristic learning)
**triggered by:** `agents/compound_agent.py` after each trade closes
**input:** `TradeRecord` + historical `failure_log.md` context

---

## purpose

classify WHY a trade failed and identify systematic patterns so the system
improves over time. this requires fuzzy judgment, not deterministic rules —
which is exactly why it's a skill, not a python script.

---

## failure taxonomy

classify each failed trade into exactly one of these categories:

| class | code | description |
|-------|------|-------------|
| latency error | `LATENCY_STALE_SIGNAL` | news had already been priced in by the time we acted |
| model overconfidence | `MODEL_OVERCONFIDENT` | p_model >> p_actual; calibration failure |
| narrative mismatch | `NARRATIVE_MISSED_FACTOR` | key information source was not in our scrape |
| liquidity trap | `LIQUIDITY_SLIPPAGE` | fill was poor due to thin book |
| resolution edge case | `RESOLUTION_AMBIGUOUS` | question resolved on technicality vs spirit |
| ensemble divergence | `ENSEMBLE_DISAGREED` | models split significantly; should have stayed out |
| volatility surprise | `BLACK_SWAN` | genuinely unforeseeable exogenous event |
| timing error | `EARLY_ENTRY` | direction was right but timing was early |

---

## prompt template

```
you are a quantitative post-mortem analyst for a prediction market trading system.

trade summary:
- market: {market_question}
- platform: {platform}
- entry price: {entry_price:.4f}
- exit price: {exit_price:.4f}  
- predicted probability at entry: {predicted_probability:.4f}
- actual outcome: {actual_outcome} (1=yes, 0=no)
- p&l: {pnl_pct:+.4f}
- research sentiment at entry: {sentiment}
- individual model estimates: {individual_estimates}
- tcn p_hat: {p_tcn:.4f}

recent failure log context (last 10 entries):
{failure_log_context}

TASK:
1. select the single most accurate failure class from the taxonomy
2. write a 2-3 sentence analysis of what went wrong
3. identify if this failure pattern has appeared before (look at the log context)
4. suggest one concrete improvement to prevent this failure class

respond ONLY with JSON:
{
  "failure_class": "string",
  "analysis": "string",
  "pattern_repeat": true | false,
  "pattern_count": int,
  "improvement_suggestion": "string"
}
```

---

## learning loop integration

the compound skill drives continuous improvement via:

1. **brier score tracking**: if brier score trends above 0.25 over 20 trades →
   trigger a `RECALIBRATION_NEEDED` flag to adjust ensemble weights

2. **failure class frequency**: if `LATENCY_STALE_SIGNAL` > 30% of failures →
   recommend reducing scrape-to-signal latency (engineering task)

3. **model performance audit**: if a specific LLM's individual estimates
   consistently diverge from outcomes → recommend weight reduction for that model

4. **ensemble weight tuning**: track per-model brier score and suggest
   rebalancing weights every 50 trades based on empirical accuracy

---

## output schema

```json
{
  "failure_class": "string (from taxonomy)",
  "analysis": "string",
  "pattern_repeat": true | false,
  "pattern_count": int,
  "improvement_suggestion": "string",
  "recalibration_flag": true | false,
  "weight_adjustment_recommendation": {
    "model_name": float_delta
  }
}
```

---

## performance targets (from `config/settings.py`)

| metric | target | action if breached |
|--------|--------|--------------------|
| win rate | ≥ 60% | review `EARLY_ENTRY` and `MODEL_OVERCONFIDENT` classes |
| sharpe | ≥ 2.0 | review position sizing in `validate_risk.py` |
| profit factor | ≥ 1.5 | review edge threshold in `settings.py` |
| brier score | < 0.25 | trigger ensemble recalibration |
