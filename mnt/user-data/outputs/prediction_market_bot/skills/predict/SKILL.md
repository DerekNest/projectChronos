# skills/predict/SKILL.md

## skill: predict agent — ensemble probability estimation

**type:** SKILL (fuzzy reasoning, multi-model consensus, calibration judgment)
**triggered by:** orchestrator after research SKILL returns sentiment payload
**input:** market snapshot + research payload + TCN model output `p_hat`

---

## purpose

estimate the **true probability** of a market resolving YES by:
1. querying multiple LLM providers in parallel with identical prompts
2. weighting their outputs by the ensemble weights from `config/settings.py`
3. combining with the TCN model's `p_hat` as a quantitative anchor
4. computing edge and expected value
5. generating a trade signal only if `edge > MIN_EDGE`

---

## ensemble weights

these weights reflect each model's track record on binary prediction tasks.
adjust them in `config/settings.py`, not here.

| model    | weight | rationale |
|----------|--------|-----------|
| grok     | 0.30   | best real-time internet access |
| claude   | 0.20   | strong reasoning, calibrated uncertainty |
| gpt-4o   | 0.20   | broad world knowledge |
| gemini   | 0.15   | strong factual retrieval |
| deepseek | 0.15   | strong quantitative reasoning |
| TCN p̂   | (anchor, not in weighted average — used as sanity check) |

---

## prompt template (identical across all models)

```
you are a probability calibration expert. provide a single number.

question: {market_question}

context:
- current market price (implied p): {market_yes_price:.3f}
- sentiment analysis: {sentiment} (confidence: {confidence:.2f})
- narrative: {narrative}
- latency opportunity: {latency_opportunity}
- tcn model estimate: {p_hat:.3f} (uncertainty: {sigma:.4f})

provide your best estimate of the probability this resolves YES.
respond ONLY with a JSON object: {"probability": float, "reasoning": "one sentence"}
probability must be between 0.0 and 1.0.
do not include any other text.
```

---

## aggregation logic

```python
# weighted average of LLM responses
p_ensemble = sum(weight_i * p_i for model_i, weight_i in ensemble_weights.items())

# edge calculation
edge = p_ensemble - p_market

# expected value (b = decimal_odds - 1 = (1/p_market) - 1)
b = (1.0 / p_market) - 1.0
ev = p_ensemble * b - (1.0 - p_ensemble)

# signal gate
if edge > MIN_EDGE and ev > 0:
    signal = "LONG_YES"
elif -edge > MIN_EDGE and ev < 0:   # only if platform supports shorting
    signal = "SHORT_YES"
else:
    signal = "NO_TRADE"
```

---

## output schema

```json
{
  "p_model": 0.0–1.0,
  "p_market": 0.0–1.0,
  "edge": float,
  "ev": float,
  "signal": "LONG_YES" | "SHORT_YES" | "NO_TRADE",
  "individual_estimates": {
    "grok": float,
    "claude": float,
    "gpt": float,
    "gemini": float,
    "deepseek": float
  },
  "p_tcn": float,
  "model_disagreement": float,
  "confidence_note": "string"
}
```

---

## disagreement handling

- if the **standard deviation** of individual model estimates > 0.12:
  → flag as `HIGH_DISAGREEMENT`, reduce effective edge by 20%
  → this prevents trading when models have genuinely divergent views
- if any single model estimate is an outlier (> 2 std deviations from mean):
  → log the outlier but keep it in the weighted average (don't silently drop)
- if TCN `p_hat` differs from ensemble by > 0.15:
  → flag `MODEL_NARRATIVE_DIVERGENCE` for compound agent review

---

## calibration guardrails

never output probabilities outside [0.05, 0.95].  
extreme probabilities (>0.95 or <0.05) suggest the market is nearly resolved
and there is no edge left — don't trade near resolution.
