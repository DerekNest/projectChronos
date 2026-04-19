# projectChronos

an end-to-end quantitative trading system for polymarket prediction markets. the system
combines order book microstructure feature engineering, a temporal convolutional network
(TCN) for signal generation, and an LLM ensemble for probability estimation — unified
under a strict deterministic/fuzzy architecture split.

---

## architecture principle

every component belongs to exactly one of two layers. **never mix them.**

| layer | format | responsibility |
|---|---|---|
| deterministic | `.py` | math, risk constraints, api routing, kelly sizing |
| fuzzy reasoning | `SKILL.md` | sentiment analysis, probability estimation, post-mortem classification |

the SKILL.md files are the "brain" — they encode reasoning heuristics for LLMs.
the `.py` files are the "skeleton" — they enforce hard constraints the LLM cannot override.

---

## pipeline

```
polymarket CLOB / gamma API
        │
        ▼
  scan_agent.py          ← discovers markets, filters by volume/liquidity
        │
        ▼
  data_pipeline.py       ← engineers 12 microstructure features per 64-tick window
        │
        ▼
  tcn_model.py           ← TCN inference → p_tcn (single float, YES probability)
        │
        ▼
  research_agent.py      ← calls research/SKILL.md → narrative + sentiment context
        │
        ▼
  predict/SKILL.md       ← LLM ensemble → p_ensemble weighted average
        │
        ▼
  validate_risk.py       ← edge gate → kelly sizing → hard position limits (SYNC)
        │
        ▼
  execution_agent.py     ← paper or live CLOB order submission
        │
        ▼
  compound_agent.py      ← post-mortem logging, brier tracking, recalibration flags
```

---

## feature engineering (12 features per tick)

the `data_pipeline.py` `BatchPipeline` produces a `(batch, 12, 64)` float32 tensor
clipped to `[-5, 5]`. features in order:

| idx | feature | description |
|-----|---------|-------------|
| 0 | `yes_price` | normalized mid price |
| 1 | `spread` | bid-ask spread |
| 2 | `book_imbalance` | (best_bid_qty - best_ask_qty) / total |
| 3 | `price_momentum` | tick-over-tick price delta |
| 4 | `volume_24h` | normalized 24h volume |
| 5 | `order_flow_imbalance` | signed order flow |
| 6 | `trade_count_15m` | trade frequency (currently stubbed 0.0) |
| 7 | `time_to_expiry` | normalized days remaining |
| 8 | `depth_ratio` | L2 depth asymmetry |
| 9 | `last_trade_size` | most recent fill size (currently stubbed 0.0) |
| 10 | `price_volatility` | rolling std of price |
| 11 | `liquidity_score` | composite book health metric |

> features [6] and [9] are currently stubbed as 0.0 pending a working `/trades` endpoint.
> the goldsky orderbook subgraph is the only confirmed working source for trade history.

---

## TCN model

- architecture: temporal convolutional network (`tcn_model.py`)
- input contract: `(batch, 12, 64)` float32
- output: single sigmoid scalar → YES probability `p_tcn ∈ [0, 1]`
- training: focal loss, `alpha=0.80` (corrects for YES=21.5% / NO=78.5% class imbalance)
- validation: market-level 80/20 split (not random window split — prevents data leakage)
- best checkpoint: `val_brier = 0.113` at epoch 5, saved to `models/tcn_weights.pt`
- the model singleton is loaded **once at startup** — not reinstantiated per inference call

---

## LLM ensemble

configured in `settings.py` (`PREDICT.ENSEMBLE_WEIGHTS`). weights must sum to 1.0
(validated at import — will raise `AssertionError` otherwise).

current config (single-model, cost-optimized):
```python
ENSEMBLE_WEIGHTS = {"claude": 1.0}   # haiku for research/predict; sonnet for compound only
```

target config (multi-model when credits available):
```python
ENSEMBLE_WEIGHTS = {
    "claude":   0.30,
    "gpt4o":    0.25,
    "gemini":   0.20,
    "grok":     0.15,
    "deepseek": 0.10,
}
```

ensemble signal logic (from `predict/SKILL.md`):
- `p_ensemble` = weighted average of individual model estimates
- if `std(estimates) > 0.12`: effective edge reduced by 20% (high disagreement penalty)
- output clipped to `[0.05, 0.95]` — no trading near-resolved markets

---

## risk management

all position sizing is deterministic in `validate_risk.py` (runs synchronously via
`ThreadPoolExecutor` to avoid blocking the async event loop).

| parameter | value | notes |
|-----------|-------|-------|
| kelly fraction | 0.5 | half-kelly |
| max position | 5% bankroll / $500 USD | whichever is smaller |
| min position | $5 USD | dust floor |
| max concurrent | 5 | open positions simultaneously |
| daily drawdown halt | 10% | halts all new orders |
| bankroll survival floor | $500 | no trading below this |
| min edge to trade | 4% | `p_model - p_market >= 0.04` |
| slippage tolerance | 2% | abandons fill if exceeded |

---

## directory structure

```
projectChronos/
├── orchestrator.py          # main async event loop — coordinates all agents
├── scan_agent.py            # market discovery (gamma API + CLOB book snapshots)
├── data_pipeline.py         # feature engineering → TCN-ready tensors
├── train_tcn.py             # training script (market-level split, focal loss)
├── historical_ingest.py     # bulk parquet download from goldsky subgraph
├── tcn_model.py             # pytorch TCN backbone
├── risk_map.py              # transfer function / position sizing utilities
├── validate_risk.py         # deterministic risk gatekeeper (synchronous)
├── research_agent.py        # parallel LLM scraping orchestrator
├── compound_agent.py        # post-mortem logger + brier tracker
├── execution_agent.py       # paper engine + live CLOB order stub
├── settings.py              # all tunable constants (frozen dataclasses)
├── SKILL.md                 # predict skill (LLM probability estimation)
├── models/
│   └── tcn_weights.pt       # trained checkpoint (val_brier=0.113)
├── data/
│   ├── historical/          # 1,080+ parquet files from goldsky ingest
│   └── logs/
│       ├── trades.jsonl     # append-only trade log
│       └── failures.md      # compound skill post-mortem log
└── requirements.txt
```

---

## quickstart

**1. install dependencies**
```bash
pip install -r requirements.txt
```

**2. set environment variables** (copy from `_env.example`)
```bash
ANTHROPIC_API_KEY=sk-...
INITIAL_BANKROLL=10000
LIVE_MODE=false          # paper trading by default — no funds needed
```

**3. ingest historical data** (only needed once, or to grow the training set)
```bash
python historical_ingest.py
```

**4. train the TCN**
```bash
python train_tcn.py
# expect: val_brier ~0.113 at epoch 5 → saves models/tcn_weights.pt
```

**5. run the bot**
```bash
python orchestrator.py
```

**6. emergency stop**
```bash
# powershell
New-Item STOP
# bash
touch STOP
```
the orchestrator checks for the `STOP` file at the top of each scan cycle and halts
immediately if it exists.

---

## known limitations / stubs

| component | status | notes |
|-----------|--------|-------|
| `_sign_order()` | stub | requires `py-clob-client` EIP-712 wiring for live trading |
| `trade_count_15m` (feature 6) | stubbed 0.0 | `/clob/trades` returns 401; goldsky subgraph is the fallback |
| `last_trade_size` (feature 9) | stubbed 0.0 | same root cause as above |
| multi-model ensemble | single model | other model API credits not yet provisioned |

---

## known bugs (resolved)

**data leakage** — original training used `random_split` on flat window indices, allowing
windows from the same market into both train and val sets. this produced an artificially
low `val_brier ≈ 0.007`. fix: market-level 80/20 split using `_market_ids` tracking.
honest val_brier is `0.113`.

**stage ordering** — in `process_market()`, the book snapshot ingest must happen *before*
the warmup gate check. original order left every market permanently stuck at 0/64 ticks.

**cold start** — `FeatureBuffer` requires 11 ticks before `raw_rows` populates, so the
effective warmup threshold is 75 (not 64). `_preseed_pipeline()` pre-seeds all markets
with 75 synthetic ticks (random walk around current `yes_price`) at startup.

---

## data sources

| source | used for | status |
|--------|----------|--------|
| gamma API | market metadata, tags, volume | working |
| polymarket CLOB | live order book snapshots | working |
| goldsky orderbook subgraph | historical trade fills | working |
| `/clob/trades` | real-time trade history | 401 — dead |
| `/data-api/activity` | activity feed | 400 — dead |
| `/clob/prices-history` | resolved market prices | empty — dead |

---

## concurrency model

```
asyncio event loop
├── scan cycle (non-blocking poll)
├── asyncio.gather() → concurrent LLM queries (latency = slowest model, not sum)
└── ThreadPoolExecutor → validate_risk.py (kept synchronous, offloaded to thread)
```

---

## environment — windows / powershell notes

- line continuation: backtick `` ` `` (not `\`)
- multi-line scripts: `@"..."@ | Out-File -Encoding utf8 script.py`
- hardware: intel core ultra 7 155u, integrated arc xe — cpu training only (no cuda)
- pytorch 2.4+: `ReduceLROnPlateau` does not accept `verbose=True`; remove it
- avoid unicode in log strings — cp1252 encoding will error; use ascii equivalents
