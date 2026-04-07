# prediction market trading bot

end-to-end ai-powered prediction market trading system exploiting information latency
with continuous learning via a multi-agent pipeline + pytorch TCN strategy.

## architecture rule

| layer | format | use case |
|---|---|---|
| deterministic logic | `.py` script | math, risk, api routing, constraints |
| fuzzy reasoning | `SKILL.md` | nlp, heuristics, agentic decisions |

## directory structure

```
prediction_market_bot/
├── agents/                    # python orchestration layer (deterministic)
│   ├── scan_agent.py          # market discovery + anomaly detection
│   ├── research_agent.py      # parallel scraping orchestrator
│   └── compound_agent.py      # post-mortem logging + metrics
├── skills/                    # SKILL.md agents (fuzzy / llm reasoning)
│   ├── research/SKILL.md      # sentiment + narrative analysis
│   ├── predict/SKILL.md       # ensemble probability estimation
│   └── compound/SKILL.md      # failure classification + calibration
├── models/
│   ├── tcn_model.py           # pytorch TCN backbone
│   └── risk_map.py            # custom transfer function / position sizing
├── scripts/
│   └── validate_risk.py       # deterministic risk gatekeeper (kelly + hard limits)
├── config/
│   └── settings.py            # all tunable constants in one place
├── data/
│   ├── logs/                  # trade logs, failure_log.md
│   └── cache/                 # scraped data cache
└── tests/
    ├── test_validate_risk.py
    └── test_tcn_model.py
```

## pipeline flow

```
scan_agent → research SKILL → predict SKILL → validate_risk.py → execution → compound SKILL
```

## quickstart

```bash
pip install -r requirements.txt
python -m agents.scan_agent          # start market scanning
touch STOP                            # emergency kill switch
```
