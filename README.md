# FraudGuard

Scores card transactions for fraud and URLs for phishing using random forests. There is a
small Flask API and a one-page dashboard. All the data is simulated unless you bring your own.

## Running it

    pip install -r requirements.txt
    python train.py            # makes fake data, trains both models (a couple of minutes)
    python red_team.py --quick # optional: attack the model, needed for the regression tests
    python -m pytest -q tests
    python app.py              # dashboard at http://127.0.0.1:5001
    python api_client.py       # in a second terminal, with app.py running
    python benchmark.py        # optional: slow vs fast feature extraction

## Files

| File | What it does |
|---|---|
| `data_gen.py` | Makes fake transactions (about 1.5% fraud, in bursts) |
| `features.py` | Per-user features: transactions in the last hour, amount vs the user's average, country mismatch, time since last transaction. One version for the API, one vectorised version for training |
| `train.py` | Trains a baseline forest and a tuned one (random search on PR-AUC, F1-based threshold, split by time) |
| `app.py` | Flask API with API-key auth, reason codes and an alert feed |
| `integrations.py` | Optional Google Safe Browsing check and alert webhook |
| `templates/index.html` | Dashboard |
| `benchmark.py`, `tests/` | Speed comparison, tests |
| `red_team.py`, `redteam/` | Simulated attacks on the model (below) |

## Red team

    python red_team.py            # about 40 s
    python red_team.py --quick    # faster
    python -m pytest -q tests     # replays the saved attacks

- `redteam/attacker.py` has 8 fraud patterns (smash-and-grab, card testing, velocity dodging, low and slow, and so on), played against copies of real customer profiles. It also has a search that mutates them using only the model's scores and keeps the ones that get through.
- `redteam/blindspots.py` fits a small decision tree to the missed transactions to get readable rules, and checks how much normal traffic falls in the same region.
- `redteam/regression.py` saves attacks to a file. Scenarios that are caught must keep being caught. Known gaps are strict xfails, so the test fails once a gap is closed and you know to re-run `red_team.py`.
- `redteam/defense_eval.py` compares candidate fixes on held-out customers and attack types, at the same false-alarm rate, with bootstrap confidence intervals and guardrails on false positives and recall.

Reports go to `reports/` and the dashboard shows the latest one. The customers and attacks are
simulated, so the numbers say nothing about real fraud. The useful part is the method. The
attacker can only exploit the features the model currently uses.

## API

Every `/api/v1/*` route needs an `X-API-Key` header. Set `FRAUDGUARD_API_KEY`, otherwise the
dev key `dev-key-change-me` is used.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness check |
| POST | `/api/v1/transactions/score` | score one transaction |
| POST | `/api/v1/transactions/batch` | up to 500, handled in order |
| POST | `/api/v1/urls/scan` | `{ "url": "..." }` |
| GET | `/api/v1/alerts` | recent HIGH-risk events |
| GET | `/api/v1/metrics` | training metrics |
| GET | `/api/v1/redteam/report` | latest red-team report |

A transaction needs `user_id, amount, merchant_category, channel, country`. `ts` and `new_device` are optional.

## Project is live on Render
- I want to check as how we make live in the corporate office this time tried with Render
Website link: https://fraudguard-2-esbt.onrender.com

## Optional environment variables

- `GSB_API_KEY` adds a Google Safe Browsing check to URL scans
- `ALERT_WEBHOOK_URL` gets a POST for every HIGH-risk alert (Slack, Teams, SIEM)

## Your own data

    python train.py --csv yours.csv

Columns: `txn_id, user_id, ts (epoch seconds), amount, merchant_category, channel, country, new_device, is_fraud`.
The Kaggle credit card dataset has no user IDs, so it won't work well with the per-user features.
PaySim or an internal export is a better fit.

## Notes

- With 1.5% fraud, accuracy is meaningless (always predicting "not fraud" gets 98.5%). Look at precision, recall, F1 and PR-AUC.
- The URL model scores about 100% on its own made-up URLs. That only shows the code works.
- User IDs `U00000` to `U02499` have history loaded at startup. Any other ID starts with no history.
