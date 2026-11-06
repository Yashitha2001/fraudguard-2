"""Saves red-team attacks to a file so pytest can replay them later.

Each scenario gets a status when it is saved:
  guarded    - at least GUARD_MIN_RATE of it was caught. The test fails if that drops by more than TOLERANCE.
  known_gap  - it wasn't caught. The test is a strict xfail, so it passes quietly while the gap is
               there and fails once a change closes it. That's the cue to re-run red_team.py.
"""
import json
import time

import numpy as np

from config import BASE_DIR
from redteam.detector import outcome

CORPUS_PATH = BASE_DIR / "tests" / "regression" / "redteam_corpus.json"
GUARD_MIN_RATE = 0.80
TOLERANCE = 0.10


def fingerprint(hist):
    return {"rows": int(len(hist)), "max_ts": int(hist["ts"].max()),
            "sum_amount": round(float(hist["amount"].sum()), 2)}


def freeze(results, detector, per_scenario=25, path=CORPUS_PATH):
    scenarios = {}
    for name in dict.fromkeys(e["scenario"] for e in results):
        eps = [e for e in results if e["scenario"] == name][:per_scenario]
        rate = float(np.mean([e["caught"] for e in eps]))
        scenarios[name] = {
            "baseline_rate": round(rate, 3),
            "status": "guarded" if rate >= GUARD_MIN_RATE else "known_gap",
            "episodes": [{"victim": e["victim"], "txns": e["txns"]} for e in eps],
        }
    corpus = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "model_version": detector.version,
              "threshold": round(float(detector.threshold), 4), "dataset": fingerprint(detector.hist),
              "scenarios": scenarios}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(corpus, indent=1))
    return {n: {"status": v["status"], "baseline_rate": v["baseline_rate"], "episodes": len(v["episodes"])}
            for n, v in scenarios.items()}


def load_corpus(path=CORPUS_PATH):
    return json.loads(path.read_text())


def evaluate_corpus(detector, corpus):
    """Replay the saved attacks against the current code and model."""
    rates = {}
    for name, entry in corpus["scenarios"].items():
        eps = detector.run_episodes([{"victim": e["victim"], "txns": e["txns"]} for e in entry["episodes"]])
        rates[name] = float(np.mean([e["caught"] for e in eps]))
    return rates
