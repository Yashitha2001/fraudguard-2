"""Replays the saved red-team attacks (see redteam/regression.py).

guarded    scenarios must still be caught, within TOLERANCE of the rate when they were saved.
known_gap  scenarios are strict xfails: they pass quietly while the gap exists and fail once
           a change closes it. When that happens, re-run `python red_team.py`."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redteam.detector import Detector
from redteam.regression import (CORPUS_PATH, GUARD_MIN_RATE, TOLERANCE, evaluate_corpus,
                                fingerprint, load_corpus)

if not CORPUS_PATH.exists():
    pytest.skip("no corpus yet: run `python red_team.py` first", allow_module_level=True)

CORPUS = load_corpus()


def _case(name):
    entry = CORPUS["scenarios"][name]
    marks = pytest.mark.xfail(strict=True, reason="known blind spot; re-run red_team.py if this starts passing") \
        if entry["status"] == "known_gap" else ()
    return pytest.param(name, marks=marks, id=f"{name}[{entry['status']}]")


@pytest.fixture(scope="module")
def rates():
    detector = Detector.load()
    if fingerprint(detector.hist) != CORPUS["dataset"]:
        pytest.skip("transactions.csv changed since the corpus was saved; re-run red_team.py")
    return evaluate_corpus(detector, CORPUS)


@pytest.mark.parametrize("name", [_case(n) for n in CORPUS["scenarios"]])
def test_scenario(rates, name):
    entry = CORPUS["scenarios"][name]
    if entry["status"] == "guarded":
        assert rates[name] >= entry["baseline_rate"] - TOLERANCE, (
            f"{name}: detection fell from {entry['baseline_rate']:.0%} to {rates[name]:.0%}")
    else:
        assert rates[name] >= GUARD_MIN_RATE
