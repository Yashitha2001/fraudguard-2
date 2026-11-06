import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redteam.attacker import SCENARIOS, make_episode, mutate
from redteam.blindspots import _render
from redteam.detector import outcome

PROFILE = {"user_id": "U1", "avg_amount": 40.0, "last_ts": 1_730_000_000, "home_country": "US",
           "usual_hour": 14, "usual_channel": "pos", "top_cats": ["grocery", "fuel", "retail"]}


def test_episodes_are_reproducible_and_after_history():
    a = make_episode(SCENARIOS["smash_and_grab"], PROFILE, np.random.default_rng(1))
    b = make_episode(SCENARIOS["smash_and_grab"], PROFILE, np.random.default_rng(1))
    assert a == b
    assert a[0]["ts"] > PROFILE["last_ts"]
    assert [t["ts"] for t in a] == sorted(t["ts"] for t in a)


def test_mimic_scenarios_look_like_the_victim():
    ep = make_episode(SCENARIOS["full_mimic"], PROFILE, np.random.default_rng(2))
    assert all(t["country"] == "US" and t["channel"] == "pos" for t in ep)
    assert all(t["merchant_category"] in PROFILE["top_cats"] for t in ep)
    assert (ep[0]["ts"] // 3600) % 24 == 14  # attack starts at the victim's usual hour


def test_card_testing_starts_small():
    ep = make_episode(SCENARIOS["card_testing"], PROFILE, np.random.default_rng(3))
    assert ep[0]["amount"] <= 3 < ep[-1]["amount"]


def test_outcome_stops_at_first_flag():
    amounts = np.array([100.0, 200.0, 300.0])
    caught = outcome(np.array([0.1, 0.9, 0.9]), amounts, 0.5)
    assert caught["caught"] and caught["first_flag"] == 1 and caught["stolen"] == 100.0
    missed = outcome(np.array([0.1, 0.2, 0.3]), amounts, 0.5)
    assert not missed["caught"] and missed["stolen"] == 600.0


def test_mutation_stays_in_bounds():
    rng = np.random.default_rng(4)
    p = SCENARIOS["low_and_slow"]
    for _ in range(200):
        p = mutate(p, rng)
        assert 0.5 <= p.amount_mult <= 15 and 2 <= p.n_txns <= 8 and 0 <= p.new_device_prob <= 1


def test_rule_rendering_is_readable():
    text = _render([("txn_count_1h", "<=", 1.5), ("country_mismatch", "<=", 0.5), ("amount_ratio", ">", 2.0)])
    assert "country_mismatch == 0" in text and "txn_count_1h <= 1.50" in text and "amount_ratio > 2.00" in text
