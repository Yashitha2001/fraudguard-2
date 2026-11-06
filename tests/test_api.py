import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_gen
from app import create_app
from features import build_features, build_features_slow, prepare

KEY = {"X-API-Key": "dev-key-change-me"}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def normal_txn(**overrides):
    txn = dict(user_id="U_TEST_OK", amount=22.0, merchant_category="grocery", channel="pos",
               country="US", ts=int(time.time()))
    txn.update(overrides)
    return txn


def test_vectorised_features_match_row_by_row():
    df = prepare(data_gen.generate(n_users=150, days=10, seed=3))
    slow, fast = build_features_slow(df), build_features(df)
    assert np.allclose(slow.to_numpy(float), fast.to_numpy(float))


def test_health_is_open(client):
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_scoring_requires_api_key(client):
    assert client.post("/api/v1/transactions/score", json=normal_txn()).status_code == 401


def test_validation_errors(client):
    r = client.post("/api/v1/transactions/score", json={"user_id": "x"}, headers=KEY)
    assert r.status_code == 400
    r = client.post("/api/v1/transactions/score", json=normal_txn(amount=-5), headers=KEY)
    assert r.status_code == 400


def test_normal_purchase_is_not_flagged(client):
    r = client.post("/api/v1/transactions/score", json=normal_txn(), headers=KEY).get_json()
    assert r["risk_level"] == "LOW" and not r["flagged"]


def test_suspicious_burst_is_flagged(client):
    now = int(time.time())
    burst = [normal_txn(user_id="U_TEST_BAD", amount=500 + 30 * i, merchant_category="gift_cards",
                        channel="web", country="NG", new_device=1, ts=now + 90 * i) for i in range(4)]
    body = client.post("/api/v1/transactions/batch", json=burst, headers=KEY).get_json()
    assert body["flagged"] >= 1
    assert body["results"][-1]["reasons"]


def test_batch_limits(client):
    assert client.post("/api/v1/transactions/batch", json=[], headers=KEY).status_code == 400
    too_many = [normal_txn(user_id=f"U{i}") for i in range(501)]
    assert client.post("/api/v1/transactions/batch", json=too_many, headers=KEY).status_code == 400


def test_url_scan(client):
    bad = client.post("/api/v1/urls/scan", json={"url": "http://secure-chase-login.xyz/verify?id=88"}, headers=KEY).get_json()
    good = client.post("/api/v1/urls/scan", json={"url": "https://www.paypal.com/help/contact"}, headers=KEY).get_json()
    assert bad["flagged"] and not good["flagged"]
    assert bad["reputation"]["status"] == "unavailable"  # no GSB key in the test environment


def test_alerts_listed_after_flag(client):
    assert len(client.get("/api/v1/alerts", headers=KEY).get_json()["alerts"]) >= 1


def test_redteam_report_endpoint(client):
    from config import BASE_DIR
    if not (BASE_DIR / "reports" / "redteam_latest.json").exists():
        pytest.skip("run red_team.py first")
    body = client.get("/api/v1/redteam/report", headers=KEY).get_json()
    assert body["scenarios"] and "corpus" in body
    assert client.get("/api/v1/redteam/report").status_code == 401
