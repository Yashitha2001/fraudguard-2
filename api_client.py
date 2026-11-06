"""Small example of another service calling the FraudGuard API. Start app.py first.

    python api_client.py
"""
import os
import time

import requests


class FraudGuardClient:
    def __init__(self, base_url="http://127.0.0.1:5001", api_key=None, timeout=5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key or os.getenv("FRAUDGUARD_API_KEY", "dev-key-change-me")

    def _post(self, path, payload):
        resp = self.session.post(self.base_url + path, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def score_transaction(self, **txn):
        return self._post("/api/v1/transactions/score", txn)

    def score_batch(self, txns):
        return self._post("/api/v1/transactions/batch", txns)

    def scan_url(self, url):
        return self._post("/api/v1/urls/scan", {"url": url})


if __name__ == "__main__":
    client = FraudGuardClient()
    now = int(time.time())

    print("Normal purchase:")
    print(" ", client.score_transaction(user_id="U00042", amount=23.5, merchant_category="grocery",
                                        channel="pos", country="US", ts=now))

    print("Suspicious burst (4 gift card buys in 6 minutes, new device, foreign country):")
    burst = [dict(user_id="U00042", amount=480 + i * 20, merchant_category="gift_cards", channel="web",
                  country="NG", new_device=1, ts=now + 90 * i) for i in range(4)]
    for r in client.score_batch(burst)["results"]:
        print(" ", r["risk_level"], r["risk_score"], r["reasons"][:2])

    print("URL scan:")
    print(" ", client.scan_url("http://secure-chase-login.xyz/verify?id=8812"))
