"""FraudGuard API and dashboard.

    python app.py                      # http://127.0.0.1:5001
    FRAUDGUARD_API_KEY=secret python app.py
    GSB_API_KEY=... ALERT_WEBHOOK_URL=https://hooks.slack.com/... python app.py   (both optional)
"""
import hmac
import logging
import os
import threading
import time
import uuid
from collections import deque
from functools import wraps

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from config import BASE_DIR, DATA_DIR, MODEL_DIR
from features import OnlineFeatureStore
from integrations import check_url_reputation, send_alert_webhook
from url_features import url_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fraudguard")

MAX_BATCH = 500
REQUIRED_TXN_FIELDS = ("user_id", "amount", "merchant_category", "channel", "country")


class FraudService:
    """Loads the models and keeps the live per-user history."""

    def __init__(self, seed_rows=40000):
        bundle = joblib.load(MODEL_DIR / "fraud_model.joblib")
        self.model, self.threshold, self.version = bundle["model"], bundle["threshold"], bundle["version"]
        self.features = bundle["features"]
        self.model.n_jobs = 1  # one transaction at a time, threads would only slow it down
        url_bundle = joblib.load(MODEL_DIR / "url_model.joblib")
        self.url_model = url_bundle["model"]
        self.url_model.n_jobs = 1

        self.store = OnlineFeatureStore()
        self.lock = threading.Lock()
        self.alerts = deque(maxlen=100)
        self._warm_start(seed_rows)

    def _warm_start(self, limit):
        """Load recent history so existing users don't start from nothing."""
        path = DATA_DIR / "transactions.csv"
        if not path.exists():
            return
        cols = ["user_id", "ts", "amount", "merchant_category", "channel", "country", "new_device"]
        hist = pd.read_csv(path, usecols=cols).sort_values("ts").tail(limit)
        for row in hist.itertuples(index=False):
            self.store.commit(row._asdict())
        log.info("loaded %d past transactions into user profiles", len(hist))

    def risk_level(self, score):
        if score >= self.threshold:
            return "HIGH"
        return "MEDIUM" if score >= self.threshold * 0.5 else "LOW"

    def score_transactions(self, txns):
        # features have to be built in order because each one updates the user's history
        with self.lock:
            feats = []
            for txn in txns:
                feats.append(self.store.compute(txn))
                self.store.commit(txn)
        frame = pd.DataFrame(feats, columns=self.features)
        scores = self.model.predict_proba(frame)[:, 1]

        results = []
        for txn, f, score in zip(txns, feats, scores):
            level = self.risk_level(score)
            result = {
                "txn_id": txn.get("txn_id") or "T-" + uuid.uuid4().hex[:10],
                "user_id": txn["user_id"],
                "risk_score": round(float(score), 4),
                "risk_level": level,
                "flagged": level == "HIGH",
                "reasons": explain(f, txn),
                "model_version": self.version,
            }
            if result["flagged"]:
                self._raise_alert(txn, result)
            results.append(result)
        return results

    def _raise_alert(self, txn, result):
        alert = {
            "time": time.strftime("%H:%M:%S"),
            "type": "transaction",
            "subject": f"{txn['user_id']} ${txn['amount']:.2f} {txn['merchant_category']}",
            "risk_score": result["risk_score"],
            "reasons": result["reasons"],
            "summary": f"High-risk transaction for {txn['user_id']} (score {result['risk_score']})",
        }
        self.alerts.appendleft(alert)
        log.warning(alert["summary"])
        send_alert_webhook(alert)

    def scan_url(self, url):
        feats = url_frame([url])
        model_score = float(self.url_model.predict_proba(feats)[0, 1])
        reputation = check_url_reputation(url)
        score = 0.99 if reputation["status"] == "malicious" else model_score
        level = "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
        result = {"url": url, "risk_score": round(score, 4), "model_score": round(model_score, 4),
                  "risk_level": level, "flagged": level == "HIGH", "reputation": reputation}
        if result["flagged"]:
            alert = {"time": time.strftime("%H:%M:%S"), "type": "url", "subject": url[:70],
                     "risk_score": result["risk_score"], "reasons": [],
                     "summary": f"Phishing URL flagged: {url[:70]}"}
            self.alerts.appendleft(alert)
            send_alert_webhook(alert)
        return result


def explain(f, txn):
    """Short reasons an analyst can read next to the score."""
    reasons = []
    if f["amount_ratio"] >= 4:
        reasons.append(f"Amount is {f['amount_ratio']:.1f}x this customer's usual spend")
    if f["txn_count_1h"] >= 3:
        reasons.append(f"{f['txn_count_1h']} transactions within the last hour")
    if f["country_mismatch"]:
        reasons.append(f"Transaction country ({txn['country']}) differs from the account's home country")
    if f["new_device"]:
        reasons.append("Unrecognised device")
    if f["is_night"]:
        reasons.append("Unusual hour (00:00-06:00 UTC)")
    if txn["merchant_category"] in ("gift_cards", "crypto", "electronics"):
        reasons.append(f"High-risk merchant category: {txn['merchant_category']}")
    return reasons


def validate_txn(raw):
    """Returns (txn, errors). txn is None if there are errors."""
    if not isinstance(raw, dict):
        return None, ["each transaction must be a JSON object"]
    missing = [k for k in REQUIRED_TXN_FIELDS if k not in raw]
    if missing:
        return None, [f"missing fields: {', '.join(missing)}"]
    try:
        amount = float(raw["amount"])
    except (TypeError, ValueError):
        return None, ["amount must be a number"]
    if amount <= 0:
        return None, ["amount must be greater than zero"]
    txn = dict(raw, amount=amount, ts=int(raw.get("ts") or time.time()),
               new_device=int(bool(raw.get("new_device", 0))))
    return txn, []


def create_app():
    app = Flask(__name__)
    app.config["API_KEY"] = os.getenv("FRAUDGUARD_API_KEY", "dev-key-change-me")
    service = FraudService()
    app.extensions["fraud_service"] = service

    def require_key(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            supplied = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(supplied, app.config["API_KEY"]):
                return jsonify(error="invalid or missing X-API-Key"), 401
            return view(*args, **kwargs)
        return wrapper

    @app.get("/")
    def dashboard():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model_version=service.version, threshold=round(service.threshold, 3))

    @app.post("/api/v1/transactions/score")
    @require_key
    def score_one():
        started = time.perf_counter()
        txn, errors = validate_txn(request.get_json(silent=True))
        if errors:
            return jsonify(error=errors), 400
        result = service.score_transactions([txn])[0]
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return jsonify(result)

    @app.post("/api/v1/transactions/batch")
    @require_key
    def score_batch():
        started = time.perf_counter()
        payload = request.get_json(silent=True)
        if not isinstance(payload, list) or not payload:
            return jsonify(error="body must be a non-empty JSON array"), 400
        if len(payload) > MAX_BATCH:
            return jsonify(error=f"max {MAX_BATCH} transactions per batch"), 400
        txns, problems = [], []
        for i, raw in enumerate(payload):
            txn, errors = validate_txn(raw)
            if errors:
                problems.append({"index": i, "errors": errors})
            else:
                txns.append(txn)
        if problems:
            return jsonify(error="validation failed", details=problems), 400
        results = service.score_transactions(txns)
        return jsonify(count=len(results), flagged=sum(r["flagged"] for r in results), results=results,
                       latency_ms=round((time.perf_counter() - started) * 1000, 1))

    @app.post("/api/v1/urls/scan")
    @require_key
    def scan_url():
        body = request.get_json(silent=True) or {}
        url = str(body.get("url", "")).strip()
        if not url:
            return jsonify(error="url is required"), 400
        return jsonify(service.scan_url(url))

    @app.get("/api/v1/alerts")
    @require_key
    def alerts():
        return jsonify(alerts=list(service.alerts))

    @app.get("/api/v1/metrics")
    @require_key
    def metrics():
        path = MODEL_DIR / "metrics.json"
        if not path.exists():
            return jsonify(error="run train.py first"), 404
        return app.response_class(path.read_text(), mimetype="application/json")

    @app.get("/api/v1/redteam/report")
    @require_key
    def redteam_report():
        path = BASE_DIR / "reports" / "redteam_latest.json"
        if not path.exists():
            return jsonify(error="run `python red_team.py` first"), 404
        return app.response_class(path.read_text(), mimetype="application/json")

    return app
app = create_app()  # Exposes the module-level instance for Gunicorn

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5001, debug=False)
