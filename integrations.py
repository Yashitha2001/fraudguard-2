"""Optional outbound calls (Safe Browsing, alert webhook). If either one is slow or down
we just move on, scoring shouldn't depend on them."""
import logging
import os
import threading

import requests

log = logging.getLogger("fraudguard.integrations")

GSB_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"


def check_url_reputation(url, timeout=2.0):
    """Look the URL up in Google Safe Browsing (needs GSB_API_KEY).

    status is "malicious", "clean" or "unavailable".
    """
    api_key = os.getenv("GSB_API_KEY")
    if not api_key:
        return {"source": "google_safe_browsing", "status": "unavailable", "detail": "no API key configured"}

    body = {
        "client": {"clientId": "fraudguard", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        resp = requests.post(GSB_ENDPOINT, params={"key": api_key}, json=body, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Safe Browsing lookup failed: %s", exc)
        return {"source": "google_safe_browsing", "status": "unavailable", "detail": "lookup failed"}

    matches = resp.json().get("matches", [])
    if matches:
        return {"source": "google_safe_browsing", "status": "malicious",
                "detail": matches[0].get("threatType", "UNKNOWN")}
    return {"source": "google_safe_browsing", "status": "clean"}


def send_alert_webhook(alert, timeout=3.0):
    """POST the alert to ALERT_WEBHOOK_URL (Slack, Teams, a SIEM...) from a background thread."""
    target = os.getenv("ALERT_WEBHOOK_URL")
    if not target:
        return

    def _post():
        try:
            requests.post(target, json={"text": f"[FraudGuard] {alert['summary']}", "alert": alert},
                          timeout=timeout)
        except requests.RequestException as exc:
            log.warning("alert webhook failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()
