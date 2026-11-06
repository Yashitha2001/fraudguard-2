"""URL features for the phishing model, plus a made-up URL dataset to train it on."""
import math
import random
import re
import string
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_WORDS = ("login", "signin", "verify", "secure", "account", "update", "confirm",
                    "bank", "password", "wallet", "otp", "billing", "suspend", "card")
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly")
RISKY_TLDS = {"xyz", "top", "click", "tk", "ml", "ga", "cf", "gq", "work", "zip", "country"}
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

URL_FEATURES = [
    "url_length", "host_length", "path_length", "num_dots", "num_hyphens", "num_at",
    "num_params", "num_subdomains", "digit_ratio", "host_entropy", "uses_https",
    "host_is_ip", "has_punycode", "uses_shortener", "risky_tld", "suspicious_words",
]


def _entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum(c / len(text) * math.log2(c / len(text)) for c in counts.values())


def extract_url_features(url):
    url = (url or "").strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    parts = host.split(".") if host else []
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(parsed.path or ""),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_at": url.count("@"),
        "num_params": (parsed.query.count("&") + 1) if parsed.query else 0,
        "num_subdomains": max(len(parts) - 2, 0),
        "digit_ratio": sum(c.isdigit() for c in url) / len(url) if url else 0.0,
        "host_entropy": _entropy(host),
        "uses_https": int(parsed.scheme == "https"),
        "host_is_ip": int(bool(IP_RE.match(host))),
        "has_punycode": int("xn--" in host),
        "uses_shortener": int(any(s in host for s in SHORTENERS)),
        "risky_tld": int(bool(parts) and parts[-1] in RISKY_TLDS),
        "suspicious_words": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
    }


def url_frame(urls):
    return pd.DataFrame([extract_url_features(u) for u in urls], columns=URL_FEATURES)


def _tok(n):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_url_dataset(n=12000, seed=7):
    random.seed(seed)
    legit_hosts = ["chase.com", "bankofamerica.com", "wellsfargo.com", "paypal.com", "citi.com",
                   "hsbc.com", "capitalone.com", "stripe.com", "wise.com", "amazon.com"]
    legit_paths = ["", "/", "/accounts", "/help/contact", "/personal/credit-cards", "/about", "/security"]
    words = ["secure", "login", "verify", "update", "account", "confirm", "card"]
    banks = ["chase", "paypal", "wellsfargo", "citi", "hsbc", "stripe"]
    rows = []
    for _ in range(n // 2):
        rows.append(("https://www." + random.choice(legit_hosts) + random.choice(legit_paths), 0))
        bank, w = random.choice(banks), "-".join(random.sample(words, random.randint(1, 3)))
        tld, style = random.choice(sorted(RISKY_TLDS)), random.random()
        if style < 0.2:
            ip = ".".join(str(random.randint(1, 250)) for _ in range(4))
            url = f"http://{ip}/{bank}/{w}.php"
        elif style < 0.5:
            url = f"http://{bank}.{w}.{_tok(8)}.{tld}/{_tok(6)}?id={_tok(10)}&session={_tok(12)}"
        elif style < 0.75:
            url = f"https://{w}-{bank}.{tld}/signin/{_tok(5)}"
        else:  # lookalike .com domains, the harder case
            url = f"https://{bank}-{w}.com/{_tok(4)}/verify"
        rows.append((url, 1))
    df = pd.DataFrame(rows, columns=["url", "label"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
