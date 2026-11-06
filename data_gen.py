"""Fake card transactions for training.

Fraud is generated as short bursts on one account (someone draining a stolen card),
which is what makes the velocity features worth having. To use real data, make a CSV
with the same columns: txn_id, user_id, ts, amount, merchant_category, channel,
country, new_device, is_fraud.
"""
import numpy as np
import pandas as pd

from config import COUNTRIES, MERCHANT_CATEGORIES, CHANNELS

START_TS = 1_730_000_000  # fixed so every run gives the same data

# how likely each hour of the day is for normal spending
HOUR_WEIGHTS = np.array([1, 1, 1, 1, 1, 2, 4, 6, 7, 7, 7, 8, 9, 8, 7, 7, 7, 8, 9, 9, 7, 5, 3, 2], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()

NORMAL_CAT_P = [0.28, 0.20, 0.10, 0.15, 0.05, 0.06, 0.07, 0.06, 0.02, 0.01]
FRAUD_CAT_P = [0.05, 0.03, 0.02, 0.10, 0.10, 0.30, 0.00, 0.05, 0.25, 0.10]
CAT_AMOUNT_MULT = {"travel": 4.0, "electronics": 3.5, "gift_cards": 1.5, "crypto": 2.5,
                   "utilities": 1.8, "fuel": 0.9, "grocery": 0.8, "restaurant": 0.7}


def _amounts(rng, scale, categories, mu, sigma):
    mult = np.array([CAT_AMOUNT_MULT.get(c, 1.0) for c in categories])
    return np.round(scale * mult * rng.lognormal(mu, sigma, len(scale)), 2)


def generate(n_users=2500, days=45, seed=42):
    rng = np.random.default_rng(seed)
    home = rng.choice(COUNTRIES[:4] + ["CA"], size=n_users, p=[0.55, 0.15, 0.1, 0.1, 0.1])
    scale = rng.lognormal(np.log(35), 0.5, n_users)
    activity = rng.lognormal(0, 0.6, n_users)
    activity /= activity.sum()

    # normal traffic
    n_normal = int(n_users * days * 0.9)
    u = rng.choice(n_users, size=n_normal, p=activity)
    day = rng.integers(0, days, n_normal)
    hour = rng.choice(24, size=n_normal, p=HOUR_WEIGHTS)
    ts = START_TS + day * 86400 + hour * 3600 + rng.integers(0, 3600, n_normal)
    cats = rng.choice(MERCHANT_CATEGORIES, size=n_normal, p=NORMAL_CAT_P)
    channel = rng.choice(CHANNELS, size=n_normal, p=[0.5, 0.25, 0.2, 0.05])
    abroad = rng.random(n_normal) < 0.04
    country = np.where(abroad, rng.choice(COUNTRIES, n_normal), home[u])
    new_dev = ((rng.random(n_normal) < 0.03) & np.isin(channel, ["web", "mobile"])).astype(int)
    normal = pd.DataFrame({
        "user_id": u, "ts": ts, "amount": _amounts(rng, scale[u], cats, 0, 0.6),
        "merchant_category": cats, "channel": channel, "country": country,
        "new_device": new_dev, "is_fraud": 0,
    })

    # fraud bursts
    n_ep = int(n_normal * 0.006)
    ep_user = rng.integers(0, n_users, n_ep)
    ep_len = rng.integers(1, 5, n_ep)
    ep_night = rng.random(n_ep) < 0.45
    ep_hour = np.where(ep_night, rng.integers(0, 6, n_ep), rng.integers(6, 24, n_ep))
    ep_start = START_TS + rng.integers(0, days, n_ep) * 86400 + ep_hour * 3600 + rng.integers(0, 3600, n_ep)
    sneaky = rng.random(n_ep) < 0.30  # slow, ordinary-looking fraud

    rows = []
    for i in range(n_ep):
        k = ep_len[i]
        if sneaky[i]:
            t = ep_start[i] + np.arange(k) * 5400
        else:
            t = ep_start[i] + np.cumsum(rng.integers(60, 720, k))
        c = rng.choice(MERCHANT_CATEGORIES, size=k, p=FRAUD_CAT_P)
        sc = np.full(k, scale[ep_user[i]])
        amt = _amounts(rng, sc, c, 0.2 if sneaky[i] else 1.3, 0.5 if sneaky[i] else 0.8)
        foreign = rng.random(k) < (0.2 if sneaky[i] else 0.55)
        other = rng.choice([x for x in COUNTRIES if x != home[ep_user[i]]], k)
        rows.append(pd.DataFrame({
            "user_id": ep_user[i], "ts": t, "amount": amt, "merchant_category": c,
            "channel": rng.choice(["web", "mobile", "pos"], k, p=[0.6, 0.35, 0.05]),
            "country": np.where(foreign, other, home[ep_user[i]]),
            "new_device": (rng.random(k) < 0.65).astype(int), "is_fraud": 1,
        }))

    df = pd.concat([normal] + rows, ignore_index=True)
    df["user_id"] = "U" + df["user_id"].astype(str).str.zfill(5)
    df = df.sort_values("ts").reset_index(drop=True)
    df.insert(0, "txn_id", ["T" + str(i).zfill(7) for i in range(len(df))])
    return df


if __name__ == "__main__":
    d = generate()
    print(len(d), "rows, fraud rate", round(d.is_fraud.mean() * 100, 2), "%")
