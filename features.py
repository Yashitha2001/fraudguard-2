"""Feature building, done two ways.

OnlineFeatureStore handles one transaction at a time and keeps per-user state. The API
uses it. build_features does the same maths with pandas/numpy for training. The row-by-row
version was too slow for training (iterrows plus a dict per user), see benchmark.py. The
tests check that both versions give the same numbers.

FEATURE_COLUMNS is what the deployed model uses. EXTENDED_COLUMNS are extra features
the red team tests in redteam/defense_eval.py.
"""
import copy
from collections import Counter, deque

import numpy as np
import pandas as pd

from config import (CATEGORY_CODES, CHANNEL_CODES, DEFAULT_AVG_AMOUNT, MAX_GAP_SECONDS,
                    RATIO_CAP, VELOCITY_WINDOW)

FEATURE_COLUMNS = [
    "amount_log", "hour", "is_night", "category_code", "channel_code", "new_device",
    "country_mismatch", "txn_count_1h", "secs_since_last", "amount_ratio",
]
# how often this user has bought this category / shopped at this time of day before
EXTENDED_COLUMNS = ["cat_prior_share", "hour_prior_share"]
ALL_COLUMNS = FEATURE_COLUMNS + EXTENDED_COLUMNS
# columns that need no user history (used for the baseline model)
RAW_COLUMNS = ["amount_log", "hour", "is_night", "category_code", "channel_code", "new_device"]

KEY_SPAN = 10 ** 11  # bigger than any epoch second, so user*SPAN + ts sorts by user then time


class OnlineFeatureStore:
    def __init__(self):
        self._users = {}

    def compute(self, txn):
        """Features for a transaction, using only what we knew before it. Doesn't change state."""
        ts = int(txn["ts"])
        state = self._users.get(txn["user_id"])
        if state is None:
            prior_mean, gap, home, recent = DEFAULT_AVG_AMOUNT, MAX_GAP_SECONDS, txn["country"], 0
            cat_share = hour_share = 0.0
        else:
            prior_mean = state["sum"] / state["n"]
            gap = min(max(ts - state["last_ts"], 0), MAX_GAP_SECONDS)
            home = state["home"]
            recent = sum(1 for t in state["recent"] if t > ts - VELOCITY_WINDOW)
            cat_share = state["cats"][txn["merchant_category"]] / state["n"]
            hour_share = state["buckets"][((ts // 3600) % 24) // 4] / state["n"]
        hour = (ts // 3600) % 24
        return {
            "amount_log": float(np.log1p(txn["amount"])),
            "hour": hour,
            "is_night": int(hour < 6),
            "category_code": CATEGORY_CODES.get(txn["merchant_category"], -1),
            "channel_code": CHANNEL_CODES.get(txn["channel"], -1),
            "new_device": int(txn.get("new_device", 0)),
            "country_mismatch": int(txn["country"] != home),
            "txn_count_1h": recent + 1,  # counts this transaction too
            "secs_since_last": gap,
            "amount_ratio": min(txn["amount"] / prior_mean, RATIO_CAP),
            "cat_prior_share": cat_share,
            "hour_prior_share": hour_share,
        }

    def commit(self, txn):
        """Add a transaction to the user's history."""
        ts = int(txn["ts"])
        state = self._users.setdefault(txn["user_id"], {
            "sum": 0.0, "n": 0, "last_ts": ts, "home": txn["country"], "recent": deque(),
            "cats": Counter(), "buckets": Counter()})
        state["sum"] += float(txn["amount"])
        state["n"] += 1
        state["last_ts"] = ts
        state["cats"][txn["merchant_category"]] += 1
        state["buckets"][((ts // 3600) % 24) // 4] += 1
        recent = state["recent"]
        while recent and recent[0] <= ts - VELOCITY_WINDOW:
            recent.popleft()
        recent.append(ts)

    # The red team copies a customer into a scratch store so fake attacks
    # don't end up in the real history.
    def export_user(self, user_id):
        state = self._users.get(user_id)
        return copy.deepcopy(state) if state is not None else None

    def import_user(self, user_id, state):
        if state is not None:
            self._users[user_id] = copy.deepcopy(state)


def prepare(df):
    """Sort by user then time. Both feature builders expect this."""
    return df.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)


def build_features_slow(df):
    """Reference version: replay the transactions one at a time."""
    store = OnlineFeatureStore()
    rows = []
    for _, row in df.iterrows():
        txn = {k: row[k] for k in ("user_id", "ts", "amount", "merchant_category",
                                   "channel", "country", "new_device")}
        rows.append(store.compute(txn))
        store.commit(txn)
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


def build_features(df):
    """Vectorised version. df has to come out of prepare() first."""
    ts = df["ts"].to_numpy(dtype=np.int64)
    user_codes = pd.factorize(df["user_id"])[0].astype(np.int64)

    # number of txns in the last hour: binary search for where each row's window starts
    key = user_codes * KEY_SPAN + ts
    window_start = np.searchsorted(key, key - VELOCITY_WINDOW, side="right")
    count_1h = np.arange(len(key)) - window_start + 1

    grouped = df.groupby("user_id", sort=False)
    gap = grouped["ts"].diff().fillna(MAX_GAP_SECONDS).clip(0, MAX_GAP_SECONDS)

    prior_n = grouped.cumcount()
    prior_sum = grouped["amount"].cumsum() - df["amount"]
    prior_mean = (prior_sum / prior_n.where(prior_n > 0)).fillna(DEFAULT_AVG_AMOUNT)

    hour = (df["ts"] // 3600) % 24
    home = grouped["country"].transform("first")

    # earlier rows from the same user with the same category / time bucket
    safe_n = prior_n.where(prior_n > 0)
    cat_prior = df.groupby(["user_id", "merchant_category"], sort=False).cumcount()
    hour_prior = df.groupby(["user_id", hour // 4], sort=False).cumcount()

    return pd.DataFrame({
        "amount_log": np.log1p(df["amount"]),
        "hour": hour,
        "is_night": (hour < 6).astype(int),
        "category_code": df["merchant_category"].map(CATEGORY_CODES).fillna(-1).astype(int),
        "channel_code": df["channel"].map(CHANNEL_CODES).fillna(-1).astype(int),
        "new_device": df["new_device"].astype(int),
        "country_mismatch": (df["country"] != home).astype(int),
        "txn_count_1h": count_1h,
        "secs_since_last": gap,
        "amount_ratio": (df["amount"] / prior_mean).clip(upper=RATIO_CAP),
        "cat_prior_share": (cat_prior / safe_n).fillna(0.0),
        "hour_prior_share": (hour_prior / safe_n).fillna(0.0),
    })[ALL_COLUMNS]
