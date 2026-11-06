"""Compares row-by-row feature extraction with the vectorised version.

    python benchmark.py

Run train.py first. Timings depend on your machine, so use your own numbers.
"""
import time

import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, MODEL_DIR
from features import FEATURE_COLUMNS, RAW_COLUMNS, build_features, build_features_slow, prepare


def timed(fn, repeats=1):
    """Run fn a few times, return its last result and the fastest time."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return out, best


def main():
    df = prepare(pd.read_csv(DATA_DIR / "transactions.csv"))
    print(f"{len(df):,} transactions\n")

    slow, t_slow = timed(lambda: build_features_slow(df))
    fast, t_fast = timed(lambda: build_features(df), repeats=3)
    same = np.allclose(slow.to_numpy(float), fast.to_numpy(float))
    print(f"feature extraction  row-by-row: {t_slow:7.2f}s  ({len(df) / t_slow:10,.0f} txn/s)")
    print(f"feature extraction  vectorised: {t_fast:7.2f}s  ({len(df) / t_fast:10,.0f} txn/s)")
    print(f"  -> {(t_slow - t_fast) / t_slow * 100:.1f}% less time, {t_slow / t_fast:.0f}x throughput, same output: {same}\n")

    baseline = joblib.load(MODEL_DIR / "baseline.joblib")
    tuned = joblib.load(MODEL_DIR / "fraud_model.joblib")["model"]
    _, t_base = timed(lambda: baseline.predict_proba(fast[RAW_COLUMNS]), repeats=3)
    _, t_tuned = timed(lambda: tuned.predict_proba(fast[FEATURE_COLUMNS]), repeats=3)
    print(f"model scoring  baseline RF: {t_base:.3f}s | tuned RF: {t_tuned:.3f}s")

    old = t_slow + t_base
    new = t_fast + t_tuned
    print(f"\nend to end (extract + score): {old:.2f}s -> {new:.2f}s  ({(old - new) / old * 100:.1f}% faster)")


if __name__ == "__main__":
    main()
