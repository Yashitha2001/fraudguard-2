"""Trains the transaction model and the URL model.

    python train.py                  # fake data
    python train.py --csv my.csv     # your own transactions

The train/test split is by time (first 75% of the timeline vs the last 25%), since that's
how the model would be used in practice. Features only look backwards, so building them
on the whole frame at once doesn't leak anything from the future.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split

import data_gen
from config import DATA_DIR, MODEL_DIR, MODEL_VERSION, RANDOM_STATE
from features import FEATURE_COLUMNS, RAW_COLUMNS, build_features, prepare
from url_features import URL_FEATURES, make_url_dataset, url_frame

PARAM_SPACE = {
    "n_estimators": [100, 150, 200],
    "max_depth": [10, 16, 24, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", 0.5],
    "class_weight": [None, "balanced_subsample"],
}


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
    }


def best_f1_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[np.argmax(f1)])


def chrono_split(df):
    """Boolean masks for the first 75% of time (train) and the rest (test)."""
    cutoff = df["ts"].quantile(0.75)
    return (df["ts"] <= cutoff).to_numpy(), (df["ts"] > cutoff).to_numpy()


def gain(old, new):
    return round((new - old) / old * 100, 1) if old else None


def train_transaction_model(df):
    t0 = time.perf_counter()
    df = prepare(df)
    X = build_features(df)
    print(f"features: {X.shape[1]} cols x {len(X):,} rows in {time.perf_counter() - t0:.2f}s")
    y = df["is_fraud"].to_numpy()

    tr, te = chrono_split(df)
    print(f"train {tr.sum():,} ({y[tr].mean():.2%} fraud) | test {te.sum():,} ({y[te].mean():.2%} fraud)")

    # baseline: plain forest, no history features, 0.5 cutoff
    baseline = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    baseline.fit(X.loc[tr, RAW_COLUMNS], y[tr])
    base = evaluate(y[te], baseline.predict_proba(X.loc[te, RAW_COLUMNS])[:, 1], 0.5)

    # tuned: all behaviour features, hyperparameters picked on PR-AUC
    # (accuracy tells you nothing when only 1.5% of rows are fraud)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        PARAM_SPACE, n_iter=8, scoring="average_precision",
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    search.fit(X.loc[tr, FEATURE_COLUMNS], y[tr])
    print("best params:", search.best_params_)

    model = search.best_estimator_
    oof = cross_val_predict(model, X.loc[tr, FEATURE_COLUMNS], y[tr],
                            cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                            method="predict_proba", n_jobs=-1)[:, 1]
    threshold = best_f1_threshold(y[tr], oof)
    tuned = evaluate(y[te], model.predict_proba(X.loc[te, FEATURE_COLUMNS])[:, 1], threshold)

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    report = {
        "model_version": MODEL_VERSION,
        "test_rows": int(te.sum()),
        "baseline": base,
        "tuned": tuned,
        "improvement_pct": {k: gain(base[k], tuned[k]) for k in ("recall", "f1", "pr_auc")},
        "best_params": search.best_params_,
        "feature_importance": importances.round(4).to_dict(),
    }
    joblib.dump(baseline, MODEL_DIR / "baseline.joblib")
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURE_COLUMNS,
                 "version": MODEL_VERSION}, MODEL_DIR / "fraud_model.joblib")
    return report


def train_url_model():
    df = make_url_dataset()
    X, y = url_frame(df["url"]), df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=2,
                                   random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr, y_tr)
    metrics = evaluate(y_te.to_numpy(), model.predict_proba(X_te)[:, 1], 0.5)
    joblib.dump({"model": model, "features": URL_FEATURES}, MODEL_DIR / "url_model.joblib")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="transactions CSV (columns are listed in data_gen.py)")
    args = parser.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv) if args.csv else data_gen.generate()
    df.to_csv(DATA_DIR / "transactions.csv", index=False)
    report = train_transaction_model(df)
    report["url_model"] = train_url_model()

    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print("\nBaseline :", report["baseline"])
    print("Tuned    :", report["tuned"])
    print("Relative improvement (%):", report["improvement_pct"])
    print("URL model:", report["url_model"])
    print("Top features:", dict(list(report["feature_importance"].items())[:5]))


if __name__ == "__main__":
    main()
