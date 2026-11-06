"""The model under attack, plus each customer's history.

Each attack runs against a copy of the customer's profile, so the shared store never
changes and every attack starts from the same state."""
import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, MODEL_DIR
from features import ALL_COLUMNS, OnlineFeatureStore

TXN_COLS = ["user_id", "ts", "amount", "merchant_category", "channel", "country", "new_device"]


def load_history(path=None):
    df = pd.read_csv(path or DATA_DIR / "transactions.csv")
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def build_profiles(hist, min_txns=15):
    """What an attacker who watched a customer would know: usual spend, hours, merchants."""
    legit = hist[hist["is_fraud"] == 0] if "is_fraud" in hist else hist
    g = legit.assign(hour=(legit["ts"] // 3600) % 24).groupby("user_id")
    prof = pd.DataFrame({
        "n": g.size(),
        "avg_amount": g["amount"].mean(),
        "last_ts": hist.groupby("user_id")["ts"].max(),
        "home_country": g["country"].agg(lambda s: s.mode().iat[0]),
        "usual_hour": g["hour"].agg(lambda s: int(s.mode().iat[0])),
        "usual_channel": g["channel"].agg(lambda s: s.mode().iat[0]),
        "top_cats": g["merchant_category"].agg(lambda s: list(s.value_counts().index[:3])),
    })
    prof = prof[prof["n"] >= min_txns]
    return {uid: dict(row, user_id=uid) for uid, row in prof.to_dict("index").items()}


def outcome(scores, amounts, threshold):
    """The account gets frozen at the first flagged transaction, so only earlier ones go through."""
    hits = np.flatnonzero(scores >= threshold)
    first = int(hits[0]) if len(hits) else None
    stolen = float(amounts[:first].sum()) if first is not None else float(amounts.sum())
    return {"caught": first is not None, "first_flag": first, "stolen": stolen,
            "attempted": float(amounts.sum()), "max_score": float(scores.max())}


class Detector:
    def __init__(self, model, threshold, features, store, hist, profiles, version="rf-v1"):
        self.model, self.threshold, self.features = model, threshold, features
        self.store, self.hist, self.profiles, self.version = store, hist, profiles, version

    @classmethod
    def load(cls, csv_path=None):
        bundle = joblib.load(MODEL_DIR / "fraud_model.joblib")
        hist = load_history(csv_path)
        store = OnlineFeatureStore()
        for row in hist[TXN_COLS].itertuples(index=False):
            store.commit(row._asdict())
        model = bundle["model"]
        model.n_jobs = 1
        return cls(model, bundle["threshold"], bundle["features"], store, hist,
                   build_profiles(hist), bundle["version"])

    def episode_features(self, victim, txns):
        scratch = OnlineFeatureStore()
        scratch.import_user(victim, self.store.export_user(victim))
        rows = []
        for txn in txns:
            rows.append(scratch.compute(txn))
            scratch.commit(txn)
        return pd.DataFrame(rows, columns=ALL_COLUMNS)

    def featurize(self, episodes):
        return [self.episode_features(e["victim"], e["txns"]) for e in episodes]

    def score_frames(self, frames):
        """Score many episodes in one model call. Returns one array of scores per episode."""
        scores = self.model.predict_proba(pd.concat(frames, ignore_index=True)[self.features])[:, 1]
        return np.split(scores, np.cumsum([len(f) for f in frames])[:-1])

    def run_episodes(self, episodes):
        """Score each episode and add the results to it (modifies the dicts in place)."""
        frames = self.featurize(episodes)
        for ep, frame, scores in zip(episodes, frames, self.score_frames(frames)):
            amounts = np.array([t["amount"] for t in ep["txns"]], dtype=float)
            ep.update(outcome(scores, amounts, self.threshold), scores=scores, frame=frame, amounts=amounts)
        return episodes
