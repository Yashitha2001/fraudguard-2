"""Tests whether a proposed fix actually helps.

How the comparison is kept fair:
  * Attacks are split by customer. Models retrained on attacks never see the customers
    they are scored on, and a couple of attack types are held out completely ("unseen").
  * Every candidate gets the same false-alarm rate as the deployed model. Without that,
    flagging more people would always look like an improvement.
  * Differences come with a paired bootstrap 95% interval over the same attacks.
  * Two guardrails: false positives and recall on real fraud must not get worse.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import MODEL_DIR, RANDOM_STATE
from features import ALL_COLUMNS, FEATURE_COLUMNS, build_features, prepare
from redteam.attacker import SCENARIOS, AttackParams, make_episode
from redteam.detector import outcome
from train import chrono_split

UNSEEN_FAMILIES = ["velocity_dodger", "low_and_slow"]  # kept out of adversarial training
FPR_GUARDRAIL = 1.15      # false-positive rate can be at most 115% of the deployed one
RECALL_GUARDRAIL = 0.02   # and recall on real fraud can drop by at most 2 points


@dataclass
class Candidate:
    name: str
    description: str
    features: list
    adversarial: bool = False
    retrain: bool = True
    fixed_threshold_scale: float = None  # only used by the lower-the-threshold option


CANDIDATES = [
    Candidate("lower_threshold", "Same model, alert threshold x0.4 (quick fix)",
              FEATURE_COLUMNS, retrain=False, fixed_threshold_scale=0.4),
    Candidate("adv_training", "Retrain with red-team episodes added as fraud examples",
              FEATURE_COLUMNS, adversarial=True),
    Candidate("novelty_features", "Add features for how unusual the merchant / hour is for this customer",
              ALL_COLUMNS),
    Candidate("novelty_plus_adv", "Novelty features + adversarial training", ALL_COLUMNS, adversarial=True),
]


def _episodes(families, victims, n, rng):
    eps = []
    for name, params in families.items():
        for i in range(n):
            v = victims[int(rng.integers(len(victims)))]
            eps.append({"scenario": name, "victim": v["user_id"], "txns": make_episode(params, v, rng)})
    return eps


def _outcomes(frames, scores, eps, threshold):
    amounts = [np.array([t["amount"] for t in e["txns"]], float) for e in eps]
    return [outcome(s, a, threshold) for s, a in zip(scores, amounts)]


def _bootstrap(delta, rng, n=2000):
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def run_comparison(detector, evolved, rng, adv_per_family=40, holdout_per_family=100, log=print):
    families = dict(SCENARIOS)
    for i, h in enumerate(evolved, 1):
        families[f"evolved_{i}"] = AttackParams(**h["params"])
    train_fams = {k: v for k, v in families.items() if k not in UNSEEN_FAMILIES}

    # normal data and the same time split train.py uses
    df = prepare(detector.hist)
    X_all, y = build_features(df), df["is_fraud"].to_numpy()
    tr, te = chrono_split(df)
    legit_te = (y[te] == 0)
    best_params = json.loads((MODEL_DIR / "metrics.json").read_text())["best_params"]

    # attacks, split by customer
    ids = sorted(detector.profiles)
    rng.shuffle(ids)
    half = len(ids) // 2
    train_victims = [detector.profiles[i] for i in ids[:half]]
    hold_victims = [detector.profiles[i] for i in ids[half:]]
    adv_eps = _episodes(train_fams, train_victims, adv_per_family, rng)
    adv_frame = pd.concat(detector.featurize(adv_eps), ignore_index=True)
    hold_eps = _episodes(families, hold_victims, holdout_per_family, rng)
    hold_frames = detector.featurize(hold_eps)
    fam_of = np.array([e["scenario"] for e in hold_eps])
    log(f"  clean train rows {tr.sum():,} | adversarial rows {len(adv_frame):,} | "
        f"held-out attack episodes {len(hold_eps)} ({len(families)} families, {len(UNSEEN_FAMILIES)} unseen)")

    # deployed model as the reference point
    base_clean = detector.model.predict_proba(X_all.loc[te, detector.features])[:, 1]
    fpr0 = float((base_clean[legit_te] >= detector.threshold).mean())
    recall0 = float((base_clean[~legit_te] >= detector.threshold).mean())
    base_scores = detector.score_frames(hold_frames)
    base_out = _outcomes(hold_frames, base_scores, hold_eps, detector.threshold)
    base_caught = np.array([o["caught"] for o in base_out], float)
    base_loss = np.array([o["stolen"] / o["attempted"] for o in base_out])

    def family_rates(caught):
        return {f: round(float(caught[fam_of == f].mean()), 3) for f in families}

    report = {"fpr_budget": round(fpr0, 5), "deployed": {
        "clean_fpr": round(fpr0, 5), "clean_recall": round(recall0, 3),
        "detection_rate": round(float(base_caught.mean()), 3),
        "loss_rate": round(float(base_loss.mean()), 3), "by_family": family_rates(base_caught)},
        "candidates": []}

    for cand in CANDIDATES:
        log(f"  evaluating {cand.name} ...")
        if cand.retrain:
            Xc, yc = X_all.loc[tr, cand.features], y[tr]
            if cand.adversarial:
                Xc = pd.concat([Xc, adv_frame[cand.features]], ignore_index=True)
                yc = np.concatenate([yc, np.ones(len(adv_frame), dtype=int)])
            model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params).fit(Xc, yc)
            model.n_jobs = 1
        else:
            model = detector.model
        clean = model.predict_proba(X_all.loc[te, cand.features])[:, 1]
        if cand.fixed_threshold_scale:
            thr = detector.threshold * cand.fixed_threshold_scale
        else:  # same false-alarm rate as the deployed model
            thr = float(np.quantile(clean[legit_te], 1 - fpr0))
        fpr = float((clean[legit_te] >= thr).mean())
        recall = float((clean[~legit_te] >= thr).mean())

        big = pd.concat(hold_frames, ignore_index=True)[cand.features]
        scores = np.split(model.predict_proba(big)[:, 1], np.cumsum([len(f) for f in hold_frames])[:-1])
        outs = _outcomes(hold_frames, scores, hold_eps, thr)
        caught = np.array([o["caught"] for o in outs], float)
        loss = np.array([o["stolen"] / o["attempted"] for o in outs])

        det_delta, loss_delta = caught - base_caught, loss - base_loss
        det_ci, loss_ci = _bootstrap(det_delta, rng), _bootstrap(loss_delta, rng)
        unseen = np.isin(fam_of, UNSEEN_FAMILIES)

        breaches = []
        if fpr > fpr0 * FPR_GUARDRAIL:
            breaches.append(f"false-positive rate {fpr:.3%} vs {fpr0:.3%} budget")
        if recall < recall0 - RECALL_GUARDRAIL:
            breaches.append(f"real-fraud recall fell {recall0:.1%} -> {recall:.1%}")
        if breaches:
            verdict = "REJECT"
        elif det_ci[0] > 0:
            verdict = "ACCEPT"
        elif det_ci[1] < 0:
            verdict, breaches = "REJECT", ["detection got worse"]
        else:
            verdict = "INCONCLUSIVE"

        report["candidates"].append({
            "name": cand.name, "description": cand.description, "verdict": verdict, "guardrail_breaches": breaches,
            "clean_fpr": round(fpr, 5), "clean_recall": round(recall, 3),
            "detection_rate": round(float(caught.mean()), 3),
            "detection_delta": round(float(det_delta.mean()), 4), "detection_delta_ci95": det_ci,
            "loss_rate": round(float(loss.mean()), 3),
            "loss_delta": round(float(loss_delta.mean()), 4), "loss_delta_ci95": loss_ci,
            "unseen_detection_delta": round(float(det_delta[unseen].mean()), 4),
            "by_family": family_rates(caught)})
    return report


def render_markdown(report):
    d = report["deployed"]
    lines = [f"Deployed model: detects {d['detection_rate']:.1%} of held-out attacks, "
             f"loses {d['loss_rate']:.1%} of attempted $, false-positive rate {d['clean_fpr']:.3%}.", "",
             "| Candidate | Verdict | Detection (delta, 95% CI) | $ loss (delta) | Unseen families | FPR |",
             "|---|---|---|---|---|---|"]
    for c in report["candidates"]:
        lines.append(f"| {c['name']} | **{c['verdict']}** | {c['detection_rate']:.1%} "
                     f"({c['detection_delta']:+.1%}, [{c['detection_delta_ci95'][0]:+.1%}, {c['detection_delta_ci95'][1]:+.1%}]) "
                     f"| {c['loss_rate']:.1%} ({c['loss_delta']:+.1%}) | {c['unseen_detection_delta']:+.1%} "
                     f"| {c['clean_fpr']:.3%} |")
        for b in c["guardrail_breaches"]:
            lines.append(f"|  | | guardrail: {b} | | | |")
    return "\n".join(lines)
