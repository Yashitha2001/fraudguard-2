"""Run the scenarios, then find where the detector misses attacks and why."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from features import ALL_COLUMNS
from redteam.attacker import make_episode, sample_victims

BINARY = {"is_night", "new_device", "country_mismatch"}
TARGET_DETECTION = 0.80


def scan(detector, scenarios, n_per_scenario, rng):
    episodes = []
    for name, params in scenarios.items():
        for victim in sample_victims(detector.profiles, n_per_scenario, rng):
            episodes.append({"scenario": name, "victim": victim["user_id"],
                             "txns": make_episode(params, victim, rng)})
    return detector.run_episodes(episodes)


def summarise(results, target=TARGET_DETECTION):
    rows = []
    for name in dict.fromkeys(e["scenario"] for e in results):
        eps = [e for e in results if e["scenario"] == name]
        attempted = sum(e["attempted"] for e in eps)
        rate = float(np.mean([e["caught"] for e in eps]))
        rows.append({
            "scenario": name,
            "episodes": len(eps),
            "detection_rate": round(rate, 3),
            "loss_rate": round(sum(e["stolen"] for e in eps) / attempted, 3),
            "avg_stolen": round(float(np.mean([e["stolen"] for e in eps])), 2),
            "blind_spot": rate < target,
        })
    return sorted(rows, key=lambda r: r["detection_rate"])


def _leaf_conditions(tree, names):
    t, leaves = tree.tree_, {}

    def walk(node, conds):
        if t.children_left[node] == -1:
            leaves[node] = conds
            return
        feat, thr = names[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], conds + [(feat, "<=", thr)])
        walk(t.children_right[node], conds + [(feat, ">", thr)])

    walk(0, [])
    return leaves


def _render(conds):
    bounds = {}
    for feat, op, thr in conds:
        lo, hi = bounds.get(feat, (-np.inf, np.inf))
        bounds[feat] = (lo, min(hi, thr)) if op == "<=" else (max(lo, thr), hi)
    parts = []
    for feat, (lo, hi) in bounds.items():
        if feat in BINARY:
            parts.append(f"{feat} == {0 if hi < 1 else 1}")
        elif np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"{lo:.2f} < {feat} <= {hi:.2f}")
        elif np.isfinite(hi):
            parts.append(f"{feat} <= {hi:.2f}")
        else:
            parts.append(f"{feat} > {lo:.2f}")
    return " AND ".join(parts) or "(everything)"


def find_blind_spots(results, threshold, clean_legit, min_missed=15):
    """Fit a small decision tree to predict which attack transactions got through, and turn its leaves into rules.

    For each rule we also check how much normal traffic falls in the same region. If a lot
    does, closing the gap would mean flagging real customers."""
    X = pd.concat([e["frame"][ALL_COLUMNS] for e in results], ignore_index=True)
    label = np.concatenate([e["scores"] < threshold for e in results])
    owner = np.concatenate([[e["scenario"]] * len(e["scores"]) for e in results])
    if label.sum() < min_missed:
        return []

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(15, len(X) // 60), random_state=7)
    tree.fit(X, label)
    attack_leaf, legit_leaf = tree.apply(X), tree.apply(clean_legit[ALL_COLUMNS])
    conditions = _leaf_conditions(tree, ALL_COLUMNS)

    spots = []
    for leaf, conds in conditions.items():
        in_leaf = attack_leaf == leaf
        missed = int((in_leaf & label).sum())
        share = missed / max(int(in_leaf.sum()), 1)
        if missed < min_missed or share < 0.6:
            continue
        legit_pct = float((legit_leaf == leaf).mean() * 100)
        by_scenario = pd.Series(owner[in_leaf & label]).value_counts().head(3).to_dict()
        if legit_pct < 2:
            verdict = "closable: very little normal traffic looks like this"
        elif legit_pct < 10:
            verdict = "costly: flagging it would add a noticeable number of false positives"
        else:
            verdict = "ambiguous: looks the same as normal customers"
        spots.append({"rule": _render(conds), "missed_txns": missed, "missed_share": round(share, 3),
                      "legit_traffic_pct": round(legit_pct, 2), "top_scenarios": by_scenario,
                      "assessment": verdict})
    return sorted(spots, key=lambda s: s["missed_txns"], reverse=True)
