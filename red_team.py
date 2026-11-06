"""Attacks our own fraud model with simulated fraud and reports what gets through.

    python red_team.py                 # full run
    python red_team.py --quick         # smaller and faster
    python red_team.py --skip-compare  # don't retrain the defence candidates

Writes reports/redteam_latest.json (the dashboard reads this) and reports/redteam_report.md,
and rewrites tests/regression/redteam_corpus.json, which pytest replays.
"""
import argparse
import json
import time

import numpy as np

from config import BASE_DIR
from features import ALL_COLUMNS, build_features, prepare
from redteam import blindspots, defense_eval, regression
from redteam.attacker import SCENARIOS, AttackParams, evolve
from redteam.detector import Detector
from train import chrono_split

REPORT_DIR = BASE_DIR / "reports"


def clean_legit_frame(hist):
    df = prepare(hist)
    X, y = build_features(df), df["is_fraud"].to_numpy()
    _, te = chrono_split(df)
    return X.loc[te & (y == 0), ALL_COLUMNS]


def markdown(report):
    out = ["# FraudGuard red-team report", "",
           f"Generated {report['generated_at']} | model {report['model_version']} | "
           f"alert threshold {report['threshold']}", "", "## 1. Detection by attack scenario", "",
           "| Scenario | Episodes | Caught | $ lost (of attempted) | Avg $ stolen/episode |", "|---|---|---|---|---|"]
    for s in report["scenarios"]:
        flag = " **BLIND SPOT**" if s["blind_spot"] else ""
        out.append(f"| {s['scenario']}{flag} | {s['episodes']} | {s['detection_rate']:.0%} | "
                   f"{s['loss_rate']:.0%} | ${s['avg_stolen']:.0f} |")
    out += ["", "## 2. Adaptive attacker vs the deployed model", "",
            "| Gen | Best attacker steals/episode | Best attacker caught | Population caught |", "|---|---|---|---|"]
    for h in report["evolution"]["history"]:
        out.append(f"| {h['generation']} | ${h['best_stolen']:.0f} | {h['best_detection_rate']:.0%} | {h['mean_detection_rate']:.0%} |")
    out += ["", "## 3. Blind spots", ""]
    if not report["blind_spots"]:
        out.append("No region of feature space met the blind-spot criteria.")
    for b in report["blind_spots"]:
        out.append(f"- `{b['rule']}`: {b['missed_txns']} missed txns ({b['missed_share']:.0%} of attacks in region), "
                   f"{b['legit_traffic_pct']}% of legitimate traffic also lands here. *{b['assessment']}.* "
                   f"Scenarios: {', '.join(b['top_scenarios'])}")
    out += ["", "## 4. Regression corpus", ""]
    for name, c in report["corpus"].items():
        out.append(f"- {name}: {c['status']} ({c['baseline_rate']:.0%} caught, {c['episodes']} episodes)")
    if report.get("defense"):
        out += ["", "## 5. Do the candidate fixes help?", "", defense_eval.render_markdown(report["defense"])]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=60, help="episodes per scenario in the scan")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--skip-compare", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.generations = 30, 4
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    print("Loading the deployed model and customer profiles...")
    detector = Detector.load()
    print(f"  {len(detector.profiles):,} customers to attack, alert threshold {detector.threshold:.3f}")

    print(f"\n[1/5] Evolving the attacker against the deployed model ({args.generations} generations)")
    evo = evolve(detector, rng, generations=args.generations)

    scenarios = dict(SCENARIOS)
    for i, h in enumerate(evo["hall_of_fame"], 1):
        scenarios[f"evolved_{i}"] = AttackParams(**h["params"])

    print(f"\n[2/5] Running {len(scenarios)} scenarios x {args.episodes} customers each")
    results = blindspots.scan(detector, scenarios, args.episodes, rng)
    summary = blindspots.summarise(results)
    for s in summary:
        mark = "  <-- blind spot" if s["blind_spot"] else ""
        print(f"  {s['scenario']:<20} caught {s['detection_rate']:>5.0%}   $ lost {s['loss_rate']:>5.0%}{mark}")

    print("\n[3/5] Looking for blind spots")
    spots = blindspots.find_blind_spots(results, detector.threshold, clean_legit_frame(detector.hist))
    for b in spots[:4]:
        print(f"  {b['rule']}\n    -> {b['missed_txns']} missed, {b['legit_traffic_pct']}% of legit traffic overlaps: {b['assessment']}")

    print("\n[4/5] Saving the regression corpus")
    corpus = regression.freeze(results, detector, per_scenario=25 if not args.quick else 15)
    print("  " + ", ".join(f"{n}={c['status']}" for n, c in corpus.items()))

    defense = None
    if not args.skip_compare:
        print("\n[5/5] Comparing possible fixes (retrains a few models)")
        defense = defense_eval.run_comparison(
            detector, evo["hall_of_fame"], rng,
            adv_per_family=20 if args.quick else 40, holdout_per_family=50 if args.quick else 100)
        print("\n" + defense_eval.render_markdown(defense))

    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model_version": detector.version,
              "threshold": round(float(detector.threshold), 3), "scenarios": summary,
              "evolution": evo, "blind_spots": spots, "corpus": corpus, "defense": defense,
              "runtime_seconds": round(time.perf_counter() - started, 1)}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "redteam_latest.json").write_text(json.dumps(report, indent=1, default=str))
    (REPORT_DIR / "redteam_report.md").write_text(markdown(report))
    print(f"\nDone in {report['runtime_seconds']}s. Reports in ./reports, corpus in tests/regression/")


if __name__ == "__main__":
    main()
