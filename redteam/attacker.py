"""Fraud scenarios, plus an evolutionary search that tries to find ways past the model.

A scenario is a bundle of settings (AttackParams). The presets below are common fraud
patterns; evolve() mutates them and keeps whichever steal the most without being caught."""
from dataclasses import asdict, dataclass, replace

import numpy as np

from config import COUNTRIES, MERCHANT_CATEGORIES

CATEGORY_MODES = ["high_risk", "victim_usual", "any"]
CHANNEL_MODES = ["web", "mobile", "pos", "victim_usual"]
COUNTRY_MODES = ["foreign", "home"]


@dataclass
class AttackParams:
    n_txns: int = 4
    amount_mult: float = 5.0        # x the victim's average ticket
    gap_seconds: float = 300.0      # time between attempts
    category_mode: str = "high_risk"
    channel_mode: str = "web"
    country_mode: str = "foreign"
    new_device_prob: float = 0.8
    start_hour: object = None       # None = any time, -1 = victim's usual hour, 0-23 = fixed
    micro_test: bool = False        # start with tiny charges to test the card


SCENARIOS = {
    "smash_and_grab":        AttackParams(4, 8.0, 180, "high_risk", "web", "foreign", 0.9),
    "card_testing":          AttackParams(6, 3.0, 120, "any", "web", "foreign", 0.8, micro_test=True),
    "velocity_dodger":       AttackParams(5, 3.0, 3900, "high_risk", "web", "foreign", 0.8),
    "low_and_slow":          AttackParams(6, 1.2, 7200, "any", "web", "home", 0.3),
    "home_mimic":            AttackParams(4, 5.0, 300, "high_risk", "web", "home", 0.8),
    "device_takeover":       AttackParams(4, 6.0, 240, "high_risk", "mobile", "foreign", 0.0),
    "daytime_camouflage":    AttackParams(3, 2.5, 1800, "victim_usual", "victim_usual", "home", 0.1, -1),
    "full_mimic":            AttackParams(3, 1.5, 5400, "victim_usual", "victim_usual", "home", 0.0, -1),
}


def sample_victims(profiles, k, rng):
    ids = list(profiles)
    picks = rng.choice(len(ids), size=k, replace=k > len(ids))
    return [profiles[ids[i]] for i in picks]


def make_episode(params, profile, rng):
    """Build the list of transactions for one attack on one customer."""
    last = int(profile["last_ts"])
    if params.start_hour is None:
        start = last + int(rng.integers(2, 48)) * 3600
    else:
        hour = profile["usual_hour"] if params.start_hour == -1 else params.start_hour
        start = (last // 86400 + int(rng.integers(1, 4))) * 86400 + int(hour) * 3600 + int(rng.integers(0, 3000))

    others = [c for c in COUNTRIES if c != profile["home_country"]]
    country = str(rng.choice(others)) if params.country_mode == "foreign" else profile["home_country"]
    n_micro = max(1, params.n_txns // 3) if params.micro_test else 0

    txns, t = [], start
    for i in range(params.n_txns):
        if i:
            t += max(30, int(params.gap_seconds * rng.uniform(0.8, 1.2)))
        if i < n_micro:
            amount = round(float(rng.uniform(1, 3)), 2)
        else:
            amount = round(float(profile["avg_amount"] * params.amount_mult * rng.lognormal(0, 0.25)), 2)
        if params.category_mode == "high_risk":
            cat = str(rng.choice(["gift_cards", "crypto", "electronics"], p=[0.4, 0.2, 0.4]))
        elif params.category_mode == "victim_usual":
            cat = str(rng.choice(profile["top_cats"]))
        else:
            cat = str(rng.choice(MERCHANT_CATEGORIES))
        channel = profile["usual_channel"] if params.channel_mode == "victim_usual" else params.channel_mode
        txns.append({"user_id": profile["user_id"], "ts": int(t), "amount": max(amount, 0.5),
                     "merchant_category": cat, "channel": channel, "country": country,
                     "new_device": int(rng.random() < params.new_device_prob)})
    return txns


def mutate(p, rng):
    q = replace(p)
    if rng.random() < 0.6:
        q.amount_mult = float(np.clip(p.amount_mult * np.exp(rng.normal(0, 0.4)), 0.5, 15))
    if rng.random() < 0.6:
        q.gap_seconds = float(np.clip(p.gap_seconds * np.exp(rng.normal(0, 0.6)), 30, 4 * 86400))
    if rng.random() < 0.3:
        q.n_txns = int(np.clip(p.n_txns + rng.integers(-1, 2), 2, 8))
    if rng.random() < 0.3:
        q.new_device_prob = float(np.clip(p.new_device_prob + rng.normal(0, 0.25), 0, 1))
    if rng.random() < 0.2:
        q.category_mode = str(rng.choice(CATEGORY_MODES))
    if rng.random() < 0.15:
        q.channel_mode = str(rng.choice(CHANNEL_MODES))
    if rng.random() < 0.15:
        q.country_mode = str(rng.choice(COUNTRY_MODES))
    if rng.random() < 0.15:
        q.start_hour = [None, -1, 2, 11, 15, 20][int(rng.integers(0, 6))]
    return q


def evolve(detector, rng, generations=6, pop_size=24, victims_per_eval=10, elite=6, log=print):
    """Evolutionary search that only sees the model's scores.

    fitness = dollars stolen + a small bonus for low scores, so there's still something to
    climb when every attempt starts out getting caught."""
    population = [mutate(p, rng) for p in SCENARIOS.values()]
    while len(population) < pop_size:
        population.append(mutate(list(SCENARIOS.values())[int(rng.integers(len(SCENARIOS)))], rng))

    history, evaluated = [], []
    for gen in range(1, generations + 1):
        victims = [v["user_id"] for v in
                   sample_victims(detector.profiles, victims_per_eval, rng)]
        episodes, owner = [], []
        for i, params in enumerate(population):
            for uid in victims:
                episodes.append({"victim": uid, "txns": make_episode(params, detector.profiles[uid], rng)})
                owner.append(i)
        detector.run_episodes(episodes)

        scored = []
        for i, params in enumerate(population):
            mine = [e for e, o in zip(episodes, owner) if o == i]
            stolen = np.mean([e["stolen"] for e in mine])
            slack = np.mean([e["attempted"] * (1 - e["max_score"]) for e in mine])
            scored.append({"params": params, "fitness": float(stolen + 0.25 * slack),
                           "stolen": float(stolen), "detection_rate": float(np.mean([e["caught"] for e in mine])),
                           "generation": gen})
        scored.sort(key=lambda s: s["fitness"], reverse=True)
        evaluated.extend(scored[:elite])
        best = scored[0]
        history.append({"generation": gen, "best_fitness": round(best["fitness"], 1),
                        "best_stolen": round(best["stolen"], 1),
                        "best_detection_rate": round(best["detection_rate"], 3),
                        "mean_detection_rate": round(float(np.mean([s["detection_rate"] for s in scored])), 3)})
        log(f"  gen {gen}: best steals ${best['stolen']:.0f}/episode, "
            f"caught {best['detection_rate']:.0%} (population mean caught {history[-1]['mean_detection_rate']:.0%})")

        parents = [s["params"] for s in scored[:elite]]
        population = list(parents) + [mutate(parents[int(rng.integers(len(parents)))], rng)
                                      for _ in range(pop_size - elite)]

    # keep the best few strategies that differ from each other
    hall, seen = [], set()
    for s in sorted(evaluated, key=lambda s: s["fitness"], reverse=True):
        key = (s["params"].category_mode, s["params"].channel_mode, s["params"].country_mode,
               round(s["params"].amount_mult), s["params"].start_hour)
        if key not in seen:
            seen.add(key)
            hall.append({"params": asdict(s["params"]), "fitness": s["fitness"],
                         "detection_rate": s["detection_rate"], "generation": s["generation"]})
        if len(hall) == 4:
            break
    return {"history": history, "hall_of_fame": hall}
