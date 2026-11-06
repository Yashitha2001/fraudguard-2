# FraudGuard red-team report

Generated 2026-11-05 16:58:53 | model rf-v1 | alert threshold 0.419

## 1. Detection by attack scenario

| Scenario | Episodes | Caught | $ lost (of attempted) | Avg $ stolen/episode |
|---|---|---|---|---|
| full_mimic **BLIND SPOT** | 30 | 0% | 100% | $294 |
| daytime_camouflage **BLIND SPOT** | 30 | 17% | 88% | $442 |
| evolved_3 **BLIND SPOT** | 30 | 23% | 86% | $1555 |
| evolved_2 **BLIND SPOT** | 30 | 33% | 82% | $1829 |
| evolved_1 **BLIND SPOT** | 30 | 40% | 72% | $1844 |
| low_and_slow **BLIND SPOT** | 30 | 47% | 79% | $311 |
| evolved_4 **BLIND SPOT** | 30 | 47% | 68% | $1600 |
| smash_and_grab | 30 | 100% | 0% | $0 |
| card_testing | 30 | 100% | 0% | $2 |
| velocity_dodger | 30 | 100% | 6% | $54 |
| home_mimic | 30 | 100% | 7% | $84 |
| device_takeover | 30 | 100% | 12% | $184 |

## 2. Adaptive attacker vs the deployed model

| Gen | Best attacker steals/episode | Best attacker caught | Population caught |
|---|---|---|---|
| 1 | $711 | 0% | 66% |
| 2 | $657 | 0% | 11% |
| 3 | $1160 | 50% | 5% |
| 4 | $1777 | 40% | 17% |

## 3. Blind spots

- `new_device == 0 AND secs_since_last > 658.50 AND category_code <= 7.50`: 764 missed txns (100% of attacks in region), 94.42% of legitimate traffic also lands here. *ambiguous: looks the same as normal customers.* Scenarios: evolved_1, evolved_3, evolved_2
- `new_device == 1 AND amount_ratio <= 1.58 AND secs_since_last > 5825.50`: 54 missed txns (76% of attacks in region), 1.09% of legitimate traffic also lands here. *closable: very little normal traffic looks like this.* Scenarios: low_and_slow, card_testing, daytime_camouflage
- `new_device == 0 AND secs_since_last > 658.50 AND category_code > 7.50`: 46 missed txns (71% of attacks in region), 2.82% of legitimate traffic also lands here. *costly: flagging it would add a noticeable number of false positives.* Scenarios: low_and_slow, velocity_dodger, home_mimic

## 4. Regression corpus

- smash_and_grab: guarded (100% caught, 15 episodes)
- card_testing: guarded (100% caught, 15 episodes)
- velocity_dodger: guarded (100% caught, 15 episodes)
- low_and_slow: known_gap (53% caught, 15 episodes)
- home_mimic: guarded (100% caught, 15 episodes)
- device_takeover: guarded (100% caught, 15 episodes)
- daytime_camouflage: known_gap (0% caught, 15 episodes)
- full_mimic: known_gap (0% caught, 15 episodes)
- evolved_1: known_gap (40% caught, 15 episodes)
- evolved_2: known_gap (20% caught, 15 episodes)
- evolved_3: known_gap (33% caught, 15 episodes)
- evolved_4: known_gap (47% caught, 15 episodes)

## 5. Do the candidate fixes help?

Deployed model: detects 59.3% of held-out attacks, loses 49.9% of attempted $, false-positive rate 0.166%.

| Candidate | Verdict | Detection (delta, 95% CI) | $ loss (delta) | Unseen families | FPR |
|---|---|---|---|---|---|
| lower_threshold | **REJECT** | 71.7% (+12.3%, [+9.8%, +15.0%]) | 38.9% (-11.0%) | +12.0% | 0.648% |
|  | | guardrail: false-positive rate 0.648% vs 0.166% budget | | | |
| adv_training | **ACCEPT** | 82.7% (+23.3%, [+19.8%, +26.8%]) | 24.1% (-25.7%) | -4.0% | 0.166% |
| novelty_features | **ACCEPT** | 60.2% (+0.8%, [+0.2%, +1.7%]) | 49.5% (-0.4%) | +0.0% | 0.166% |
| novelty_plus_adv | **ACCEPT** | 83.5% (+24.2%, [+20.8%, +27.5%]) | 23.2% (-26.7%) | -1.0% | 0.166% |