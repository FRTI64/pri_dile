# Protocol v3 baseline

- catalog: `00b435c7387ae1e2ffd07f105f6ea1ed56edf165a4b800dc8c8d577144b45d05`
- dev: 20 fields × 299 opponents, seed 20261027
- dev tiers: `dev_param`, `dev_family`, `dev_complex`, `dev_ood`
- selection rule: top 3 by `dev_param` mean, pre-registered before the test run
- test_v6: 40 fields × 299 opponents, seed 20261027

## 1. Dev (all candidates)

| # | candidate | dev_param | dev_family | dev_complex | dev_ood | min_norm | worst param | worst complex |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | RL-Repair | 532.44 | 582.92 | 420.38 | 489.84 | 0.9905 | 345.10 | 281.28 |
| 2 | RL-PPO-Hard60 | 527.97 | 582.94 | 410.09 | 485.12 | 0.9989 | 348.03 | 260.98 |
| 3 | Adv-RL | 526.58 | 583.20 | 415.19 | 487.22 | 0.9828 | 350.28 | 248.80 |
| 4 | RL-PPO | 524.37 | 582.72 | 397.90 | 478.61 | - | 348.42 | 240.95 |
| 5 | Final-tuned | 523.39 | 583.13 | 413.64 | 484.57 | 0.9473 | 330.05 | 228.91 |
| 6 | Final-base | 522.55 | 583.03 | 402.20 | 479.29 | 0.9183 | 330.92 | 220.87 |
| 7 | Resp3-base | 521.54 | 582.65 | 399.45 | 478.86 | 0.8924 | 330.25 | 228.41 |
| 8 | Resp3-tuned | 517.38 | 583.11 | 396.13 | 474.88 | 0.8917 | 330.63 | 215.22 |
| 9 | Triple-majority | 517.07 | 583.14 | 410.02 | 480.14 | 0.8357 | 331.62 | 225.37 |
| 10 | Final+Resp3 | 515.99 | 583.16 | 404.36 | 477.48 | 0.8708 | 332.09 | 211.17 |
| 11 | Neural | 515.27 | 583.18 | 408.54 | 479.38 | 0.8508 | 331.58 | 223.61 |
| 12 | GRU | 515.26 | 583.16 | 409.39 | 479.19 | 0.8401 | 331.80 | 231.27 |
| 13 | Adv-MLP | 515.24 | 583.18 | 406.65 | 478.26 | 0.8618 | 350.65 | 220.72 |
| 14 | SelfPlay-GRU | 513.85 | 582.45 | 390.57 | 471.19 | 0.8380 | 349.04 | 221.23 |
| 15 | Adv-GRU | 513.73 | 583.13 | 406.90 | 477.47 | 0.8690 | 350.62 | 229.83 |
| 16 | Harvest-tuned | 502.93 | 536.35 | 390.34 | 461.31 | 0.7922 | 339.23 | 220.10 |
| 17 | Bayes-base | 502.21 | 583.55 | 388.47 | 467.58 | 0.6675 | 351.34 | 214.40 |
| 18 | Bayes-tuned | 501.18 | 583.61 | 387.93 | 466.73 | 0.6675 | 351.34 | 214.01 |
| 19 | Harvest45 | 489.35 | 461.79 | 402.33 | 451.03 | 0.5389 | 325.51 | 222.28 |
| 20 | GapStart | 462.18 | 197.24 | 448.77 | 424.30 | 0.1846 | 111.51 | 269.58 |

Top 3 by `dev_param` mean: RL-Repair, RL-PPO-Hard60, Adv-RL.

`dev_complex` is the discriminating validation tier: standard families saturate near 583, while complex families separate the top candidates. `dev_ood` is a deliberately shifted mixture.

## 2. test_v6 (top 3, one-shot)

| candidate | mean | min | p05 | worst family |
|---|---:|---:|---:|---:|
| RL-Repair | 484.32 | 466.99 | 468.78 | 121.52 |
| RL-PPO-Hard60 | 476.54 | 459.33 | 463.12 | 136.57 |
| Adv-RL | 471.44 | 454.91 | 457.89 | 131.79 |

### Paired field-level differences

| comparison | delta | 95% bootstrap CI | significant |
|---|---:|---|---|
| RL-Repair - RL-PPO-Hard60 | +7.78 | [+7.02, +8.58] | yes |
| RL-Repair - Adv-RL | +12.88 | [+11.65, +14.08] | yes |
| RL-PPO-Hard60 - Adv-RL | +5.10 | [+3.86, +6.33] | yes |

### test_v6 family means

| family | RL-Repair | RL-PPO-Hard60 | Adv-RL |
|---|---:|---:|---:|
| adaptive | 600.81 | 593.55 | 561.08 |
| adaptive_mix | 316.05 | 294.21 | 317.06 |
| adaptive_probe | 659.90 | 656.31 | 607.00 |
| anchor | 481.33 | 480.97 | 483.13 |
| defect | 773.97 | 773.88 | 757.18 |
| delayed_extort | 398.56 | 380.54 | 394.59 |
| early_harvest | 121.52 | 136.57 | 131.79 |
| early_probe | 546.39 | 508.30 | 462.43 |
| extort | 428.33 | 397.26 | 384.23 |
| forgiving | 595.00 | 595.00 | 595.00 |
| frequent_kamikaze | 175.56 | 175.61 | 202.57 |
| hyper_extort | 215.11 | 164.09 | 156.95 |
| late | 430.22 | 430.09 | 427.18 |
| memory_two | 595.00 | 595.00 | 595.00 |
| meta_mix | 512.06 | 492.16 | 491.56 |
| nice | 595.00 | 595.00 | 595.00 |
| noisy | 472.59 | 471.62 | 473.14 |
| phase_switch | 386.24 | 382.53 | 379.36 |
| policy | 595.00 | 594.91 | 594.99 |
| probe_exploit | 272.82 | 248.50 | 236.18 |
| probe_harvest | 417.99 | 414.43 | 412.77 |
| punitive | 604.00 | 604.00 | 604.00 |
| random | 680.53 | 678.63 | 653.37 |
| temporal | 483.84 | 464.84 | 443.89 |
| wild | 422.92 | 411.68 | 417.78 |

## 3. Red exploitability

| candidate | dev_red blue | dev_red red win | test_red blue | test_red red win |
|---|---:|---:|---:|---:|
| RL-Repair | 378.05 | 100.0% | 314.65 | 100.0% |
| RL-PPO-Hard60 | 378.00 | 100.0% | 319.00 | 100.0% |
| Adv-RL | 375.90 | 100.0% | 342.95 | 100.0% |

Red-v5 / Red-v6 still win 100% of matches; anti-exploitation remains unsolved.

## 4. Usage ledger

- `test_v6` / `RL-Repair` — 2026-10-27T16:13:16 — clean
- `test_v6` / `RL-PPO-Hard60` — 2026-10-27T16:13:16 — clean
- `test_v6` / `Adv-RL` — 2026-10-27T16:13:16 — clean
- `test_red` / `RL-Repair` — 2026-10-27T16:13:26 — clean
- `test_red` / `RL-PPO-Hard60` — 2026-10-27T16:13:26 — clean
- `test_red` / `Adv-RL` — 2026-10-27T16:13:26 — clean

## Notes

- Protocol v2 / official / blind numbers are legacy and cannot be mixed with this table.
- `test_v6` was run once for the pre-registered top 3; do not tune on it. A new candidate needs a new test suite (`test_v7`).
- test_v6 composition: 35% seen-family test, 20% test-only adversarial, 20% complex, 15% policy, 10% anchors.
- `dev_family` is saturated for most generalist policies; it is a pathology gate. `dev_complex` is the main complex-behaviour signal.
- Anti-exploitation remains unsolved: use the red pool / league harness separately.
