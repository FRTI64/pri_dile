# Protocol v2 baseline

- catalog: `f6705fbb2ced34dd66b3aec1b8ee2cddc629a8d2c6b54fc470ef009e8cc4e36a`
- dev: 20 fields × 299 opponents, seed 20261027
- selection rule: top 3 by `dev_param` mean, pre-registered before the test run
- test_v5: 30 fields × 299 opponents, seed 20261027

## 1. Dev (all candidates)

| # | candidate | dev_param | min | p05 | min_norm | worst family | dev_family |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Adv-RL | 520.19 | 496.78 | 504.19 | 0.9904 | 218.91 | 583.03 |
| 2 | RL-Repair | 520.09 | 497.56 | 506.04 | 0.9898 | 295.47 | 582.71 |
| 3 | RL-PPO-Hard60 | 514.18 | 492.74 | 498.25 | 0.9982 | 238.42 | 582.73 |
| 4 | Adv-GRU | 512.71 | 489.71 | 494.29 | 0.9002 | 186.70 | 582.96 |
| 5 | Adv-MLP | 512.18 | 489.68 | 497.11 | 0.8982 | 186.29 | 582.98 |
| 6 | Bayes-base | 510.95 | 486.14 | 492.15 | 0.7846 | 162.72 | 583.39 |
| 7 | Bayes-tuned | 510.69 | 485.93 | 491.24 | 0.7677 | 159.23 | 583.39 |
| 8 | GRU | 510.68 | 490.01 | 494.34 | 0.8856 | 183.69 | 582.98 |
| 9 | RL-PPO | 510.55 | 488.00 | 494.82 | - | 207.41 | 582.53 |
| 10 | Neural | 509.92 | 485.76 | 493.12 | 0.8927 | 185.15 | 582.97 |
| 11 | SelfPlay-GRU | 506.58 | 484.85 | 492.69 | 0.9148 | 189.73 | 582.34 |
| 12 | Triple-majority | 505.63 | 479.80 | 481.19 | 0.8776 | 186.24 | 582.88 |
| 13 | Final+Resp3 | 503.19 | 479.17 | 487.33 | 0.8214 | 170.37 | 582.94 |
| 14 | Resp3-tuned | 499.57 | 473.86 | 480.90 | 0.8049 | 183.35 | 582.89 |
| 15 | Final-tuned | 493.08 | 471.99 | 476.93 | 0.7159 | 189.31 | 582.85 |
| 16 | Resp3-base | 486.38 | 471.02 | 474.81 | 0.6120 | 210.03 | 582.45 |
| 17 | Final-base | 485.73 | 466.80 | 473.66 | 0.6125 | 186.85 | 582.83 |
| 18 | Harvest-tuned | 479.21 | 462.11 | 465.14 | 0.7379 | 171.32 | 537.27 |
| 19 | Harvest45 | 467.48 | 452.34 | 453.93 | 0.5315 | 180.90 | 460.38 |
| 20 | GapStart | 466.34 | 445.30 | 450.94 | 0.1772 | 107.00 | 209.49 |

Top 3 by `dev_param` mean: Adv-RL, RL-Repair, RL-PPO-Hard60.

`dev_family` is saturated for most generalist policies (all near 583); it is a pathology gate, not a ranking metric. `dev_param` is dominated by the `extort` family, so `min_norm` / per-family numbers matter more than the mean alone.

## 2. test_v5 (top 3, one-shot)

| candidate | mean | min | p05 | worst family |
|---|---:|---:|---:|---:|
| Adv-RL | 467.95 | 447.26 | 449.29 | 126.98 |
| RL-Repair | 477.05 | 459.11 | 461.38 | 115.52 |
| RL-PPO-Hard60 | 469.91 | 451.01 | 453.36 | 130.91 |

### Paired field-level differences

| comparison | delta | 95% bootstrap CI | significant |
|---|---:|---|---|
| Adv-RL - RL-Repair | -9.10 | [-10.42, -7.84] | yes |
| Adv-RL - RL-PPO-Hard60 | -1.96 | [-3.17, -0.71] | yes |
| RL-Repair - RL-PPO-Hard60 | +7.14 | [+6.44, +7.88] | yes |

### test_v5 family means

| family | Adv-RL | RL-Repair | RL-PPO-Hard60 |
|---|---:|---:|---:|
| adaptive | 604.00 | 604.00 | 604.00 |
| adaptive_probe | 618.41 | 666.86 | 664.67 |
| anchor | 487.11 | 484.51 | 484.46 |
| defect | 784.68 | 800.97 | 800.89 |
| early_harvest | 126.98 | 115.52 | 130.91 |
| early_probe | 500.58 | 550.97 | 511.41 |
| extort | 341.37 | 388.44 | 348.25 |
| forgiving | 595.00 | 595.00 | 595.00 |
| frequent_kamikaze | 192.20 | 171.81 | 172.57 |
| hyper_extort | 156.09 | 221.24 | 161.95 |
| late | 321.05 | 313.70 | 321.38 |
| nice | 604.00 | 604.00 | 604.00 |
| noisy | 421.90 | 444.55 | 433.54 |
| probe_harvest | 393.40 | 394.91 | 393.19 |
| punitive | 604.00 | 604.00 | 604.00 |
| random | 558.76 | 565.27 | 564.57 |
| temporal | 485.49 | 465.47 | 452.25 |
| wild | 424.14 | 454.01 | 430.98 |

## 3. Red exploitability

| candidate | dev_red blue | dev_red red win | test_red blue | test_red red win |
|---|---:|---:|---:|---:|
| Adv-RL | 375.90 | 100.0% | 342.95 | 100.0% |
| RL-Repair | 378.05 | 100.0% | 314.65 | 100.0% |
| RL-PPO-Hard60 | 378.00 | 100.0% | 319.00 | 100.0% |

Red-v5 / Red-v6 still win 100% of matches; anti-exploitation remains unsolved.

## 4. Usage ledger

- `test_v5` / `Adv-RL` — 2026-10-27T15:14:59 — clean
- `test_v5` / `RL-Repair` — 2026-10-27T15:14:59 — clean
- `test_v5` / `RL-PPO-Hard60` — 2026-10-27T15:14:59 — clean
- `test_red` / `Adv-RL` — 2026-10-27T15:15:08 — clean
- `test_red` / `RL-Repair` — 2026-10-27T15:15:08 — clean
- `test_red` / `RL-PPO-Hard60` — 2026-10-27T15:15:08 — clean

## Notes

- Old `official` / `blind` numbers are legacy and cannot be mixed with this table.
- `test_v5` was run once for the pre-registered top 3; do not tune on it. A new candidate needs a new test suite (`test_v6`).
- The 25% test-only component now contains adversarial families (`early_harvest`, `frequent_kamikaze`, `probe_harvest`, `hyper_extort`, `early_probe`, `adaptive_probe`).
- Anti-exploitation remains unsolved: use the red pool / league harness separately; `test_v5` is not a red best-response test.
