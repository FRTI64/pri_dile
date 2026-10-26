# e1_robust_league_20261026 — cross-play / league report

- **Name:** e1_robust_league_20261026
- **Created:** 2026-10-26T12:31:45
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-PPO
- **Fingerprint:** `ede7e2590c7873087797055409aec4edcc70f62fe46e45d11f7b01a318dd39e8`

E1 post-training comparison for the domain-randomized A60 fine-tune. Same suites/seeds as E0; self-play candidates omitted for runtime. heldout_v2 is report-only.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| Final-tuned | rule | 542.24 | 0.8540 | 360.92 | 334.05 | 1.000 |
| Resp3-tuned | rule | 541.83 | 0.8698 | 358.40 | 326.20 | 0.950 |
| Triple-majority | rule | 543.80 | 0.8739 | 357.34 | 337.55 | 1.000 |
| RL-PPO | learned | 550.80 | 1.0000 | 387.26 | 310.55 | 1.000 |
| RL-PPO-Hard60 | learned | 552.11 | 0.9928 | 391.13 | 299.20 | 1.000 |
| RL-Robust | learned | 552.65 | 0.9964 | 388.98 | 311.70 | 1.000 |
| Adv-RL | learned | 551.69 | 0.9815 | 387.14 | 325.55 | 0.950 |

## Per-suite means

| Candidate | official | blind | validation | heldout_v2 |
|---|---:|---:|---:|---:|
| Final-tuned | 546.35 | 547.75 | 489.23 | 490.04 |
| Resp3-tuned | 545.98 | 546.38 | 489.55 | 492.31 |
| Triple-majority | 547.37 | 551.33 | 491.21 | 491.94 |
| RL-PPO | 553.22 | 562.80 | 504.45 | 494.97 |
| RL-PPO-Hard60 | 554.27 | 563.87 | 507.85 | 500.07 |
| RL-Robust | 554.71 | 564.63 | 508.47 | 501.43 |
| Adv-RL | 554.05 | 562.97 | 501.87 | 502.39 |

## Cross-play payoff matrix

| row \ col | Final-tuned | Resp3-tuned | Triple-majority | RL-PPO | RL-PPO-Hard60 | RL-Robust | Adv-RL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final-tuned | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| Resp3-tuned | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| Triple-majority | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| RL-PPO | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| RL-PPO-Hard60 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| RL-Robust | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| Adv-RL | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |

## Meta-game

| Candidate | Elo | Nash weight | Pool exploitability | Best response |
|---|---:|---:|---:|---|
| Final-tuned | 1000.0 | 0.1423 | 0.0000 | Triple-majority |
| Resp3-tuned | 1000.0 | 0.1454 | 0.0000 | Triple-majority |
| Triple-majority | 1000.0 | 0.1431 | 0.0000 | Resp3-tuned |
| RL-PPO | 1000.0 | 0.1379 | 0.0000 | Triple-majority |
| RL-PPO-Hard60 | 1000.0 | 0.1427 | 0.0000 | Triple-majority |
| RL-Robust | 1000.0 | 0.1426 | 0.0000 | Triple-majority |
| Adv-RL | 1000.0 | 0.1459 | 0.0000 | Triple-majority |

Nash exploitability (score-difference units): **0.0000**; iterations=20000.

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| Final-tuned | 334.05 | 607.55 | +273.50 | 1.000 | E4-red-BC |
| Resp3-tuned | 326.20 | 572.70 | +246.50 | 0.950 | E4-red-BC |
| Triple-majority | 337.55 | 606.55 | +269.00 | 1.000 | E4-red-BC |
| RL-PPO | 310.55 | 654.05 | +343.50 | 1.000 | E4-red-BC |
| RL-PPO-Hard60 | 299.20 | 651.20 | +352.00 | 1.000 | E4-red-BC |
| RL-Robust | 311.70 | 670.70 | +359.00 | 1.000 | E4-red-BC |
| Adv-RL | 325.55 | 608.55 | +283.00 | 0.950 | E4-red-BC |

## Held-out v2 per family

| Candidate | Family | Mean |
|---|---|---:|
| RL-PPO | heldout_v2:reciprocal_v2 | 599.15 |
| RL-PPO | heldout_v2:punitive_v2 | 600.62 |
| RL-PPO | heldout_v2:noisy_v2 | 421.54 |
| RL-PPO | heldout_v2:extort_v2 | 419.15 |
| RL-PPO | heldout_v2:random_v2 | 409.11 |
| RL-PPO | heldout_v2:temporal_v2 | 447.57 |
| RL-PPO | heldout_v2:adaptive_v2 | 582.78 |
| RL-PPO-Hard60 | heldout_v2:reciprocal_v2 | 599.15 |
| RL-PPO-Hard60 | heldout_v2:punitive_v2 | 600.62 |
| RL-PPO-Hard60 | heldout_v2:noisy_v2 | 425.65 |
| RL-PPO-Hard60 | heldout_v2:extort_v2 | 441.87 |
| RL-PPO-Hard60 | heldout_v2:random_v2 | 420.57 |
| RL-PPO-Hard60 | heldout_v2:temporal_v2 | 450.90 |
| RL-PPO-Hard60 | heldout_v2:adaptive_v2 | 582.81 |
| Adv-RL | heldout_v2:reciprocal_v2 | 599.15 |
| Adv-RL | heldout_v2:punitive_v2 | 600.62 |
| Adv-RL | heldout_v2:noisy_v2 | 418.32 |
| Adv-RL | heldout_v2:extort_v2 | 421.75 |
| Adv-RL | heldout_v2:random_v2 | 442.27 |
| Adv-RL | heldout_v2:temporal_v2 | 462.74 |
| Adv-RL | heldout_v2:adaptive_v2 | 581.34 |

## Gate

- Data complete: **True**
- Cross-play complete: **True**
- Red complete: **True**
- Status: **PASS**

## Caveats

- This is a synthetic opponent pool; cross-play Elo/Nash is a pool-level
  approximation, not proof against an arbitrary best response.
- `heldout_v2` was created for E0 and must not be reused for iterative
  model selection. The older `final` suite is now validation.
- E4 red is a held-out stress test for non-BC candidates, but it was
  trained against a BC blue. It does not solve anti-exploitation.
- No candidate is promoted by this report. The main weights are unchanged.
