# blue_vs_red_comp_eval_20261026 — cross-play / league report

- **Name:** blue_vs_red_comp_eval_20261026
- **Created:** 2026-10-26T13:54:13
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-PPO
- **Fingerprint:** `3dd8666ef6d4921fa3c5e61c60e3451550050144ac883913aa018e92fb7d61eb`

Evaluate the competitive/mixed blue response against training red and held-out E4 red.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| Final-tuned | rule | 0.00 | - | 0.00 | 340.93 | 1.000 |
| Triple-majority | rule | 0.00 | - | 0.00 | 333.40 | 1.000 |
| RL-PPO | blue | 0.00 | - | 0.00 | 317.20 | 1.000 |
| RL-PPO-Hard60 | blue | 0.00 | - | 0.00 | 320.98 | 1.000 |
| RL-Repair | blue | 0.00 | - | 0.00 | 304.45 | 1.000 |
| Blue-vs-Red-Comp | blue | 0.00 | - | 0.00 | 317.48 | 1.000 |
| Adv-RL | blue | 0.00 | - | 0.00 | 332.80 | 0.975 |
| SelfPlay-GRU | blue | 0.00 | - | 0.00 | 312.05 | 1.000 |

## Per-suite means

| Candidate |  |
|---|
| Final-tuned |  |
| Triple-majority |  |
| RL-PPO |  |
| RL-PPO-Hard60 |  |
| RL-Repair |  |
| Blue-vs-Red-Comp |  |
| Adv-RL |  |
| SelfPlay-GRU |  |

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| Final-tuned | 340.93 | 620.92 | +280.00 | 1.000 | Red-RL-v1 |
| Triple-majority | 333.40 | 598.90 | +265.50 | 1.000 | Red-RL-v1 |
| RL-PPO | 317.20 | 673.95 | +356.75 | 1.000 | Red-RL-v1 |
| RL-PPO-Hard60 | 320.98 | 689.48 | +368.50 | 1.000 | Red-RL-v1 |
| RL-Repair | 304.45 | 715.70 | +411.25 | 1.000 | Red-RL-v1 |
| Blue-vs-Red-Comp | 317.48 | 686.98 | +369.50 | 1.000 | Red-RL-v1 |
| Adv-RL | 332.80 | 637.30 | +304.50 | 0.975 | Red-RL-v1 |
| SelfPlay-GRU | 312.05 | 684.05 | +372.00 | 1.000 | Red-RL-v1 |

## Held-out v2 per family

| Candidate | Family | Mean |
|---|---|---:|

## Gate

- Data complete: **True**
- Cross-play complete: **False**
- Red complete: **True**
- Status: **INCOMPLETE**

## Caveats

- This is a synthetic opponent pool; cross-play Elo/Nash is a pool-level
  approximation, not proof against an arbitrary best response.
- `heldout_v2` was created for E0 and must not be reused for iterative
  model selection. The older `final` suite is now validation.
- E4 red is a held-out stress test for non-BC candidates, but it was
  trained against a BC blue. It does not solve anti-exploitation.
- No candidate is promoted by this report. The main weights are unchanged.
