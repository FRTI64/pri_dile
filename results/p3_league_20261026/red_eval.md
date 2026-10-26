# p3_red_pool_20261026 — cross-play / league report

- **Name:** p3_red_pool_20261026
- **Created:** 2026-10-26T14:48:32
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-Repair
- **Fingerprint:** `50071ade431946d69e91f49562943b979c2da6612efc4b838e0d7b9a9fcc576b`

P3 red pool v1-v6 vs training/held-out blue candidates. Used to compute maximin red weights; not a blue promotion report.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| RL-PPO | blue | 0.00 | - | 0.00 | 338.99 | 1.000 |
| RL-PPO-Hard60 | blue | 0.00 | - | 0.00 | 338.16 | 1.000 |
| RL-Repair | blue | 0.00 | - | 0.00 | 336.57 | 1.000 |
| Adv-RL | blue | 0.00 | - | 0.00 | 348.83 | 1.000 |
| SelfPlay-GRU | blue | 0.00 | - | 0.00 | 336.79 | 1.000 |
| Final-tuned | rule | 0.00 | - | 0.00 | 362.88 | 1.000 |
| Triple-majority | rule | 0.00 | - | 0.00 | 357.39 | 0.992 |
| Blue-vs-Red-Comp-v1 | blue | 0.00 | - | 0.00 | 342.68 | 1.000 |

## Per-suite means

| Candidate |  |
|---|
| RL-PPO |  |
| RL-PPO-Hard60 |  |
| RL-Repair |  |
| Adv-RL |  |
| SelfPlay-GRU |  |
| Final-tuned |  |
| Triple-majority |  |
| Blue-vs-Red-Comp-v1 |  |

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| RL-PPO | 338.99 | 719.91 | +380.92 | 1.000 | Red-RL-v3 |
| RL-PPO-Hard60 | 338.16 | 721.66 | +383.50 | 1.000 | Red-RL-v3 |
| RL-Repair | 336.57 | 730.48 | +393.92 | 1.000 | Red-RL-v3 |
| Adv-RL | 348.83 | 686.75 | +337.92 | 1.000 | Red-RL-v4 |
| SelfPlay-GRU | 336.79 | 728.62 | +391.83 | 1.000 | Red-RL-v3 |
| Final-tuned | 362.88 | 651.47 | +288.58 | 1.000 | Red-RL-v2 |
| Triple-majority | 357.39 | 644.89 | +287.50 | 0.992 | Red-RL-v5-heldout |
| Blue-vs-Red-Comp-v1 | 342.68 | 714.84 | +372.17 | 1.000 | Red-RL-v3 |

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
