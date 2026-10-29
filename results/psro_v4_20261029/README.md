# PSRO — psro_v4_20261029

- **Status:** experimental; not promoted
- **Created:** 2026-10-29T17:50:01
- **Config:** `/home/frti64/pri_dile/configs/psro/psro_v4_20261029.json`
- **Initial exploitability:** **290.002**
- **Final exploitability:** **0.000**
- **Final blue worst-case:** 33.125
- **Best red response:** `Red-RL-v1`
- **Deploy:** `/home/frti64/pri_dile/weights/psro_v4/psro_policy.json` (PSRO-BR-br_01)

## Blue population

| Blue | Gen | Meta weight | Expected vs red | Worst case vs red | Standalone JSON |
|---|---:|---:|---:|---:|---|
| RL-Switch | 0 | 0.0000 | -385.94 | -457.19 | - |
| RL-Wide160 | 0 | 0.0000 | -388.28 | -448.28 | - |
| RL-Pattern | 0 | 0.0000 | -389.54 | -465.00 | - |
| RL-Repair | 0 | 0.0000 | -402.66 | -478.28 | - |
| RL-PPO | 0 | 0.0000 | -375.63 | -472.50 | - |
| Final-tuned | 0 | 0.0000 | -276.56 | -290.00 | - |
| Triple-majority | 0 | 0.0000 | -285.93 | -290.94 | - |
| Adv-RL | 0 | 0.0000 | -326.25 | -353.12 | - |
| PSRO-BR-br_01 | 1 | 1.0000 | 33.14 | 33.12 | yes |
| PSRO-BR-br_02 | 2 | 0.0000 | -73.28 | -281.88 | yes |
| PSRO-BR-br_03 | 3 | 0.0000 | -10.00 | -79.84 | yes |

A `-` in **Standalone JSON** means a rule candidate that is deployed
through the meta-policy rather than a single learned JSON file.

## Red population

| Red | Meta weight | Training weight | Expected vs blue |
|---|---:|---:|---:|
| Red-RL-v1 | 1.0000 | 0.9000 | -33.12 |
| Red-RL-v3 | 0.0001 | 0.0500 | -265.94 |
| Red-RL-v4 | 0.0000 | 0.0500 | -203.75 |

## Rounds

| Round | Before exploitability | After exploitability | Delta | Added |
|---:|---:|---:|---:|---|
| 1 | 290.002 | 0.000 | -290.002 | PSRO-BR-br_01 |
| 2 | 0.000 | 0.000 | 0.000 | PSRO-BR-br_02 |
| 3 | 0.000 | 0.000 | 0.000 | PSRO-BR-br_03 |

## Caveats

- The meta-game is estimated from a finite seed set, not solved exactly.
- The best-response oracle is the existing PPO trainer; it is not a
  proof of a true best response.
- Blue exploitability is the red best-response gain against the maximin
  blue mixture.  Lower is better; a positive delta means the new
  best-response policy did not enter the maximin support.
- Red-v5 / Red-v6 and Protocol v4 `test_v7` remain held out.  A PSRO
  candidate is only promotable after the normal dev/test and C++ checks.
- The main weights (`weights/rl_policy_weights.json`) are unchanged.

## Diagnostic evaluation

These are **small diagnostic runs**, not a full Protocol v4 baseline.  The
10-field `dev_param` run uses 1024 opponents; the other tiers use 3 fields ×
512.  Held-out red uses 10 seeds.  See `eval_*.json` and `run_eval.sh`.

### Protocol v4 dev_param (10 fields × 1024)

| Candidate | dev_param | min_norm | worst family |
|---|---:|---:|---:|
| RL-Repair (reference) | **546.47** | 1.0000 | late 305.98 |
| **PSRO-V4-BR** | 486.93 | 0.5599 | noisy 260.04 |
| PSRO-V4-Meta | 487.50 | 0.5708 | noisy 265.11 |

Per-family trade-off (10×1024): PSRO-V4-BR gives up `noisy` (260.0 vs
464.4), `adaptive` (397.1 vs 585.8) and `extort` (366.2 vs 496.8), while
keeping `defect` (680.0), `wild` (456.4 vs 527.5), `late` (312.4 vs
306.0) and `axelrod` (472.1 vs 557.4).  The drop is concentrated in the
families that reward exploitation, which is the expected cost of the
anti-red objective.

Other tiers (3 fields × 512, same checkpoint):

| Tier | PSRO-V4-BR | RL-Repair |
|---|---:|---:|
| dev_complex | **436.43** | 434.28 |
| dev_ood | 477.72 | 515.19 |
| dev_axelrod | 498.11 | 582.46 |
| dev_family | 567.74 | - |

### Held-out Red-v5/v6 (10 seeds)

| Candidate | blue mean | red mean | diff | red win |
|---|---:|---:|---:|---:|
| **PSRO-V4-BR** | **530.55** | 135.05 | **-395.50** | **0.000** |
| PSRO-V4-Meta | 502.05 | 275.05 | -227.00 | 0.300 |
| Final-tuned | 376.65 | 671.15 | +294.50 | 1.000 |
| Triple-majority | 370.50 | 669.00 | +298.50 | 1.000 |
| RL-Repair | 348.35 | 726.85 | +378.50 | 1.000 |
| RL-Wide160 | 350.20 | 720.20 | +370.00 | 1.000 |
| RL-Switch | 348.15 | 726.15 | +378.00 | 1.000 |

`dev_red` (10 seeds): PSRO-V4-BR blue 422.9 / red 145.9, red win **0.0**;
RL-Repair blue 379.2 / red 715.2, red win 1.0.  PSRO-V4-Meta is 20.0%
because the per-match member RNG is reseeded and the only member is a
stochastic policy; the single exported best response is the stable
anti-red artifact.

### Decision

**No-Go for promotion.**  The stronger oracle is a real improvement over
`psro_20261027`: the round-1 best response enters the blue maximin support
with weight 1.0 and reduces in-population exploitability from **290.002 to
0.000**; it also flips both `dev_red` and held-out Red-v5/v6 to red win
**0.0**.  However, `dev_param` is **486.93 vs 546.47** for RL-Repair
(**-59.5**), `min_norm` is **0.5599**, and the `noisy` family falls to
260.0.  This is an anti-exploitation specialist, not a main-weight
replacement.  `weights/rl_policy_weights.json` is unchanged; `psro-v4` /
`psro-v4-br` are experimental candidates only.

The next step should be a constrained / multi-objective oracle (official
family floor plus the red objective), or a red oracle (double oracle) so
both sides grow; do not repeat the same 3-round blue-only configuration.
Red-v5/v6 and Protocol v4 `test_v7` remain held out and were not used for
selection.
