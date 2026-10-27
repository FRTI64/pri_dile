# PSRO — psro_scratch_20261027

- **Status:** experimental; not promoted
- **Created:** 2026-10-27T20:38:59
- **Config:** `/home/frti64/pri_dile/configs/psro/psro_scratch_20261027.json`
- **Initial exploitability:** **62.812**
- **Final exploitability:** **0.000**
- **Final blue worst-case:** 0.000
- **Best red response:** `RL-Repair`
- **Deploy:** `/home/frti64/pri_dile/weights/psro_scratch/psro_policy.json` (PSRO-BR-br_02)

## Blue population

| Blue | Gen | Meta weight | Expected vs red | Worst case vs red | Standalone JSON |
|---|---:|---:|---:|---:|---|
| Scratch-Init | 0 | 0.0000 | -0.00 | -62.81 | yes |
| Scratch-Base | 1 | 0.0000 | -0.02 | -365.94 | yes |
| PSRO-BR-br_01 | 1 | 0.0016 | 0.01 | -0.31 | yes |
| PSRO-BR-br_02 | 2 | 0.9984 | 0.01 | 0.00 | yes |

A `-` in **Standalone JSON** means a rule candidate that is deployed
through the meta-policy rather than a single learned JSON file.

## Red population

| Red | Meta weight | Training weight | Expected vs blue |
|---|---:|---:|---:|
| RL-Repair | 0.1461 | 0.1378 | 0.00 |
| RL-PPO-Hard60 | 0.1386 | 0.1314 | 0.00 |
| RL-PPO | 0.1450 | 0.1369 | 0.00 |
| Adv-RL | 0.1416 | 0.1340 | 0.00 |
| Final-tuned | 0.1437 | 0.1358 | 0.00 |
| Triple-majority | 0.1459 | 0.1376 | 0.00 |
| SelfPlay-GRU | 0.0019 | 0.0152 | -1.87 |
| Blue-Meta-v2 | 0.1373 | 0.1303 | 0.00 |
| Red-RL-v1 | 0.0001 | 0.0137 | -78.86 |
| Red-RL-v3 | 0.0000 | 0.0136 | -196.08 |
| Red-RL-v4 | 0.0000 | 0.0136 | -210.77 |

## Rounds

| Round | Before exploitability | After exploitability | Delta | Added |
|---:|---:|---:|---:|---|
| 1 | 62.812 | 0.108 | -62.705 | PSRO-BR-br_01 |
| 2 | 0.108 | 0.000 | -0.108 | PSRO-BR-br_02 |

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

## From-scratch setup

- `Scratch-Init`: random 85-dim MLP; only feature normalisation is inherited
  from the BC template.  No policy weights are copied.
- `Scratch-Base`: 30-iteration PPO trained from random on
  `original / late / grim30 / wide` only.  Official validation avg **553.67**;
  Protocol v4 `dev_param` (4×512) **528.99**.  This is still a from-scratch
  policy; no existing model is used.
- Red/reference pool in the PSRO config contains existing models and rules.
  The blue population contains only `Scratch-Init`, `Scratch-Base` and
  `PSRO-BR-*` policies.
- Oracle: initialised from the scratch base checkpoint, `bc_kl=0.05`,
  `opponent_fraction=0.4`, `opponent_penalty=1.0`, 20 iterations per round,
  `--no-value-load`, `value_bias=0`.  The oracle never loads an existing
  candidate.

## Diagnostic evaluation

These are small diagnostics (4 dev fields × 512; 10 red seeds), not a full
Protocol v4 baseline.  See `eval_dev_param_smoke.json`,
`eval_heldout_red.json`, `eval_dev_red.json` and `run_eval.sh`.

| Candidate | dev_param | min_norm | dev_red red win | Red-v5/v6 red win |
|---|---:|---:|---:|---:|
| **PSRO-Scratch** (`psro-scratch`) | 480.43 | 0.5501 | **0.0** | **0.0** |
| **PSRO-Scratch-BR** (`psro-scratch-br`) | 481.11 | 0.5437 | **0.0** | **0.0** |
| Scratch-Base (from-scratch, no PSRO) | 528.99 | - | - | 1.0 |
| RL-Repair | **543.35** | 1.0000 | 1.0 | 1.0 |
| Final-tuned | 525.73 | 0.9159 | - | 1.0 |
| Triple-majority | 530.61 | 0.9375 | - | 1.0 |

Red diagnostics:

```text
dev_red (protocol red): PSRO-Scratch blue 436.2 / red 210.2 / red_win 0.0
                        PSRO-Scratch-BR blue 427.2 / red 195.2 / red_win 0.0
                        RL-Repair blue 379.2 / red 715.2 / red_win 1.0

held-out Red-v5/v6 (league): PSRO-Scratch blue 491.4 / red 197.4 / diff -294.0 / red_win 0.0
                             PSRO-Scratch-BR blue 508.4 / red 196.3 / diff -312.0 / red_win 0.0
                             Scratch-Base blue 351.8 / red 711.8 / diff +360.0 / red_win 1.0
                             RL-Repair blue 347.4 / red 730.0 / diff +382.5 / red_win 1.0
```

**This is the first PSRO candidate that flips the held-out red outcome**: the
red exploiters win 0% instead of 100%, and the blue score is higher than the
red score on both Red-v5 and Red-v6.  The cost is a large official drop:
`dev_param` 480.43 vs RL-Repair 543.35, `min_norm` 0.55, worst family
`noisy` 249.1.

## Ablations

| Run | Blue learner | Oracle | dev_param | Red-v5/v6 red win |
|---|---|---|---:|---:|
| pilot (`results/psro_scratch_pilot_20261027/`) | random + BRs | 40% red, 30 iter, score-diff | 473.27 | 0.0 |
| v1 main (this report) | random + scratch base + BRs | 40% red, 20 iter, bc_kl 0.05 | **480.43** | **0.0** |
| v2 (`results/psro_scratch_v2_20261027/`) | random + scratch base + BRs | 3 reds only, 20 iter | 477.74 | 0.0 |
| v3 (`results/psro_scratch_v3_20261027/`) | random + scratch base + anchored BRs | 20% red, own-score, bc_kl 0.5 | 532.13 (BR) / 476.65 (deployed random)* | 1.0 (BR) / 0.2 (deployed random) |
| scratch base | no PSRO | official only | 528.99 | 1.0 |

`*` v3 deployed the random init because it stayed in the maximin support; the
anchored single BR itself scored 532.13 on dev_param and did not beat the red
exploiters (red win 1.0).

## Decision

**No-Go for promotion.**  The from-scratch PSRO model achieves the
anti-exploitation objective on the held-out reds (0% red win), but it gives up
~63 `dev_param` points and has `min_norm` 0.55.  It is an experimental
anti-exploitation specialist, not a replacement for RL-Repair.  The scratch
base (528.99, red win 1.0) is the safer from-scratch-only model if official
score matters.

The next step should not be more of the same fine-tune: either use a
multi-objective/constrained oracle that keeps official families above a floor
while training against reds, or evaluate on a fresh held-out red before
declaring the anti-exploitation result robust.  The main weights are unchanged.
