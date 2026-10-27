# PSRO — psro_scratch_pilot_20261027

- **Status:** experimental; not promoted
- **Created:** 2026-10-27T20:04:20
- **Config:** `/home/frti64/pri_dile/configs/psro/psro_scratch_pilot_20261027.json`
- **Initial exploitability:** **50.312**
- **Final exploitability:** **0.000**
- **Final blue worst-case:** 0.000
- **Best red response:** `RL-Repair`
- **Deploy:** `/home/frti64/pri_dile/weights/psro_scratch_pilot/psro_policy.json` (PSRO-BR-br_01)

## Blue population

| Blue | Gen | Meta weight | Expected vs red | Worst case vs red | Standalone JSON |
|---|---:|---:|---:|---:|---|
| Scratch-Init | 0 | 0.0002 | 0.00 | -50.31 | yes |
| PSRO-BR-br_01 | 1 | 0.9998 | 0.00 | 0.00 | yes |

A `-` in **Standalone JSON** means a rule candidate that is deployed
through the meta-policy rather than a single learned JSON file.

## Red population

| Red | Meta weight | Training weight | Expected vs blue |
|---|---:|---:|---:|
| RL-Repair | 0.1287 | 0.1230 | 0.00 |
| RL-PPO-Hard60 | 0.1275 | 0.1220 | 0.00 |
| RL-PPO | 0.1273 | 0.1219 | 0.00 |
| Adv-RL | 0.1215 | 0.1169 | 0.00 |
| Final-tuned | 0.1222 | 0.1175 | 0.00 |
| Triple-majority | 0.1272 | 0.1218 | 0.00 |
| SelfPlay-GRU | 0.1240 | 0.1190 | 0.00 |
| Blue-Meta-v2 | 0.1216 | 0.1170 | 0.00 |
| Red-RL-v1 | 0.0000 | 0.0136 | -139.65 |
| Red-RL-v3 | 0.0000 | 0.0136 | -290.87 |
| Red-RL-v4 | 0.0000 | 0.0136 | -306.51 |

## Rounds

| Round | Before exploitability | After exploitability | Delta | Added |
|---:|---:|---:|---:|---|
| 1 | 50.312 | 0.000 | -50.312 | PSRO-BR-br_01 |

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
