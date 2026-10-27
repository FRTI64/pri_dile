# PSRO — psro_scratch_v2_20261027

- **Status:** experimental; not promoted
- **Created:** 2026-10-27T20:48:56
- **Config:** `/home/frti64/pri_dile/configs/psro/psro_scratch_v2_20261027.json`
- **Initial exploitability:** **65.000**
- **Final exploitability:** **0.000**
- **Final blue worst-case:** 113.125
- **Best red response:** `Red-RL-v1`
- **Deploy:** `/home/frti64/pri_dile/weights/psro_scratch_v2/psro_policy.json` (PSRO-BR-br_01)

## Blue population

| Blue | Gen | Meta weight | Expected vs red | Worst case vs red | Standalone JSON |
|---|---:|---:|---:|---:|---|
| Scratch-Init | 0 | 0.0000 | -65.00 | -65.00 | yes |
| Scratch-Base | 1 | 0.0000 | -312.19 | -382.50 | yes |
| PSRO-BR-br_01 | 1 | 1.0000 | 113.12 | 113.12 | yes |
| PSRO-BR-br_02 | 2 | 0.0000 | 88.12 | 88.12 | yes |

A `-` in **Standalone JSON** means a rule candidate that is deployed
through the meta-policy rather than a single learned JSON file.

## Red population

| Red | Meta weight | Training weight | Expected vs blue |
|---|---:|---:|---:|
| Red-RL-v1 | 1.0000 | 0.9000 | -113.12 |
| Red-RL-v3 | 0.0000 | 0.0500 | -193.12 |
| Red-RL-v4 | 0.0000 | 0.0500 | -207.81 |

## Rounds

| Round | Before exploitability | After exploitability | Delta | Added |
|---:|---:|---:|---:|---|
| 1 | 65.000 | 0.000 | -65.000 | PSRO-BR-br_01 |
| 2 | 0.000 | 0.000 | 0.000 | PSRO-BR-br_02 |

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
