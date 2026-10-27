# PSRO — psro_scratch_v3_20261027

- **Status:** experimental; not promoted
- **Created:** 2026-10-27T20:58:41
- **Config:** `/home/frti64/pri_dile/configs/psro/psro_scratch_v3_20261027.json`
- **Initial exploitability:** **60.000**
- **Final exploitability:** **60.000**
- **Final blue worst-case:** -60.000
- **Best red response:** `Red-RL-v1`
- **Deploy:** `/home/frti64/pri_dile/weights/psro_scratch_v3/psro_policy.json` (Scratch-Init)

## Blue population

| Blue | Gen | Meta weight | Expected vs red | Worst case vs red | Standalone JSON |
|---|---:|---:|---:|---:|---|
| Scratch-Init | 0 | 1.0000 | -60.00 | -60.00 | yes |
| Scratch-Base | 1 | 0.0000 | -338.44 | -366.25 | yes |
| PSRO-BR-br_01 | 1 | 0.0000 | -330.00 | -385.00 | yes |
| PSRO-BR-br_02 | 2 | 0.0000 | -317.82 | -389.06 | yes |

A `-` in **Standalone JSON** means a rule candidate that is deployed
through the meta-policy rather than a single learned JSON file.

## Red population

| Red | Meta weight | Training weight | Expected vs blue |
|---|---:|---:|---:|
| Red-RL-v1 | 1.0000 | 0.9000 | 60.00 |
| Red-RL-v3 | 0.0001 | 0.0500 | 38.44 |
| Red-RL-v4 | 0.0000 | 0.0500 | -21.25 |

## Rounds

| Round | Before exploitability | After exploitability | Delta | Added |
|---:|---:|---:|---:|---|
| 1 | 60.000 | 60.000 | 0.000 | PSRO-BR-br_01 |
| 2 | 60.000 | 60.000 | 0.000 | PSRO-BR-br_02 |

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
