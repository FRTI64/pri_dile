# PSRO — psro_20261027

- **Status:** experimental; not promoted
- **Created:** 2026-10-27T19:44:15
- **Config:** `/home/frti64/pri_dile/configs/psro/psro_20261027.json`
- **Initial exploitability:** **284.797**
- **Final exploitability:** **284.797**
- **Final blue worst-case:** -284.797
- **Best red response:** `Red-RL-v4`
- **Deploy:** `/home/frti64/pri_dile/weights/psro/psro_policy.json` (PSRO-BR-br_02)

## Blue population

| Blue | Gen | Meta weight | Expected vs red | Worst case vs red | Standalone JSON |
|---|---:|---:|---:|---:|---|
| RL-Repair | 0 | 0.0000 | -395.08 | -478.75 | - |
| RL-PPO-Hard60 | 0 | 0.0000 | -386.48 | -471.88 | - |
| RL-PPO | 0 | 0.0000 | -392.19 | -473.75 | - |
| Adv-RL | 0 | 0.0000 | -344.89 | -350.94 | - |
| Final-tuned | 0 | 0.7251 | -284.78 | -285.31 | - |
| Triple-majority | 0 | 0.2749 | -284.81 | -287.50 | - |
| SelfPlay-GRU | 0 | 0.0000 | -388.51 | -469.06 | - |
| Blue-Meta-v2 | 0 | 0.0000 | -380.59 | -453.44 | - |
| PSRO-BR-br_01 | 1 | 0.0000 | -372.91 | -424.38 | yes |
| PSRO-BR-br_02 | 2 | 0.0000 | -375.89 | -441.56 | yes |

A `-` in **Standalone JSON** means a rule candidate that is deployed through
the meta-policy rather than a single learned JSON file.

## Red population

| Red | Meta weight | Training weight | Expected vs blue |
|---|---:|---:|---:|
| Red-RL-v1 | 0.3393 | 0.3384 | 284.78 |
| Red-RL-v3 | 0.0001 | 0.0500 | 254.00 |
| Red-RL-v4 | 0.6606 | 0.6115 | 284.80 |

## Rounds

| Round | Before exploitability | After exploitability | Delta | Added |
|---:|---:|---:|---:|---|
| 1 | 284.799 | 284.799 | 0.000 | PSRO-BR-br_01 |
| 2 | 284.804 | 284.804 | 0.000 | PSRO-BR-br_02 |

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

These are **small diagnostic runs**, not a full Protocol v4 baseline.  They use
4 dev fields × 512 opponents and 10 held-out-red seeds.  See
`eval_dev_param_smoke.json`, `eval_heldout_red.json` and
`run_eval.sh`.

### Protocol v4 `dev_param` (4 fields × 512)

| Candidate | dev_param | min_norm | worst family |
|---|---:|---:|---:|
| RL-Repair (reference) | **543.35** | 1.0000 | late 323.32 |
| Triple-majority | 530.61 | 0.9375 | late 310.07 |
| **PSRO-Meta** | 527.71 | 0.9106 | late 308.04 |
| PSRO-BR | 525.58 | 0.7066 | late 327.21 |
| Final-tuned | 525.73 | 0.9159 | late 306.92 |

PSRO-Meta is the blue maximin mixture (Final-tuned 72.5%, Triple-majority
27.5%).  It is essentially a weighted rule policy and loses about 15.6 points
on `dev_param` versus RL-Repair; `psro-br` loses more and its `min_norm` is
0.71.

### Held-out Red-v5 / Red-v6 (10 seeds)

Higher blue mean is better.  All reds still win 100% of matches.

| Candidate | blue mean vs held-out red |
|---|---:|
| Final-tuned | 375.70 |
| **PSRO-Meta** | **373.70** |
| Triple-majority | 373.05 |
| PSRO-BR | 354.45 |
| RL-Repair | 348.35 |

PSRO-Meta beats RL-Repair by **+25.4 blue points** on these two held-out
exploiters.  In red-minus-blue units the mean diff falls from 378.5 to 297.5
(**-21.4%**; Red-v5 0%, Red-v6 -37.3%), so the exploitability-reduction target
is met on this diagnostic.  It is still **not a Gate D pass**: official/dev
drops by 15.6 and the family guardrail fails, the reds still win every match,
and this is only a 10-seed / 4-field diagnostic.

## Decision

**No-Go for promotion.**  The PSRO loop is implemented and reproducible, and
the meta-policy is a useful anti-exploitation trade-off artifact, but it does
not beat RL-Repair on the current dev protocol.  `psro` and `psro-br` are
registered as experimental candidates only; `weights/rl_policy_weights.json` is
untouched.

The oracle did not enter the maximin support: exploitability stayed at
284.80 and both RL best responses received weight 0.  The meta-policy is
therefore a reweighted rule ensemble, not a new learned policy.  The next PSRO
experiment should either (a) train a stronger blue oracle (more iterations / a
rule-like initialisation), or (b) add a red oracle so both sides of the game
grow (`double oracle`).  Repeating the same two-round PPO oracle is not useful.
