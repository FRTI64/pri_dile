# e1c_final_heldout_v3_20261026 — cross-play / league report

- **Name:** e1c_final_heldout_v3_20261026
- **Created:** 2026-10-26T13:02:22
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-PPO
- **Fingerprint:** `8a96c3a85c4e047edf6b05919bcb3f294a233807a3a2771953fb66ebc7307ca0`

One-shot E1c final test on heldout_v3. Do not use for model selection.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| Final-tuned | rule | 489.77 | 0.8989 | 385.32 | 341.85 | 0.950 |
| Triple-majority | rule | 492.29 | 0.8791 | 376.83 | 334.70 | 1.000 |
| RL-PPO | learned | 502.31 | 1.0000 | 401.43 | 295.60 | 1.000 |
| RL-PPO-Hard60 | learned | 505.97 | 0.9999 | 405.65 | 310.00 | 1.000 |
| RL-Repair | learned | 509.88 | 0.9996 | 405.58 | 306.35 | 1.000 |
| Adv-RL | learned | 504.84 | 0.9807 | 407.62 | 328.30 | 1.000 |

## Per-suite means

| Candidate | heldout_v3 |
|---|---:|
| Final-tuned | 489.77 |
| Triple-majority | 492.29 |
| RL-PPO | 502.31 |
| RL-PPO-Hard60 | 505.97 |
| RL-Repair | 509.88 |
| Adv-RL | 504.84 |

## Cross-play payoff matrix

| row \ col | Final-tuned | Triple-majority | RL-PPO | RL-PPO-Hard60 | RL-Repair | Adv-RL |
|---|---:|---:|---:|---:|---:|---:|
| Final-tuned | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| Triple-majority | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| RL-PPO | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| RL-PPO-Hard60 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| RL-Repair | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |
| Adv-RL | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 | 595.0 |

## Meta-game

| Candidate | Elo | Nash weight | Pool exploitability | Best response |
|---|---:|---:|---:|---|
| Final-tuned | 1000.0 | 0.1689 | 0.0000 | Triple-majority |
| Triple-majority | 1000.0 | 0.1676 | 0.0000 | RL-Repair |
| RL-PPO | 1000.0 | 0.1655 | 0.0000 | Triple-majority |
| RL-PPO-Hard60 | 1000.0 | 0.1620 | 0.0000 | Triple-majority |
| RL-Repair | 1000.0 | 0.1707 | 0.0000 | Triple-majority |
| Adv-RL | 1000.0 | 0.1654 | 0.0000 | Triple-majority |

Nash exploitability (score-difference units): **0.0000**; iterations=20000.

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| Final-tuned | 341.85 | 615.85 | +274.00 | 0.950 | E4-red-BC |
| Triple-majority | 334.70 | 605.70 | +271.00 | 1.000 | E4-red-BC |
| RL-PPO | 295.60 | 629.10 | +333.50 | 1.000 | E4-red-BC |
| RL-PPO-Hard60 | 310.00 | 665.00 | +355.00 | 1.000 | E4-red-BC |
| RL-Repair | 306.35 | 705.85 | +399.50 | 1.000 | E4-red-BC |
| Adv-RL | 328.30 | 621.30 | +293.00 | 1.000 | E4-red-BC |

## Held-out v2 per family

| Candidate | Family | Mean |
|---|---|---:|

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
