# e2c_checkpoint_selection — cross-play / league report

- **Name:** e2c_checkpoint_selection
- **Created:** 2026-10-26T13:23:49
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-PPO-Hard60
- **Fingerprint:** `7f0a6fcb09472c7e80a43c38452f8bdc6101c88cd86932b477a901ccfa7501b9`

Validation-only checkpoint selection on heldout_v2 with a fixed policy seed; E2c DRO (eta=0.5, q=0.5, blend=0.5).

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| RL-PPO-Hard60 | baseline | 499.50 | 1.0000 | 418.85 | - | - |
| RL-Repair | baseline | 504.40 | 0.9972 | 424.19 | - | - |
| E2c-005 | checkpoint | 504.27 | 0.9998 | 424.51 | - | - |
| E2c-010 | checkpoint | 505.08 | 0.9998 | 424.31 | - | - |
| E2c-015 | checkpoint | 505.33 | 0.9997 | 425.59 | - | - |
| E2c-020 | checkpoint | 504.51 | 0.9958 | 425.57 | - | - |

## Per-suite means

| Candidate | heldout_v2 |
|---|---:|
| RL-PPO-Hard60 | 499.50 |
| RL-Repair | 504.40 |
| E2c-005 | 504.27 |
| E2c-010 | 505.08 |
| E2c-015 | 505.33 |
| E2c-020 | 504.51 |

## Held-out v2 per family

| Candidate | Family | Mean |
|---|---|---:|
| RL-PPO-Hard60 | heldout_v2:reciprocal_v2 | 599.15 |
| RL-PPO-Hard60 | heldout_v2:punitive_v2 | 600.62 |
| RL-PPO-Hard60 | heldout_v2:noisy_v2 | 429.89 |
| RL-PPO-Hard60 | heldout_v2:extort_v2 | 438.58 |
| RL-PPO-Hard60 | heldout_v2:random_v2 | 418.85 |
| RL-PPO-Hard60 | heldout_v2:temporal_v2 | 449.59 |
| RL-PPO-Hard60 | heldout_v2:adaptive_v2 | 581.05 |

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
