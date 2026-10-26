# e2_checkpoint_selection — cross-play / league report

- **Name:** e2_checkpoint_selection
- **Created:** 2026-10-26T13:12:40
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-PPO-Hard60
- **Fingerprint:** `c778236d516d53cb256884486ba6d3457ee9dc430d0eee29eaae57ca418eec57`

Validation-only checkpoint selection on heldout_v2 with a fixed policy seed. Intermediate checkpoints were not committed.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| RL-PPO-Hard60 | baseline | 499.50 | 1.0000 | 418.85 | - | - |
| RL-Repair | baseline | 504.40 | 0.9972 | 424.19 | - | - |
| E2-005 | checkpoint | 504.84 | 0.9972 | 425.81 | - | - |
| E2-010 | checkpoint | 504.80 | 0.9951 | 425.21 | - | - |
| E2-015 | checkpoint | 505.30 | 0.9973 | 428.24 | - | - |
| E2-020 | checkpoint | 504.83 | 0.9961 | 424.77 | - | - |

## Per-suite means

| Candidate | heldout_v2 |
|---|---:|
| RL-PPO-Hard60 | 499.50 |
| RL-Repair | 504.40 |
| E2-005 | 504.84 |
| E2-010 | 504.80 |
| E2-015 | 505.30 |
| E2-020 | 504.83 |

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
