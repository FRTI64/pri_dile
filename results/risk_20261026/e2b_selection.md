# e2b_checkpoint_selection — cross-play / league report

- **Name:** e2b_checkpoint_selection
- **Created:** 2026-10-26T13:15:44
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-PPO-Hard60
- **Fingerprint:** `91f91e67d58752f143e6dc79ce29eb0a34fe5e7b5fd07a3497a51f6f9b367c47`

Validation-only checkpoint selection on heldout_v2 with a fixed policy seed; E2b risk-coef=3.0, quantile=0.1.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| RL-PPO-Hard60 | baseline | 499.50 | 1.0000 | 418.85 | - | - |
| RL-Repair | baseline | 504.40 | 0.9972 | 424.19 | - | - |
| E2b-005 | checkpoint | 505.78 | 0.9999 | 428.89 | - | - |
| E2b-010 | checkpoint | 505.66 | 0.9995 | 423.29 | - | - |
| E2b-015 | checkpoint | 505.67 | 0.9994 | 423.01 | - | - |
| E2b-020 | checkpoint | 505.48 | 0.9998 | 420.62 | - | - |

## Per-suite means

| Candidate | heldout_v2 |
|---|---:|
| RL-PPO-Hard60 | 499.50 |
| RL-Repair | 504.40 |
| E2b-005 | 505.78 |
| E2b-010 | 505.66 |
| E2b-015 | 505.67 |
| E2b-020 | 505.48 |

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
