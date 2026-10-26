# e1b_checkpoint_selection — cross-play / league report

- **Name:** e1b_checkpoint_selection
- **Created:** 2026-10-26T12:43:49
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-PPO-Hard60
- **Fingerprint:** `545f9d139cc9fd57c98404d4619d7b65e7a1525e461b65234166727b66c52008`

Validation-only checkpoint selection on heldout_v2 with a fixed policy seed.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| RL-PPO-Hard60 | baseline | 499.50 | 1.0000 | 418.85 | - | - |
| RL-PPO | baseline | 496.70 | 0.9544 | 411.13 | - | - |
| E1b-005 | checkpoint | 500.89 | 0.9910 | 415.09 | - | - |
| E1b-010 | checkpoint | 501.37 | 0.9814 | 411.07 | - | - |
| E1b-015 | checkpoint | 501.93 | 0.9850 | 412.56 | - | - |
| E1b-020 | checkpoint | 501.63 | 0.9875 | 413.62 | - | - |
| E1b-025 | checkpoint | 502.14 | 0.9827 | 411.61 | - | - |
| E1b-030 | checkpoint | 502.06 | 0.9810 | 410.90 | - | - |
| E1b-035 | checkpoint | 502.32 | 0.9883 | 413.94 | - | - |
| E1b-040 | checkpoint | 502.54 | 0.9967 | 417.48 | - | - |

## Per-suite means

| Candidate | heldout_v2 |
|---|---:|
| RL-PPO-Hard60 | 499.50 |
| RL-PPO | 496.70 |
| E1b-005 | 500.89 |
| E1b-010 | 501.37 |
| E1b-015 | 501.93 |
| E1b-020 | 501.63 |
| E1b-025 | 502.14 |
| E1b-030 | 502.06 |
| E1b-035 | 502.32 |
| E1b-040 | 502.54 |

## Held-out v2 per family

| Candidate | Family | Mean |
|---|---|---:|
| RL-PPO | heldout_v2:reciprocal_v2 | 599.15 |
| RL-PPO | heldout_v2:punitive_v2 | 600.62 |
| RL-PPO | heldout_v2:noisy_v2 | 427.11 |
| RL-PPO | heldout_v2:extort_v2 | 418.57 |
| RL-PPO | heldout_v2:random_v2 | 411.13 |
| RL-PPO | heldout_v2:temporal_v2 | 451.39 |
| RL-PPO | heldout_v2:adaptive_v2 | 581.54 |
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
