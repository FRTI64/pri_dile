# e1c_checkpoint_selection — cross-play / league report

- **Name:** e1c_checkpoint_selection
- **Created:** 2026-10-26T13:00:54
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-PPO-Hard60
- **Fingerprint:** `9c80e24e441f5deb52c4b5a96bdb8a3455ec114bef415877590567365c66c6cc`

Validation-only checkpoint selection on heldout_v2 with a fixed policy seed. Intermediate checkpoints were not committed; selected iter 15 equals weights/rl_policy_checkpoint_repair.json.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| RL-PPO-Hard60 | baseline | 499.50 | 1.0000 | 418.85 | - | - |
| RL-PPO | baseline | 496.70 | 0.9544 | 411.13 | - | - |
| RL-Robust-B | baseline | 502.54 | 0.9967 | 417.48 | - | - |
| E1c-005 | checkpoint | 502.90 | 0.9946 | 416.59 | - | - |
| E1c-010 | checkpoint | 504.00 | 0.9981 | 420.43 | - | - |
| E1c-015 | checkpoint | 503.67 | 0.9956 | 424.49 | - | - |
| E1c-020 | checkpoint | 504.40 | 0.9972 | 424.19 | - | - |

## Per-suite means

| Candidate | heldout_v2 |
|---|---:|
| RL-PPO-Hard60 | 499.50 |
| RL-PPO | 496.70 |
| RL-Robust-B | 502.54 |
| E1c-005 | 502.90 |
| E1c-010 | 504.00 |
| E1c-015 | 503.67 |
| E1c-020 | 504.40 |

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
