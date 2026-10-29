# psro_v4_red_check — cross-play / league report

- **Name:** psro_v4_red_check
- **Created:** 2026-10-29T17:50:36
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-Repair
- **Fingerprint:** `0fa6a850e0b99dbcaba3c5a5ee2cd6601626e792173c16834974879e3dcc28ca`

Diagnostic only: compare the PSRO v4 meta-policy / final best response with current candidates on held-out Red-v5/v6. Not a selection run and not a Gate D evaluation.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| PSRO-V4-Meta |  | 0.00 | - | 0.00 | 502.05 | 0.300 |
| PSRO-V4-BR |  | 0.00 | - | 0.00 | 530.55 | 0.000 |
| RL-Repair |  | 0.00 | - | 0.00 | 348.35 | 1.000 |
| RL-Wide160 |  | 0.00 | - | 0.00 | 350.20 | 1.000 |
| RL-Switch |  | 0.00 | - | 0.00 | 348.15 | 1.000 |
| Final-tuned |  | 0.00 | - | 0.00 | 376.65 | 1.000 |
| Triple-majority |  | 0.00 | - | 0.00 | 370.50 | 1.000 |

## Per-suite means

| Candidate |  |
|---|
| PSRO-V4-Meta |  |
| PSRO-V4-BR |  |
| RL-Repair |  |
| RL-Wide160 |  |
| RL-Switch |  |
| Final-tuned |  |
| Triple-majority |  |

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| PSRO-V4-Meta | 502.05 | 275.05 | -227.00 | 0.300 | Red-v5-heldout |
| PSRO-V4-BR | 530.55 | 135.05 | -395.50 | 0.000 | Red-v5-heldout |
| RL-Repair | 348.35 | 726.85 | +378.50 | 1.000 | Red-v6-heldout |
| RL-Wide160 | 350.20 | 720.20 | +370.00 | 1.000 | Red-v6-heldout |
| RL-Switch | 348.15 | 726.15 | +378.00 | 1.000 | Red-v6-heldout |
| Final-tuned | 376.65 | 671.15 | +294.50 | 1.000 | Red-v5-heldout |
| Triple-majority | 370.50 | 669.00 | +298.50 | 1.000 | Red-v5-heldout |

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
