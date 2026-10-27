# psro_red_check — cross-play / league report

- **Name:** psro_red_check
- **Created:** 2026-10-27T19:45:02
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-Repair
- **Fingerprint:** `89c5740d300c67f842bbf956900d4897989617acd48373036cc1eb65347a8c4e`

Diagnostic only: compare the PSRO meta-policy / final best response with the current candidates on held-out Red-v5/v6. Not a selection run and not a Gate D evaluation.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| PSRO-Meta |  | 0.00 | - | 0.00 | 373.70 | 1.000 |
| PSRO-BR |  | 0.00 | - | 0.00 | 354.45 | 1.000 |
| RL-Repair |  | 0.00 | - | 0.00 | 348.35 | 1.000 |
| Final-tuned |  | 0.00 | - | 0.00 | 375.70 | 1.000 |
| Triple-majority |  | 0.00 | - | 0.00 | 373.05 | 1.000 |

## Per-suite means

| Candidate |  |
|---|
| PSRO-Meta |  |
| PSRO-BR |  |
| RL-Repair |  |
| Final-tuned |  |
| Triple-majority |  |

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| PSRO-Meta | 373.70 | 671.20 | +297.50 | 1.000 | Red-v5-heldout |
| PSRO-BR | 354.45 | 717.45 | +363.00 | 1.000 | Red-v6-heldout |
| RL-Repair | 348.35 | 726.85 | +378.50 | 1.000 | Red-v6-heldout |
| Final-tuned | 375.70 | 672.20 | +296.50 | 1.000 | Red-v5-heldout |
| Triple-majority | 373.05 | 667.05 | +294.00 | 1.000 | Red-v5-heldout |

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
