# psro_scratch_red_check — cross-play / league report

- **Name:** psro_scratch_red_check
- **Created:** 2026-10-27T21:00:10
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-Repair
- **Fingerprint:** `3b73d86e844c7d49da08703693d635d55075c367c7e226d35bd0d6d7e9127c90`

Diagnostic only: from-scratch PSRO variants vs held-out Red-v5/v6. Not a selection run and not a Gate D evaluation.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| PSRO-Scratch |  | 0.00 | - | 0.00 | 491.45 | 0.000 |
| PSRO-Scratch-BR |  | 0.00 | - | 0.00 | 508.35 | 0.000 |
| Scratch-Init |  | 0.00 | - | 0.00 | 425.25 | 0.200 |
| Scratch-Base |  | 0.00 | - | 0.00 | 351.80 | 1.000 |
| RL-Repair |  | 0.00 | - | 0.00 | 347.45 | 1.000 |
| Final-tuned |  | 0.00 | - | 0.00 | 376.65 | 1.000 |
| Triple-majority |  | 0.00 | - | 0.00 | 370.50 | 1.000 |

## Per-suite means

| Candidate |  |
|---|
| PSRO-Scratch |  |
| PSRO-Scratch-BR |  |
| Scratch-Init |  |
| Scratch-Base |  |
| RL-Repair |  |
| Final-tuned |  |
| Triple-majority |  |

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| PSRO-Scratch | 491.45 | 197.45 | -294.00 | 0.000 | Red-v5-heldout |
| PSRO-Scratch-BR | 508.35 | 196.35 | -312.00 | 0.000 | Red-v5-heldout |
| Scratch-Init | 425.25 | 390.25 | -35.00 | 0.200 | Red-v6-heldout |
| Scratch-Base | 351.80 | 711.80 | +360.00 | 1.000 | Red-v6-heldout |
| RL-Repair | 347.45 | 729.95 | +382.50 | 1.000 | Red-v6-heldout |
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
