# p3_blue_checkpoint_selection_20261026 — cross-play / league report

- **Name:** p3_blue_checkpoint_selection_20261026
- **Created:** 2026-10-26T15:08:36
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-Repair
- **Fingerprint:** `428cdb17145b59181fd8c7fd23561684125b94dd8ba87e2881b1984598af1d8f`

Choose a blue checkpoint using official/blind/heldout_v2 validation and the TRAINING reds v3/v4. Held-out reds are excluded. RL-Repair is the baseline for official/family checks.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| RL-Repair | baseline | 552.88 | 1.0000 | 391.82 | 301.77 | 1.000 |
| Blue-Meta-b1-15 | candidate | 548.53 | 0.8670 | 377.15 | 316.18 | 1.000 |
| Blue-Meta-b1-20 | candidate | 544.84 | 0.8164 | 365.88 | 322.02 | 0.975 |
| Blue-Meta-b2-15 | candidate | 551.12 | 0.8735 | 385.41 | 311.43 | 1.000 |
| Blue-Meta-b2-20 | candidate | 548.09 | 0.8313 | 364.53 | 319.10 | 0.950 |

## Per-suite means

| Candidate | official | blind | validation | heldout_v2 |
|---|---:|---:|---:|---:|
| RL-Repair | 554.76 | 564.61 | 510.37 | 504.40 |
| Blue-Meta-b1-15 | 551.92 | 557.22 | 501.96 | 490.65 |
| Blue-Meta-b1-20 | 548.82 | 551.63 | 496.79 | 486.01 |
| Blue-Meta-b2-15 | 553.93 | 562.85 | 502.14 | 492.07 |
| Blue-Meta-b2-20 | 551.86 | 557.29 | 498.60 | 484.13 |

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| RL-Repair | 301.77 | 741.27 | +439.50 | 1.000 | Red-RL-v3 |
| Blue-Meta-b1-15 | 316.18 | 693.92 | +377.75 | 1.000 | Red-RL-v3 |
| Blue-Meta-b1-20 | 322.02 | 666.27 | +344.25 | 0.975 | Red-RL-v4 |
| Blue-Meta-b2-15 | 311.43 | 705.17 | +393.75 | 1.000 | Red-RL-v3 |
| Blue-Meta-b2-20 | 319.10 | 660.35 | +341.25 | 0.950 | Red-RL-v3 |

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
