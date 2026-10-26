# p3_blue_vs_heldout_red_20261026 — cross-play / league report

- **Name:** p3_blue_vs_heldout_red_20261026
- **Created:** 2026-10-26T15:09:32
- **Schema:** ipd-league-report-v1
- **Baseline:** RL-Repair
- **Fingerprint:** `042b1281ad90cf1b63504df7ecba6133297804f03f4db07e8c1ca6a2af70c5e4`

Gate D: compare RL-Repair, the maximin blue response and the previous league-lite response against held-out E4/Red-v5/Red-v6. Official/blind/heldout_v2 checked for family regressions.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| RL-Repair | baseline | 552.88 | 1.0000 | 391.82 | 330.18 | 1.000 |
| Blue-Meta-v2 | candidate | 552.90 | 0.9679 | 393.90 | 334.83 | 1.000 |
| Blue-vs-Red-Comp-v1 | previous | 552.45 | 0.9411 | 391.98 | 336.47 | 1.000 |

## Per-suite means

| Candidate | official | blind | validation | heldout_v2 |
|---|---:|---:|---:|---:|
| RL-Repair | 554.76 | 564.61 | 510.37 | 504.40 |
| Blue-Meta-v2 | 555.00 | 564.34 | 508.98 | 502.94 |
| Blue-vs-Red-Comp-v1 | 554.73 | 563.94 | 507.45 | 500.14 |

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| RL-Repair | 330.18 | 720.52 | +390.33 | 1.000 | Red-RL-v6-heldout |
| Blue-Meta-v2 | 334.83 | 702.17 | +367.33 | 1.000 | Red-RL-v6-heldout |
| Blue-vs-Red-Comp-v1 | 336.47 | 706.80 | +370.33 | 1.000 | Red-RL-v6-heldout |

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
