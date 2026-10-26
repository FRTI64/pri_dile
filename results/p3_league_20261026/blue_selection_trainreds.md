# p3_blue_selection_trainreds_20261026 — cross-play / league report

- **Name:** p3_blue_selection_trainreds_20261026
- **Created:** 2026-10-26T15:02:45
- **Schema:** ipd-league-report-v1
- **Baseline:** Blue-Meta-b1
- **Fingerprint:** `c4bcd5ee497060f39eaf5a05951e2c1a090d983e8c3e2c0a24f73935ddf7cc51`

Select among blue responses using official/blind validation and the TRAINING reds v3/v4 only. Held-out reds are not referenced.

## Candidate summary

| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |
|---|---|---:|---:|---:|---:|---:|
| Blue-Meta-b1 | candidate | 552.77 | 1.0000 | 393.11 | 307.95 | 1.000 |
| Blue-Meta-b2 | candidate | 552.63 | 0.9511 | 391.60 | 312.90 | 1.000 |
| Blue-Meta-b3 | candidate | 553.02 | 0.9776 | 393.44 | 302.93 | 1.000 |

## Per-suite means

| Candidate | official | blind | validation | heldout_v2 |
|---|---:|---:|---:|---:|
| Blue-Meta-b1 | 554.79 | 564.28 | 509.84 | 503.14 |
| Blue-Meta-b2 | 554.92 | 564.49 | 506.42 | 499.53 |
| Blue-Meta-b3 | 554.79 | 565.29 | 510.43 | 504.08 |

## Held-out red pressure

| Blue | Blue mean | Red mean | Diff | Red win | Worst red |
|---|---:|---:|---:|---:|---|
| Blue-Meta-b1 | 307.95 | 723.20 | +415.25 | 1.000 | Red-RL-v3 |
| Blue-Meta-b2 | 312.90 | 718.90 | +406.00 | 1.000 | Red-RL-v3 |
| Blue-Meta-b3 | 302.93 | 744.92 | +442.00 | 1.000 | Red-RL-v3 |

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
