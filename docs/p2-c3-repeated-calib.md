# P2 C3 — repeated counterfactual calibration

- **Status:** offline screen complete — **No-Go**（2026-10-26）
- **Code:** `ipd/training/opponent_model.py` (`repeats`,
  `three_way_calibrated_label`), `ipd/policies/moe.py`,
  `ipd/evaluation/moe_eval.py`
- **Entry points:** `python3 -m ipd train opponent-model`,
  `python3 -m ipd moe-eval`
- **Base / fallback:** `weights/rl_policy_weights_repair.json` (RL-Repair)
- **Promotion:** none; no candidate registered, no weights replaced
- **Results:** `results/p2_repeated_calib_20261026/`

## 1. Motivation

The one-sample counterfactual response label is noisy, and the
family+excitation three-way gate had safe recall only 31.7% on heldout
counterfactual states.  Repeated C/D branches estimate `p(response)` per
state and let us calibrate:

- `safe`: `p(response) <= 0.0`
- `risky`: `p(response) >= 0.5`
- `unknown`: otherwise

## 2. Implementation

`collect_counterfactual_dataset(..., repeats=10)` deep-copies the same
opponent state for each repeat, reseeds the common C/D RNG per repeat, and
records the fraction of repeats with a different reply sequence.  The
post-branch training samples are stored only once (first repeat) to avoid
duplicates; the response dataset stores one row per state with
`response_probabilities`.

Training uses the mixed base + exploratory + counterfactual data:

```bash
python3 -m ipd train opponent-model --data-mode mixed \
  --counterfactual-repeats 10 \
  --threeway-source counterfactual \
  --calibrated-safe-threshold 0.0 \
  --calibrated-risky-threshold 0.5
```

## 3. Gate metrics

| Trace / labels | accuracy | safe precision | safe recall |
|---|---:|---:|---:|
| heldout_v2 counterfactual, calibrated | **67.7%** | **74.7%** | **62.1%** |
| heldout_v2 counterfactual, family+excitation | 86.6% | 82.6% | 31.7% |

Calibration raises safe recall from 31.7% to 62.1%, but lowers precision
from 82.6% to 74.7%.  The remaining false positives are enough to make an
AllD expert catastrophic on extort.

## 4. MoE screen

`safe → AllD`, `risky/unknown → RL-Repair`:

| Config | seeds | validation Δ | heldout_v2 Δ | min_norm | safe rate |
|---|---:|---:|---:|---:|---:|
| no veto, t0.6/day5 | 1 | -10.62 | **-25.87** | 0.704 | 40.4% |
| no veto, t0.6/day10 | 1 | -8.56 | **-9.46** | 0.888 | 34.7% |
| veto 0.2, t0.6/day10 | 1 | -0.42 | +3.56 | 0.985 | 2.0% |
| veto 0.2, t0.6/day10 | **5** | **-0.31** | **+2.93** | **0.9988** | 2.5% |

The response veto (C2 head, `p(response) <= 0.2`) removes the extort
collapse but shrinks safe coverage to 2.5%.  Heldout family deltas (5
seeds): noisy +7.07, random +9.50, temporal +3.05, extort -0.55.

Held-out red: E4 +0.35 (safe rate 1.05%), v5/v6 0.00.

## 5. Bayesian / EV probe control

`bayes_probe_screen.py` lets the C1 belief planner override once on a
selected day and otherwise uses RL-Repair.  Best 1-seed result is day30
+1.29; with 3 seeds it is **-5.22** (random -33.1).  Unconditional early
probing is -10.7~-20.8.  A real probe needs explicit value-of-information,
not a single planner EV override.

## 6. Decision

**No-Go.**  Repeated calibration improves recall but not precision; the
safe variant is only +2.93 on heldout_v2 with validation -0.31 and a 2.5%
safe rate.  Gate C (+10 on held-out red/dynamic/phase) is not met.  Do not
register or change main weights; `heldout_v4` remains unused.

If C3 continues, the next step should be an explicit Bayesian
value-of-information probe (probe only when expected information value
exceeds cost) or a better posterior/feature representation, not another
threshold sweep.

## 7. Reproduction

```bash
bash results/p2_repeated_calib_20261026/run.sh
PYTHONPATH=. .venv/bin/python \
  results/p2_repeated_calib_20261026/bayes_probe_screen.py
python3 -m unittest tests.test_opponent_model tests.test_moe -v
```

## 8. Artifacts

```text
ipd/training/opponent_model.py
weights/opponent_gate_calibrated.json
results/p2_repeated_calib_20261026/
docs/p2-c3-repeated-calib.md
```
