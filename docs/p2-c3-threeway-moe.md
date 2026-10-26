# P2 C3 — three-way safe/risky/unknown gate + MoE

- **Status:** offline screen complete — **No-Go**（2026-10-26）
- **Code:** `ipd/training/opponent_model.py`（`ThreeWayGate` /
  `three_way_gate_label`）, `ipd/policies/moe.py`（`ThreeWayMoE`）,
  `ipd/evaluation/moe_eval.py`
- **Entry points:** `python3 -m ipd train opponent-model`,
  `python3 -m ipd moe-eval`
- **Base / fallback:** `weights/rl_policy_weights_repair.json` (RL-Repair)
- **Promotion:** none; no candidate registered, no weights replaced
- **Results:** `results/p2_threeway_moe_20261026/`

## 1. Why three-way

The binary C2/C3 gate was precise on random/late but still had
adaptive/noisy false positives, and its aggressive planner collapsed on
extort.  A three-way gate adds an **abstain** route:

- `safe`: allowed family plus enough response evidence;
- `risky`: responsive family or a C2 response veto;
- `unknown`: low excitation / low confidence → fallback.

`ThreeWayMoE` routes `safe` to an exploitation expert and both `risky` and
`unknown` to RL-Repair.  The tested exploit experts were always-defect (AllD)
and the conservative belief planner.

## 2. Gate metrics

Training data is the mixed exploratory + counterfactual dataset from
`results/p2_excitation_20261026/`.  Three-way gate: train 95.9%,
validation 94.3%.

| Suite trace | 3-way accuracy | safe precision | safe recall |
|---|---:|---:|---:|
| heldout_v2 base | 84.5% | 65.6% | 20.7% |
| heldout_v2 explore | 88.4% | **82.1%** | **36.9%** |
| heldout_v2 counterfactual | 86.6% | **82.6%** | **31.7%** |

The three-way gate improves precision on excited states but **abstains too
often**: safe recall on deployment-style base trajectories is only 20.7%.

## 3. MoE screen

Fair-RNG; safe → AllD, risky/unknown → RL-Repair:

| Config | seeds | validation Δ | heldout_v2 Δ | min_norm | safe rate |
|---|---:|---:|---:|---:|---:|
| t0.6, day5 | 1 | +1.35 | +3.64 | 0.996 | 13.4% |
| t0.6, day5 | 3 | +2.56 | +0.51 | 0.992 | 12.9% |
| t0.6, day10 | 3 | +2.56 | +0.62 | 0.998 | 13.4% |
| t0.6, day10 | 5 | +0.53 | **+4.04** | 0.997 | 13.2% |
| t0.65, day10 | 5 | +0.90 | +1.41 | 0.998 | 11.8% |
| safe → planner, t0.6, day5 | 3 | +1.18 | **-1.28** | 0.987 | 10.6% |

The 5-seed AllD t0.6/day10 heldout gains are random +14.93, noisy +9.73,
temporal +3.55, adaptive +1.30, extort -1.23.  Validation is only +0.53;
the result is not stable across seeds and is far below Gate C's +10.

Held-out red E4/v5/v6: **0.00** blue delta, safe rate 0.0% — the three-way
gate does not misfire on red, but also provides no anti-exploitation gain.

## 4. Active-probe control

`results/p2_threeway_moe_20261026/probe_cost.py` forces one early D and
then returns to RL-Repair.  On heldout_v2 the overall cost is **-10.7 to
-20.8**; per-family losses are reciprocal -49 to -74 and punitive -35 to
-61.  Unconditional early probing is therefore not viable without a
high-precision safe gate or a Bayesian probe-value model.

## 5. Diagnosis

1. **Safe recall is the bottleneck**, not planner depth.  The three-way
   gate correctly abstains on adaptive/noisy, but also abstains on many
   random/late states where AllD would gain.
2. **AllD is very sensitive to false positives.**  The 1-seed extort
   behavior and the seed variance show that loosening the safe threshold
   trades a few random points for extort risk.
3. **Planner expert is worse than AllD** on this routing: the belief
   planner still damages random/extort when it overrides.
4. Gate C (+10 on held-out red/dynamic/phase) is not met; no promotion.

## 6. Next steps

- Repeat each counterfactual branch multiple times and train a calibrated
  safe/risky/unknown probability model; current labels are one-sample.
- Fuse the three-way gate with the belief planner posterior instead of
  85-dim history only.
- If probing is revisited, use expected-value probing (only when the
  Bayesian value of information is positive), not an unconditional early D.
- Keep RL-Repair as fallback; do not change main weights; `heldout_v4`
  remains unused.

## 7. Reproduction

```bash
bash results/p2_threeway_moe_20261026/run.sh
python3 -m unittest tests.test_moe tests.test_opponent_model -v
```

## 8. Artifacts

```text
ipd/training/opponent_model.py
ipd/policies/moe.py
ipd/evaluation/moe_eval.py
tests/test_moe.py
weights/opponent_gate_threeway.json
results/p2_threeway_moe_20261026/
docs/p2-c3-threeway-moe.md
```
