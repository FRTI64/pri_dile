# P2 C2/C3 — learned opponent model + confidence gate

- **Status:** offline screen complete — **No-Go**（2026-10-26）;
  excitation follow-up: data-layer Go / planner still No-Go（2026-10-26）
- **Code:** `ipd/training/opponent_model.py`,
  `ipd/policies/belief_planner.py`（gate hook）,
  `ipd/evaluation/planner_eval.py`
- **Entry point:** `python3 -m ipd train opponent-model`
- **Base / fallback:** `weights/rl_policy_weights_repair.json` (RL-Repair)
- **Promotion:** none; no candidate registered, no weights replaced

## 1. Why C2/C3

C1 showed that the hand-written 38-type particle filter confuses noisy/extort
opponents with non-responsive exploiters.  C2/C3 replace the gate with a
learned model:

- **family MLP**: 85-dim history features → 9 coarse families;
- **gate MLP**: 85-dim features → `allow` (defect/late/random) vs `deny`;
- **next-action linear model**: opponent next action C/D (auxiliary);

Only the gate is used by the planner.  It must be high-confidence before an
override is allowed.  Training data is `TYPE_BANK` only; `heldout_v2` is used
for evaluation.

## 2. Classifier results

Training: 67 TYPE_BANK types / 13,400 samples; validation: 18 held-out types /
3,600 samples.  Held-out suite: `heldout_v2` (74 opponents, 1 episode).

| Model | validation | heldout_v2 | notes |
|---|---:|---:|---|
| family MLP（9 classes） | 42.6% | 32.1% | multiclass routing too weak |
| gate MLP（allow/deny） | **86.6%** | **70.6%** | precision 70.2%, recall 38.7% |
| next-action linear | 81.9% | - | auxiliary only |

Gate behavior on heldout_v2 by family:

| Family | allow rate | precision | recall |
|---|---:|---:|---:|
| random_v2 | 75.0% | 1.00 | 0.75 |
| temporal_v2（late） | 24.1% | 1.00 | 0.24 |
| adaptive_v2（含 extort） | 19.7% | **0.00** | 0 |
| noisy_v2 | 16.0% | **0.00** | 0 |

The gate is precise on random/late, but all false positives come from
adaptive/noisy.  This is exactly the C1 failure mode, now measurable.

## 3. Planner screens（fair RNG）

Each arm gets an independent same-seed base policy; one seed per opponent.

| Planner | validation Δ | heldout_v2 Δ | min_norm | override |
|---|---:|---:|---:|---:|
| ungated aggressive（h6/top8/conf0.75） | -12.11 | -16.15 | 0.846 | 6.9–10.1% |
| ungated conservative（h4/top6/conf0.9/margin10） | -2.19 | +2.06 | 0.993 | 0.2–0.3% |
| **gated conservative（gate 0.7）** | **0.00** | **+1.20** | **0.999** | 0.1% |
| gated aggressive（h6/top8/conf0.75） | -7.40 | **-21.91** | 0.768 | 2.6% |

Gated conservative family deltas: adaptive +8.75, noisy **-0.33**,
extort/random/temporal/punitive/reciprocal 0.  Held-out red (fair RNG):
E4 0.00, v5 0.00, v6 0.00 — the gate blocks all red overrides.

Gated aggressive is worse: extort **-107.6**, temporal -21.9, random -37.1.
The gate’s adaptive/noisy false positives are expensive when planning is
allowed to act.

## 4. Gate C assessment

Gate C asks for +10 on held-out red / dynamic / phase, with official not down
more than 2:

- held-out red: 0.00 — **fail**;
- dynamic/phase proxy: temporal 0.00 in the gated conservative run;
- heldout_v2 mean: +1.20 — **far below +10**;
- validation: 0.00 (safe), family min_norm 0.999 (no collapse).

**No-Go.**  The learned gate is real (86.6% on held-out types) but too
conservative at deployment; its positive predictions are precise for
random/late, while adaptive/noisy false positives make aggressive planning
unsafe.  A two-stage `risky` veto (adaptive+noisy) also has false positives on
late/random, so it does not solve the problem by itself.

## 5. Diagnosis and next steps

1. **Excitation is the bottleneck.**  RL-Repair rarely defects early, so many
   opponent families are observationally equivalent in the training history.
   The classifier sees few response events.  Train the opponent model on
   exploratory/counterfactual blue policies, not only RL-Repair trajectories.
2. **Three-way gate / MoE.**  Replace `allow/deny` with
   `{safe_to_exploit, risky, unknown}` and route only high-confidence
   safe-to-exploit states to the planner; `unknown` falls back to RL-Repair.
3. **C2 auxiliary head.**  Train next-action / responsiveness prediction jointly
   with the gate and evaluate on held-out response events (our D → their next
   move), not just marginal action accuracy.
4. Do not change the main weights.  Use `heldout_v4` only after a candidate
   passes validation and gets one-shot final approval.

## 6. Excitation follow-up（2026-10-26）

The first next step was implemented in
`results/p2_excitation_20261026/` and
`docs/research/p2-c2c3-excitation.md`:

- Open-loop exploratory schedules (`single_probes`, `block_probes`,
  `random`, `alternating`, `late_block`) raise the overall `excited`
  rate from 25.8% to 59.3%; cooperative/forgiving go from 0% to
  56.9%/48.6%.
- Counterfactual C/D branches (536 pairs) give a direct responsiveness
  signal; cooperative/noisy are 80.8%/89.6% responsive, late/random
  are 0%/10%.
- Gate: base+explore 72.0/71.1/43.7 on heldout_v2 (old
  70.6/70.2/38.7); mixed + counterfactual 73.1/69.0/52.7, and
  77.1/73.7/61.5 on the counterfactual suite.  Responsiveness head
  reaches 71.8% on heldout counterfactual.
- Planner is still No-Go: mixed CF gate conservative is +0.88 on
  heldout_v2 with 3 seeds (1-seed +3.04, temporal +11.0, but temporal
  falls to +0.45 with 3 seeds); held-out red stays 0.00; aggressive
  still collapses extort (-107.62).  Training excitation is fixed, but
  deployment trajectories remain low-excitation because RL-Repair
  rarely defects early.
- Next: three-way `safe/risky/unknown` gate + MoE, and/or an
  early-game active probe that creates response events before
  enabling exploitation.  The three-way follow-up is now also
  **No-Go**: see `docs/research/p2-c3-threeway-moe.md`; unconditional early
  probing costs -10.7~-20.8 on heldout_v2.

## 7. Reproduction

```bash
python3 -m ipd train opponent-model --episodes 2 --epochs 300 \
  --seed 20261048 \
  --out weights/opponent_classifier.json \
  --gate-out weights/opponent_gate.json \
  --next-out weights/opponent_next_action.json \
  --report-out results/p2_opponent_model_20261026/opponent_model_report.json

bash results/p2_opponent_model_20261026/run_planner_screens.sh
PYTHONPATH=. .venv/bin/python results/p2_opponent_model_20261026/red_screen.py
```

## 8. Artifacts

```text
ipd/training/opponent_model.py
ipd/policies/belief_planner.py       # gate hook + diagnostics
ipd/evaluation/planner_eval.py       # --gate-model / --gate-threshold
tests/test_opponent_model.py
weights/opponent_classifier.json     # family MLP
weights/opponent_gate.json           # binary gate MLP
weights/opponent_next_action.json    # linear next-action model
results/p2_opponent_model_20261026/
  README.md
  opponent_model_report.json
  planner_ungated_aggressive.json
  planner_ungated_conservative.json
  planner_gated_t07.json
  planner_gated_t06.json
  planner_gated_aggressive.json
  planner_gated_red.json
  run_planner_screens.sh / red_screen.py
  checksums.txt
docs/research/p2-c2c3-excitation.md         # exploratory / counterfactual follow-up
results/p2_excitation_20261026/    # data, gate/response reports, planner screens
```

Main weights are unchanged.
