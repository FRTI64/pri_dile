#!/usr/bin/env bash
# Reproduce the C2/C3 exploratory/counterfactual data screen.
set -euo pipefail
OUT=results/p2_excitation_20261026
mkdir -p "${OUT}"

# 1. Base + exploratory trajectories (best practical gate at deployment).
.venv/bin/python -m ipd train opponent-model \
  --episodes 2 --explore-episodes 2 --epochs 300 --seed 20261048 \
  --data-mode base+explore --eval-mode both \
  --out weights/opponent_classifier_explore.json \
  --gate-out weights/opponent_gate_explore.json \
  --next-out weights/opponent_next_action_explore.json \
  --report-out "${OUT}/opponent_model_explore_report.json"

# 2. Full exploratory + counterfactual branches + responsiveness head.
.venv/bin/python -m ipd train opponent-model \
  --episodes 2 --explore-episodes 2 --counterfactual-episodes 1 \
  --counterfactual-branches 8 --counterfactual-horizon 6 \
  --counterfactual-prefix explore,base \
  --epochs 300 --seed 20261048 --data-mode mixed --eval-mode all \
  --out weights/opponent_classifier_cf.json \
  --gate-out weights/opponent_gate_cf.json \
  --next-out weights/opponent_next_action_cf.json \
  --response-out weights/opponent_responsiveness.json \
  --report-out "${OUT}/opponent_model_mixed_report.json"

# 3. Conservative planner screens (fair RNG; 1 seed and 3-seed check).
.venv/bin/python -m ipd planner-eval \
  --suite validation,heldout_v2 --seeds 1 \
  --horizon 4 --top-k 6 --fallback-confidence 0.9 --margin 10 \
  --gate-model weights/opponent_gate_cf.json --gate-allowed allow \
  --gate-threshold 0.6 --gate-min-day 5 \
  --out "${OUT}/planner_cf_gated_t06.json"

.venv/bin/python -m ipd planner-eval \
  --suite validation,heldout_v2 --seeds 3 \
  --horizon 4 --top-k 6 --fallback-confidence 0.9 --margin 10 \
  --gate-model weights/opponent_gate_cf.json --gate-allowed allow \
  --gate-threshold 0.6 --gate-min-day 5 \
  --out "${OUT}/planner_cf_gated_t06_seeds3.json"

.venv/bin/python -m ipd planner-eval \
  --suite validation,heldout_v2 --seeds 3 \
  --horizon 4 --top-k 6 --fallback-confidence 0.9 --margin 10 \
  --gate-model weights/opponent_gate_explore.json --gate-allowed allow \
  --gate-threshold 0.7 --gate-min-day 5 \
  --out "${OUT}/planner_explore_gated_t07_seeds3.json"

# 4. Additional threshold / aggressive / ungated screens (ablations).
screen() {
  local name=$1
  shift
  .venv/bin/python -m ipd planner-eval \
    --suite validation,heldout_v2 "$@" --out "${OUT}/${name}.json"
}

screen planner_explore_gated_t07 --seeds 1 \
  --horizon 4 --top-k 6 --fallback-confidence 0.9 --margin 10 \
  --gate-model weights/opponent_gate_explore.json --gate-allowed allow \
  --gate-threshold 0.7 --gate-min-day 5
screen planner_cf_gated_aggressive --seeds 1 \
  --horizon 6 --top-k 8 --fallback-confidence 0.75 \
  --gate-model weights/opponent_gate_cf.json --gate-allowed allow \
  --gate-threshold 0.7 --gate-min-day 5
screen planner_ungated_conservative_seeds3 --seeds 3 \
  --horizon 4 --top-k 6 --fallback-confidence 0.9 --margin 10

# 5. Held-out red screen (20 seeds, same planner).
PYTHONPATH=. .venv/bin/python "${OUT}/red_screen.py" \
  --gate-model weights/opponent_gate_cf.json --threshold 0.6 --min-day 5 \
  --out "${OUT}/planner_cf_gated_red.json"
