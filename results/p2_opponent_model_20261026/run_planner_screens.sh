#!/usr/bin/env bash
# Fair-RNG reruns of the P2 planner screens (fresh base for base/planner).
set -euo pipefail
OUT=results/p2_opponent_model_20261026
mkdir -p "${OUT}"

.venv/bin/python -m ipd planner-eval --suite validation,heldout_v2 --seeds 1 \
  --horizon 6 --top-k 8 --fallback-confidence 0.75 \
  --out "${OUT}/planner_ungated_aggressive.json" \
  2>&1 | tee "${OUT}/planner_ungated_aggressive.log"

.venv/bin/python -m ipd planner-eval --suite validation,heldout_v2 --seeds 1 \
  --horizon 4 --top-k 6 --fallback-confidence 0.9 --margin 10 \
  --out "${OUT}/planner_ungated_conservative.json" \
  2>&1 | tee "${OUT}/planner_ungated_conservative.log"

.venv/bin/python -m ipd planner-eval --suite validation,heldout_v2 --seeds 1 \
  --horizon 4 --top-k 6 --fallback-confidence 0.9 --margin 10 \
  --gate-model weights/opponent_gate.json --gate-allowed allow \
  --gate-threshold 0.7 --gate-min-day 5 \
  --out "${OUT}/planner_gated_t07.json" \
  2>&1 | tee "${OUT}/planner_gated_t07.log"
