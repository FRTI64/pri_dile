#!/usr/bin/env bash
# Reproduce the P2 C3 three-way gate + MoE screen.
set -euo pipefail
OUT=results/p2_threeway_moe_20261026
mkdir -p "${OUT}"

# 1. Train the mixed-data heads; only the 3-way gate is kept in weights/.
.venv/bin/python -m ipd train opponent-model \
  --episodes 2 --explore-episodes 2 --counterfactual-episodes 1 \
  --counterfactual-branches 8 --counterfactual-horizon 6 \
  --counterfactual-prefix explore,base \
  --epochs 300 --seed 20261048 --data-mode mixed --eval-mode all \
  --out /tmp/ipd_threeway_classifier.json \
  --gate-out /tmp/ipd_threeway_binary.json \
  --threeway-out weights/opponent_gate_threeway.json \
  --next-out /tmp/ipd_threeway_next.json \
  --response-out /tmp/ipd_threeway_response.json \
  --report-out "${OUT}/opponent_model_report.json"

# 2. MoE screens: safe -> AllD, risky/unknown -> RL-Repair.
.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 1 \
  --gate-model weights/opponent_gate_threeway.json --safe-expert alld \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 5 \
  --out "${OUT}/moe_alld_t06_d5.json"

.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 3 \
  --gate-model weights/opponent_gate_threeway.json --safe-expert alld \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 5 \
  --out "${OUT}/moe_alld_t06_d5_seeds3.json"

.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 3 \
  --gate-model weights/opponent_gate_threeway.json --safe-expert alld \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 10 \
  --out "${OUT}/moe_alld_t06_d10_seeds3.json"

.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 5 \
  --gate-model weights/opponent_gate_threeway.json --safe-expert alld \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 10 \
  --out "${OUT}/moe_alld_t06_d10_seeds5.json"

.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 5 \
  --gate-model weights/opponent_gate_threeway.json --safe-expert alld \
  --safe-threshold 0.65 --risky-threshold 0.6 --min-day 10 \
  --out "${OUT}/moe_alld_t065_d10_seeds5.json"

# 3. Planner expert ablation.
.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 1 \
  --gate-model weights/opponent_gate_threeway.json --safe-expert planner \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 5 \
  --planner-horizon 4 --planner-top-k 6 \
  --planner-fallback-confidence 0.9 --planner-margin 10 \
  --out "${OUT}/moe_planner_t06_d5.json"

.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 3 \
  --gate-model weights/opponent_gate_threeway.json --safe-expert planner \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 5 \
  --planner-horizon 4 --planner-top-k 6 \
  --planner-fallback-confidence 0.9 --planner-margin 10 \
  --out "${OUT}/moe_planner_t06_d5_seeds3.json"

# 4. Held-out red screen and active-probe cost control.
PYTHONPATH=. .venv/bin/python "${OUT}/red_screen.py" \
  --gate-model weights/opponent_gate_threeway.json \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 10 \
  --out "${OUT}/moe_alld_red.json"
PYTHONPATH=. .venv/bin/python "${OUT}/probe_cost.py" \
  --out "${OUT}/probe_cost.json"
