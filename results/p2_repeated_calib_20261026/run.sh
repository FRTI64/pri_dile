#!/usr/bin/env bash
# Reproduce the repeated-counterfactual calibration screen.
set -euo pipefail
OUT=results/p2_repeated_calib_20261026
mkdir -p "${OUT}"

# 1. Repeated counterfactual data + calibrated 3-way gate.
.venv/bin/python -m ipd train opponent-model \
  --episodes 2 --explore-episodes 2 --counterfactual-episodes 1 \
  --counterfactual-branches 8 --counterfactual-horizon 6 \
  --counterfactual-repeats 10 --counterfactual-prefix explore,base \
  --epochs 300 --seed 20261048 --data-mode mixed --eval-mode all \
  --threeway-source counterfactual \
  --calibrated-safe-threshold 0.0 --calibrated-risky-threshold 0.5 \
  --out /tmp/ipd_rc_classifier.json \
  --gate-out /tmp/ipd_rc_binary.json \
  --threeway-out weights/opponent_gate_calibrated.json \
  --next-out /tmp/ipd_rc_next.json \
  --response-out /tmp/ipd_rc_response.json \
  --report-out "${OUT}/opponent_model_report.json"

# 2. Calibrated safe -> AllD without a response veto (shows extort collapse).
.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 1 \
  --gate-model weights/opponent_gate_calibrated.json --safe-expert alld \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 5 \
  --out "${OUT}/moe_alld_t06_d5.json"
.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 1 \
  --gate-model weights/opponent_gate_calibrated.json --safe-expert alld \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 10 \
  --out "${OUT}/moe_alld_t06_d10.json"

# 3. Add the C2 response veto; this is the safe variant.
.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 1 \
  --gate-model weights/opponent_gate_calibrated.json --safe-expert alld \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 10 \
  --response-model weights/opponent_responsiveness.json \
  --max-response-probability 0.2 \
  --out "${OUT}/moe_alld_t06_d10_resp02.json"
.venv/bin/python -m ipd moe-eval \
  --suite validation,heldout_v2 --seeds 5 \
  --gate-model weights/opponent_gate_calibrated.json --safe-expert alld \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 10 \
  --response-model weights/opponent_responsiveness.json \
  --max-response-probability 0.2 \
  --out "${OUT}/moe_alld_t06_d10_resp02_seeds5.json"

# 4. Held-out red screen for the safe variant.
PYTHONPATH=. .venv/bin/python results/p2_threeway_moe_20261026/red_screen.py \
  --gate-model weights/opponent_gate_calibrated.json \
  --safe-threshold 0.6 --risky-threshold 0.6 --min-day 10 \
  --response-model weights/opponent_responsiveness.json \
  --max-response-probability 0.2 \
  --out "${OUT}/moe_alld_red.json"
