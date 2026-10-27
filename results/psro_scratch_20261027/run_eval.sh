#!/usr/bin/env bash
# Diagnostic evaluation for the from-scratch PSRO artifacts.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 -m ipd protocol eval \
  --candidates psro-scratch,psro-scratch-br,psro,psro-br,rl-repair,final-tuned,triple-majority \
  --tiers dev_param --fields 4 --n 512 --workers 20 --seed 20261027 \
  --baseline RL-Repair \
  --out results/psro_scratch_20261027/eval_dev_param_smoke.json

python3 -m ipd league \
  --config configs/psro/psro_scratch_red_check.json \
  --skip-suites --skip-cross-play --workers 20 --red-seeds 10 \
  --out results/psro_scratch_20261027/eval_heldout_red.json \
  --md-out results/psro_scratch_20261027/eval_heldout_red.md
