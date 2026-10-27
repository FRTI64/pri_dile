#!/usr/bin/env bash
# Diagnostic evaluation for the PSRO artifacts.
# These are deliberately small: 4 fields x 512 for dev_param and 10 seeds on
# held-out Red-v5/v6.  They are not a full Protocol v4 baseline and must not be
# used for promotion.
set -euo pipefail

cd "$(dirname "$0")/../.."

python3 -m ipd protocol eval \
  --candidates psro,psro-br,rl-repair,final-tuned,triple-majority \
  --tiers dev_param --fields 4 --n 512 --workers 20 --seed 20261027 \
  --baseline RL-Repair \
  --out results/psro_20261027/eval_dev_param_smoke.json

python3 -m ipd league \
  --config configs/psro/psro_red_check.json \
  --skip-suites --skip-cross-play --workers 20 --red-seeds 10 \
  --out results/psro_20261027/eval_heldout_red.json \
  --md-out results/psro_20261027/eval_heldout_red.md
