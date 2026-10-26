#!/usr/bin/env bash
# Evaluate the expanded red pool and compute maximin blue-training weights.
set -euo pipefail

OUT=results/p3_league_20261026
mkdir -p "${OUT}"

.venv/bin/python -m ipd league \
  --config configs/eval/p3_red_pool_20261026.json \
  --skip-suites --skip-cross-play --workers 20 \
  --out "${OUT}/red_eval.json" \
  --md-out "${OUT}/red_eval.md" \
  2>&1 | tee "${OUT}/red_eval.log"

.venv/bin/python -m ipd meta \
  --config configs/eval/p3_red_pool_20261026.json \
  --report "${OUT}/red_eval.json" \
  --train-reds Red-RL-v1,Red-RL-v2,Red-RL-v3,Red-RL-v4 \
  --heldout-reds Red-RL-v5-heldout,Red-RL-v6-heldout \
  --blues RL-PPO,RL-PPO-Hard60,RL-Repair,Adv-RL,SelfPlay-GRU,Blue-vs-Red-Comp-v1 \
  --uniform-mix 0.4 --iterations 20000 --seed 20261047 \
  --out "${OUT}/blue_meta_pool.json" \
  --summary-out "${OUT}/meta_weights.json" \
  2>&1 | tee "${OUT}/meta.log"
