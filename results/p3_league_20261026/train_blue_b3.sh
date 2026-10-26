#!/usr/bin/env bash
# Third blue response: target the strongest RL-family exploiter (Red-v3).
set -euo pipefail

OUT=results/p3_league_20261026
mkdir -p /tmp/p3_league "${OUT}"

.venv/bin/python -m ipd train rl --device auto --seed 20261046 \
  --iterations 25 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,repair \
  --dist-weights 1,1,1,2,2 \
  --opponent-pool "${OUT}/blue_meta_pool_v3heavy.json" \
  --opponent-fraction 0.6 --opponent-penalty 1.0 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --save-all-checkpoints /tmp/p3_league/blue_b3_checkpoints \
  --out weights/red_pool/blue_meta_b3.json \
  --checkpoint-out /tmp/p3_league/blue_b3_checkpoint.json \
  --history-out "${OUT}/blue_meta_b3_train.json" \
  2>&1 | tee "${OUT}/blue_meta_b3_train.log"
