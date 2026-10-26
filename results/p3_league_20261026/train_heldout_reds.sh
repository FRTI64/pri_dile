#!/usr/bin/env bash
# Train the two strictly held-out reds in parallel.  They are never used by
# the blue response.  A slightly smaller field budget than v2-v4 keeps the
# total compute bounded after the first red-pool pass.
set -euo pipefail

mkdir -p /tmp/p3_league results/p3_league_20261026

train_red() {
  local name="$1" seed="$2" pool="$3"
  echo "=== training Red-${name} seed=${seed} pool=${pool} ==="
  .venv/bin/python -m ipd train rl --device auto --seed "${seed}" --from-scratch \
    --iterations 25 --fields-per-dist 12 --n 80 --workers 10 \
    --opponent-pool "${pool}" --allow-probe --no-validation \
    --val-fields-per-dist 1 --val-n 32 \
    --lr 2e-4 --ent-coef 0.01 --target-kl 0.03 \
    --out "weights/red_pool/red_rl_${name}.json" \
    --checkpoint-out "/tmp/p3_league/red_${name}_checkpoint.json" \
    --history-out "results/p3_league_20261026/red_${name}_train.json" \
    2>&1 | tee "results/p3_league_20261026/red_${name}_train.log"
}

train_red v5_heldout 20261041 configs/league/p3_red_v5_heldout_blues.json &
train_red v6_heldout 20261042 configs/league/p3_red_v6_heldout_blues.json &
wait
