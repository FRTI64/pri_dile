#!/usr/bin/env bash
# Train two PSRO-lite blue responses against the maximin red mixture.
# Candidate selection uses official validation only; held-out reds are not
# consulted here.
set -euo pipefail

OUT=results/p3_league_20261026
POOL="${OUT}/blue_meta_pool.json"
mkdir -p /tmp/p3_league "${OUT}"

train_blue() {
  local name="$1" seed="$2" frac="$3" iters="$4"
  echo "=== training Blue-Meta-${name} seed=${seed} red_frac=${frac} iters=${iters} ==="
  .venv/bin/python -m ipd train rl --device auto --seed "${seed}" \
    --iterations "${iters}" --fields-per-dist 5 --n 80 --workers 20 \
    --train-dists original,late,grim30,wide,repair \
    --dist-weights 1,1,1,2,2 \
    --opponent-pool "${POOL}" \
    --opponent-fraction "${frac}" --opponent-penalty 1.0 \
    --init-weights weights/rl_policy_checkpoint_repair.json \
    --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
    --val-fields-per-dist 5 --val-n 100 --val-every 5 \
    --select-metric avg \
    --save-all-checkpoints "/tmp/p3_league/blue_${name}_checkpoints" \
    --out "weights/red_pool/blue_meta_${name}.json" \
    --checkpoint-out "/tmp/p3_league/blue_${name}_checkpoint.json" \
    --history-out "${OUT}/blue_meta_${name}_train.json" \
    2>&1 | tee "${OUT}/blue_meta_${name}_train.log"
}

train_blue b1 20261044 0.5 25
train_blue b2 20261045 0.6 25
