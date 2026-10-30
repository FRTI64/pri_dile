#!/usr/bin/env bash
# Fix attempt 1: restore the 128/128 gentle stage that the original
# docs/research/rl-wide.md lineage used before the 4-iter balance.
set -euo pipefail
cd /home/frti64/pri_dile
. .venv/bin/activate
EXP=results/retrain_matrix_20261029
LOG="$EXP/fix_lineage.log"

run() {
  echo "===== $(date -Iseconds) :: $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

# 1) 128/128 gentle (8 iter, legacy + train_v4, no extra PB)
run python3 -m ipd train rl --device auto --hidden 128 128 --iterations 8 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261035 \
  --init-weights "$EXP/wide128_ac.json" \
  --train-dists original,late,grim30,wide,train_v4 \
  --dist-weights 1,1,1,2,2 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints "$EXP/fix_wide128_ckpts" \
  --bc-kl-coef 0.1 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out "$EXP/fix_wide128_weights.json" \
  --checkpoint-out "$EXP/fix_wide128_ac.json" \
  --history-out "$EXP/fix_wide128_history.json"

test -f "$EXP/fix_wide128_ckpts/iter_004.json"

# 2) 128/128 balance, 4 iter, PB weight 1/11; original used iter_004
run python3 -m ipd train rl --device auto --hidden 128 128 --iterations 4 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261036 \
  --init-weights "$EXP/fix_wide128_ckpts/iter_004.json" \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints "$EXP/fix_wide_balance_ckpts" \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out "$EXP/fix_wide_balance_weights.json" \
  --checkpoint-out "$EXP/fix_wide_balance_ac.json" \
  --history-out "$EXP/fix_wide_balance_history.json"

test -f "$EXP/fix_wide_balance_ckpts/iter_004.json"

# 3) 160/128, 2 iter balance, take iter_001
run python3 -m ipd train expand \
  --input "$EXP/fix_wide_balance_ckpts/iter_004.json" \
  --hidden 160 128 \
  --out "$EXP/fix_wide160_ac.json" \
  --seed 20261039

run python3 -m ipd train rl --device auto --hidden 160 128 --iterations 2 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261040 \
  --init-weights "$EXP/fix_wide160_ac.json" \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints "$EXP/fix_wide160_ckpts" \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out "$EXP/fix_wide160_weights.json" \
  --checkpoint-out "$EXP/fix_wide160_ac_best.json" \
  --history-out "$EXP/fix_wide160_history.json"

test -f "$EXP/fix_wide160_ckpts/iter_001.json"
echo "FIX_LINEAGE_DONE $(date -Iseconds)" | tee -a "$LOG"
