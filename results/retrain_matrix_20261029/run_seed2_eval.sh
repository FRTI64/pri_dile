#!/usr/bin/env bash
# Independent evaluation seed (20261029), per-tier with resume.
set -euo pipefail
cd /home/frti64/pri_dile
. .venv/bin/activate
EXP=results/retrain_matrix_20261029
LOG="$EXP/seed2_dev.log"
for tier in dev_param dev_family dev_complex dev_axelrod dev_ood; do
  out="$EXP/seed2_dev_${tier}.json"
  if [ -f "$out" ]; then
    echo "skip existing $out" | tee -a "$LOG"
    continue
  fi
  echo "===== $(date -Iseconds) tier=$tier" | tee -a "$LOG"
  python3 "$EXP/eval_weights.py" \
    --specs "$EXP/seed2_specs.json" \
    --tiers "$tier" --fields 30 --n 2048 --workers 20 \
    --seed 20261029 --baseline RL-PPO --out "$out" 2>&1 | tee -a "$LOG"
done
echo "SEED2_EVAL_DONE $(date -Iseconds)" | tee -a "$LOG"
