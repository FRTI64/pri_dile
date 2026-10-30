#!/usr/bin/env bash
# GRU-DAgger distillation of the retrained RL-Switch teacher.
# Teacher = retrained RL-Wide160 (iter_001) + existing RL-Repair.
set -euo pipefail
cd /home/frti64/pri_dile
. .venv/bin/activate
EXP=results/retrain_matrix_20261029
WEXP=weights/retrain_matrix_20261029
mkdir -p "$WEXP"
LOG="$EXP/dagger.log"

run() {
  echo "===== $(date -Iseconds) :: $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run python3 "$EXP/dagger_switch_retrained.py" \
  --student gru --rounds 2 --bc-fields 12 --dagger-fields 6 \
  --val-fields 3 --n 48 --hidden 64 --layers 1 \
  --epochs 30 --lr 1e-3 --seed 20261043 \
  --base-weights "$EXP/wide160_ckpts/iter_001.json" \
  --alt-weights weights/rl_policy_weights_repair.json \
  --out "$WEXP/rl_dagger_gru_weights.json" \
  --history-out "$EXP/dagger_history.json"

run python3 "$EXP/dagger_switch_retrained.py" \
  --student gru --rounds 2 --bc-fields 12 --dagger-fields 6 \
  --val-fields 3 --n 48 --hidden 64 --layers 1 \
  --epochs 30 --lr 1e-3 --seed 20261044 \
  --base-weights "$EXP/wide160_ckpts/iter_001.json" \
  --alt-weights weights/rl_policy_weights_repair.json \
  --out "$WEXP/rl_dagger_gru_weights_seed20261044.json" \
  --history-out "$EXP/dagger_history_seed20261044.json"

echo "DAGGER_DONE $(date -Iseconds)" | tee -a "$LOG"
