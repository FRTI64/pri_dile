#!/usr/bin/env bash
set -euo pipefail
EXP=/home/frti64/pri_dile/results/retrain_matrix_20261029
ARGS="--candidates rl,rl-pattern,rl-wide160 --tiers dev_param,dev_axelrod,dev_ood --fields 15 --n 2048 --workers 20 --baseline RL-PPO"
cd /home/frti64/pri_dile
. .venv/bin/activate
python3 -m ipd protocol eval $ARGS --out "$EXP/new_matrix_control_small.json" \
  > "$EXP/new_matrix_control_small.log" 2>&1
cd /tmp/opencode/pri_dile_head
python3 -m ipd protocol eval $ARGS --out "$EXP/old_matrix_control_small.json" \
  > "$EXP/old_matrix_control_small.log" 2>&1
echo MATRIX_CONTROL_DONE
