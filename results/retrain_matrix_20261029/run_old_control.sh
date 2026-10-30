#!/usr/bin/env bash
# Same-protocol control with the pre-migration Axelrod internal matrix (HEAD).
# HEAD keeps the project 6/0/10/1 payoff but the old Axelrod adapter does not
# pass a custom game, so Axelrod strategies use their 3/0/5/1 default.
set -euo pipefail
cd /tmp/opencode/pri_dile_head
. /home/frti64/pri_dile/.venv/bin/activate
OUT=/home/frti64/pri_dile/results/retrain_matrix_20261029/old_matrix_control_small.json
python3 -m ipd protocol eval \
  --candidates rl,rl-pattern,rl-wide160 \
  --tiers dev_param,dev_axelrod,dev_ood \
  --fields 15 --n 2048 --workers 20 \
  --baseline RL-PPO --out "$OUT" \
  > /home/frti64/pri_dile/results/retrain_matrix_20261029/old_matrix_control_small.log 2>&1
echo OLD_CONTROL_DONE
