#!/usr/bin/env bash
# Phase 2: clean-lineage retraining after the Axelrod internal-matrix change.
# All new artifacts stay under results/retrain_matrix_20261029/ and
# weights/retrain_matrix_20261029/.  Existing main weights are never touched.
set -euo pipefail
cd /home/frti64/pri_dile
. .venv/bin/activate

EXP=results/retrain_matrix_20261029
WEXP=weights/retrain_matrix_20261029
mkdir -p "$EXP" "$WEXP"
LOG="$EXP/phase2.log"

run() {
  echo "===== $(date -Iseconds) :: $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

# ---------------------------------------------------------------- 2.1 pattern
run python3 -m ipd train rl --device auto --iterations 20 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261029 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --train-dists original,late,grim30,wide,train_complex,train_axelrod \
  --dist-weights 1,1,1,2,2,1 \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --no-validation \
  --out "$EXP/pattern_warm_weights.json" \
  --checkpoint-out "$EXP/pattern_warm_ac.json" \
  --history-out "$EXP/pattern_warm_history.json"

run python3 -m ipd train rl --device auto --iterations 8 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261032 \
  --init-weights "$EXP/pattern_warm_ac.json" \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,4 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints "$EXP/pattern_ckpts" \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out "$EXP/pattern_weights.json" \
  --checkpoint-out "$EXP/pattern_ac.json" \
  --history-out "$EXP/pattern_history.json"

test -f "$EXP/pattern_ckpts/iter_006.json"
python3 - <<'PY'
import json
from pathlib import Path
exp = Path("results/retrain_matrix_20261029")
h = json.loads((exp / "pattern_history.json").read_text())
rows = {r["iteration"]: r.get("val", {}).get("selection")
        for r in h["history"] if r.get("val")}
(exp / "pattern_selection.json").write_text(json.dumps({
    "fixed_choice": "iter_006",
    "reason": "prescribed rl-pattern recipe",
    "validation_selection_by_iter": rows,
    "history_best_iteration": h.get("meta", {}).get("best_iteration"),
}, indent=2) + "\n")
print("pattern validation selection:", rows)
PY

# ---------------------------------------------------------------- 2.2 wide
run python3 -m ipd train expand \
  --input "$EXP/pattern_ckpts/iter_006.json" \
  --hidden 128 64 \
  --out "$EXP/wide_ac.json" \
  --weights-out "$EXP/wide_weights.json" \
  --seed 20261033

run python3 -m ipd train rl --device auto --hidden 128 64 --iterations 8 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261034 \
  --init-weights "$EXP/wide_ac.json" \
  --train-dists original,late,grim30,wide,train_v4 \
  --dist-weights 1,1,1,2,2 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints "$EXP/wide_gentle_ckpts" \
  --bc-kl-coef 0.1 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out "$EXP/wide_gentle_weights.json" \
  --checkpoint-out "$EXP/wide_gentle_ac.json" \
  --history-out "$EXP/wide_gentle_history.json"

test -f "$EXP/wide_gentle_ckpts/iter_008.json"
run python3 -m ipd train expand \
  --input "$EXP/wide_gentle_ckpts/iter_008.json" \
  --hidden 128 128 \
  --out "$EXP/wide128_ac.json" \
  --seed 20261035

run python3 -m ipd train rl --device auto --hidden 128 128 --iterations 4 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261036 \
  --init-weights "$EXP/wide128_ac.json" \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints "$EXP/wide_balance_ckpts" \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out "$EXP/wide_balance_weights.json" \
  --checkpoint-out "$EXP/wide_balance_ac.json" \
  --history-out "$EXP/wide_balance_history.json"

WIDE_SEL=$(python3 - <<'PY'
import json
from pathlib import Path
exp = Path("results/retrain_matrix_20261029")
h = json.loads((exp / "wide_balance_history.json").read_text())
best = int(h["meta"]["best_iteration"])
path = exp / "wide_balance_ckpts" / f"iter_{best:03d}.json"
if not path.is_file():
    available = sorted((exp / "wide_balance_ckpts").glob("iter_*.json"))
    path = available[-1]
    best = int(path.stem.split("_")[1])
(exp / "wide_balance_selection.json").write_text(json.dumps({
    "history_best_iteration": best,
    "chosen_checkpoint": str(path),
    "selection_by_iter": {r["iteration"]: r.get("val", {}).get("selection")
                          for r in h["history"] if r.get("val")},
}, indent=2) + "\n")
print(path)
PY
)
echo "wide balance selected: $WIDE_SEL" | tee -a "$LOG"

run python3 -m ipd train expand \
  --input "$WIDE_SEL" \
  --hidden 160 128 \
  --out "$EXP/wide160_ac.json" \
  --seed 20261039

# Final width stage, main training seed.  Keep iter_001.
run python3 -m ipd train rl --device auto --hidden 160 128 --iterations 2 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261040 \
  --init-weights "$EXP/wide160_ac.json" \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints "$EXP/wide160_ckpts" \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out "$EXP/wide160_weights.json" \
  --checkpoint-out "$EXP/wide160_ac_best.json" \
  --history-out "$EXP/wide160_history.json"
test -f "$EXP/wide160_ckpts/iter_001.json"

# Independent training seed for the final width stage.
run python3 -m ipd train expand \
  --input "$WIDE_SEL" \
  --hidden 160 128 \
  --out "$EXP/wide160_ac_seed2.json" \
  --seed 20261042
run python3 -m ipd train rl --device auto --hidden 160 128 --iterations 2 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261041 \
  --init-weights "$EXP/wide160_ac_seed2.json" \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints "$EXP/wide160_seed2_ckpts" \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out "$EXP/wide160_seed2_weights.json" \
  --checkpoint-out "$EXP/wide160_seed2_ac_best.json" \
  --history-out "$EXP/wide160_seed2_history.json"
test -f "$EXP/wide160_seed2_ckpts/iter_001.json"

echo "PHASE2_DONE $(date -Iseconds)" | tee -a "$LOG"
