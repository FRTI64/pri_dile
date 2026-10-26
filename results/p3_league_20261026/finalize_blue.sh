#!/usr/bin/env bash
# One-shot held-out red / Gate D evaluation for the selected blue candidate.
# Requires weights/red_pool/blue_meta_v2.json.
set -euo pipefail

OUT=results/p3_league_20261026
mkdir -p "${OUT}/benchmark" "${OUT}/blind"

.venv/bin/python -m ipd league \
  --config configs/eval/p3_blue_vs_heldout_red_20261026.json \
  --skip-cross-play --workers 20 \
  --out "${OUT}/blue_vs_heldout_red.json" \
  --md-out "${OUT}/blue_vs_heldout_red.md" \
  2>&1 | tee "${OUT}/blue_vs_heldout_red.log"

.venv/bin/python -m ipd benchmark --arch rl \
  --weights weights/red_pool/blue_meta_v2.json \
  --fields 20 --workers 20 \
  --out "${OUT}/benchmark/blue_meta_v2.json" \
  2>&1 | tee "${OUT}/benchmark/blue_meta_v2.log"

.venv/bin/python -m ipd blind --arch rl \
  --weights weights/red_pool/blue_meta_v2.json \
  --fields 20 --workers 20 \
  --out "${OUT}/blind/blue_meta_v2.json" \
  2>&1 | tee "${OUT}/blind/blue_meta_v2.log"

.venv/bin/python - <<'PY'
import json
from ipd.training.league import gate_d

path = "results/p3_league_20261026/blue_vs_heldout_red.json"
report = json.load(open(path, encoding="utf-8"))
result = gate_d(
    report, "RL-Repair", "Blue-Meta-v2",
    ["E4-red-BC", "Red-RL-v5-heldout", "Red-RL-v6-heldout"],
)
out = "results/p3_league_20261026/gate_d.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"written {out}")
PY
