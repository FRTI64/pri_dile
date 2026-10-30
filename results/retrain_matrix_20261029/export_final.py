#!/usr/bin/env python3
"""Export selected actor-critic checkpoints to deployable policy JSON."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.policies.rl import RLPolicy

EXP = ROOT / "results/retrain_matrix_20261029"
WEXP = ROOT / "weights/retrain_matrix_20261029"


def export(checkpoint: Path, name: str) -> dict:
    policy = RLPolicy(str(checkpoint))
    payload = policy.export_policy_weights()
    out = WEXP / name
    out.write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"checkpoint": str(checkpoint), "weights": str(out),
            "sha256": digest, "format": payload.get("format"),
            "hidden": payload.get("hidden")}


def main() -> None:
    wide_sel = json.loads(
        (EXP / "wide_balance_selection.json").read_text())["chosen_checkpoint"]
    wide = Path(wide_sel) if Path(wide_sel).is_absolute() else ROOT / wide_sel
    rows = {
        "rl_pattern_rm": export(EXP / "pattern_ckpts/iter_006.json",
                                "rl_policy_weights_pattern_rm.json"),
        "rl_wide_rm": export(wide, "rl_policy_weights_wide_rm.json"),
        "rl_wide160_rm": export(EXP / "wide160_ckpts/iter_001.json",
                                "rl_policy_weights_wide160_rm.json"),
        "rl_wide160_rm_s2": export(EXP / "wide160_seed2_ckpts/iter_001.json",
                                   "rl_policy_weights_wide160_rm_s2.json"),
    }
    out = EXP / "export_manifest.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
