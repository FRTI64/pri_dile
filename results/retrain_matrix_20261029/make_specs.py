#!/usr/bin/env python3
"""Build the candidate-spec JSON files consumed by eval_weights.py."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results/retrain_matrix_20261029"
WEXP = ROOT / "weights/retrain_matrix_20261029"


def spec(name: str, kind: str, weights: str | None = None) -> dict:
    row = {"name": name, "kind": kind, "mode": "sample"}
    if weights:
        row["weights"] = str(Path(weights).resolve())
    return row


def require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return path


def main() -> None:
    pattern = require(EXP / "pattern_ckpts/iter_006.json")
    wide_sel = json.loads(
        (EXP / "wide_balance_selection.json").read_text())["chosen_checkpoint"]
    wide = require(Path(wide_sel) if Path(wide_sel).is_absolute()
                   else ROOT / wide_sel)
    wide160 = require(EXP / "wide160_ckpts/iter_001.json")
    wide160_s2 = require(EXP / "wide160_seed2_ckpts/iter_001.json")
    dagger = require(WEXP / "rl_dagger_gru_weights.json")
    dagger_s2 = require(WEXP / "rl_dagger_gru_weights_seed20261044.json")

    base = [
        spec("RL-PPO", "rl"),
        spec("RL-Pattern", "rl-pattern"),
        spec("RL-Wide160", "rl-wide160"),
        spec("RL-Switch", "rl-switch"),
        spec("GRU-DAgger", "rl-dagger-gru"),
    ]
    retrained = [
        spec("RL-Pattern-RM", "rl-pattern", pattern),
        spec("RL-Wide-RM", "rl-wide", wide),
        spec("RL-Wide160-RM", "rl-wide160", wide160),
        spec("RL-Wide160-RM-s2", "rl-wide160", wide160_s2),
        spec("RL-Switch-RM", "rl-switch", wide160),
        spec("GRU-DAgger-RM", "rl-dagger-gru", dagger),
        spec("GRU-DAgger-RM-s2", "rl-dagger-gru", dagger_s2),
    ]
    (EXP / "retrained_specs.json").write_text(json.dumps({
        "note": "baseline + retrained candidates; RL-PPO is the min_norm base",
        "candidates": base + retrained,
    }, indent=2) + "\n")
    print("wrote", EXP / "retrained_specs.json")
    print("wide selected:", wide)
    for row in base + retrained:
        print(f"  {row['name']:18} {row['kind']:14} {row.get('weights', '')}")


if __name__ == "__main__":
    main()
