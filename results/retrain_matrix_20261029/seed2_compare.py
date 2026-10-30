#!/usr/bin/env python3
"""Merge the independent-seed current-candidate run and compare."""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
TIERS = ("dev_param", "dev_family", "dev_complex", "dev_axelrod", "dev_ood")
PAIRS = (
    ("RL-Pattern-RM", "RL-Pattern"),
    ("RL-Wide-RM", "RL-Wide160"),
    ("RL-Wide160-RM", "RL-Wide160"),
    ("RL-Wide160-RM-s2", "RL-Wide160"),
    ("RL-Switch-RM", "RL-Switch"),
    ("GRU-DAgger-RM", "GRU-DAgger"),
    ("GRU-DAgger-RM-s2", "GRU-DAgger"),
)


def main() -> None:
    merged = None
    for tier in TIERS:
        payload = json.loads((EXP / f"seed2_current_{tier}.json").read_text())
        if merged is None:
            merged = {"seed": payload["seed"], "tiers": [],
                      "results": {n: {} for n in payload["results"]}}
        merged["tiers"].append(tier)
        for name, rows in payload["results"].items():
            merged["results"][name].update(rows)
    (EXP / "seed2_current.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    retrained = json.loads((EXP / "seed2_dev.json").read_text())["results"]
    print("| 候选 | 对照 | param | family | complex | axelrod | ood |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for new, old in PAIRS:
        cells = [f"{retrained[new][t]['mean'] - merged['results'][old][t]['mean']:+.2f}"
                 for t in TIERS]
        print(f"| {new} | {old} | " + " | ".join(cells) + " |")
    print("\nabsolute at seed 20261029:")
    for name in ("RL-PPO",) + tuple(old for _, old in PAIRS):
        if name in merged["results"]:
            rows = merged["results"][name]
            print(f"{name:18} " + "  ".join(
                f"{t.split('_', 1)[1]}={rows[t]['mean']:.2f}" for t in TIERS))


if __name__ == "__main__":
    main()
