#!/usr/bin/env python3
"""Merge per-tier eval JSONs into one retrained_dev.json."""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
TIERS = ("dev_param", "dev_family", "dev_complex", "dev_axelrod", "dev_ood")


def main() -> None:
    merged = None
    for tier in TIERS:
        path = EXP / f"retrained_dev_{tier}.json"
        payload = json.loads(path.read_text())
        if merged is None:
            merged = {
                "protocol": payload.get("protocol"),
                "wrapper": payload.get("wrapper"),
                "seed": payload["seed"],
                "fields": payload["fields"],
                "n": payload["n"],
                "rounds": payload["rounds"],
                "tiers": [],
                "no_consume": True,
                "selection_eligible": False,
                "baseline": payload.get("baseline"),
                "candidates": payload["candidates"],
                "results": {name: {} for name in payload["results"]},
            }
        merged["tiers"].append(tier)
        for name, rows in payload["results"].items():
            merged["results"][name].update(rows)
    out = EXP / "retrained_dev.json"
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    print("wrote", out)
    for name, rows in merged["results"].items():
        print(f"{name:18} " + "  ".join(
            f"{t.split('_', 1)[1]}={rows[t]['mean']:.2f}" for t in TIERS))


if __name__ == "__main__":
    main()
