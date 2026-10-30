#!/usr/bin/env python3
"""Print the Phase 1/3 tables used in README.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

TIERS = ("dev_param", "dev_family", "dev_complex", "dev_axelrod", "dev_ood")
SHORT = {"dev_param": "param", "dev_family": "family",
         "dev_complex": "complex", "dev_axelrod": "axelrod",
         "dev_ood": "ood"}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    paths = sys.argv[1:]
    data = {Path(p).stem: load(p) for p in paths}
    for label, payload in data.items():
        print(f"\n## {label} (seed={payload.get('seed')}, "
              f"fields={payload.get('fields')}, n={payload.get('n')})")
        names = list(payload["results"])
        print("| candidate | " + " | ".join(SHORT[t] for t in TIERS)
              + " | min_norm |")
        print("|---|" + "---:|" * (len(TIERS) + 1))
        for name in names:
            rows = payload["results"][name]
            vals = []
            norms = []
            for tier in TIERS:
                vals.append(f"{rows[tier]['mean']:.2f}")
                if rows[tier].get("min_norm") is not None:
                    norms.append(rows[tier]["min_norm"])
            mn = f"{min(norms):.4f}" if norms else "baseline"
            print(f"| {name} | " + " | ".join(vals) + f" | {mn} |")

        if "dev_axelrod" in payload.get("tiers", []):
            print("\n### dev_axelrod per family")
            base = names[0]
            fams = sorted(payload["results"][base]["dev_axelrod"]["families"])
            print("| candidate | " + " | ".join(fams) + " |")
            print("|---|" + "---:|" * len(fams))
            for name in names:
                fam = payload["results"][name]["dev_axelrod"]["families"]
                print(f"| {name} | " + " | ".join(
                    f"{fam[f]:.2f}" for f in fams) + " |")

        if "dev_complex" in payload.get("tiers", []):
            print("\n### dev_complex per family")
            base = names[0]
            fams = sorted(payload["results"][base]["dev_complex"]["families"])
            print("| candidate | " + " | ".join(fams) + " |")
            print("|---|" + "---:|" * len(fams))
            for name in names:
                fam = payload["results"][name]["dev_complex"]["families"]
                print(f"| {name} | " + " | ".join(
                    f"{fam[f]:.2f}" for f in fams) + " |")


if __name__ == "__main__":
    main()
