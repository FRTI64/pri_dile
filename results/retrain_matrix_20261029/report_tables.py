#!/usr/bin/env python3
"""Emit the markdown tables used in README.md."""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
TIERS = ("dev_param", "dev_family", "dev_complex", "dev_axelrod", "dev_ood")
PRED = {
    "RL-Pattern-RM": "RL-Pattern",
    "RL-Wide-RM": "RL-Wide160",
    "RL-Wide160-RM": "RL-Wide160",
    "RL-Wide160-RM-s2": "RL-Wide160",
    "RL-Switch-RM": "RL-Switch",
    "GRU-DAgger-RM": "GRU-DAgger",
    "GRU-DAgger-RM-s2": "GRU-DAgger",
}
RETRAINED = tuple(PRED)


def load(name: str):
    return json.loads((EXP / name).read_text())


def table_base(data: dict) -> str:
    lines = ["| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | min_norm |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, rows in data["results"].items():
        vals = " | ".join(f"{rows[t]['mean']:.2f}" for t in TIERS)
        norms = [rows[t]["min_norm"] for t in TIERS
                 if rows[t].get("min_norm") is not None]
        mn = f"{min(norms):.3f}" if norms else "baseline"
        lines.append(f"| {name} | {vals} | {mn} |")
    return "\n".join(lines)


def table_diffs(data: dict) -> str:
    lines = ["| 候选 | 对照 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name in RETRAINED:
        pred = PRED[name]
        cells = []
        for tier in TIERS:
            delta = data["results"][name][tier]["mean"] \
                - data["results"][pred][tier]["mean"]
            cells.append(f"{delta:+.2f}")
        lines.append(f"| {name} | {pred} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_families(data: dict, tier: str, families: list[str]) -> str:
    lines = ["| 候选 | " + " | ".join(families) + " |",
             "|---|" + "---:|" * len(families)]
    for name in RETRAINED:
        pred = PRED[name]
        for label, key in (("旧", pred), ("新", name)):
            fam = data["results"][key][tier]["families"]
            lines.append(f"| {key} | " + " | ".join(
                f"{fam.get(f, float('nan')):.2f}" for f in families) + " |")
        lines.append(f"| {name} diff | " + " | ".join(
            f"{data['results'][name][tier]['families'].get(f, 0) - data['results'][pred][tier]['families'].get(f, 0):+.2f}"
            for f in families) + " |")
    return "\n".join(lines)


def main() -> None:
    base = load("baseline_dev.json")
    retrained = load("retrained_dev.json")
    print("## baseline\n" + table_base(base))
    print("\n## retrained\n" + table_base(retrained))
    print("\n## diff vs predecessor\n" + table_diffs(retrained))
    print("\n## dev_axelrod families\n"
          + table_families(retrained, "dev_axelrod",
                           ["axelrod", "axelrod_extra", "AxelrodMix",
                            "anchor"]))
    print("\n## dev_complex families\n"
          + table_families(retrained, "dev_complex",
                           ["pattern_breaker", "lookahead", "adaptive_mix",
                            "phase_switch", "probe_exploit"]))
    seed2 = EXP / "seed2_dev.json"
    if seed2.is_file():
        print("\n## independent seed\n" + table_base(load("seed2_dev.json")))
    test = EXP / "test_v7_provisional.json"
    if test.is_file():
        payload = load("test_v7_provisional.json")
        print("\n## test_v7\n| 候选 | test_v7 | min field | p05 | worst family |")
        print("|---|---:|---:|---:|---:|")
        for name, row in payload["results"].items():
            r = row["test_v7"]
            print(f"| {name} | {r['mean']:.2f} | {r['min']:.2f} | "
                  f"{r['p05']:.2f} | {r['worst_family']:.2f} |")


if __name__ == "__main__":
    main()
