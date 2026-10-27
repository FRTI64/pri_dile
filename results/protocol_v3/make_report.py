#!/usr/bin/env python3
"""Build results/protocol_v3/README.md from the v3 JSON results."""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Sequence

BASE = Path(__file__).resolve().parent
DEV = BASE / "protocol_v3_dev.json"
TEST = BASE / "protocol_v3_test.json"
DEV_RED = BASE / "protocol_v3_dev_red.json"
TEST_RED = BASE / "protocol_v3_test_red.json"
USAGE = BASE / "usage.json"
OUT = BASE / "README.md"

DEV_TIERS = ("dev_param", "dev_family", "dev_complex", "dev_ood")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def bootstrap_ci(values: Sequence[float], n: int = 20000,
                 alpha: float = 0.05, seed: int = 20261027
                 ) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    size = len(values)
    for _ in range(n):
        means.append(statistics.fmean(rng.choices(values, k=size)))
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[int((1 - alpha / 2) * n) - 1]
    return lo, hi


def paired_delta(a: Sequence[float], b: Sequence[float]) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs)
    return {
        "delta": statistics.fmean(diffs),
        "lo": lo,
        "hi": hi,
        "significant": lo > 0 or hi < 0,
    }


def build_rows(dev: dict) -> list[dict]:
    rows = []
    for name, tiers in dev["results"].items():
        dp = tiers["dev_param"]
        rows.append({
            "name": name,
            "dev_param": dp["mean"],
            "dev_family": tiers["dev_family"]["mean"],
            "dev_complex": tiers["dev_complex"]["mean"],
            "dev_ood": tiers["dev_ood"]["mean"],
            "min_norm": dp["min_norm"],
            "worst_param": dp["worst_family"],
            "worst_complex": tiers["dev_complex"]["worst_family"],
            "worst_ood": tiers["dev_ood"]["worst_family"],
        })
    rows.sort(key=lambda row: (-row["dev_param"], -row["dev_ood"],
                               row["name"]))
    return rows


def main() -> int:
    dev = load(DEV)
    test = load(TEST) if TEST.exists() else None
    rows = build_rows(dev)
    lines: list[str] = [
        "# Protocol v3 baseline",
        "",
        f"- catalog: `{dev['catalog_sha256']}`",
        f"- dev: {dev['fields']} fields × {dev['n']} opponents, "
        f"seed {dev['seed']}",
        "- dev tiers: " + ", ".join(f"`{x}`" for x in DEV_TIERS),
        "- selection rule: top 3 by `dev_param` mean, pre-registered before "
        "the test run",
    ]
    if test:
        lines.append(
            f"- test_v6: {test['fields']} fields × {test['n']} opponents, "
            f"seed {test['seed']}")
    lines += ["", "## 1. Dev (all candidates)", "",
              "| # | candidate | dev_param | dev_family | dev_complex | "
              "dev_ood | min_norm | worst param | worst complex |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['name']} | {fmt(row['dev_param'])} | "
            f"{fmt(row['dev_family'])} | {fmt(row['dev_complex'])} | "
            f"{fmt(row['dev_ood'])} | {fmt(row['min_norm'], 4)} | "
            f"{fmt(row['worst_param'])} | {fmt(row['worst_complex'])} |")
    lines += [
        "",
        "Top 3 by `dev_param` mean: " + ", ".join(
            row["name"] for row in rows[:3]) + ".",
        "",
        "`dev_complex` is the discriminating validation tier: standard "
        "families saturate near 583, while complex families separate the "
        "top candidates. `dev_ood` is a deliberately shifted mixture.",
        "",
    ]

    if test:
        test_results = test["results"]
        names = [row["name"] for row in rows if row["name"] in test_results]
        lines += ["## 2. test_v6 (top 3, one-shot)", "",
                  "| candidate | mean | min | p05 | worst family |",
                  "|---|---:|---:|---:|---:|"]
        for name in names:
            row = test_results[name]["test_v6"]
            lines.append(
                f"| {name} | {fmt(row['mean'])} | {fmt(row['min'])} | "
                f"{fmt(row['p05'])} | {fmt(row['worst_family'])} |")
        lines += ["", "### Paired field-level differences", "",
                  "| comparison | delta | 95% bootstrap CI | significant |",
                  "|---|---:|---|---|"]
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a = test_results[left]["test_v6"].get("values", [])
                b = test_results[right]["test_v6"].get("values", [])
                if not a or not b:
                    continue
                stat = paired_delta(a, b)
                lines.append(
                    f"| {left} - {right} | {stat['delta']:+.2f} | "
                    f"[{stat['lo']:+.2f}, {stat['hi']:+.2f}] | "
                    f"{'yes' if stat['significant'] else 'no'} |")
        lines += ["", "### test_v6 family means", ""]
        families = sorted(test_results[names[0]]["test_v6"]["families"])
        header = "| family | " + " | ".join(names) + " |"
        sep = "|---|" + "---:|" * len(names)
        lines += [header, sep]
        for family in families:
            cells = [fmt(test_results[name]["test_v6"]["families"].get(family))
                     for name in names]
            lines.append(f"| {family} | " + " | ".join(cells) + " |")
        lines += [""]

    if DEV_RED.exists() or TEST_RED.exists():
        dev_red = load(DEV_RED)["results"] if DEV_RED.exists() else {}
        test_red = load(TEST_RED)["results"] if TEST_RED.exists() else {}
        red_names = list(dev_red or test_red)
        lines += ["## 3. Red exploitability", "",
                  "| candidate | dev_red blue | dev_red red win | "
                  "test_red blue | test_red red win |",
                  "|---|---:|---:|---:|---:|"]
        for name in red_names:
            dev_row = dev_red.get(name, {})
            test_row = test_red.get(name, {})
            lines.append(
                f"| {name} | {fmt(dev_row.get('blue_mean'))} | "
                f"{dev_row.get('red_win', 0):.1%} | "
                f"{fmt(test_row.get('blue_mean'))} | "
                f"{test_row.get('red_win', 0):.1%} |")
        lines += ["", "Red-v5 / Red-v6 still win 100% of matches; "
                      "anti-exploitation remains unsolved.", ""]

    if USAGE.exists():
        usage = load(USAGE)
        lines += ["## 4. Usage ledger", ""]
        for entry in usage.get("entries", []):
            flag = "contaminated" if entry.get("contaminated") else "clean"
            lines.append(
                f"- `{entry['tier']}` / `{entry['candidate']}` — "
                f"{entry['created']} — {flag}")
        lines += [""]

    lines += [
        "## Notes",
        "",
        "- Protocol v2 / official / blind numbers are legacy and cannot be "
        "mixed with this table.",
        "- `test_v6` was run once for the pre-registered top 3; do not tune "
        "on it. A new candidate needs a new test suite (`test_v7`).",
        "- test_v6 composition: 35% seen-family test, 20% test-only "
        "adversarial, 20% complex, 15% policy, 10% anchors.",
        "- `dev_family` is saturated for most generalist policies; it is a "
        "pathology gate. `dev_complex` is the main complex-behaviour signal.",
        "- Anti-exploitation remains unsolved: use the red pool / league "
        "harness separately.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
