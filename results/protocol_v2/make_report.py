#!/usr/bin/env python3
"""Build results/protocol_v2/README.md from the Protocol v2 JSON results."""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Sequence

BASE = Path(__file__).resolve().parent
DEV = BASE / "protocol_v2_dev.json"
TEST = BASE / "protocol_v2_test.json"
DEV_RED = BASE / "protocol_v2_dev_red.json"
TEST_RED = BASE / "protocol_v2_test_red.json"
USAGE = BASE / "usage.json"
OUT = BASE / "README.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def bootstrap_ci(values: Sequence[float], n: int = 10000,
                 alpha: float = 0.05, seed: int = 20261027) -> tuple[float, float]:
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
        df = tiers.get("dev_family", {})
        rows.append({
            "name": name,
            "dp_mean": dp["mean"],
            "dp_min": dp["min"],
            "dp_p05": dp["p05"],
            "dp_worst": dp["worst_family"],
            "dp_min_norm": dp["min_norm"],
            "df_mean": df.get("mean"),
            "df_min": df.get("min"),
            "dp_values": dp.get("values", []),
        })
    rows.sort(key=lambda row: (-row["dp_mean"], -(row["df_mean"] or 0),
                               row["name"]))
    return rows


def main() -> int:
    dev = load(DEV)
    test = load(TEST) if TEST.exists() else None
    rows = build_rows(dev)
    lines: list[str] = []
    lines += [
        "# Protocol v2 baseline",
        "",
        f"- catalog: `{dev['catalog_sha256']}`",
        f"- dev: {dev['fields']} fields × {dev['n']} opponents, "
        f"seed {dev['seed']}",
        "- selection rule: top 3 by `dev_param` mean, pre-registered before "
        "the test run",
    ]
    if test:
        lines.append(
            f"- test_v5: {test['fields']} fields × {test['n']} opponents, "
            f"seed {test['seed']}")
    lines += ["", "## 1. Dev (all candidates)", "",
              "| # | candidate | dev_param | min | p05 | min_norm | worst family | dev_family |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['name']} | {fmt(row['dp_mean'])} | "
            f"{fmt(row['dp_min'])} | {fmt(row['dp_p05'])} | "
            f"{fmt(row['dp_min_norm'], 4)} | {fmt(row['dp_worst'])} | "
            f"{fmt(row['df_mean'])} |")
    lines += [
        "",
        "Top 3 by `dev_param` mean: " + ", ".join(
            row["name"] for row in rows[:3]) + ".",
        "",
        "`dev_family` is saturated for most generalist policies (all near "
        "583); it is a pathology gate, not a ranking metric. `dev_param` is "
        "dominated by the `extort` family, so `min_norm` / per-family numbers "
        "matter more than the mean alone.",
        "",
    ]

    if test:
        test_results = test["results"]
        names = [row["name"] for row in rows if row["name"] in test_results]
        lines += ["## 2. test_v5 (top 3, one-shot)", "",
                  "| candidate | mean | min | p05 | worst family |",
                  "|---|---:|---:|---:|---:|"]
        for name in names:
            row = test_results[name]["test_v5"]
            lines.append(
                f"| {name} | {fmt(row['mean'])} | {fmt(row['min'])} | "
                f"{fmt(row['p05'])} | {fmt(row['worst_family'])} |")
        lines += ["", "### Paired field-level differences", "",
                  "| comparison | delta | 95% bootstrap CI | significant |",
                  "|---|---:|---|---|"]
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a = test_results[left]["test_v5"].get("values", [])
                b = test_results[right]["test_v5"].get("values", [])
                if not a or not b:
                    continue
                stat = paired_delta(a, b)
                lines.append(
                    f"| {left} - {right} | {stat['delta']:+.2f} | "
                    f"[{stat['lo']:+.2f}, {stat['hi']:+.2f}] | "
                    f"{'yes' if stat['significant'] else 'no'} |")
        lines += ["", "### test_v5 family means", ""]
        families = sorted(test_results[names[0]]["test_v5"]["families"])
        header = "| family | " + " | ".join(names) + " |"
        sep = "|---|" + "---:|" * len(names)
        lines += [header, sep]
        for family in families:
            cells = [fmt(test_results[name]["test_v5"]["families"].get(family))
                     for name in names]
            lines.append(f"| {family} | " + " | ".join(cells) + " |")
        lines += [""]

    if DEV_RED.exists() or TEST_RED.exists():
        dev_red = load(DEV_RED)["results"] if DEV_RED.exists() else {}
        test_red = load(TEST_RED)["results"] if TEST_RED.exists() else {}
        red_names = [name for name in (dev_red or test_red)]
        lines += ["## 3. Red exploitability", "",
                  "| candidate | dev_red blue | dev_red red win | "
                  "test_red blue | test_red red win |",
                  "|---|---:|---:|---:|---:|"]
        for name in red_names:
            dev = dev_red.get(name, {})
            test = test_red.get(name, {})
            lines.append(
                f"| {name} | {fmt(dev.get('blue_mean'))} | "
                f"{dev.get('red_win', 0):.1%} | "
                f"{fmt(test.get('blue_mean'))} | "
                f"{test.get('red_win', 0):.1%} |")
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
        "- Old `official` / `blind` numbers are legacy and cannot be mixed "
        "with this table.",
        "- `test_v5` was run once for the pre-registered top 3; do not tune on "
        "it. A new candidate needs a new test suite (`test_v6`).",
        "- The 25% test-only component now contains adversarial families "
        "(`early_harvest`, `frequent_kamikaze`, `probe_harvest`, "
        "`hyper_extort`, `early_probe`, `adaptive_probe`).",
        "- Anti-exploitation remains unsolved: use the red pool / league "
        "harness separately; `test_v5` is not a red best-response test.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
