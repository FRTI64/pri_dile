#!/usr/bin/env python3
"""Validate the cross-repo 977 outputs."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
A_COUNT = 30
B_COUNT = 977
REPS = 20
ROUNDS = 100


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    result = json.loads((OUT / "run_result.json").read_text())
    config = json.loads((OUT / "run_config.json").read_text())
    errors: list[str] = []

    raw = read_csv("raw_reps.csv")
    summary = read_csv("matchups.csv")
    weighted = read_csv("weighted_summary.csv")

    if len(raw) != A_COUNT * B_COUNT * REPS:
        errors.append(f"raw rows {len(raw)} != {A_COUNT * B_COUNT * REPS}")
    if len(summary) != A_COUNT * B_COUNT:
        errors.append(f"summary rows {len(summary)} != {A_COUNT * B_COUNT}")
    if len(weighted) != A_COUNT:
        errors.append(f"weighted rows {len(weighted)} != {A_COUNT}")

    raw_errors = [r for r in raw if r.get("error")]
    if raw_errors:
        errors.append(f"raw error rows: {len(raw_errors)}")

    candidates = Counter(r["candidate"] for r in summary)
    opponents = Counter(r["opponent_key"] for r in summary)
    if len(candidates) != A_COUNT:
        errors.append(f"candidates in summary: {len(candidates)}")
    if any(n != B_COUNT for n in candidates.values()):
        bad = {k: n for k, n in candidates.items() if n != B_COUNT}
        errors.append(f"candidate opponent counts != {B_COUNT}: {bad}")
    if len(opponents) != B_COUNT:
        errors.append(f"opponents in summary: {len(opponents)}")
    if any(n != A_COUNT for n in opponents.values()):
        bad = {k: n for k, n in opponents.items() if n != A_COUNT}
        errors.append(f"opponent candidate counts != {A_COUNT}: {bad}")

    raw_pairs = Counter((r["candidate"], r["opponent_key"]) for r in raw)
    bad_reps = {k: n for k, n in raw_pairs.items() if n != REPS}
    if bad_reps:
        errors.append(f"pairs without {REPS} reps: {len(bad_reps)}")

    for row in summary:
        for col in ("score_a_mean", "score_b_mean"):
            if row[col] == "":
                errors.append(f"empty {col}: {row['candidate']} "
                              f"{row['opponent_key']}")
                continue
            value = float(row[col])
            if not (0.0 <= value <= 10.0 * ROUNDS):
                errors.append(f"out-of-range {col}={value}: "
                              f"{row['candidate']} {row['opponent_key']}")

    # Recompute the weighted summaries from matchups.csv.
    acc: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "w4": 0.0, "w5": 0.0, "s4": 0.0, "s5": 0.0,
                 "eq": 0.0})
    for row in summary:
        item = acc[row["candidate"]]
        s = float(row["score_a_mean"])
        w4 = float(row["weight_v4"])
        w5 = float(row["weight_v5"])
        item["n"] += 1
        item["w4"] += w4
        item["w5"] += w5
        item["s4"] += s * w4
        item["s5"] += s * w5
        item["eq"] += s
    wrows = {r["candidate"]: r for r in weighted}
    for candidate, item in acc.items():
        if candidate not in wrows:
            errors.append(f"missing weighted row: {candidate}")
            continue
        row = wrows[candidate]
        for col, value in (
            ("v4_weighted_score", item["s4"] / item["w4"]),
            ("v5_weighted_score", item["s5"] / item["w5"]),
            ("equal_weight_score", item["eq"] / item["n"]),
        ):
            got = float(row[col])
            if abs(got - value) > 1e-6:
                errors.append(f"{candidate} {col}: {got} != {value}")

    print(f"raw rows: {len(raw):,}")
    print(f"summary rows: {len(summary):,}")
    print(f"candidates: {len(candidates)}; opponents: {len(opponents)}")
    print(f"raw errors: {len(raw_errors)}")
    print(f"run_result.complete: {result.get('complete')}")
    print(f"missing shards: {len(result.get('missing_shards', []))}")
    print(f"task failures: {len(result.get('failures', []))}")
    print(f"config: A={config['a_count']} B={config['b_count']} "
          f"reps={config['reps']} rounds={config['rounds']}")
    if errors:
        print("\nFAILED:")
        for error in errors[:100]:
            print(" -", error)
        if len(errors) > 100:
            print(f" ... {len(errors) - 100} more")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
