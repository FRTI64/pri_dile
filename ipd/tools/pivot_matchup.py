#!/usr/bin/env python3
"""Pivot a long matchup CSV into candidate × opponent score tables.

Input rows (``results/matchup_all.csv``)::

    candidate,type,family,our,opponent,diff,reps

Outputs:

* ``matchup_all_wide.csv``: one row per candidate, one column per opponent
  type, each cell is ``candidate_score|opponent_score``.
* ``matchup_all_scores.csv``: long format with a single ``score`` column in
  the same ``candidate_score|opponent_score`` form.

Run from the repository root::

    python3 ipd/tools/pivot_matchup.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _fmt(value: str) -> str:
    number = float(value)
    return f"{number:g}"


def _opponent_order(csv_path: Path, json_path: Path) -> list[str]:
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        names = [row["name"] for row in data.get("opponents", [])]
        if names:
            return names
    seen: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["type"] not in seen:
                seen.append(row["type"])
    return seen


def pivot(csv_path: Path, json_path: Path, wide_path: Path,
          score_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    candidates: list[str] = []
    cells: dict[tuple[str, str], str] = {}
    long_rows: list[list[str]] = []
    for row in rows:
        name = row["candidate"]
        if name not in candidates:
            candidates.append(name)
        score = f"{_fmt(row['our'])}|{_fmt(row['opponent'])}"
        cells[(name, row["type"])] = score
        long_rows.append([name, row["type"], row["family"], score])

    opponents = _opponent_order(csv_path, json_path)
    # Keep any type that only appears in the CSV (defensive).
    for _cand, typ in cells:
        if typ not in opponents:
            opponents.append(typ)

    wide_path.parent.mkdir(parents=True, exist_ok=True)
    with open(wide_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["candidate", *opponents])
        for cand in candidates:
            writer.writerow([cand, *[cells.get((cand, typ), "")
                                     for typ in opponents]])

    with open(score_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["candidate", "type", "family", "score"])
        writer.writerows(long_rows)

    print(f"written {wide_path} ({len(candidates)} candidates × "
          f"{len(opponents)} opponents)")
    print(f"written {score_path} ({len(long_rows)} rows)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Pivot matchup_all.csv into candidate × opponent tables.")
    ap.add_argument("--input", default=str(ROOT / "results" / "matchup_all.csv"))
    ap.add_argument("--json", default=str(ROOT / "results" / "matchup_all.json"),
                    help="optional JSON used for opponent column order")
    ap.add_argument("--wide-out",
                    default=str(ROOT / "results" / "matchup_all_wide.csv"))
    ap.add_argument("--scores-out",
                    default=str(ROOT / "results" / "matchup_all_scores.csv"))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pivot(Path(args.input), Path(args.json), Path(args.wide_out),
          Path(args.scores_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
