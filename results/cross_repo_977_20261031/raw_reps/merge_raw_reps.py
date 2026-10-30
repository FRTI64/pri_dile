#!/usr/bin/env python3
"""Merge raw_reps/<candidate>.csv into raw_reps.csv."""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "raw_reps.csv"
EXPECTED = 586_200


def main() -> int:
    files = sorted(p for p in HERE.glob("*.csv") if p.name != "raw_reps.csv")
    if not files:
        raise SystemExit("no candidate CSV files found")
    with files[0].open(newline="", encoding="utf-8") as fh:
        fields = csv.DictReader(fh).fieldnames
    if fields is None:
        raise SystemExit(f"missing header: {files[0]}")
    total = 0
    tmp = OUT.with_name(OUT.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for path in files:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames != fields:
                    raise SystemExit(f"header mismatch: {path}")
                for row in reader:
                    writer.writerow(row)
                    total += 1
    if total != EXPECTED:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"row count {total} != {EXPECTED}")
    tmp.replace(OUT)
    print(f"wrote {OUT} ({total:,} rows from {len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
