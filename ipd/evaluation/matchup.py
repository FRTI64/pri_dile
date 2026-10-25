#!/usr/bin/env python3
"""Unified head-to-head matchup matrix against every opponent archetype."""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

from ..core.opponents_extended import TYPE_BANK, TYPE_BY_NAME
from ..core.simulator import play_match
from ..paths import result_path
from ..policies.registry import BASE_NAMES, candidate_specs

SCHEMA = "ipd-matchup-v1"


def type_task(task: dict) -> list[tuple[str, str, str, float, float, float, int]]:
    from ..policies.registry import CandidateSpec

    type_name = task["type"]
    spec = TYPE_BY_NAME[type_name]
    reps = int(task["reps"])
    base_seed = int(task["seed"])
    specs = [CandidateSpec.from_dict(x) for x in task["candidates"]]
    rows = []
    for rep in range(reps):
        opp_seed = (base_seed + rep * 1000
                    + zlib.crc32(type_name.encode("utf-8")) % 997)
        for cand_spec in specs:
            opponent = spec.factory(random.Random(opp_seed))
            opponent.name = type_name
            candidate = cand_spec.build()
            result = play_match(candidate, opponent)
            rows.append((cand_spec.name, type_name, spec.family,
                         result.score_a, result.score_b,
                         result.score_a - result.score_b, 1))
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Per-opponent-type matchup matrix for one or more candidates.")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated names (default: base rule candidates)")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--out", default=result_path("matchup_results.csv"))
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    specs = candidate_specs(args.candidates, default=BASE_NAMES)
    tasks = [{"type": spec.name, "reps": args.reps, "seed": args.seed,
              "candidates": [x.to_dict() for x in specs]}
             for spec in TYPE_BANK]

    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rows in ex.map(type_task, tasks, chunksize=1):
            for cand, typ, family, our, opp, diff, n in rows:
                key = (cand, typ, family)
                bucket = agg[key]
                bucket[0] += our
                bucket[1] += opp
                bucket[2] += diff
                bucket[3] += n

    out = []
    for (cand, typ, family), (our, opp, diff, n) in agg.items():
        out.append((cand, typ, family, our / n, opp / n, diff / n, n))
    out.sort(key=lambda x: (x[0], x[2], x[1]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate", "type", "family", "our", "opponent",
                         "diff", "reps"])
        writer.writerows(out)

    by_cand = defaultdict(list)
    by_family = defaultdict(list)
    for cand, typ, family, our, opp, diff, n in out:
        by_cand[cand].append(our)
        by_family[(cand, family)].append(our)
    print("\nEqual-type mean")
    for cand, vals in sorted(by_cand.items(),
                             key=lambda kv: statistics.fmean(kv[1]),
                             reverse=True):
        print(f"{cand:16}{statistics.fmean(vals):8.2f}")
    families = sorted({family for _, family in by_family})
    print("\nFamily means")
    print(f"{'strategy':16}" + "".join(f"{f:>12}" for f in families))
    for spec in specs:
        print(f"{spec.name:16}"
              + "".join(f"{statistics.fmean(by_family[(spec.name, f)]):12.2f}"
                        for f in families))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
