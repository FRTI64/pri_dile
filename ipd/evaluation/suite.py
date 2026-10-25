#!/usr/bin/env python3
"""Unified nine-suite benchmark (family-weighted opponent mixtures)."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

from ..paths import result_path
from ..policies.registry import SUITE_NAMES, candidate_specs, resolve_names
from .common import evaluate_field
from .distributions import FAMILY_MULT, sample_suite_field

SCHEMA = "ipd-suite-v1"


def suite_task(task: dict) -> tuple[str, int, dict[str, float]]:
    from ..policies.registry import CandidateSpec

    suite = task["suite"]
    seed = int(task["seed"])
    n = int(task["n"])
    specs = [CandidateSpec.from_dict(x) for x in task["specs"]]
    field = sample_suite_field(suite, seed, n)
    out = {}
    for spec in specs:
        policy = spec.build()
        # Fresh opponent RNG state for every candidate.
        out[spec.name] = evaluate_field(policy, copy.deepcopy(field)).mean
    return suite, seed, out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Benchmark candidates on nine suites.")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated names or core|base|suite|learned|all")
    ap.add_argument("--suites", default="all",
                    help="comma-separated suite names or 'all'")
    ap.add_argument("--fields", type=int, default=12)
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out", default=result_path("suite_results.json"))
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    specs = candidate_specs(args.candidates, default=SUITE_NAMES)
    names = [spec.name for spec in specs]

    suites = (list(FAMILY_MULT) if args.suites.strip() == "all"
              else [x.strip() for x in args.suites.split(",") if x.strip()])
    unknown = [s for s in suites if s not in FAMILY_MULT]
    if unknown:
        ap.error(f"unknown suites: {unknown}; choose from {list(FAMILY_MULT)}")

    import random
    rng = random.Random(args.seed)
    seeds = {suite: [rng.randrange(1 << 30) for _ in range(args.fields)]
             for suite in suites}
    tasks = [
        {"suite": suite, "seed": seed, "n": args.n,
         "specs": [spec.to_dict() for spec in specs]}
        for suite in suites for seed in seeds[suite]
    ]

    started = time.time()
    values: dict[str, dict[str, list[float]]] = {
        name: {suite: [] for suite in suites} for name in names
    }
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for suite, seed, result in ex.map(suite_task, tasks, chunksize=1):
            for name, value in result.items():
                values[name][suite].append(value)

    print("suites", suites)
    print(f"{'strategy':18}" + "".join(f"{s:>11}" for s in suites))
    summary: dict[str, dict] = {}
    for spec in specs:
        row = {}
        for suite in suites:
            vals = values[spec.name][suite]
            row[suite] = {"mean": statistics.fmean(vals), "min": min(vals),
                          "values": vals}
        summary[spec.name] = row
        print(f"{spec.name:18}"
              + "".join(f"{row[s]['mean']:11.2f}" for s in suites))

    payload = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "seed": args.seed,
        "fields": args.fields,
        "n": args.n,
        "suites": suites,
        "candidates": [spec.to_dict() for spec in specs],
        "summary": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
