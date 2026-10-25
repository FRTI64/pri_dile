#!/usr/bin/env python3
"""Unified four-distribution benchmark.

This unifies the legacy four-distribution benchmark scripts.  Every candidate
is referenced by a registry name, so adding a policy does not require changing
this file.

Examples::

    python3 -m ipd benchmark
    python3 -m ipd benchmark --candidates core,neural,gru
    python3 -m ipd benchmark --candidates all --fields 20 --workers 20
    python3 -m ipd benchmark --arch rl --fields 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

from ..paths import result_path
from ..policies.registry import (
    CORE_NAMES, candidate_specs, legacy_arch_names, resolve_names,
)
from .common import eval_task, seed_map
from .distributions import DISTS

SCHEMA = "ipd-benchmark-v1"
_LEGACY_ARCHES = ("mlp", "gru", "rl", "adversarial", "adv-rl",
                  "adv-mlp", "adv-gru")
_ARCH_OUT = {
    "mlp": "neural_benchmark.json",
    "gru": "gru_benchmark.json",
    "rl": "rl_benchmark.json",
    "adversarial": "adversarial_benchmark.json",
    "adv-mlp": "adversarial_benchmark.json",
    "adv-gru": "adversarial_gru_benchmark.json",
    "adv-rl": "adversarial_rl_benchmark.json",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Benchmark any registered candidate on the four official "
                    "distributions.")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated names or core|base|suite|learned|all")
    ap.add_argument("--arch", default=None, choices=_LEGACY_ARCHES,
                    help="legacy shortcut: add one learned policy to the core set")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--neural-mode", "--mode", dest="mode", default="sample",
                    choices=("sample", "argmax"))
    # Legacy final_benchmark flags.
    ap.add_argument("--neural", action="store_true",
                    help="legacy: include the default neural policy")
    ap.add_argument("--neural-arch", default="mlp",
                    choices=_LEGACY_ARCHES)
    ap.add_argument("--neural-weights", default=None)
    ap.add_argument("--neural-config", default=None)
    ap.add_argument("--include-baselines", action="store_true", default=True)
    ap.add_argument("--no-baselines", dest="include_baselines",
                    action="store_false")
    ap.add_argument("--dists", default=",".join(DISTS))
    ap.add_argument("--fields", type=int, default=8)
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--out", default=None)
    return ap


def _selected_names(args: argparse.Namespace) -> list[str]:
    if args.candidates:
        names = resolve_names(args.candidates, default=CORE_NAMES)
        if not args.include_baselines and args.arch:
            names = []
    else:
        names = list(CORE_NAMES) if args.include_baselines else []

    extra: list[str] = []
    if args.arch:
        extra.extend(legacy_arch_names(args.arch))
    if args.neural:
        extra.extend(legacy_arch_names(args.neural_arch))
    return list(dict.fromkeys(names + extra))


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    dists = tuple(x.strip() for x in args.dists.split(",") if x.strip())
    unknown = [d for d in dists if d not in DISTS]
    if unknown:
        ap.error(f"unknown distributions: {unknown}; choose from {DISTS}")

    names = _selected_names(args)
    if not names:
        ap.error("no candidates selected")
    mode = args.mode
    weights = args.weights or args.neural_weights
    config = args.config or args.neural_config
    specs = candidate_specs(names, weights=weights, config=config, mode=mode)

    out = args.out
    if out is None:
        arch = args.arch or (args.neural_arch if args.neural else None)
        out = result_path(_ARCH_OUT.get(arch, "benchmark.json"))

    seeds = seed_map(args.seed, dists, args.fields)
    tasks = [
        {
            "spec": spec.to_dict(),
            "name": spec.name,
            "dist": dist,
            "seeds": seeds[dist],
            "n": args.n,
            "rounds": args.rounds,
        }
        for spec in specs
        for dist in dists
    ]

    started = time.time()
    results: dict[str, dict[str, list[float]]] = {spec.name: {} for spec in specs}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, dist, values in ex.map(eval_task, tasks, chunksize=1):
            results[name][dist] = values
            print(f"{name:18} {dist:9} "
                  f"mean={statistics.fmean(values):7.2f} "
                  f"min={min(values):7.2f}", flush=True)

    print("\nSummary (mean score per opponent):")
    header = f"{'strategy':20}" + "".join(f"{d:>10}" for d in dists)
    print(header + f"{'avg':>10}{'worst':>10}{'min':>10}")
    summary: dict[str, dict] = {}
    for spec in specs:
        rows = results[spec.name]
        means = [statistics.fmean(rows[d]) for d in dists]
        row = {
            "kind": spec.kind,
            "dists": {
                d: {
                    "mean": statistics.fmean(rows[d]),
                    "min": min(rows[d]),
                    "p05": sorted(rows[d])[max(0, int(0.05 * len(rows[d])))],
                    "values": rows[d],
                }
                for d in dists
            },
            "avg": statistics.fmean(means),
            "worst": min(means),
            "min": min(min(rows[d]) for d in dists),
        }
        summary[spec.name] = row
        print(f"{spec.name:20}" + "".join(f"{m:10.2f}" for m in means)
              + f"{row['avg']:10.2f}{row['worst']:10.2f}{row['min']:10.2f}")

    payload = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "seed": args.seed,
        "fields": args.fields,
        "n": args.n,
        "rounds": args.rounds,
        "dists": list(dists),
        "candidates": [spec.to_dict() for spec in specs],
        "summary": summary,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
