#!/usr/bin/env python3
"""Evaluate arbitrary weight files with the Protocol v5 dev harness.

The stock ``python3 -m ipd protocol eval`` CLI resolves candidates from the
registry.  This wrapper builds one ``CandidateSpec`` per checkpoint and reuses
the same worker helpers and seed schedule as the CLI, so numbers are directly
comparable to ``protocol_v5_dev.json``.

It never touches the test usage ledger; every run is provisional.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.evaluation.protocol import (  # noqa: E402
    DEV_TIERS, PROTOCOL_NAME, PROTOCOL_VERSION, TEST_TIER,
)
from ipd.evaluation.protocol_cli import (  # noqa: E402
    _aggregate, _eval_field_task,
)
from ipd.evaluation.scheduler import chunk_specs  # noqa: E402
from ipd.policies.registry import CandidateSpec  # noqa: E402

TIERS = DEV_TIERS + (TEST_TIER,)


def _parse_specs(path: str) -> list[CandidateSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    specs: list[CandidateSpec] = []
    for row in data["candidates"]:
        params = {"mode": row.get("mode", "sample")}
        if row.get("weights"):
            params["weights"] = str(row["weights"])
        if row.get("config"):
            params["config"] = str(row["config"])
        specs.append(CandidateSpec(kind=str(row["kind"]),
                                   name=str(row["name"]),
                                   params=params))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True,
                    help="JSON with {'candidates': [{name,kind,weights,...}]}")
    ap.add_argument("--tiers", default=",".join(DEV_TIERS))
    ap.add_argument("--fields", type=int, default=30)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--candidate-chunk", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--baseline", default=None,
                    help="candidate name used for per-family min_norm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-consume", action="store_true",
                    help="kept for command parity; this wrapper never consumes")
    args = ap.parse_args()

    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        raise SystemExit(f"unknown tiers: {unknown}")
    if TEST_TIER in tiers:
        print(f"WARNING: {TEST_TIER} is provisional; this wrapper never writes "
              "the usage ledger and never enables selection", flush=True)
    specs = _parse_specs(args.specs)
    names = [s.name for s in specs]
    if args.baseline and args.baseline not in names:
        raise SystemExit(f"baseline {args.baseline!r} not in {names}")

    seeds = [args.seed + i * 7919 for i in range(args.fields)]
    chunks = chunk_specs(specs, args.candidate_chunk)
    tasks = []
    for tier in tiers:
        for seed in seeds:
            for chunk in chunks:
                tasks.append({
                    "specs": [s.to_dict() for s in chunk],
                    "tier": tier, "seed": seed, "n": args.n,
                    "rounds": args.rounds, "field_reuse": "auto",
                    "opponent_mode": "sample",
                })
    print(f"candidates={names} tiers={tiers} fields={args.fields} "
          f"n={args.n} seed={args.seed} tasks={len(tasks)}", flush=True)

    rows: dict[tuple[str, str], list[dict]] = {
        (name, tier): [] for name in names for tier in tiers
    }
    started = time.time()
    workers = max(1, min(args.workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_eval_field_task, task) for task in tasks]
        done = 0
        for fut in as_completed(futures):
            payload = fut.result()
            for row in payload["results"]:
                rows[(row["candidate"], payload["tier"])].append(row)
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"progress {done}/{len(tasks)} "
                      f"elapsed={time.time() - started:.0f}s", flush=True)

    aggregated = {}
    base_by_tier = {}
    if args.baseline:
        for tier in tiers:
            base_by_tier[tier] = _aggregate(rows[(args.baseline, tier)])
    for name in names:
        aggregated[name] = {}
        for tier in tiers:
            aggregated[name][tier] = _aggregate(
                rows[(name, tier)], baseline=base_by_tier.get(tier))

    payload = {
        "schema": "ipd-protocol-eval-v1",
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "split_version": PROTOCOL_VERSION,
        "sample_rng_version": "4",
        "wrapper": "results/protocol_v5/eval_weights.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "seed": args.seed,
        "fields": args.fields,
        "n": args.n,
        "rounds": args.rounds,
        "tiers": tiers,
        "seeds": seeds,
        "no_consume": True,
        "selection_eligible": False,
        "test_consumption_locked": True,
        "baseline": args.baseline,
        "candidates": [s.to_dict() for s in specs],
        "results": aggregated,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nProtocol v5 evaluation ({args.fields} fields, n={args.n})")
    for name in names:
        cells = "  ".join(f"{t}={aggregated[name][t]['mean']:.2f}"
                          for t in tiers)
        print(f"  {name:24} {cells}")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
