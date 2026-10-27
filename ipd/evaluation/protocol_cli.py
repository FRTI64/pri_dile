#!/usr/bin/env python3
"""CLI for Protocol v2: audit, freeze and evaluate train/dev/test splits."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from ..paths import result_path
from ..policies.registry import CandidateSpec, candidate_specs
from .common import evaluate_field_details, quantile, run_match
from .protocol import (
    PROTOCOL_NAME, SPLIT_TIERS, audit_protocol, build_protocol_field,
    build_protocol_red_field, catalog_sha256, guard_test_usage,
    record_usage, verify_manifest, write_manifest,
)

DEV_DEFAULT = "dev_param,dev_family"
ALL_DEFAULT = "dev_param,dev_family"


def _parse_tiers(raw: str) -> list[str]:
    text = raw.strip()
    if text in ("dev", "dev_all"):
        return ["dev_param", "dev_family"]
    if text == "all":
        return list(SPLIT_TIERS)
    tiers = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in tiers if x not in SPLIT_TIERS]
    if unknown:
        raise SystemExit(f"unknown tiers: {unknown}; choose from {SPLIT_TIERS}")
    return tiers


def _eval_task(task: dict[str, Any]) -> dict[str, Any]:
    spec = CandidateSpec.from_dict(task["spec"])
    tier = str(task["tier"])
    seed = int(task["seed"])
    n = int(task["n"])
    rounds = int(task.get("rounds", 100))
    field = build_protocol_field(tier, seed, n)
    policy = spec.build()
    stats, by_opponent = evaluate_field_details(policy, field, rounds)
    family_sums: dict[str, float] = defaultdict(float)
    family_counts: dict[str, int] = defaultdict(int)
    for name, scores in by_opponent.items():
        family = name.split(":", 1)[0]
        family_sums[family] += float(sum(scores))
        family_counts[family] += len(scores)
    return {
        "candidate": spec.name,
        "tier": tier,
        "seed": seed,
        "mean": stats.mean,
        "min": min(stats.scores) if stats.scores else 0.0,
        "scores": list(stats.scores),
        "family_sums": dict(family_sums),
        "family_counts": dict(family_counts),
    }


def _aggregate(rows: Sequence[dict[str, Any]],
               baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    field_means = [float(row["mean"]) for row in rows]
    family_sums: dict[str, float] = defaultdict(float)
    family_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for family, value in row["family_sums"].items():
            family_sums[family] += float(value)
        for family, count in row["family_counts"].items():
            family_counts[family] += int(count)
    families = {
        family: family_sums[family] / family_counts[family]
        for family in sorted(family_sums) if family_counts[family]
    }
    normalised: dict[str, float] = {}
    if baseline:
        base_families = baseline.get("families", {})
        for family, value in families.items():
            base_value = base_families.get(family)
            if base_value:
                normalised[family] = value / float(base_value)
    return {
        "fields": len(field_means),
        "mean": statistics.fmean(field_means) if field_means else 0.0,
        "min": min(field_means) if field_means else 0.0,
        "p05": quantile(field_means, 0.05),
        "max": max(field_means) if field_means else 0.0,
        "std": (statistics.pstdev(field_means)
                if len(field_means) > 1 else 0.0),
        "worst_family": min(families.values()) if families else 0.0,
        "min_norm": min(normalised.values()) if normalised else None,
        "families": families,
        "family_norm": normalised,
        "values": field_means,
    }


def _audit(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit Protocol v2 splits.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    report = audit_protocol()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"written {path}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"protocol: {report['protocol']}")
        print(f"catalog:  {report['catalog_size']} configs, "
              f"sha256 {report['catalog_sha256'][:16]}...")
        print(f"manifest: {report['manifest']['reason']}")
        print("\ncounts by group/family:")
        for group, families in report["counts"].items():
            total = sum(families.values())
            print(f"  {group:12} {total:4d}  "
                  + ", ".join(f"{k}={v}" for k, v in sorted(families.items())))
        print("\nheld-out families:")
        print("  dev-only : " + ", ".join(report["tier_families"]["dev_family"]))
        print("  test-only: " + ", ".join(report["tier_families"]["test_only"]))
        if report["warnings"]:
            print("\nwarnings:")
            for warning in report["warnings"]:
                print(f"  - {warning}")
        if report["errors"]:
            print("\nerrors:")
            for error in report["errors"]:
                print(f"  - {error}")
        print("\nOK" if report["ok"] else "\nFAILED")
    return 0 if report["ok"] else 1


def _manifest(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Write or verify Protocol v2 manifest.")
    ap.add_argument("--write", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    path = write_manifest(args.out)
    check = verify_manifest(args.out)
    print(f"written {path}")
    print(f"catalog_sha256 {check['catalog_sha256']}")
    print("OK" if check["ok"] else f"FAILED: {check['reason']}")
    return 0 if check["ok"] else 1


def _eval(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate candidates on Protocol v2 dev/test tiers.")
    ap.add_argument("--candidates", default="all",
                    help="comma-separated registry names or sets")
    ap.add_argument("--tiers", default=DEV_DEFAULT,
                    help=f"{DEV_DEFAULT}, all, or a comma list from {SPLIT_TIERS}")
    ap.add_argument("--fields", type=int, default=None,
                    help="fields per tier (default: 20 dev, 30 test)")
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None,
                    help="candidate name used for per-family min_norm")
    ap.add_argument("--acknowledge-once", action="store_true",
                    help="required for test_v5; records one-shot usage")
    ap.add_argument("--force", action="store_true",
                    help="rerun a consumed test tier and mark it contaminated")
    args = ap.parse_args(argv)

    tiers = _parse_tiers(args.tiers)
    if "test_v5" in tiers and not args.acknowledge_once:
        raise SystemExit("test_v5 requires --acknowledge-once")
    if "test_v5" in tiers and args.fields is None:
        fields = 30
    else:
        fields = args.fields if args.fields is not None else 20
    if fields <= 0:
        raise SystemExit("--fields must be positive")

    specs = candidate_specs(args.candidates, default="all")
    if not specs:
        raise SystemExit("no candidates selected")
    names = [spec.name for spec in specs]
    if args.baseline and args.baseline not in names:
        raise SystemExit(f"baseline {args.baseline!r} is not in the candidate list")

    if "test_v5" in tiers:
        for name in names:
            try:
                guard_test_usage(name, force=args.force)
            except RuntimeError as exc:
                raise SystemExit(str(exc))

    seeds = [args.seed + i * 7919 for i in range(fields)]
    tasks = [
        {"spec": spec.to_dict(), "tier": tier, "seed": seed,
         "n": args.n, "rounds": args.rounds}
        for spec in specs
        for tier in tiers
        for seed in seeds
    ]

    started = time.time()
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {
        (name, tier): [] for name in names for tier in tiers
    }
    total = len(tasks)
    progress_every = max(1, total // 20)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for done, result in enumerate(ex.map(_eval_task, tasks, chunksize=1), 1):
            rows[(result["candidate"], result["tier"])].append(result)
            if done % progress_every == 0 or done == total:
                print(f"progress {done}/{total} field tasks", flush=True)

    aggregated: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in names
    }
    for name in names:
        for tier in tiers:
            aggregated[name][tier] = _aggregate(rows[(name, tier)])
    if args.baseline:
        for name in names:
            if name == args.baseline:
                continue
            for tier in tiers:
                aggregated[name][tier] = _aggregate(
                    rows[(name, tier)],
                    baseline=aggregated[args.baseline][tier],
                )

    out = args.out
    if out is None:
        if "test_v5" in tiers:
            out = result_path(f"protocol_v2/{PROTOCOL_NAME}_test.json")
        else:
            out = result_path(f"protocol_v2/{PROTOCOL_NAME}_dev.json")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "ipd-protocol-eval-v1",
        "protocol": PROTOCOL_NAME,
        "catalog_sha256": catalog_sha256(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "tiers": tiers,
        "fields": fields,
        "n": args.n,
        "rounds": args.rounds,
        "seed": args.seed,
        "baseline": args.baseline,
        "candidates": [spec.to_dict() for spec in specs],
        "results": aggregated,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\nProtocol v2 evaluation ({fields} fields, n={args.n})")
    for name in names:
        cells = []
        for tier in tiers:
            row = aggregated[name][tier]
            cells.append(f"{tier}={row['mean']:.2f}")
        print(f"  {name:20} " + "  ".join(cells))
    print(f"\nwritten {out_path}")

    if "test_v5" in tiers:
        for name in names:
            record_usage("test_v5", name, str(out_path), force=args.force)
        print("test_v5 usage recorded")
    return 0


def _red_task(task: dict[str, Any]) -> dict[str, Any]:
    spec = CandidateSpec.from_dict(task["spec"])
    tier = str(task["tier"])
    seed = int(task["seed"])
    rounds = int(task.get("rounds", 100))
    red = build_protocol_red_field(tier, seed, 1)[0]
    blue = spec.build()
    result = run_match(blue, red, rounds=rounds)
    return {
        "candidate": spec.name,
        "tier": tier,
        "seed": seed,
        "blue": float(result.score),
        "red": float(result.opponent_score),
        "red_win": bool(result.opponent_score > result.score),
    }


def _red(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate candidates against Protocol v2 red pools.")
    ap.add_argument("--candidates", default="all")
    ap.add_argument("--tier", default="dev_red",
                    choices=("dev_red", "test_red"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=None)
    ap.add_argument("--acknowledge-once", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.tier == "test_red" and not args.acknowledge_once:
        raise SystemExit("test_red requires --acknowledge-once")
    specs = candidate_specs(args.candidates, default="all")
    names = [spec.name for spec in specs]
    if args.tier == "test_red":
        for name in names:
            try:
                guard_test_usage(name, force=args.force, tier="test_red")
            except RuntimeError as exc:
                raise SystemExit(str(exc))

    seeds = [args.seed + i * 7919 for i in range(args.seeds)]
    tasks = [
        {"spec": spec.to_dict(), "tier": args.tier, "seed": seed,
         "rounds": args.rounds}
        for spec in specs for seed in seeds
    ]
    started = time.time()
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for result in ex.map(_red_task, tasks, chunksize=1):
            rows[result["candidate"]].append(result)

    summary: dict[str, dict[str, Any]] = {}
    for name in names:
        values = rows[name]
        blue = [row["blue"] for row in values]
        red = [row["red"] for row in values]
        summary[name] = {
            "seeds": len(values),
            "blue_mean": statistics.fmean(blue),
            "blue_min": min(blue),
            "blue_p05": quantile(blue, 0.05),
            "red_mean": statistics.fmean(red),
            "diff": statistics.fmean(b - r for b, r in zip(blue, red)),
            "red_win": sum(1 for row in values if row["red_win"]) / len(values),
        }

    out = args.out
    if out is None:
        out = result_path(f"protocol_v2/protocol_v2_{args.tier}.json")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ipd-protocol-red-eval-v1",
        "protocol": PROTOCOL_NAME,
        "catalog_sha256": catalog_sha256(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "tier": args.tier,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "seed": args.seed,
        "results": summary,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n{args.tier}: {args.seeds} seeds")
    for name in names:
        row = summary[name]
        print(f"  {name:20} blue={row['blue_mean']:7.2f} "
              f"red={row['red_mean']:7.2f} diff={row['diff']:+7.2f} "
              f"red_win={row['red_win']:.1%}")
    print(f"written {out_path}")
    if args.tier == "test_red":
        for name in names:
            record_usage("test_red", name, str(out_path), force=args.force)
        print("test_red usage recorded")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print("usage: python3 -m ipd protocol <audit|manifest|eval|red> [options]")
        return 0
    command, rest = args[0], args[1:]
    if command == "audit":
        return _audit(rest)
    if command == "manifest":
        return _manifest(rest)
    if command in ("eval", "evaluate"):
        return _eval(rest)
    if command in ("red", "red-eval"):
        return _red(rest)
    print(f"unknown protocol command: {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
