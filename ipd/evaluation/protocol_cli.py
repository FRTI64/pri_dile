#!/usr/bin/env python3
"""CLI for Protocol v4: audit, freeze and evaluate train/dev/test splits."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed,
)
from pathlib import Path
from typing import Any, Sequence

from ..paths import result_path
from ..policies.registry import CandidateSpec, candidate_specs
from .common import FieldStats, evaluate_field_details, quantile, run_match
from .protocol import (
    COMPLEXITY_TIERS, DEV_FIELDS, DEV_TIERS, LEVEL_TIERS,
    PROTOCOL_N_PLAYERS, PROTOCOL_NAME, SPLIT_TIERS, TEST_FIELDS,
    TEST_POLICY_PATH, TEST_TIER, audit_protocol, build_protocol_field,
    build_protocol_red_field, catalog_sha256, field_reuse_safe,
    guard_test_usage, record_usage, restore_field_rng, snapshot_field_rng,
    test_consumption_allowed, verify_manifest, write_manifest,
)
from .scheduler import (
    MemorySampler, ProfileAccumulator, ProgressTracker, chunk_specs,
    peak_rss_mb, worker_budget,
)

DEV_DEFAULT = "dev_param,dev_family,dev_complex,dev_axelrod,dev_ood"
ALL_DEFAULT = DEV_DEFAULT

_PLAIN_GPU_KINDS = frozenset({
    "neural", "gru", "rl", "rl-hard60", "rl-repair", "selfplay",
    "psro-br", "psro-scratch-br",
})
_ADV_GPU_KINDS = frozenset({"adv-mlp", "adv-gru", "adv-rl"})
_PSRO_META_KINDS = frozenset({"psro", "psro-scratch"})


_LEARNED_OPPONENT_TYPES = frozenset({
    "NeuralPolicy", "RLPolicy", "GRUPolicy", "SelfPlayGRUPolicy",
})


def _apply_opponent_mode(field: Sequence[Any], mode: str) -> None:
    """Switch learned policy opponents to the requested action mode."""
    if mode != "argmax":
        return
    for opponent in field:
        if type(opponent).__name__ in _LEARNED_OPPONENT_TYPES:
            opponent.mode = "argmax"


def _gpu_spec_supported(spec: CandidateSpec) -> bool:
    """Whether a candidate can use the batched argmax path.

    PSRO meta-policies need a population inspection because their blue mixture
    may contain stochastic rule members; those stay on CPU to preserve the
    Protocol v4 RNG stream.
    """
    kind = str(spec.kind)
    if kind in _PLAIN_GPU_KINDS or kind in _ADV_GPU_KINDS:
        return True
    if kind in _PSRO_META_KINDS:
        try:
            from .gpu.blue import supports_gpu_policy

            return supports_gpu_policy(spec.build())
        except Exception:
            return False
    return False


def _resolve_gpu(requested: str, candidate_mode: str,
                 *, quiet: bool = False) -> tuple[bool, str]:
    """Resolve ``--gpu auto|on|off`` and return ``(enabled, reason)``."""
    if requested == "off":
        return False, "disabled by --gpu off"
    try:
        import torch
    except ImportError:
        if requested == "on" and not quiet:
            print("WARNING: --gpu on requested but torch is unavailable; "
                  "using the CPU path", flush=True)
        return False, "torch unavailable"
    if not torch.cuda.is_available():
        if requested == "on" and not quiet:
            print("WARNING: --gpu on requested but CUDA is unavailable; "
                  "using the CPU path", flush=True)
        return False, "CUDA unavailable"
    if candidate_mode != "argmax":
        if requested == "on" and not quiet:
            print("WARNING: --gpu on only supports --mode argmax; "
                  "sample mode stays on CPU (Protocol v4 RNG contract)",
                  flush=True)
        return False, "sample mode stays on CPU"
    return True, "CUDA available"


def _parse_torch_dtype(name: str):
    import torch

    if name == "float64":
        return torch.float64
    return torch.float32


def _parse_tiers(raw: str) -> list[str]:
    text = raw.strip()
    if text in ("dev", "dev_all"):
        return ["dev_param", "dev_family", "dev_complex", "dev_axelrod",
                "dev_ood"]
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


def _field_result_rows(
        spec: CandidateSpec, tier: str, seed: int,
        stats: Any, by_opponent: dict[str, list[int]],
        timings: dict[str, float],
) -> dict[str, Any]:
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
        "timings": timings,
    }


def _eval_field_task(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a candidate chunk on one field.

    The field is built once per task.  Before every candidate after the first
    the opponent RNG states are restored to their post-build values, which is
    what makes reuse bit-identical to rebuilding the field.  If a field
    contains an object that cannot be safely reused, it is rebuilt for each
    candidate instead of risking RNG leakage.
    """
    specs = [CandidateSpec.from_dict(item) for item in task["specs"]]
    tier = str(task["tier"])
    seed = int(task["seed"])
    n = int(task["n"])
    rounds = int(task.get("rounds", 100))
    reuse_mode = str(task.get("field_reuse", "auto"))
    opponent_mode = str(task.get("opponent_mode", "sample"))

    started = time.perf_counter()
    field = build_protocol_field(tier, seed, n)
    _apply_opponent_mode(field, opponent_mode)
    field_build_seconds = time.perf_counter() - started
    reuse = reuse_mode != "off" and field_reuse_safe(field)
    snapshot = snapshot_field_rng(field) if reuse else None

    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        if index:
            if snapshot is not None:
                restore_field_rng(field, snapshot)
            else:
                started = time.perf_counter()
                field = build_protocol_field(tier, seed, n)
                _apply_opponent_mode(field, opponent_mode)
                field_build_seconds += time.perf_counter() - started

        started = time.perf_counter()
        policy = spec.build()
        candidate_build_seconds = time.perf_counter() - started

        family_seconds: dict[str, float] = {}
        started = time.perf_counter()
        stats, by_opponent = evaluate_field_details(
            policy, field, rounds, timings=family_seconds)
        eval_seconds = time.perf_counter() - started
        results.append(_field_result_rows(
            spec, tier, seed, stats, by_opponent, {
                "eval_seconds": eval_seconds,
                "candidate_build_seconds": candidate_build_seconds,
                "family_seconds": family_seconds,
            }))

    return {
        "tier": tier,
        "seed": seed,
        "field_build_seconds": field_build_seconds,
        "field_reuse": bool(snapshot is not None and len(specs) > 1),
        "results": results,
        "worker_pid": os.getpid(),
        "peak_rss_mb": _worker_peak_rss_mb(),
    }


def _eval_field_batch_task(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a candidate chunk on several fields with batched GPU rows.

    The task returns one result row per ``(candidate, field)`` so the existing
    Protocol v4 aggregation and result schema are unchanged.
    """
    specs = [CandidateSpec.from_dict(item) for item in task["specs"]]
    tier = str(task["tier"])
    seeds = [int(seed) for seed in task["seeds"]]
    n = int(task["n"])
    rounds = int(task.get("rounds", 100))
    reuse_mode = str(task.get("field_reuse", "auto"))
    gpu = dict(task.get("gpu") or {})

    started = time.perf_counter()
    fields = [build_protocol_field(tier, seed, n) for seed in seeds]
    for field in fields:
        _apply_opponent_mode(field, "argmax")
    field_build_seconds = time.perf_counter() - started
    reuse = (reuse_mode != "off"
             and all(field_reuse_safe(field) for field in fields))
    snapshots = ([snapshot_field_rng(field) for field in fields]
                 if reuse else None)

    results: list[dict[str, Any]] = []
    fallback = False
    for index, spec in enumerate(specs):
        if index:
            if snapshots is not None:
                for field, snapshot in zip(fields, snapshots):
                    restore_field_rng(field, snapshot)
            else:
                started = time.perf_counter()
                fields = [build_protocol_field(tier, seed, n) for seed in seeds]
                for field in fields:
                    _apply_opponent_mode(field, "argmax")
                field_build_seconds += time.perf_counter() - started

        started = time.perf_counter()
        policy = spec.build()
        candidate_build_seconds = time.perf_counter() - started

        try:
            from .gpu.simulator import evaluate_fields_gpu

            simulations = evaluate_fields_gpu(
                policy, fields, rounds=rounds,
                device=str(gpu.get("device", "cuda")),
                dtype=_parse_torch_dtype(str(gpu.get("dtype", "float32"))),
                max_batch=int(gpu.get("max_batch", 32768)),
                profile=bool(gpu.get("profile", False)))
            for seed, simulation in zip(seeds, simulations):
                stats = FieldStats(sum(simulation.scores), simulation.scores,
                                   simulation.names)
                results.append(_field_result_rows(
                    spec, tier, seed, stats, simulation.by_opponent, {
                        "eval_seconds": simulation.seconds,
                        "candidate_build_seconds": candidate_build_seconds,
                        "family_seconds": simulation.family_seconds,
                        "gpu_seconds": (simulation.blue_seconds
                                        + simulation.opponent_seconds),
                        "blue_seconds": simulation.blue_seconds,
                        "opponent_seconds": simulation.opponent_seconds,
                        "device": str(gpu.get("device", "cuda")),
                        "dtype": str(gpu.get("dtype", "float32")),
                        "mode": "argmax",
                    }))
        except Exception as exc:  # pragma: no cover - hardware/torch failure
            fallback = True
            if not gpu.get("quiet_fallback", False):
                print(f"WARNING: GPU evaluation failed for {spec.name}: "
                      f"{exc}; rebuilding fields and using CPU", flush=True)
            fields = [build_protocol_field(tier, seed, n) for seed in seeds]
            for field in fields:
                _apply_opponent_mode(field, "argmax")
            snapshots = ([snapshot_field_rng(field) for field in fields]
                         if reuse else None)
            for seed, field in zip(seeds, fields):
                cpu_policy = spec.build()
                family_seconds: dict[str, float] = {}
                cpu_started = time.perf_counter()
                stats, by_opponent = evaluate_field_details(
                    cpu_policy, field, rounds, timings=family_seconds)
                cpu_seconds = time.perf_counter() - cpu_started
                results.append(_field_result_rows(
                    spec, tier, seed, stats, by_opponent, {
                        "eval_seconds": cpu_seconds,
                        "candidate_build_seconds": candidate_build_seconds,
                        "family_seconds": family_seconds,
                        "device": "cpu-fallback",
                        "mode": "cpu-fallback",
                    }))

    return {
        "tier": tier,
        "seed": seeds[0] if seeds else 0,
        "seeds": seeds,
        "field_build_seconds": field_build_seconds,
        "field_reuse": bool(snapshots is not None and len(specs) > 1),
        "results": results,
        "worker_pid": os.getpid(),
        "peak_rss_mb": _worker_peak_rss_mb(),
        "gpu": {"enabled": not fallback, "fallback": fallback,
                "device": str(gpu.get("device", "cuda")),
                "dtype": str(gpu.get("dtype", "float32"))},
    }


def _worker_peak_rss_mb() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Unix
        return 0.0
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                 / divisor, 2)


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
    ap = argparse.ArgumentParser(description="Audit Protocol v4 splits.")
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
        policy = "locked" if not test_consumption_allowed() else "UNLOCKED"
        print(f"test policy: {policy} ({TEST_POLICY_PATH})")
        print("\ncounts by group/family:")
        for group, families in report["counts"].items():
            total = sum(families.values())
            print(f"  {group:12} {total:4d}  "
                  + ", ".join(f"{k}={v}" for k, v in sorted(families.items())))
        print("\nfamily groups:")
        for key in ("dev_param", "dev_family", "dev_complex",
                    "dev_axelrod", "train_complex", "train_axelrod",
                    "test_complex", "test_policy", "test_axelrod",
                    "test_only"):
            values = report["tier_families"].get(key, [])
            print(f"  {key:13}: " + ", ".join(values))
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
    ap = argparse.ArgumentParser(description="Write or verify Protocol v4 manifest.")
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
        description="Evaluate candidates on Protocol v4 dev/test tiers.")
    ap.add_argument("--candidates", default="all",
                    help="comma-separated registry names or sets")
    ap.add_argument("--tiers", default=DEV_DEFAULT,
                    help=f"{DEV_DEFAULT}, all, or a comma list from {SPLIT_TIERS}")
    ap.add_argument("--fields", type=int, default=None,
                    help=f"fields per tier (default: {DEV_FIELDS} dev, "
                         f"{TEST_FIELDS} test)")
    ap.add_argument("--n", type=int, default=PROTOCOL_N_PLAYERS)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--workers", default="20",
                    help="worker count or 'auto' "
                         "(min(20, cpu_count-2, RAM budget))")
    ap.add_argument("--candidate-chunk", type=int, default=6,
                    help="candidates evaluated per field build (default: 6)")
    ap.add_argument("--field-reuse", choices=("auto", "on", "off"),
                    default="auto",
                    help="reuse a field across a candidate chunk when RNG "
                         "state can be restored (default: auto)")
    ap.add_argument("--mode", choices=("sample", "argmax"), default="sample",
                    help="candidate action mode; GPU batching only supports "
                         "argmax, sample stays on CPU (default: sample)")
    ap.add_argument("--opponent-mode", choices=("sample", "argmax"),
                    default=None,
                    help="action mode for learned policy opponents; defaults "
                         "to --mode (GPU tasks always use argmax)")
    ap.add_argument("--gpu", choices=("auto", "on", "off"), default="auto",
                    help="batched GPU inference (default: auto; requires "
                         "--mode argmax)")
    ap.add_argument("--gpu-device", default="cuda")
    ap.add_argument("--gpu-dtype", choices=("float32", "float64"),
                    default="float32",
                    help="GPU arithmetic precision; use float64 for parity "
                         "checks (default: float32)")
    ap.add_argument("--gpu-max-batch", type=int, default=32768,
                    help="maximum rows per GPU chunk (default: 32768)")
    ap.add_argument("--gpu-min-batch", type=int, default=2048,
                    help="minimum aggregate batch for --gpu auto "
                         "(default: 2048)")
    ap.add_argument("--gpu-field-batch", type=int, default=4,
                    help="fields simulated together for cross-field policy "
                         "aggregation (default: 4)")
    ap.add_argument("--gpu-workers", type=int, default=8,
                    help="concurrent worker processes allowed to use CUDA "
                         "when CPU and GPU tasks are mixed (default: 8)")
    ap.add_argument("--profile", action="store_true",
                    help="write per-family/per-candidate timings to "
                         "results/protocol_v4/profile_*.json")
    ap.add_argument("--profile-out", default=None,
                    help="override the profile output path")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None,
                    help="candidate name used for per-family min_norm")
    ap.add_argument("--acknowledge-once", action="store_true",
                    help="cannot unlock test_v7 while test consumption is "
                         "locked; ignored for provisional runs")
    ap.add_argument("--no-consume", action="store_true",
                    help="evaluate test tier without writing usage ledger; "
                         "required while test consumption is locked")
    ap.add_argument("--force", action="store_true",
                    help="cannot override the test consumption lock; ignored "
                         "for provisional runs")
    args = ap.parse_args(argv)

    tiers = _parse_tiers(args.tiers)
    test_run = TEST_TIER in tiers
    consumption_locked = test_run and not test_consumption_allowed()
    no_consume = args.no_consume or consumption_locked
    if test_run and consumption_locked and not args.no_consume:
        raise SystemExit(
            f"{TEST_TIER} is locked (test consumption disabled); "
            "use --no-consume for a provisional run")
    if test_run and not consumption_locked and not args.acknowledge_once \
            and not args.no_consume:
        raise SystemExit(
            f"{TEST_TIER} requires --acknowledge-once or --no-consume")
    if TEST_TIER in tiers and args.fields is None:
        fields = TEST_FIELDS
    else:
        fields = args.fields if args.fields is not None else DEV_FIELDS
    if fields <= 0:
        raise SystemExit("--fields must be positive")
    if args.candidate_chunk <= 0:
        raise SystemExit("--candidate-chunk must be positive")
    if args.gpu_max_batch <= 0:
        raise SystemExit("--gpu-max-batch must be positive")
    if args.gpu_min_batch <= 0:
        raise SystemExit("--gpu-min-batch must be positive")
    if args.gpu_field_batch <= 0:
        raise SystemExit("--gpu-field-batch must be positive")
    if args.gpu_workers <= 0:
        raise SystemExit("--gpu-workers must be positive")
    try:
        budget = worker_budget(args.workers)
    except ValueError as exc:
        raise SystemExit(str(exc))
    workers = int(budget["resolved"])

    args.opponent_mode = (args.opponent_mode
                          or ("argmax" if args.mode == "argmax"
                              else "sample"))
    gpu_enabled, gpu_reason = _resolve_gpu(args.gpu, args.mode)
    aggregate = int(args.n) * int(args.gpu_field_batch)
    if gpu_enabled and args.gpu == "auto" and aggregate < args.gpu_min_batch:
        gpu_enabled = False
        gpu_reason = (f"aggregate batch {aggregate} < "
                      f"--gpu-min-batch {args.gpu_min_batch}")
    specs = candidate_specs(args.candidates, default="all", mode=args.mode)
    if not specs:
        raise SystemExit("no candidates selected")
    names = [spec.name for spec in specs]
    gpu_specs = ([spec for spec in specs if _gpu_spec_supported(spec)]
                 if gpu_enabled else [])
    cpu_specs = [spec for spec in specs if spec not in gpu_specs]
    if gpu_enabled and not gpu_specs:
        print("WARNING: --gpu resolved to GPU but no selected candidate "
              "supports batched argmax inference; using CPU", flush=True)
        gpu_enabled = False
        gpu_reason = "no supported candidates"
    if gpu_enabled and cpu_specs:
        print(f"GPU routing: {len(gpu_specs)} candidate(s) batched, "
              f"{len(cpu_specs)} candidate(s) on CPU", flush=True)
    if args.baseline and args.baseline not in names:
        raise SystemExit(f"baseline {args.baseline!r} is not in the candidate list")

    if consumption_locked:
        print(f"WARNING: test consumption is locked by {TEST_POLICY_PATH}; "
              "running provisionally (selection_eligible=false)", flush=True)
        if args.acknowledge_once or args.force:
            print("WARNING: --acknowledge-once/--force cannot unlock test "
                  "consumption and are ignored for this run", flush=True)
    elif test_run and args.no_consume:
        print("WARNING: --no-consume enabled; test set is NOT consumed and "
              "results are provisional, not selection-eligible", flush=True)

    if test_run and not no_consume:
        for name in names:
            try:
                guard_test_usage(name, force=args.force, tier=TEST_TIER)
            except RuntimeError as exc:
                raise SystemExit(str(exc))

    seeds = [args.seed + i * 7919 for i in range(fields)]
    cpu_tasks: list[dict[str, Any]] = []
    gpu_tasks: list[dict[str, Any]] = []
    cpu_chunks = chunk_specs(cpu_specs, args.candidate_chunk)
    for tier in tiers:
        for seed in seeds:
            for chunk in cpu_chunks:
                cpu_tasks.append({
                    "specs": [spec.to_dict() for spec in chunk], "tier": tier,
                    "seed": seed, "n": args.n, "rounds": args.rounds,
                    "field_reuse": args.field_reuse,
                    "opponent_mode": args.opponent_mode,
                })
    gpu_chunks = chunk_specs(gpu_specs, args.candidate_chunk)
    gpu_field_batch = int(args.gpu_field_batch) if gpu_enabled else 1
    gpu_config = {
        "device": args.gpu_device,
        "dtype": args.gpu_dtype,
        "max_batch": args.gpu_max_batch,
        "profile": bool(args.profile),
    }
    for tier in tiers:
        for start in range(0, len(seeds), gpu_field_batch):
            seed_group = seeds[start:start + gpu_field_batch]
            for chunk in gpu_chunks:
                gpu_tasks.append({
                    "specs": [spec.to_dict() for spec in chunk], "tier": tier,
                    "seeds": seed_group, "seed": seed_group[0],
                    "n": args.n, "rounds": args.rounds,
                    "field_reuse": args.field_reuse, "gpu": gpu_config,
                    "gpu_batch": True,
                })
    # Spread GPU tasks through the queue so the GPU and the CPU rule-heavy
    # tasks overlap instead of leaving the GPU idle until the CPU queue drains.
    tasks: list[dict[str, Any]] = []
    if gpu_tasks and cpu_tasks:
        stride = len(cpu_tasks) / len(gpu_tasks)
        next_gpu = 0.0
        gpu_index = 0
        for index, task in enumerate(cpu_tasks):
            tasks.append(task)
            if gpu_index < len(gpu_tasks) and index + 1 >= next_gpu:
                tasks.append(gpu_tasks[gpu_index])
                gpu_index += 1
                next_gpu += stride
        tasks.extend(gpu_tasks[gpu_index:])
    else:
        tasks = cpu_tasks + gpu_tasks

    started = time.time()
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {
        (name, tier): [] for name in names for tier in tiers
    }
    total = len(tasks)
    # Do not start more processes than there are tasks.
    workers = max(1, min(workers, total))
    profile = ProfileAccumulator(
        workers=workers, workers_requested=str(budget["requested"]),
        candidate_chunk=args.candidate_chunk, field_reuse=args.field_reuse)
    tracker = ProgressTracker(tasks)
    ram = budget["ram_available_gb"]
    ram_text = "unknown" if ram is None else f"{ram:.1f}GB"
    print(f"workers {workers} (requested={budget['requested']}, "
          f"cpu={budget['cpu_count']}, ram_available={ram_text}, "
          f"chunks={len(cpu_chunks)}+{len(gpu_chunks)}x"
          f"{args.candidate_chunk}, field_reuse={args.field_reuse})",
          flush=True)
    print(f"mode={args.mode} gpu={args.gpu} enabled={gpu_enabled} "
          f"({gpu_reason}); gpu candidates={len(gpu_specs)}, "
          f"cpu candidates={len(cpu_specs)}", flush=True)
    worker_rss: dict[int, float] = {}
    lock = threading.Lock()
    completed = 0
    sampler = MemorySampler()
    sampler.start()

    def handle(result: dict[str, Any]) -> None:
        nonlocal completed
        with lock:
            completed += 1
            for row in result["results"]:
                rows[(row["candidate"], row["tier"])].append(row)
            profile.add(result)
            pid = int(result.get("worker_pid", 0))
            worker_rss[pid] = max(worker_rss.get(pid, 0.0),
                                  float(result.get("peak_rss_mb", 0.0)))
            tracker.update(result, force=completed == total)

    def run_pool(task_list: Sequence[dict[str, Any]], count: int,
                 function: Any) -> None:
        if not task_list:
            return
        count = max(1, min(int(count), len(task_list)))
        with ProcessPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(function, item) for item in task_list]
            for future in as_completed(futures):
                handle(future.result())

    if gpu_enabled and cpu_tasks and gpu_tasks:
        gpu_workers = max(1, min(args.gpu_workers, len(gpu_tasks)))
        cpu_workers = max(1, workers - gpu_workers)
        print(f"split pools: cpu={cpu_workers} gpu={gpu_workers} "
              f"(CUDA concurrency limited to {gpu_workers})", flush=True)
        with ThreadPoolExecutor(max_workers=2) as both:
            runs = [
                both.submit(run_pool, cpu_tasks, cpu_workers,
                            _eval_field_task),
                both.submit(run_pool, gpu_tasks, gpu_workers,
                            _eval_field_batch_task),
            ]
            for run in runs:
                run.result()
    elif gpu_tasks and not cpu_tasks:
        run_pool(gpu_tasks, min(args.gpu_workers, len(gpu_tasks)),
                 _eval_field_batch_task)
    else:
        run_pool(cpu_tasks or tasks, workers, _eval_field_task)
    wall_seconds = time.time() - started
    memory = sampler.stop()
    worker_rss_total = sum(worker_rss.values())
    rss_mb = max(peak_rss_mb(), worker_rss_total)
    profile_payload = profile.payload(
        wall_seconds=wall_seconds, peak_rss=rss_mb)
    # Completion order is nondeterministic; keep field means in seed order so
    # result files are reproducible across worker counts.
    for values in rows.values():
        values.sort(key=lambda row: int(row["seed"]))

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
        if TEST_TIER in tiers:
            suffix = "test_provisional" if no_consume else "test"
            out = result_path(f"protocol_v4/{PROTOCOL_NAME}_{suffix}.json")
        elif len(tiers) == 1 and tiers[0] in LEVEL_TIERS + COMPLEXITY_TIERS:
            out = result_path(
                f"protocol_v4/{PROTOCOL_NAME}_{tiers[0]}.json")
        else:
            out = result_path(f"protocol_v4/{PROTOCOL_NAME}_dev.json")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "ipd-protocol-eval-v1",
        "protocol": PROTOCOL_NAME,
        "catalog_sha256": catalog_sha256(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(wall_seconds, 2),
        "tiers": tiers,
        "fields": fields,
        "n": args.n,
        "rounds": args.rounds,
        "seed": args.seed,
        "baseline": args.baseline,
        "workers": workers,
        "worker_budget": budget,
        "candidate_chunk": args.candidate_chunk,
        "field_reuse": args.field_reuse,
        "mode": args.mode,
        "opponent_mode": (args.opponent_mode if not gpu_enabled
                          else "argmax (GPU tasks)"),
        "gpu": {
            "enabled": bool(gpu_enabled),
            "requested": args.gpu,
            "reason": gpu_reason,
            "device": args.gpu_device if gpu_enabled else None,
            "dtype": args.gpu_dtype if gpu_enabled else None,
            "max_batch": args.gpu_max_batch if gpu_enabled else None,
            "min_batch": args.gpu_min_batch,
            "workers": (max(1, min(args.gpu_workers, len(gpu_tasks)))
                        if gpu_enabled and gpu_tasks else 0),
            "field_batch": gpu_field_batch if gpu_enabled else 1,
            "candidates": [spec.name for spec in gpu_specs],
            "cpu_candidates": [spec.name for spec in cpu_specs],
        },
        "field_tasks": total,
        "peak_rss_mb": rss_mb,
        "peak_pss_mb": memory["peak_pss_mb"],
        "peak_anon_mb": memory["peak_anon_mb"],
        "memory": memory,
        "workers_rss_mb": round(worker_rss_total, 2),
        "workers_peak_rss_mb": {
            str(pid): round(value, 2)
            for pid, value in sorted(worker_rss.items())
        },
        "profile": profile_payload,
        "usage_consumed": bool(test_run and not no_consume),
        "selection_eligible": not no_consume,
        "no_consume": no_consume,
        "test_consumption_locked": consumption_locked,
        "candidates": [spec.to_dict() for spec in specs],
        "results": aggregated,
    }
    if args.profile:
        profile_out = (Path(args.profile_out) if args.profile_out
                       else out_path.with_name(
                           f"profile_{out_path.stem}.json"))
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        profile_payload = dict(profile_payload)
        profile_payload.update({
            "protocol": PROTOCOL_NAME,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tiers": tiers,
            "fields": fields,
            "n": args.n,
            "seed": args.seed,
            "candidates": names,
            "out": str(out_path),
        })
        profile_out.write_text(
            json.dumps(profile_payload, ensure_ascii=False, indent=2),
            encoding="utf-8")
        payload["profile_path"] = str(profile_out)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\nProtocol v4 evaluation ({fields} fields, n={args.n}, "
          f"workers={workers})")
    for name in names:
        cells = []
        for tier in tiers:
            row = aggregated[name][tier]
            cells.append(f"{tier}={row['mean']:.2f}")
        print(f"  {name:20} " + "  ".join(cells))
    print(f"\nwritten {out_path}")
    if args.profile:
        print(f"profile {payload['profile_path']}")

    if test_run and not no_consume:
        for name in names:
            record_usage(TEST_TIER, name, str(out_path), force=args.force)
        print(f"{TEST_TIER} usage recorded")
    elif test_run:
        reason = "locked by policy" if consumption_locked else "--no-consume"
        print(f"{TEST_TIER} NOT consumed ({reason}); "
              "results are provisional")
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
        description="Evaluate candidates against Protocol v4 red pools.")
    ap.add_argument("--candidates", default="all")
    ap.add_argument("--tier", default="dev_red",
                    choices=("dev_red", "test_red"))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--workers", default="20",
                    help="worker count or 'auto' "
                         "(min(20, cpu_count-2, RAM budget))")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=None)
    ap.add_argument("--acknowledge-once", action="store_true",
                    help="cannot unlock test_red while test consumption is "
                         "locked; ignored for provisional runs")
    ap.add_argument("--no-consume", action="store_true",
                    help="evaluate test_red without writing usage ledger; "
                         "required while test consumption is locked")
    ap.add_argument("--force", action="store_true",
                    help="cannot override the test consumption lock; ignored "
                         "for provisional runs")
    args = ap.parse_args(argv)

    test_run = args.tier == "test_red"
    consumption_locked = test_run and not test_consumption_allowed()
    no_consume = args.no_consume or consumption_locked
    if test_run and consumption_locked and not args.no_consume:
        raise SystemExit(
            "test_red is locked (test consumption disabled); "
            "use --no-consume for a provisional run")
    if test_run and not consumption_locked and not args.acknowledge_once \
            and not args.no_consume:
        raise SystemExit("test_red requires --acknowledge-once or --no-consume")
    try:
        budget = worker_budget(args.workers)
    except ValueError as exc:
        raise SystemExit(str(exc))
    workers = int(budget["resolved"])
    specs = candidate_specs(args.candidates, default="all")
    names = [spec.name for spec in specs]
    if consumption_locked:
        print(f"WARNING: test consumption is locked by {TEST_POLICY_PATH}; "
              "running provisionally (selection_eligible=false)", flush=True)
        if args.acknowledge_once or args.force:
            print("WARNING: --acknowledge-once/--force cannot unlock test "
                  "consumption and are ignored for this run", flush=True)
    elif test_run and args.no_consume:
        print("WARNING: --no-consume enabled; test_red is NOT consumed",
              flush=True)

    if test_run and not no_consume:
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
    workers = max(1, min(workers, len(tasks)))
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    print(f"workers {workers} (requested={budget['requested']})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for result in ex.map(_red_task, tasks, chunksize=1):
            rows[result["candidate"]].append(result)
    wall_seconds = time.time() - started

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
        suffix = "_provisional" if no_consume else ""
        out = result_path(
            f"protocol_v4/protocol_v4_{args.tier}{suffix}.json")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ipd-protocol-red-eval-v1",
        "protocol": PROTOCOL_NAME,
        "catalog_sha256": catalog_sha256(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(wall_seconds, 2),
        "tier": args.tier,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "seed": args.seed,
        "workers": workers,
        "worker_budget": budget,
        "peak_rss_mb": peak_rss_mb(),
        "usage_consumed": bool(test_run and not no_consume),
        "selection_eligible": not no_consume,
        "no_consume": no_consume,
        "test_consumption_locked": consumption_locked,
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
    if test_run and not no_consume:
        for name in names:
            record_usage("test_red", name, str(out_path), force=args.force)
        print("test_red usage recorded")
    elif test_run:
        reason = "locked by policy" if consumption_locked else "--no-consume"
        print(f"test_red NOT consumed ({reason}); results are provisional")
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
