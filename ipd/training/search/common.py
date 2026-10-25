"""Shared utilities for the P0/P1/P2 parameter searches.

Policy classes and field builders live in their proper layers; this module
keeps the historical search API as thin adapters.
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import ipd.core.opponents_extended as ext  # noqa: F401  (legacy re-export)
from ipd.core.early import EarlyProbeMachine
from ipd.evaluation.common import evaluate_field as _evaluate_field
from ipd.evaluation.distributions import (
    HIGH_GRIM_BASE, N_PLAYERS, ORIGINAL_SCENARIO_WEIGHTS, build_field,
)
from ipd.policies.rules import (
    FinalPolicy, HarvestPolicy, ParamHarvest, ParamPolicy,
)

__all__ = [
    "N_PLAYERS", "HIGH_GRIM_BASE", "ORIGINAL_SCENARIO_WEIGHTS",
    "FinalPolicy", "HarvestPolicy", "ParamPolicy", "ParamHarvest",
    "build_field", "make_policy", "evaluate_field", "FieldSet",
    "make_fieldsets", "evaluate_parallel", "evaluate_serial", "mean_per_opp",
    "ext", "main",
]


def make_policy(spec: dict, seed: int = 20261024):
    kind = spec.get("kind", "final")
    params = {k: v for k, v in spec.items() if k != "kind"}
    if kind == "final":
        return FinalPolicy(seed=seed)
    if kind == "harvest":
        return HarvestPolicy(seed=seed, **params)
    if kind == "early":
        return EarlyProbeMachine(seed=seed)
    if kind == "param":
        return ParamPolicy(seed=seed, **params)
    if kind == "param_harvest":
        return ParamHarvest(seed=seed, **params)
    raise ValueError(kind)


def evaluate_field(spec: dict, kind: str, seed: int,
                   n: int = N_PLAYERS) -> int:
    field, _ = build_field(kind, seed, n)
    return _evaluate_field(make_policy(spec), field).total


def _worker(args):
    spec, kind, seed, n = args
    return evaluate_field(spec, kind, seed, n)


@dataclass
class FieldSet:
    specs: list[tuple[str, int]]
    label: str = ""

    @property
    def size(self):
        return len(self.specs)


def make_fieldsets(seed=2026, n_orig=24, n_late=24):
    rng = random.Random(seed)
    orig = [("original", rng.randrange(1 << 30)) for _ in range(n_orig)]
    late = [("late", rng.randrange(1 << 30)) for _ in range(n_late)]
    val_rng = random.Random(seed + 10000)
    orig_val = [("original", val_rng.randrange(1 << 30))
                for _ in range(n_orig)]
    late_val = [("late", val_rng.randrange(1 << 30)) for _ in range(n_late)]
    return {
        "train_original": FieldSet(orig, "train_original"),
        "train_late": FieldSet(late, "train_late"),
        "val_original": FieldSet(orig_val, "val_original"),
        "val_late": FieldSet(late_val, "val_late"),
    }


def evaluate_parallel(spec, fields: FieldSet, workers: int = 12,
                      n: int = N_PLAYERS) -> list[int]:
    args = [(spec, kind, seed, n) for kind, seed in fields.specs]
    if workers <= 1:
        return [_worker(a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, args, chunksize=1))


def evaluate_serial(spec, fields: FieldSet) -> list[int]:
    return [_worker((spec, kind, seed, N_PLAYERS))
            for kind, seed in fields.specs]


def mean_per_opp(field_scores, n=N_PLAYERS):
    return sum(field_scores) / len(field_scores) / n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--fields", type=int, default=40)
    args = ap.parse_args(argv)
    fs = FieldSet([("original", i * 7919 + 13) for i in range(args.fields)])
    spec = {"kind": "final"}

    t = time.perf_counter()
    serial = evaluate_serial(spec, fs)
    t_serial = time.perf_counter() - t
    t = time.perf_counter()
    parallel = evaluate_parallel(spec, fs, args.workers)
    t_par = time.perf_counter() - t
    print(f"fields={len(serial)} serial={t_serial:.2f}s "
          f"parallel({args.workers})={t_par:.2f}s "
          f"speedup={t_serial / t_par:.2f}x")
    print(f"mean/opp serial={mean_per_opp(serial):.2f} "
          f"parallel={mean_per_opp(parallel):.2f}")
    return 0


if __name__ == "__main__":
    main()
