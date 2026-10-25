#!/usr/bin/env python3
"""Shared simulator utilities for P0/P1/P2.

P0: deterministic field generation + multiprocessing evaluator.
P1/P2: parameterised policies and fast tournament evaluation.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ipd_sim import C, D, payoff
from ipd_strategy import Machine
from ipd_harvest import HarvestMachine
from ipd_early import EarlyProbeMachine
import ipd_extended as ext

N_PLAYERS = 299
# Base weights for a fixed high-Grim population. Late harvesters remain a
# significant block, so raising Grim tests the safety/exploitation tradeoff.
HIGH_GRIM_BASE = {
    "AllC": 8, "TFT": 18, "TFT2": 6, "GTFT.05": 8, "GTFT.10": 5,
    "Pavlov": 5, "AllD": 5, "Random.50": 5, "Tester": 2, "Prober": 2,
    "LateD5": 3, "LateD10": 4, "LateD15": 3, "LateD20": 3,
    "LateD25": 3, "LateD30": 3, "LateD40": 3, "LateD50": 3,
    "Mistrust": 3, "Grudger2": 2, "Grudger3": 1,
}
ORIGINAL_SCENARIO_WEIGHTS = {
    "balanced": 22, "cooperative": 8, "defector": 8, "naive": 8,
    "sophisticated": 22, "noisy": 12, "punitive": 10, "wild": 10,
}


class FinalPolicy(Machine):
    name = "Final"
    def reset(self): super().reset()
    def act(self, t, mine, theirs): return self.move(t + 1, mine, theirs)


class HarvestPolicy(HarvestMachine):
    name = "Harvest"
    def __init__(self, seed=20261024, **params):
        for k, v in params.items():
            setattr(self, k, v)
        super().__init__(seed)
    def reset(self): super().reset()
    def act(self, t, mine, theirs): return self.move(t + 1, mine, theirs)


class ParamPolicy(Machine):
    name = "Param"
    def __init__(self, seed=20261024, **params):
        for k, v in params.items():
            setattr(self, k, v)
        super().__init__(seed)
    def reset(self): super().reset()
    def act(self, t, mine, theirs): return self.move(t + 1, mine, theirs)


class ParamHarvest(HarvestMachine):
    name = "ParamHarvest"
    def __init__(self, seed=20261024, **params):
        for k, v in params.items():
            setattr(self, k, v)
        super().__init__(seed)
    def reset(self): super().reset()
    def act(self, t, mine, theirs): return self.move(t + 1, mine, theirs)


def make_policy(spec: dict[str, Any], seed: int = 20261024):
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


def build_field(kind: str, seed: int, n: int = N_PLAYERS):
    rng = random.Random(seed)
    if kind == "original":
        scen = rng.choices(list(ORIGINAL_SCENARIO_WEIGHTS),
                           weights=list(ORIGINAL_SCENARIO_WEIGHTS.values()),
                           k=1)[0]
        specs, probs = ext.sample_mixture(rng, scen)
        pop = []
        for _ in range(n):
            s = rng.choices(specs, weights=probs, k=1)[0]
            o = s.factory(rng); o.name = s.name; pop.append(o)
        return pop, scen
    if kind == "late":
        return ext.sample_population(n, rng, "late_harvest"), "late_harvest"
    if kind == "mixed":
        return ext.sample_mixed_population(n, rng)
    if kind.startswith("grim"):
        grim = int(kind[4:])
        weights = dict(HIGH_GRIM_BASE)
        weights["Grim"] = grim
        specs = [ext.TYPE_BY_NAME[name] for name in weights]
        weights_list = list(weights.values())
        pop = []
        for _ in range(n):
            spec = rng.choices(specs, weights=weights_list, k=1)[0]
            obj = spec.factory(rng)
            obj.name = spec.name
            pop.append(obj)
        return pop, kind
    raise ValueError(kind)


def evaluate_field(spec: dict[str, Any], kind: str, seed: int,
                   n: int = N_PLAYERS) -> int:
    field, _ = build_field(kind, seed, n)
    me = make_policy(spec)
    total = 0
    for opp in field:
        me.reset()
        opp.reset()
        mh: list[int] = []
        oh: list[int] = []
        s = 0
        for t in range(100):
            x = me.act(t, mh, oh)
            y = opp.act(t, oh, mh)
            mh.append(x)
            oh.append(y)
            s += payoff(x, y)
        total += s
    return total


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
    orig_val = [("original", val_rng.randrange(1 << 30)) for _ in range(n_orig)]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--fields", type=int, default=40)
    args = ap.parse_args()
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
          f"speedup={t_serial/t_par:.2f}x")
    print(f"mean/opp serial={mean_per_opp(serial):.2f} "
          f"parallel={mean_per_opp(parallel):.2f}")


if __name__ == "__main__":
    main()
