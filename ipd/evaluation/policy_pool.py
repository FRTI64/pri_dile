"""Frozen policy pool for P3 red best-response training.

The pool is intentionally format-agnostic: it can sample RL actor-critic
checkpoints, deployable MLPs, GRU policies and self-play GRUs.  Policies are
constructed fresh per episode so their internal RNG/state never leaks between
opponents.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Sequence


@lru_cache(maxsize=128)
def load_policy_data(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    fmt = data.get("format")
    if fmt not in ("ipd-rl-policy-v1", "ipd-neural-policy-v1",
                   "ipd-gru-policy-v1", "ipd-selfplay-gru-v1"):
        raise ValueError(f"unsupported policy format {fmt!r} in {path}")
    return data


def build_policy(data: dict, *, seed: int = 20261024,
                 mode: str = "sample", name: str | None = None):
    fmt = data.get("format")
    if fmt == "ipd-rl-policy-v1":
        from ipd.policies.rl import RLPolicy

        return RLPolicy(data, seed=seed, mode=mode, name=name)
    if fmt == "ipd-neural-policy-v1":
        from ipd.policies.neural import NeuralPolicy

        return NeuralPolicy(data, seed=seed, mode=mode, name=name)
    if fmt == "ipd-gru-policy-v1":
        from ipd.policies.gru import GRUPolicy

        return GRUPolicy(data, seed=seed, mode=mode, name=name)
    if fmt == "ipd-selfplay-gru-v1":
        from ipd.policies.selfplay_gru import SelfPlayGRUPolicy

        return SelfPlayGRUPolicy(data, seed=seed, mode=mode, name=name)
    raise ValueError(f"unsupported policy format {fmt!r}")


def load_policy(path: str, *, seed: int = 20261024, mode: str = "sample",
                name: str | None = None):
    return build_policy(load_policy_data(path), seed=seed, mode=mode,
                        name=name)


def build_policy_field(paths: Sequence[str], *, seed: int, n: int,
                       mode: str = "sample") -> list:
    """Sample ``n`` independent frozen opponents from ``paths``."""
    if not paths or n <= 0:
        return []
    rng = random.Random(seed)
    data = {path: load_policy_data(path) for path in set(paths)}
    field = []
    for i in range(int(n)):
        path = rng.choice(list(paths))
        label = f"blue:{Path(path).stem}:{i}"
        policy = build_policy(data[path], seed=rng.randrange(1 << 30),
                              mode=mode, name=label)
        field.append(policy)
    return field


__all__ = ["load_policy_data", "build_policy", "load_policy",
           "build_policy_field"]
