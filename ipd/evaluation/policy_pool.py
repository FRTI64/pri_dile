"""Frozen policy pool for P3 red best-response / league training.

The pool is intentionally format-agnostic: an entry can be a JSON policy path
(RL actor-critic checkpoint, deployable MLP, GRU, self-play GRU) or a
serialisable :class:`~ipd.policies.registry.CandidateSpec` such as Adv-RL,
Final-tuned or Triple-majority.  Entries carry an optional sampling weight, so
a meta-Nash / maximin mixture can be passed directly to the RL learner.

Policies are constructed fresh per episode so their internal RNG/state never
leaks between opponents.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

PolicyEntry = str | Mapping[str, Any]


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


def _entry_weight(entry: PolicyEntry) -> float:
    if isinstance(entry, Mapping):
        return max(0.0, float(entry.get("weight", 1.0)))
    return 1.0


def _entry_name(entry: PolicyEntry) -> str:
    if isinstance(entry, Mapping):
        if entry.get("name"):
            return str(entry["name"])
        spec = entry.get("spec")
        if isinstance(spec, Mapping) and spec.get("name"):
            return str(spec["name"])
        if entry.get("path"):
            return Path(str(entry["path"])).stem
        if entry.get("kind"):
            return str(entry["kind"])
    return Path(str(entry)).stem


def _spec_from_entry(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    spec = entry.get("spec")
    if isinstance(spec, Mapping):
        return dict(spec)
    if entry.get("kind"):
        return {
            "kind": str(entry["kind"]),
            "name": str(entry.get("name") or entry["kind"]),
            "params": dict(entry.get("params") or {}),
        }
    return None


def build_policy_entry(entry: PolicyEntry, *, seed: int = 20261024,
                       mode: str = "sample", name: str | None = None):
    """Build one pool entry (a JSON path or a candidate spec)."""
    if isinstance(entry, str):
        return load_policy(entry, seed=seed, mode=mode,
                           name=name or Path(entry).stem)
    if not isinstance(entry, Mapping):
        raise TypeError(f"unsupported pool entry: {entry!r}")
    if entry.get("path"):
        return load_policy(str(entry["path"]), seed=seed, mode=mode,
                           name=name or _entry_name(entry))
    spec_data = _spec_from_entry(entry)
    if spec_data is None:
        raise ValueError(f"pool entry needs path/spec/kind: {entry!r}")
    from ipd.policies.registry import CandidateSpec, canonical_name

    kind = canonical_name(str(spec_data["kind"]))
    params = dict(spec_data.get("params") or {})
    params["seed"] = int(seed)
    params.setdefault("mode", mode)
    label = name or str(spec_data.get("name") or kind)
    return CandidateSpec(kind=kind, name=label, params=params).build()


def _pool_weights(entries: Sequence[PolicyEntry],
                  weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        raw = [_entry_weight(entry) for entry in entries]
    else:
        if len(weights) != len(entries):
            raise ValueError("weights must match entries in length")
        raw = [max(0.0, float(value)) for value in weights]
    total = sum(raw)
    if total <= 0.0:
        return [1.0] * len(entries)
    return raw


def build_policy_field(entries: Sequence[PolicyEntry], *, seed: int, n: int,
                       mode: str = "sample",
                       weights: Sequence[float] | None = None) -> list:
    """Sample ``n`` independent frozen opponents from weighted entries.

    The RNG sequence is deterministic for a fixed ``seed``/``n``/entry order.
    Zero-weight entries are never sampled; if every weight is zero the sampler
    falls back to a uniform mixture.
    """
    if not entries or n <= 0:
        return []
    entry_list = list(entries)
    mix = _pool_weights(entry_list, weights)
    rng = random.Random(seed)
    indices = rng.choices(range(len(entry_list)), weights=mix, k=int(n))
    field = []
    for i, index in enumerate(indices):
        entry = entry_list[index]
        base = _entry_name(entry)
        field.append(build_policy_entry(
            entry, seed=rng.randrange(1 << 30), mode=mode,
            name=f"{base}:{i}"))
    return field


__all__ = ["PolicyEntry", "load_policy_data", "build_policy", "load_policy",
           "build_policy_entry", "build_policy_field"]
