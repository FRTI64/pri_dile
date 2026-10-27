"""Deployable PSRO meta-policy.

A PSRO result is a population plus a blue meta-strategy.  This module turns
that result into a normal tournament policy: at ``reset()`` it samples one
pure policy from the blue meta-strategy and delegates the whole match to it.
Sampling once per match (rather than per day) is the standard interpretation
of a mixed strategy in a repeated game and keeps the member policy's state
consistent.

Only the standard library is used at inference time.  Candidate specs are
built lazily through :mod:`ipd.policies.registry`; JSON policies use the
existing RL / MLP / GRU / SelfPlay inference implementations.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..paths import weights_path
from .base import ensure_policy
from .gru import GRUPolicy
from .neural import NeuralPolicy
from .rl import RLPolicy
from .selfplay_gru import SelfPlayGRUPolicy

DEFAULT_META = "psro/meta_strategy.json"


def _load_policy(path: str, *, seed: int, mode: str, name: str):
    with open(path, encoding="utf-8") as fh:
        fmt = json.load(fh).get("format")
    if fmt == "ipd-rl-policy-v1":
        policy = RLPolicy(path, seed=seed, mode=mode, name=name)
    elif fmt == "ipd-neural-policy-v1":
        policy = NeuralPolicy(path, seed=seed, mode=mode, name=name)
    elif fmt == "ipd-gru-policy-v1":
        policy = GRUPolicy(path, seed=seed, mode=mode, name=name)
    elif fmt == "ipd-selfplay-gru-v1":
        policy = SelfPlayGRUPolicy(path, seed=seed, mode=mode, name=name)
    else:
        raise ValueError(f"unsupported PSRO member format {fmt!r} in {path}")
    return ensure_policy(policy, name=name)


def build_member(entry: Mapping[str, Any], *, seed: int, mode: str,
                 name: str | None = None):
    """Build one population member from a policy-pool entry."""
    label = name or str(entry.get("name") or entry.get("kind") or "member")
    if entry.get("path"):
        return _load_policy(str(entry["path"]), seed=seed, mode=mode,
                            name=label)
    if entry.get("spec"):
        from .registry import CandidateSpec

        return CandidateSpec.from_dict(entry["spec"]).build()
    kind = str(entry.get("kind") or "")
    if not kind:
        raise ValueError(f"PSRO member {label!r} has no path/spec/kind")
    if kind == "psro" or kind == "psro-meta":
        raise ValueError("nested PSRO meta-policies are not supported")
    from .registry import make_policy

    params = dict(entry.get("params") or {})
    params.setdefault("seed", seed)
    params.setdefault("mode", mode)
    return make_policy(kind, name=label, **params)


def load_meta_strategy(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path or weights_path(DEFAULT_META))
    with source.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, Mapping):
        raise ValueError(f"invalid PSRO meta-strategy: {source}")
    population = data.get("population")
    if not isinstance(population, list) or not population:
        raise ValueError(f"PSRO meta-strategy has no population: {source}")
    return dict(data)


class PSROMetaPolicy:
    """Sample one blue policy from ``blue_meta.weights_by_name`` per match."""

    name = "PSRO-Meta"

    def __init__(self, meta_path: str | Path | None = None, *,
                 seed: int = 20261024, mode: str = "sample",
                 name: str | None = None, weights: str | None = None):
        self.meta_path = str(meta_path or weights or weights_path(DEFAULT_META))
        self.data = load_meta_strategy(self.meta_path)
        population = [row for row in self.data["population"]
                      if str(row.get("role") or "blue") == "blue"]
        if not population:
            population = list(self.data["population"])
        self.population = population
        meta = (self.data.get("blue_meta") or self.data.get("meta") or {})
        weight_map = meta.get("weights_by_name") or {}
        raw = [max(0.0, float(weight_map.get(str(row.get("name")), 0.0)))
               for row in population]
        if sum(raw) <= 0.0:
            raw = [1.0] * len(population)
        total = sum(raw)
        self.weights = [value / total for value in raw]
        self.rng = random.Random(int(seed))
        self.mode = mode
        self._seed = int(seed)
        self._current = None
        self._current_name: str | None = None
        self._members: dict[str, Any] = {}
        if name:
            self.name = name

    @property
    def current_name(self) -> str | None:
        return self._current_name

    def reset(self) -> None:
        index = self.rng.choices(
            range(len(self.population)), weights=self.weights, k=1)[0]
        row = self.population[index]
        label = str(row.get("name") or f"member-{index}")
        child_seed = self.rng.randrange(1 << 30)
        member = self._members.get(label)
        if member is None:
            member = build_member(
                row.get("entry") or row, seed=child_seed, mode=self.mode,
                name=label)
            self._members[label] = member
        self._current = member
        self._current_name = label
        self._current.reset()

    def _ensure(self) -> None:
        if self._current is None:
            self.reset()

    def act(self, t: int, mine: Sequence[int],
            theirs: Sequence[int]) -> int:
        self._ensure()
        return int(self._current.act(t, mine, theirs))

    def move(self, day: int, mine: Sequence[int],
             theirs: Sequence[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def observe(self, mine: Sequence[int], theirs: Sequence[int]) -> None:
        self._ensure()
        observe = getattr(self._current, "observe", None)
        if observe is not None:
            observe(mine, theirs)

    def set_rounds(self, rounds: int) -> None:
        if self._current is None:
            return
        setter = getattr(self._current, "set_rounds", None)
        if setter is not None:
            setter(rounds)

    def __repr__(self) -> str:
        return self.name


__all__ = [
    "DEFAULT_META", "PSROMetaPolicy", "build_member", "load_meta_strategy",
]
