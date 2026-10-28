"""Batched policy-opponent routing for the GPU simulator.

A protocol field is heterogeneous: most rows are hand-written rules or Axelrod
adapters, a small subset are learned policy opponents.  This module builds one
batched actor per learned ``(type, weights)`` group and keeps the remaining
rows on their original CPU objects.  Routing is by row, so the field order and
the score accounting stay unchanged.

Only argmax inference is batched.  ``PolicyFinal`` / ``PolicyTriple`` and every
other stochastic rule remain on the CPU path: their Protocol v4 semantics are
defined by a sequential Mersenne Twister stream, which a GPU batch cannot
reproduce without a protocol version bump.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .features import FeatureState21, FeatureState85
from .policies import BatchGRU, BatchMLP, BatchSelfPlayGRU


@dataclass
class _Group:
    rows: list[int]
    kind: str
    policy: Any
    state: Any
    safe: bool = True
    temperature: float = 1.0

    def __post_init__(self) -> None:
        self.rows_t = torch.tensor(self.rows, dtype=torch.long,
                                   device=self.state.device)
        self.policy.reset(len(self.rows))


def unwrap_policy(obj: Any) -> Any:
    """Return the concrete policy behind an optional PolicyAdapter."""
    seen = set()
    while hasattr(obj, "policy") and id(obj) not in seen:
        seen.add(id(obj))
        inner = getattr(obj, "policy")
        if inner is obj or not hasattr(inner, "act"):
            break
        obj = inner
    return obj


def _policy_kind(obj: Any) -> str | None:
    obj = unwrap_policy(obj)
    name = type(obj).__name__
    if name in ("NeuralPolicy", "RLPolicy"):
        return "mlp"
    if name == "GRUPolicy":
        return "gru"
    if name == "SelfPlayGRUPolicy":
        return "selfplay"
    return None


def _family_name(obj: Any) -> str:
    name = str(getattr(obj, "name", type(obj).__name__))
    return name.split(":", 1)[0]


def _weights_key(obj: Any, kind: str) -> tuple:
    obj = unwrap_policy(obj)
    path = getattr(obj, "_gpu_weights_path", None)
    weights = getattr(obj, "weights", None)
    if path is not None:
        identity: Any = str(path)
    elif isinstance(weights, dict):
        identity = id(weights)
    else:
        identity = id(obj)
    safe = bool(getattr(obj, "safe", True))
    temperature = float(getattr(obj, "temperature", 1.0))
    return (kind, identity, safe, temperature)


def build_batch_policy(obj: Any, device: torch.device,
                       dtype: torch.dtype) -> Any:
    obj = unwrap_policy(obj)
    kind = _policy_kind(obj)
    weights = getattr(obj, "weights", None)
    if kind == "mlp":
        return BatchMLP(weights, device=device, dtype=dtype,
                        safe=bool(getattr(obj, "safe", True)))
    if kind == "gru":
        return BatchGRU(weights, device=device, dtype=dtype)
    if kind == "selfplay":
        return BatchSelfPlayGRU(weights, device=device, dtype=dtype)
    raise TypeError(f"not a batched learned policy: {type(obj).__name__}")


class BatchOpponentField:
    """Per-row opponent actor for one field chunk.

    ``rows`` are global row indices; ``opponents`` are the objects for those
    rows in the same order.  CPU opponents are reset once at construction, as
    ``run_match`` does before each match.
    """

    def __init__(self, opponents: Sequence[Any], rows: Sequence[int],
                 *, device: torch.device, dtype: torch.dtype,
                 rounds: int = 100, profile: bool = False) -> None:
        if len(opponents) != len(rows):
            raise ValueError("opponents and rows must have the same length")
        self.device = torch.device(device)
        self.dtype = dtype
        self.rounds = int(rounds)
        self.profile = bool(profile)
        self.family_seconds: dict[str, float] = {}
        self._events: list[tuple[str, Any, Any]] = []
        self.n = len(rows)
        self.names = [str(getattr(obj, "name", type(obj).__name__))
                      for obj in opponents]
        self.groups: list[_Group] = []
        self.cpu: list[tuple[int, Any, list[int], list[int]]] = []
        grouped: dict[tuple, list[tuple[int, Any]]] = {}
        for row, obj in zip(rows, opponents):
            kind = _policy_kind(obj)
            if kind is None:
                self.cpu.append((int(row), obj, [], []))
                continue
            grouped.setdefault(_weights_key(obj, kind), []).append(
                (int(row), obj))

        for (kind, _identity, safe, temperature), members in grouped.items():
            group_rows = [row for row, _ in members]
            template = members[0][1]
            state_cls = FeatureState21 if kind == "selfplay" else FeatureState85
            kwargs = {}
            if kind == "selfplay":
                kwargs["version"] = getattr(
                    template, "observation_version", "selfplay-21-v1")
            state = state_cls(len(group_rows), device=self.device,
                              dtype=dtype, rounds=self.rounds, **kwargs)
            group = _Group(
                rows=group_rows, kind=kind,
                policy=build_batch_policy(template, self.device, dtype),
                state=state, safe=safe, temperature=temperature)
            self.groups.append(group)

        for _, obj, _, _ in self.cpu:
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()
            setter = getattr(obj, "set_rounds", None)
            if callable(setter):
                setter(self.rounds)

    def reset(self) -> None:
        for group in self.groups:
            group.state.reset()
            group.policy.reset(len(group.rows))
        for _, obj, own, blue in self.cpu:
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()
            own.clear()
            blue.clear()

    @torch.no_grad()
    def act(self, t: int, blue_actions: torch.Tensor
            ) -> torch.Tensor:
        """Return opponent actions for this chunk at round ``t``."""
        # Fill a CPU list first: assigning individual elements of a CUDA
        # tensor in the rule-heavy loop costs one synchronising kernel per
        # opponent per round.
        actions = [0] * self.n
        for group in self.groups:
            if self.profile and self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            features = group.state.features()
            if group.kind in ("gru", "selfplay"):
                logits = group.policy.step(features)
            else:
                logits = group.policy.logits(features)
            no_defect = ~group.state.any_defect("theirs")
            action = group.policy.act_from_logits(logits, t, no_defect)
            for row, value in zip(group.rows,
                                  action.detach().cpu().tolist()):
                actions[row] = int(value)
            if self.profile and self.device.type == "cuda":
                end.record()
                self._events.append(("policy", start, end))

        if self.cpu:
            for row, obj, own, blue in self.cpu:
                started = (time.perf_counter() if self.profile else 0.0)
                y = obj.act(t, own, blue)
                if self.profile:
                    family = _family_name(obj)
                    self.family_seconds[family] = (
                        self.family_seconds.get(family, 0.0)
                        + time.perf_counter() - started)
                if y not in (0, 1):
                    raise ValueError(
                        f"{getattr(obj, 'name', type(obj).__name__)} "
                        f"returned invalid action {y!r}")
                actions[row] = int(y)
        return torch.tensor(actions, dtype=torch.int8, device=self.device)

    def finish_profile(self) -> dict[str, float]:
        """Resolve recorded CUDA events and return per-family seconds."""
        if self.profile and self._events:
            torch.cuda.synchronize(self.device)
            for family, start, end in self._events:
                self.family_seconds[family] = (
                    self.family_seconds.get(family, 0.0)
                    + start.elapsed_time(end) / 1000.0)
            self._events.clear()
        return dict(self.family_seconds)

    def observe(self, blue_actions: torch.Tensor,
                opponent_actions: torch.Tensor) -> None:
        for group in self.groups:
            group.state.append(opponent_actions[group.rows_t],
                               blue_actions[group.rows_t])
        if self.cpu:
            b = blue_actions.detach().cpu().tolist()
            o = opponent_actions.detach().cpu().tolist()
            for row, _, own, blue in self.cpu:
                own.append(int(o[row]))
                blue.append(int(b[row]))


__all__ = ["BatchOpponentField", "build_batch_policy"]
