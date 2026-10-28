"""Batched policy-opponent routing for the GPU simulator.

A protocol field is heterogeneous: most rows are hand-written rules or Axelrod
adapters, a small subset are learned policy opponents.  This module builds one
batched actor per learned ``(type, weights)`` group and keeps the remaining
rows on their original CPU objects.  Routing is by row, so the field order and
the score accounting stay unchanged.

Argmax inference is batched for every learned policy.  Under the opt-in
Protocol v5 sample contract, learned policies are batched too: each row's
uniform stream is drawn from that opponent's own ``random.Random`` state before
the match, and the kernel advances a per-row draw pointer only on rounds where
a hard rule does not decide the action.  This reproduces the lazy CPU draws
exactly.  ``PolicyFinal`` / ``PolicyTriple`` and every other stochastic rule
remain on the CPU path: their Protocol v4 semantics are defined by a sequential
Mersenne Twister stream, which a GPU batch cannot reproduce without a protocol
version bump.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ..fast.bank import RuleBank
from ..fast.spec import get_fast_spec
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
    uniforms: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.rows_t = torch.tensor(self.rows, dtype=torch.long,
                                   device=self.state.device)
        self.policy.reset(len(self.rows))
        self.draw_idx = torch.zeros(len(self.rows), dtype=torch.long,
                                    device=self.state.device)


def _policy_rng(obj: Any) -> random.Random | None:
    """Return the per-opponent stream used by sample-mode policy inference."""
    rng = getattr(unwrap_policy(obj), "rng", None)
    return rng if isinstance(rng, random.Random) else None


def _sample_uniforms(obj: Any, rounds: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor | None:
    """Pre-draw one match worth of uniforms from an opponent's own RNG."""
    rng = _policy_rng(obj)
    if rng is None:
        return None
    values = [rng.random() for _ in range(int(rounds))]
    return torch.tensor(values, dtype=dtype, device=device)


def _reset_opponent(obj: Any) -> None:
    reset = getattr(obj, "reset", None)
    if callable(reset):
        reset()


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


def policy_kind(obj: Any) -> str | None:
    """Public kind lookup shared by the evaluator and the training env."""
    obj = unwrap_policy(obj)
    name = type(obj).__name__
    if name in ("NeuralPolicy", "RLPolicy"):
        return "mlp"
    if name == "GRUPolicy":
        return "gru"
    if name == "SelfPlayGRUPolicy":
        return "selfplay"
    return None


def policy_weights_key(obj: Any, kind: str) -> tuple:
    """Public identity key used to group identical learned policies."""
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


def _policy_kind(obj: Any) -> str | None:
    return policy_kind(obj)


def _family_name(obj: Any) -> str:
    name = str(getattr(obj, "name", type(obj).__name__))
    return name.split(":", 1)[0]


def _weights_key(obj: Any, kind: str) -> tuple:
    return policy_weights_key(obj, kind)


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
                 rounds: int = 100, profile: bool = False,
                 fast_rules: bool | str = "auto",
                 fast_rules_threshold: float = 0.90,
                 policy_mode: str = "argmax") -> None:
        if len(opponents) != len(rows):
            raise ValueError("opponents and rows must have the same length")
        self.device = torch.device(device)
        self.dtype = dtype
        self.rounds = int(rounds)
        self.profile = bool(profile)
        self.fast_rules_mode = fast_rules
        self.fast_rules_threshold = float(fast_rules_threshold)
        self.policy_mode = str(policy_mode)
        self.family_seconds: dict[str, float] = {}
        self._events: list[tuple[str, Any, Any]] = []
        self.n = len(rows)
        self.names = [str(getattr(obj, "name", type(obj).__name__))
                      for obj in opponents]
        self.groups: list[_Group] = []
        self.cpu: list[tuple[int, Any, list[int], list[int]]] = []
        self.rules: RuleBank | None = None
        self.rule_rows: list[int] = []
        self.rule_seconds = 0.0
        self._current_t = 0
        grouped: dict[tuple, list[tuple[int, Any]]] = {}
        rule_members: list[tuple[int, Any]] = []
        for row, obj in zip(rows, opponents):
            kind = _policy_kind(obj)
            if kind is not None:
                if (self.policy_mode == "sample"
                        and _policy_rng(obj) is None):
                    # No captured stream means we cannot reproduce the CPU
                    # draws; keep this row on the exact Python path.
                    self.cpu.append((int(row), obj, [], []))
                else:
                    grouped.setdefault(_weights_key(obj, kind), []).append(
                        (int(row), obj))
                continue
            if (self.fast_rules_mode is not False
                    and get_fast_spec(obj) is not None):
                rule_members.append((int(row), obj))
                continue
            self.cpu.append((int(row), obj, [], []))

        if rule_members:
            # ``auto`` declines a field where the fast rules are a small
            # minority; the Python fallback would dominate and the bank's
            # setup overhead would be a net regression (test_v7 is ~70% rules
            # by row count but only ~4% by cost).
            enable = self.fast_rules_mode is True or self.fast_rules_mode == "on"
            if self.fast_rules_mode == "auto":
                enable = (len(rule_members) / max(1, self.n)
                          >= self.fast_rules_threshold)
            bank = RuleBank([obj for _, obj in rule_members],
                            rounds=self.rounds, enable=enable)
            if bank.enabled:
                self.rules = bank
                self.rule_rows = [row for row, _ in rule_members]
            else:
                for row, obj in rule_members:
                    self.cpu.append((row, obj, [], []))

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
            uniforms = None
            if self.policy_mode == "sample":
                # Match ``run_match``: reset the opponent once, then capture
                # the stream it would consume lazily during this match.
                rows_uniforms = []
                for _, obj in members:
                    _reset_opponent(obj)
                    row_uniforms = _sample_uniforms(
                        obj, self.rounds, self.device, dtype)
                    if row_uniforms is None:  # pragma: no cover - checked above
                        raise TypeError(
                            "sample-mode policy opponent has no RNG stream")
                    rows_uniforms.append(row_uniforms)
                uniforms = torch.stack(rows_uniforms)
            group = _Group(
                rows=group_rows, kind=kind,
                policy=build_batch_policy(template, self.device, dtype),
                state=state, safe=safe, temperature=temperature,
                uniforms=uniforms)
            self.groups.append(group)

        for _, obj, _, _ in self.cpu:
            _reset_opponent(obj)
            setter = getattr(obj, "set_rounds", None)
            if callable(setter):
                setter(self.rounds)

    def reset(self) -> None:
        self._current_t = 0
        for group in self.groups:
            group.state.reset()
            group.policy.reset(len(group.rows))
            group.draw_idx.zero_()
        if self.rules is not None:
            self.rules.reset()
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
        self._current_t = int(t)
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
            if group.uniforms is not None:
                action, group.draw_idx = group.policy.act_from_logits_sample(
                    logits, t, no_defect, group.uniforms, group.draw_idx)
            else:
                action = group.policy.act_from_logits(logits, t, no_defect)
            for row, value in zip(group.rows,
                                  action.detach().cpu().tolist()):
                actions[row] = int(value)
            if self.profile and self.device.type == "cuda":
                end.record()
                self._events.append(("policy", start, end))

        if self.rules is not None:
            started = (time.perf_counter() if self.profile else 0.0)
            blue = blue_actions.detach().cpu().tolist()
            rule_actions = self.rules.act(t, blue)
            if self.profile:
                self.rule_seconds += time.perf_counter() - started
            for row, value in zip(self.rule_rows, rule_actions):
                actions[row] = int(value)

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
        if self.rule_seconds:
            self.family_seconds["rulebank"] = self.rule_seconds
        return dict(self.family_seconds)

    def observe(self, blue_actions: torch.Tensor,
                opponent_actions: torch.Tensor) -> None:
        for group in self.groups:
            group.state.append(opponent_actions[group.rows_t],
                               blue_actions[group.rows_t])
        if self.rules is not None:
            b = blue_actions.detach().cpu().tolist()
            o = opponent_actions.detach().cpu().tolist()
            # RuleBank rows are local (0..len(rule_rows)-1), while the field
            # rows can have gaps for policies/Axelrod fallbacks.
            self.rules.observe(
                self._current_t,
                [b[row] for row in self.rule_rows],
                [o[row] for row in self.rule_rows])
        if self.cpu:
            b = blue_actions.detach().cpu().tolist()
            o = opponent_actions.detach().cpu().tolist()
            for row, _, own, blue in self.cpu:
                own.append(int(o[row]))
                blue.append(int(b[row]))


__all__ = ["BatchOpponentField", "build_batch_policy", "policy_kind",
           "policy_weights_key", "unwrap_policy"]
