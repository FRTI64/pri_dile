"""Batched blue-policy adapters for the GPU simulator.

Supported candidates:

* plain ``NeuralPolicy`` / ``RLPolicy`` / ``GRUPolicy`` / ``SelfPlayGRUPolicy``
* the constrained ``Adv-MLP`` / ``Adv-GRU`` / ``Adv-RL`` wrappers around one of
  those plain policies
* ``PSROMetaPolicy`` whose blue population contains only plain learned
  policies (``psro-scratch``); the mixed rule/learned PSRO meta-policy stays on
  the CPU path

Everything else (Final/Triple/rule candidates) is CPU-only.  Plain learned
policies and ``psro-scratch`` support both ``argmax`` and the opt-in v5
per-match ``sample`` contract.  ``Adv-*`` wrappers are argmax-only on GPU;
v5 sample keeps them on the CPU so their sequential v4 RNG stream, and the
candidate score, stays exactly the same as the pure-Python baseline.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import torch

from .features import FeatureState21, FeatureState85
from .policies import BatchGRU, BatchMLP, BatchSelfPlayGRU
from .policy_opponents import _policy_kind, build_batch_policy, unwrap_policy

_PLAIN_KINDS = frozenset({"neural", "gru", "rl", "rl-hard60", "rl-repair",
                          "rl-pattern", "rl-wide", "rl-wide160",
                          "psro-br", "psro-scratch-br", "selfplay"})
_ADV_KINDS = frozenset({"adv-mlp", "adv-gru", "adv-rl"})


def _psro_member_supported(entry: dict[str, Any]) -> bool:
    path = entry.get("path")
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                fmt = json.load(fh).get("format")
        except (OSError, ValueError):
            return False
        return fmt in ("ipd-neural-policy-v1", "ipd-rl-policy-v1",
                       "ipd-gru-policy-v1", "ipd-selfplay-gru-v1")
    spec = entry.get("spec") if isinstance(entry.get("spec"), dict) else entry
    kind = str(spec.get("kind") or "")
    return kind in _PLAIN_KINDS


def supports_gpu_policy(policy: Any) -> bool:
    """Return whether a candidate can be evaluated by the batched path."""
    policy = unwrap_policy(policy)
    if _policy_kind(policy) is not None:
        return True
    name = type(policy).__name__
    if name in ("AdvMLPPolicy", "AdvGRUPolicy", "AdvRLPolicy"):
        return _policy_kind(getattr(policy, "base", None)) is not None
    if name == "PSROMetaPolicy":
        population = list(getattr(policy, "population", []))
        return bool(population) and all(
            _psro_member_supported(row.get("entry") or row)
            for row in population)
    return False


class _PlainActor:
    def __init__(self, policy: Any, rows: Sequence[int],
                 device: torch.device, dtype: torch.dtype,
                 rounds: int, mode: str = "argmax",
                 uniforms: Any = None) -> None:
        kind = _policy_kind(policy)
        if kind is None:
            raise TypeError(f"unsupported blue policy: {type(policy).__name__}")
        self.kind = kind
        self.rows = list(rows)
        self.rows_t = torch.tensor(self.rows, dtype=torch.long, device=device)
        self.mode = str(mode)
        self.uniforms = (uniforms.to(device)
                         if uniforms is not None else None)
        self.draw_idx = torch.zeros(len(self.rows), dtype=torch.long,
                                    device=device)
        state_cls = FeatureState21 if kind == "selfplay" else FeatureState85
        kwargs = {}
        if kind == "selfplay":
            kwargs["version"] = getattr(
                policy, "observation_version", "selfplay-21-v1")
        self.state = state_cls(len(self.rows), device=device, dtype=dtype,
                               rounds=rounds, **kwargs)
        self.policy = build_batch_policy(policy, device, dtype)
        self.policy.reset(len(self.rows))

    def act(self, t: int) -> torch.Tensor:
        features = self.state.features()
        if self.kind in ("gru", "selfplay"):
            logits = self.policy.step(features)
        else:
            logits = self.policy.logits(features)
        no_defect = ~self.state.any_defect("theirs")
        if self.mode == "argmax":
            return self.policy.act_from_logits(logits, t, no_defect)
        if self.uniforms is None:
            raise ValueError("sample mode requires per-match uniforms")
        action, self.draw_idx = self.policy.act_from_logits_sample(
            logits, t, no_defect, self.uniforms, self.draw_idx)
        return action

    def observe(self, mine: torch.Tensor, theirs: torch.Tensor) -> None:
        self.state.append(mine[self.rows_t], theirs[self.rows_t])


class _AdvActor:
    """Vectorised form of the shared Adv-MLP/GRU/RL wrapper rules."""

    def __init__(self, policy: Any, rows: Sequence[int],
                 device: torch.device, dtype: torch.dtype,
                 rounds: int, mode: str = "argmax",
                 uniforms: Any = None) -> None:
        self.base = _PlainActor(policy.base, rows, device, dtype, rounds,
                                mode=mode, uniforms=uniforms)
        self.rows = list(rows)
        self.rows_t = torch.tensor(self.rows, dtype=torch.long, device=device)
        self.device = device
        self.gap = None if policy.gap is None else float(policy.gap)
        self.min_c = int(policy.min_c)
        self.min_d = int(policy.min_d)
        self.late_day = (None if policy.late_day is None
                         else int(policy.late_day))
        self.repair = bool(policy.repair)
        self.punish = torch.zeros(len(self.rows), dtype=torch.bool,
                                  device=device)
        self.first_d = torch.full((len(self.rows),), -1, dtype=torch.long,
                                  device=device)
        self.rounds = int(rounds)

    def act(self, t: int) -> torch.Tensor:
        base_action = self.base.act(t)
        state = self.base.state
        theirs = state.theirs
        b = len(self.rows)
        dev = self.device
        punish = self.punish
        final = torch.full((b,), t >= 99, dtype=torch.bool, device=dev)
        update_first = (~punish) & (~final)
        if t > 0:
            is_d = theirs[:, t - 1] == 1
            self.first_d = torch.where(
                update_first & (self.first_d < 0) & is_d,
                torch.full_like(self.first_d, t - 1), self.first_d)

        late = torch.zeros(b, dtype=torch.bool, device=dev)
        if self.late_day is not None:
            first = self.first_d
            valid = (first >= 0) & (t >= first + 2)
            idx = (first + 1).clamp(0, self.rounds - 1)
            followed = theirs.gather(1, idx.unsqueeze(1)).squeeze(1) == 1
            late = valid & (first >= self.late_day) & followed
            late &= ~punish

        gap_hit = torch.zeros(b, dtype=torch.bool, device=dev)
        if self.gap is not None:
            n_c = state.n_c.to(torch.float64)
            n_d = state.n_d.to(torch.float64)
            d_c = state.d_c.to(torch.float64)
            d_d = state.d_d.to(torch.float64)
            enough = (n_c >= self.min_c) & (n_d >= self.min_d)
            p_c = d_c / n_c.clamp_min(1.0)
            p_d = d_d / n_d.clamp_min(1.0)
            gap_hit = enough & ((p_d - p_c) <= self.gap) & (~punish)

        start = (~punish) & (late | gap_hit) & (t < 99)
        repair_exit = (punish & self.repair & (t < 99) & (t >= 2)
                       & (theirs[:, t - 1] == 0)
                       & (theirs[:, t - 2] == 0))

        action = base_action
        action = torch.where(punish & (~repair_exit),
                             torch.ones(b, dtype=torch.int8, device=dev),
                             action)
        action = torch.where(start,
                             torch.ones(b, dtype=torch.int8, device=dev),
                             action)
        action = torch.where(final,
                             torch.ones(b, dtype=torch.int8, device=dev),
                             action)
        self.punish = (punish & ~repair_exit) | start
        return action

    def observe(self, mine: torch.Tensor, theirs: torch.Tensor) -> None:
        self.base.observe(mine, theirs)


class _PSROActor:
    def __init__(self, policy: Any, rows: Sequence[int],
                 device: torch.device, dtype: torch.dtype,
                 rounds: int, mode: str = "argmax",
                 uniforms: Any = None) -> None:
        if mode != "argmax":
            raise TypeError("PSRO meta-policy sample mode is not batched")
        self.rows = list(rows)
        # Reproduce the per-match member draw in field order.  Argmax members
        # have no stochastic state, so only the selected weights matter.
        by_label: dict[str, list[int]] = {}
        members: dict[str, Any] = {}
        for local, _row in enumerate(self.rows):
            policy.reset()
            member = getattr(policy, "_current", None)
            label = str(getattr(policy, "_current_name", ""))
            if member is None or _policy_kind(member) is None:
                raise TypeError("PSRO member is not a plain learned policy")
            by_label.setdefault(label, []).append(local)
            members.setdefault(label, member)
        self.groups = [
            _PlainActor(members[label], label_rows, device, dtype, rounds)
            for label, label_rows in by_label.items()
        ]
        self.n = len(self.rows)
        self.device = torch.device(device)

    def act(self, t: int) -> torch.Tensor:
        out = torch.zeros(self.n, dtype=torch.int8, device=self.device)
        for group in self.groups:
            out[group.rows_t] = group.act(t)
        return out

    def observe(self, mine: torch.Tensor, theirs: torch.Tensor) -> None:
        for group in self.groups:
            group.observe(mine, theirs)


class GpuBlueField:
    """Blue actor for one chunk of rows."""

    def __init__(self, policy: Any, rows: Sequence[int],
                 device: torch.device, dtype: torch.dtype,
                 rounds: int = 100, mode: str = "argmax",
                 uniforms: Any = None) -> None:
        policy = unwrap_policy(policy)
        self.device = torch.device(device)
        self.n = len(rows)
        self.rows = list(rows)
        self.kind = type(policy).__name__
        if _policy_kind(policy) is not None:
            self.actor: Any = _PlainActor(
                policy, rows, self.device, dtype, rounds, mode=mode,
                uniforms=uniforms)
        elif self.kind in ("AdvMLPPolicy", "AdvGRUPolicy", "AdvRLPolicy"):
            self.actor = _AdvActor(
                policy, rows, self.device, dtype, rounds, mode=mode,
                uniforms=uniforms)
        elif self.kind == "PSROMetaPolicy":
            self.actor = _PSROActor(
                policy, rows, self.device, dtype, rounds, mode=mode,
                uniforms=uniforms)
        else:
            raise TypeError(f"unsupported GPU blue policy: {self.kind}")

    def act(self, t: int) -> torch.Tensor:
        return self.actor.act(t)

    def observe(self, mine: torch.Tensor, theirs: torch.Tensor) -> None:
        self.actor.observe(mine, theirs)


def supports_gpu_sample_policy(policy: Any) -> bool:
    """Whether a blue policy supports the v5 batched sample path."""
    policy = unwrap_policy(policy)
    if _policy_kind(policy) is not None:
        return True
    # Adv wrappers and PSRO meta-policies keep their exact CPU sample stream.
    return False


__all__ = ["GpuBlueField", "supports_gpu_policy",
           "supports_gpu_sample_policy"]
