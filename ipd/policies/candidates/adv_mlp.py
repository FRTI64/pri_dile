"""Candidate: Adv-MLP.

Self-contained MLP + constrained exploitation candidate.  The base network
class comes from ``ipd/policies/neural.py``; the wrapper rules are implemented
locally so this candidate does not import another candidate.
"""
from __future__ import annotations

import json
import os

from ...core.simulator import C, D
from ...paths import config_path, weights_path
from ..neural import NeuralPolicy

KIND = "adv-mlp"
DISPLAY = "Adv-MLP"
DEFAULT_CONFIG = "adversarial_policy_config.json"


def _resolve_base_path(base_weights: str, config_file: str) -> str:
    if os.path.isabs(base_weights):
        return base_weights
    for candidate in (
        os.path.join(os.path.dirname(os.path.abspath(config_file)), base_weights),
        weights_path(base_weights),
    ):
        if os.path.exists(candidate):
            return candidate
    return base_weights


class AdvMLPPolicy:
    """Base MLP policy + constrained late/non-responsive exploitation."""

    name = DISPLAY

    def __init__(self, base, gap: float | None = 0.10,
                 min_c: int = 8, min_d: int = 3,
                 late_day: int | None = 50, repair: bool = False,
                 name: str | None = None):
        if gap is None and late_day is None:
            raise ValueError("enable at least one exploitation rule")
        self.base = base
        self.gap = gap
        self.min_c = int(min_c)
        self.min_d = int(min_d)
        self.late_day = late_day
        self.repair = bool(repair)
        if name:
            self.name = name
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.punish = False
        self.first_d: int | None = None

    def _observe(self, mine: list[int], theirs: list[int]) -> None:
        observe = getattr(self.base, "observe", None)
        if observe is not None:
            observe(mine, theirs)

    def _start_punishment(self, mine: list[int], theirs: list[int]) -> int:
        self.punish = True
        self._observe(mine, theirs)
        return D

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t >= 99:
            self._observe(mine, theirs)
            return D

        if self.punish:
            if (self.repair and len(theirs) >= 2
                    and theirs[-1] == C and theirs[-2] == C):
                self.punish = False
                return self.base.act(t, mine, theirs)
            self._observe(mine, theirs)
            return D

        if self.first_d is None and D in theirs:
            self.first_d = len(theirs) - 1

        if (self.late_day is not None and self.first_d is not None
                and self.first_d >= self.late_day
                and len(theirs) >= self.first_d + 2
                and theirs[self.first_d + 1] == D):
            return self._start_punishment(mine, theirs)

        if self.gap is not None:
            n_c = n_d = d_c = d_d = 0
            for i in range(1, len(theirs)):
                if mine[i - 1] == C:
                    n_c += 1
                    d_c += theirs[i] == D
                else:
                    n_d += 1
                    d_d += theirs[i] == D
            if n_c >= self.min_c and n_d >= self.min_d:
                p_c = d_c / n_c
                p_d = d_d / n_d
                if p_d - p_c <= self.gap:
                    return self._start_punishment(mine, theirs)

        return self.base.act(t, mine, theirs)

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def __repr__(self) -> str:
        return self.name


def build(*, seed: int = 20261024, weights: str | None = None,
          config: str | None = None, mode: str = "sample",
          name: str | None = None, **_: object) -> AdvMLPPolicy:
    config_file = config or config_path(DEFAULT_CONFIG)
    with open(config_file, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg.get("format") != "ipd-adversarial-config-v1":
        raise ValueError(f"unsupported config format: {cfg.get('format')}")
    arch = cfg.get("base_arch", "mlp")
    if arch != "mlp":
        raise ValueError(f"adv-mlp requires base_arch=mlp, got {arch!r}")
    base_path = _resolve_base_path(weights or cfg["base_weights"], config_file)
    base = NeuralPolicy(base_path, seed=seed, mode=mode)
    return AdvMLPPolicy(
        base,
        gap=cfg.get("gap"),
        min_c=cfg.get("min_c", 8),
        min_d=cfg.get("min_d", 3),
        late_day=cfg.get("late_day"),
        repair=cfg.get("repair", False),
        name=name or DISPLAY,
    )
