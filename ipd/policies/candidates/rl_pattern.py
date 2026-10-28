"""Candidate: pattern-robust RL fine-tune (PB).

Starts from the RL-Repair / entropy-regularized checkpoint and mixes the full
``train_v4`` catalog with targeted ``train_pattern_breaker`` fields.  The
resulting policy keeps the safe opening rules but learns a less predictable
reaction after the opponent has defected, which closes the large
``pattern_breaker`` hole in Protocol v4 while retaining the v4 families.  It is
kept as a separate candidate so RL-Repair and the main RL-PPO weights stay
untouched.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "rl-pattern"
DISPLAY = "RL-Pattern"
DEFAULT_WEIGHTS = "rl_policy_weights_pattern.json"


def _load_base(weights: str, seed: int, mode: str):
    with open(weights, encoding="utf-8") as fh:
        fmt = json.load(fh).get("format")
    cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
    return cls(weights, seed=seed, mode=mode)


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> object:
    policy = _load_base(weights or weights_path(DEFAULT_WEIGHTS), seed, mode)
    policy.name = name or DISPLAY
    return policy


__all__ = ["KIND", "DISPLAY", "DEFAULT_WEIGHTS", "build"]
