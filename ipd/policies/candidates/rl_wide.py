"""Candidate: wider 128/128 RL policy (net2wider warm start).

Starts from the pattern-robust 64/32 checkpoint, expands both hidden layers to
128 with net2wider, and continues PPO on the v4 training distribution without
the extra pattern-heavy weighting.  The wider trunk gains capacity for the
complex / Axelrod families while keeping the dev_param and test_v7 behavior of
RL-Pattern.  It is kept as a separate candidate so the 64/32 deployment path
and all earlier weights remain untouched.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "rl-wide"
DISPLAY = "RL-Wide"
DEFAULT_WEIGHTS = "rl_policy_weights_wide.json"


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
