"""Candidate: 160/128 RL policy (net2wider width sweet spot).

Starts from the balanced 128/128 RL-Wide checkpoint, expands the first hidden
layer to 160, and runs two short balance PPO iterations (PB weight 1/11).
The wider first layer keeps the dev_param / test behavior of RL-Wide while
raising dev_complex and, unlike 192/128, keeps every family min_norm near 1.0.
It is kept as a separate candidate so the 128/128 and earlier weights remain
untouched.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "rl-wide160"
DISPLAY = "RL-Wide160"
DEFAULT_WEIGHTS = "rl_policy_weights_wide160.json"


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
