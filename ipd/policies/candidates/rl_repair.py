"""Candidate: RL-PPO repair fine-tune (E1c).

Starts from the E1b domain-randomized checkpoint and repairs the random-family
regression with the ``repair`` training mixture.  It is a separate candidate so
the A60 and main RL-PPO weights remain untouched.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "rl-repair"
DISPLAY = "RL-Repair"
DEFAULT_WEIGHTS = "rl_policy_weights_repair.json"


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
