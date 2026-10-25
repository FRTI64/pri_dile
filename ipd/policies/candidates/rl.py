"""Candidate: RL-PPO.

Self-contained candidate definition.  It accepts either a deployable MLP export
or an actor-critic checkpoint, matching the historical registry behavior.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "rl"
DISPLAY = "RL-PPO"
DEFAULT_WEIGHTS = "rl_policy_checkpoint.json"


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
