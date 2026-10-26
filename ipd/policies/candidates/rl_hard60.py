"""Candidate: RL-PPO fine-tuned with the hard-opponent pool (A60).

This is the A60 experiment from ``results/hard2_20261026``: the existing
RL-PPO checkpoint fine-tuned for 60 iterations with ``hard`` weight 2,
KL-to-base 0.1 and lr 1e-4.  It is a separate candidate so the original
``rl`` weights remain untouched.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "rl-hard60"
DISPLAY = "RL-PPO-Hard60"
DEFAULT_WEIGHTS = "rl_policy_weights_hard60.json"


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
