"""Candidate: from-scratch PSRO best response.

This is the final single policy exported by the scratch PSRO oracle.  It starts
from random weights; the only inherited state is the frozen feature
normalisation.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "psro-scratch-br"
DISPLAY = "PSRO-Scratch-BR"
DEFAULT_WEIGHTS = "psro_scratch/psro_policy.json"


def _load_base(path: str, seed: int, mode: str):
    with open(path, encoding="utf-8") as fh:
        fmt = json.load(fh).get("format")
    cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
    return cls(path, seed=seed, mode=mode)


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> object:
    policy = _load_base(weights or weights_path(DEFAULT_WEIGHTS), seed, mode)
    policy.name = name or DISPLAY
    return policy


__all__ = ["KIND", "DISPLAY", "DEFAULT_WEIGHTS", "build"]
