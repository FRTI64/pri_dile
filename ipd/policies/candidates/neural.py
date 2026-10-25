"""Candidate: Neural (MLP).

Self-contained candidate definition.  The network architecture lives in
``ipd/policies/neural.py``; this file selects the candidate's weights and mode.
"""
from __future__ import annotations

from ...paths import weights_path
from ..neural import NeuralPolicy

KIND = "neural"
DISPLAY = "Neural"
DEFAULT_WEIGHTS = "neural_policy_weights.json"


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> NeuralPolicy:
    policy = NeuralPolicy(
        weights or weights_path(DEFAULT_WEIGHTS), seed=seed, mode=mode)
    policy.name = name or DISPLAY
    return policy
