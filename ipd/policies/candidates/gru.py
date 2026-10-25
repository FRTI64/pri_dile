"""Candidate: GRU.

Self-contained candidate definition.  The recurrent network architecture lives
in ``ipd/policies/gru.py``; this file selects the candidate's weights and mode.
"""
from __future__ import annotations

from ...paths import weights_path
from ..gru import GRUPolicy

KIND = "gru"
DISPLAY = "GRU"
DEFAULT_WEIGHTS = "gru_policy_weights.json"


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> GRUPolicy:
    policy = GRUPolicy(
        weights or weights_path(DEFAULT_WEIGHTS), seed=seed, mode=mode)
    policy.name = name or DISPLAY
    return policy
