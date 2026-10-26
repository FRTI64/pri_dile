"""Candidate: self-play GRU policy (trained by ipd.training.selfplay)."""
from __future__ import annotations

from ...paths import weights_path
from ..selfplay_gru import SelfPlayGRUPolicy

KIND = "selfplay"
DISPLAY = "SelfPlay-GRU"
DEFAULT_WEIGHTS = "selfplay_gru_policy.json"


def build(*, seed: int = 20261026, weights: str | None = None,
          mode: str = "sample", name: str | None = None, **_: object):
    path = weights or weights_path(DEFAULT_WEIGHTS)
    policy = SelfPlayGRUPolicy(path, seed=seed, mode=mode, name=name)
    return policy


__all__ = ["KIND", "DISPLAY", "DEFAULT_WEIGHTS", "build"]
