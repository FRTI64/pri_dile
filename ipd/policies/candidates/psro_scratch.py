"""Candidate: from-scratch PSRO meta-policy.

Only the blue population (random init + from-scratch PPO best responses) is
sampled.  Existing models and rules are red/reference opponents in the training
config and are never blue members.
"""
from __future__ import annotations

from ...paths import weights_path
from ..psro import PSROMetaPolicy

KIND = "psro-scratch"
DISPLAY = "PSRO-Scratch"
DEFAULT_META = "psro_scratch/meta_strategy.json"


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> object:
    return PSROMetaPolicy(
        weights or weights_path(DEFAULT_META), seed=seed, mode=mode,
        name=name or DISPLAY)


__all__ = ["KIND", "DISPLAY", "DEFAULT_META", "build"]
