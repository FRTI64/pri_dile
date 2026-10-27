"""Candidate: PSRO meta-policy.

Samples one blue policy per match from the blue maximin mixture written by
``ipd.training.psro``.  The population and weights live in
``weights/psro/meta_strategy.json``; the tournament policy itself uses only
standard-library inference.
"""
from __future__ import annotations

from ...paths import weights_path
from ..psro import PSROMetaPolicy

KIND = "psro"
DISPLAY = "PSRO-Meta"
DEFAULT_META = "psro/meta_strategy.json"


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> object:
    return PSROMetaPolicy(
        weights or weights_path(DEFAULT_META), seed=seed, mode=mode,
        name=name or DISPLAY)


__all__ = ["KIND", "DISPLAY", "DEFAULT_META", "build"]
