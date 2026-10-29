"""Candidate: PSRO-v4 meta-policy.

Samples one blue policy per match from the blue maximin mixture written by the
2026-10-29 PSRO v4 run.  The population and weights live in
``weights/psro_v4/meta_strategy.json``; inference uses only the standard
library plus the project's policy adapters.
"""
from __future__ import annotations

from ...paths import weights_path
from ..psro import PSROMetaPolicy

KIND = "psro-v4"
DISPLAY = "PSRO-V4-Meta"
DEFAULT_META = "psro_v4/meta_strategy.json"


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> object:
    return PSROMetaPolicy(
        weights or weights_path(DEFAULT_META), seed=seed, mode=mode,
        name=name or DISPLAY)


__all__ = ["KIND", "DISPLAY", "DEFAULT_META", "build"]
