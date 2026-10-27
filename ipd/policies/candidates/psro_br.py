"""Candidate: the final PSRO best-response policy.

This is the single learned policy exported by the last PSRO oracle round.  It
is kept separate from ``psro`` (the meta-policy) so the two PSRO outputs can be
compared on the same evaluation protocol.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "psro-br"
DISPLAY = "PSRO-BR"
DEFAULT_WEIGHTS = "psro/psro_policy.json"


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
