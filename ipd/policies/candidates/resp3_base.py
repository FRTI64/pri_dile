"""Candidate: Resp3-base.

Self-contained Responsive3 candidate with the core default parameters.
"""
from __future__ import annotations

import random

from ...core.opponents.reciprocal import Responsive3

KIND = "resp3-base"
DISPLAY = "Resp3-base"


class Resp3BasePolicy(Responsive3):
    name = DISPLAY

    def __init__(self, seed: int = 20261024):
        super().__init__(rng=random.Random(seed))
        self.name = DISPLAY


def build(*, seed: int = 20261024, name: str | None = None,
          **_: object) -> Resp3BasePolicy:
    policy = Resp3BasePolicy(seed=seed)
    policy.name = name or DISPLAY
    return policy
