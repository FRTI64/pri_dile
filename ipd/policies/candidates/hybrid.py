"""Candidate: Final+Resp3 (hybrid).

Self-contained hybrid candidate.  It constructs the core Final and Responsive3
machines locally with the tuned Final parameters; it does not import another
candidate file.
"""
from __future__ import annotations

import random

from ...core.machine import Machine
from ...core.opponents.reciprocal import Responsive3
from ...core.simulator import C, D

KIND = "hybrid"
DISPLAY = "Final+Resp3"
SEED = 20261025

FINAL_TUNED = {
    "P_UNPROVOKED": 0.2395699546614404,
    "P_MUTUAL": 0.6399155368157192,
    "P_RESPONSIVE": 0.9544467968114831,
    "MIN_C": 5,
    "MIN_D": 2,
    "RESP_GAP": 0.3871088255996472,
    "DEFECT_RATE": 0.1170465108152342,
    "PEACE_TESTS": 1,
    "PEACE_COOLDOWN": 9,
}


class HybridPolicy:
    """Final + Responsive3; agree on action, otherwise sample/defect."""

    name = DISPLAY

    def __init__(self, seed: int = SEED):
        self.f = Machine(seed=20261024)
        for key, value in FINAL_TUNED.items():
            setattr(self.f, key, value)
        # The historical tune selected Responsive3 defaults for the second
        # component (empty resp_cfg), with p=0.0.
        self.r = Responsive3(rng=random.Random(20261024))
        self.p = 0.0
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.f.reset()
        self.r.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        a = self.f.move(t + 1, mine, theirs)
        b = self.r.act(t, mine, theirs)
        if a == b:
            return a
        return C if self.rng.random() < self.p else D


def build(*, name: str | None = None, **_: object) -> HybridPolicy:
    policy = HybridPolicy()
    policy.name = name or DISPLAY
    return policy
