"""Candidate: Resp3-tuned.

Embeds the tuned Responsive3 parameters; it does not import another candidate.
"""
from __future__ import annotations

import random

from ...core.opponents.reciprocal import Responsive3

KIND = "resp3-tuned"
DISPLAY = "Resp3-tuned"

G_UNPROV = 0.31711381663242244
G_MUTUAL = 0.26173130084075913
G_RESP = 0.9804446329081643
MIN_C = 5
MIN_D = 2
RESP = 0.46788262082576104
RATE = 0.197064176521839
MAX_TESTS = 3
COOLDOWN = 14


class Resp3TunedPolicy(Responsive3):
    name = DISPLAY

    def __init__(self, seed: int = 20261024):
        super().__init__(
            g_unprov=G_UNPROV,
            g_mutual=G_MUTUAL,
            g_resp=G_RESP,
            min_c=MIN_C,
            min_d=MIN_D,
            resp=RESP,
            rate=RATE,
            max_tests=MAX_TESTS,
            cooldown=COOLDOWN,
            rng=random.Random(seed),
        )
        self.name = DISPLAY


def build(*, seed: int = 20261024, name: str | None = None,
          **_: object) -> Resp3TunedPolicy:
    policy = Resp3TunedPolicy(seed=seed)
    policy.name = name or DISPLAY
    return policy
