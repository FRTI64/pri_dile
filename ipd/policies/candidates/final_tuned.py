"""Candidate: Final-tuned.

This file embeds the tuned Final thresholds so the candidate can be translated
without importing another policy.  Values mirror results/*.json and
:mod:`ipd.policies.tuned`; keep both in sync if retraining changes them.
"""
from __future__ import annotations

from ...core.machine import Machine

KIND = "final-tuned"
DISPLAY = "Final-tuned"

P_UNPROVOKED = 0.2395699546614404
P_MUTUAL = 0.6399155368157192
P_RESPONSIVE = 0.9544467968114831
MIN_C = 5
MIN_D = 2
RESP_GAP = 0.3871088255996472
DEFECT_RATE = 0.1170465108152342
PEACE_TESTS = 1
PEACE_COOLDOWN = 9


class FinalTunedPolicy(Machine):
    name = DISPLAY
    P_UNPROVOKED = P_UNPROVOKED
    P_MUTUAL = P_MUTUAL
    P_RESPONSIVE = P_RESPONSIVE
    MIN_C = MIN_C
    MIN_D = MIN_D
    RESP_GAP = RESP_GAP
    DEFECT_RATE = DEFECT_RATE
    PEACE_TESTS = PEACE_TESTS
    PEACE_COOLDOWN = PEACE_COOLDOWN

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        return self.move(t + 1, mine, theirs)


def build(*, seed: int = 20261024, name: str | None = None,
          **_: object) -> FinalTunedPolicy:
    policy = FinalTunedPolicy(seed=seed)
    policy.name = name or DISPLAY
    return policy
