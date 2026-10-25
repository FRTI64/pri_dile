"""Candidate: Harvest-tuned.

Embeds the tuned Harvest parameters.  It does not import another candidate.
"""
from __future__ import annotations

from ...core.harvest import HarvestMachine

KIND = "harvest-tuned"
DISPLAY = "Harvest-tuned"

P_UNPROVOKED = 0.10034922303092272
P_MUTUAL = 0.6374190528551432
P_RESPONSIVE = 0.9318802080916426
MIN_C = 3
MIN_D = 2
RESP_GAP = 0.5198267076998384
DEFECT_RATE = 0.2140943179339193
PEACE_TESTS = 1
PEACE_COOLDOWN = 4
HARVEST_START = 75


class HarvestTunedPolicy(HarvestMachine):
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
    HARVEST_START = HARVEST_START


def build(*, seed: int = 20261024, name: str | None = None,
          **_: object) -> HarvestTunedPolicy:
    policy = HarvestTunedPolicy(seed=seed)
    policy.name = name or DISPLAY
    return policy
