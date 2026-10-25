"""Candidate: Harvest45.

Self-contained endgame-harvest candidate.  It uses the core HarvestMachine and
sets only its own parameters.
"""
from __future__ import annotations

from ...core.harvest import HarvestMachine

KIND = "harvest45"
DISPLAY = "Harvest45"


class Harvest45Policy(HarvestMachine):
    name = DISPLAY
    HARVEST_START = 45


def build(*, seed: int = 20261024, name: str | None = None,
          **_: object) -> Harvest45Policy:
    policy = Harvest45Policy(seed=seed)
    policy.name = name or DISPLAY
    return policy
