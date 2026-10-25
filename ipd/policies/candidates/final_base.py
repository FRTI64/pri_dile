"""Candidate: Final-base.

Self-contained submission/evaluation candidate.  It uses only the core Final
state-machine implementation; it does not import another candidate file.
"""
from __future__ import annotations

from ...core.machine import Machine

KIND = "final-base"
DISPLAY = "Final-base"


class FinalBasePolicy(Machine):
    name = DISPLAY

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        return self.move(t + 1, mine, theirs)


def build(*, seed: int = 20261024, name: str | None = None,
          **_: object) -> FinalBasePolicy:
    policy = FinalBasePolicy(seed=seed)
    policy.name = name or DISPLAY
    return policy
