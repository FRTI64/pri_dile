"""Candidate: GapStart.

Self-contained gap-threshold start policy.
"""
from __future__ import annotations

from ...core.features import FEATURES, feature_vector
from ...core.simulator import C, D

KIND = "gap-start"
DISPLAY = "GapStart"
START = 1
THRESHOLD = 0.5


class GapStartPolicy:
    """Cooperate until ``start``; then defect iff the gap is small."""

    name = DISPLAY

    def __init__(self, start: int | None = START, threshold: float = THRESHOLD):
        self.start = start
        self.threshold = threshold

    def reset(self) -> None:
        pass

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t >= 99:
            return D
        if self.start is not None and t + 1 < self.start:
            return C
        gap = feature_vector(t + 1, mine, theirs)[FEATURES.index("gap")]
        return D if gap <= self.threshold else C


def build(*, name: str | None = None, **_: object) -> GapStartPolicy:
    policy = GapStartPolicy()
    policy.name = name or DISPLAY
    return policy
