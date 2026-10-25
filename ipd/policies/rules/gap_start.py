"""Gap-based start policy used by the search experiments."""
from __future__ import annotations

from ...core.features import FEATURES, feature_vector
from ...core.simulator import C, D


class GapStartPolicy:
    """Cooperate until ``start``; then defect iff the gap is small."""

    name = "GapStart"

    def __init__(self, start: int | None = 1, threshold: float = 0.55):
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
