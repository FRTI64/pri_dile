"""Fundamental IPD types shared by the simulator and opponent modules."""
from __future__ import annotations

import random

C = 0
D = 1


def payoff(me: int, them: int) -> int:
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1


class Strategy:
    name = "base"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.reset()

    def reset(self) -> None:
        pass

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        """Return C or D. t is zero-based, before the current move."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


__all__ = ["C", "D", "payoff", "Strategy"]
