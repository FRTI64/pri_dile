"""Fundamental IPD types shared by the simulator and opponent modules."""
from __future__ import annotations

import random

C = 0
D = 1

# Canonical pri_dile prisoner's dilemma matrix, in Press-Dyson notation:
#   T=10 (temptation), R=6 (reward), P=1 (punishment), S=0 (sucker).
R, S, T, P = 6, 0, 10, 1


def payoff(me: int, them: int) -> int:
    if me == C:
        return R if them == C else S
    return T if them == C else P


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


__all__ = ["C", "D", "R", "S", "T", "P", "payoff", "Strategy"]
