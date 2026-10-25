"""Defecting, late-harvesting, and autonomous opponent strategies."""
from __future__ import annotations

from ..primitives import C, D, Strategy


class Alternator(Strategy):
    name = "Alternator"

    def act(self, t, mine, theirs):
        return C if t % 2 == 0 else D


class LateDefector(Strategy):
    def __init__(self, k=20, rng=None):
        self.k = k
        super().__init__(rng)
        self.name = f"LateD{k}"

    def act(self, t, mine, theirs):
        return D if t >= 100 - self.k else C


class EarlyDefector(Strategy):
    def __init__(self, k=20, rng=None):
        self.k = k
        super().__init__(rng)
        self.name = f"EarlyD{k}"

    def act(self, t, mine, theirs):
        return D if t < self.k else C


class Gradual(Strategy):
    """Punish n times after the n-th defection, then two cooperative moves."""

    def reset(self):
        self.punish_left = 0
        self.coop_left = 0
        self.total = 0

    name = "Gradual"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if theirs[-1] == D and mine[-1] == C:
            self.total += 1
            self.punish_left = self.total
        if self.punish_left:
            self.punish_left -= 1
            return D
        if self.coop_left:
            self.coop_left -= 1
            return C
        if self.punish_left == 0 and self.total and theirs[-1] == C and mine[-1] == D:
            self.coop_left = 1
            return C
        return C


class Grudger2(Strategy):
    name = "Grudger2"

    def act(self, t, mine, theirs):
        return D if theirs.count(D) >= 2 else C


class RandomP(Strategy):
    def __init__(self, p, rng=None):
        self.p = p
        super().__init__(rng)
        self.name = f"Random({p:g})"

    def act(self, t, mine, theirs):
        return C if self.rng.random() < self.p else D


__all__ = [
    "Alternator", "LateDefector", "EarlyDefector", "Gradual", "Grudger2",
    "RandomP",
]
