"""Canonical simple opponents used by the simulator and the extended bank."""
from __future__ import annotations

from ..primitives import C, D, Strategy, payoff


class AllC(Strategy):
    name = "AllC"

    def act(self, t, mine, theirs):
        return C


class AllD(Strategy):
    name = "AllD"

    def act(self, t, mine, theirs):
        return D


class TFT(Strategy):
    name = "TFT"

    def act(self, t, mine, theirs):
        return C if t == 0 else theirs[-1]


class TFTLastD(TFT):
    name = "TFT-lastD"

    def act(self, t, mine, theirs):
        return D if t == 99 else super().act(t, mine, theirs)


class SuspiciousTFT(Strategy):
    name = "sTFT"

    def act(self, t, mine, theirs):
        return D if t == 0 else theirs[-1]


class Grim(Strategy):
    name = "Grim"

    def act(self, t, mine, theirs):
        return D if D in theirs else C


class GTFT(Strategy):
    """Copy the opponent, but forgive an isolated defection with prob g."""

    def __init__(self, g=0.1, rng=None):
        self.g = g
        super().__init__(rng)
        self.name = f"GTFT({g:g})"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if theirs[-1] == C:
            return C
        return C if self.rng.random() < self.g else D


class TFT2(Strategy):
    name = "TFT2"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        return D if len(theirs) >= 2 and theirs[-1] == D and theirs[-2] == D else C


class Pavlov(Strategy):
    name = "Pavlov"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        p = payoff(mine[-1], theirs[-1])
        return mine[-1] if p >= 3 else (C if mine[-1] == D else D)


class RandomMoves(Strategy):
    def __init__(self, p=0.5, rng=None):
        self.p = p
        super().__init__(rng)
        self.name = f"Random({p:g})"

    def act(self, t, mine, theirs):
        return C if self.rng.random() < self.p else D


class Joss(Strategy):
    """TFT with a small probability of defection."""

    def __init__(self, p=0.1, rng=None):
        self.p = p
        super().__init__(rng)
        self.name = f"Joss({p:g})"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if theirs[-1] == D:
            return D
        return D if self.rng.random() < self.p else C


class Prober(Strategy):
    """Classic prober: C,C,D, then TFT if punished, else exploit."""

    name = "Prober"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if t == 1:
            return C
        if t == 2:
            return D
        if D in theirs[:3]:
            return theirs[-1]
        return D if (t - 3) % 4 < 2 else C


class Tester(Strategy):
    """Defect immediately; reciprocate if punished, exploit if not."""

    name = "Tester"

    def act(self, t, mine, theirs):
        if t == 0:
            return D
        if theirs[0] == D:
            return theirs[-1]
        return D if (t - 1) % 4 < 2 else C


class FirmButFair(Strategy):
    name = "FirmButFair"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if theirs[-1] == D:
            return D
        return C


class Mistrust(Strategy):
    """Cooperate unless the opponent has defected twice in total."""

    name = "Mistrust"

    def act(self, t, mine, theirs):
        return D if theirs.count(D) >= 2 else C


class Grudge2(Strategy):
    name = "Grudge2"

    def act(self, t, mine, theirs):
        return D if theirs.count(D) >= 2 else C


class Bully(Strategy):
    """Cooperate, but if the opponent ever retaliates, back off; probe occasionally."""

    name = "Bully"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if t % 17 == 16:
            return D
        if theirs[-1] == D:
            return C
        return C


class WinStayLoseShift2(Strategy):
    name = "WSLS2"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        p = payoff(mine[-1], theirs[-1])
        if mine[-1] == C:
            return C if p >= 6 else D
        return D if p >= 1 else C


__all__ = [
    "AllC", "AllD", "TFT", "TFTLastD", "SuspiciousTFT", "Grim", "GTFT",
    "TFT2", "Pavlov", "RandomMoves", "Joss", "Prober", "Tester",
    "FirmButFair", "Mistrust", "Grudge2", "Bully", "WinStayLoseShift2",
]
