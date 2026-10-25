#!/usr/bin/env python3
"""Iterated prisoner's dilemma tournament simulator.

Scoring per round (row player):
    CC = 6, CD = 0, DC = 10, DD = 1
where the first letter is our move.

The simulator is deliberately dependency-free so it is easy to run locally.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Sequence

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
        # If the probe was punished, be a normal reciprocator.
        if D in theirs[:3]:
            return theirs[-1]
        # Otherwise assume an unconditional cooperator and exploit in bursts.
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


ALL_STRATEGIES = [
    AllC,
    AllD,
    TFT,
    TFTLastD,
    SuspiciousTFT,
    Grim,
    lambda rng: GTFT(0.05, rng),
    lambda rng: GTFT(0.15, rng),
    TFT2,
    Pavlov,
    lambda rng: RandomMoves(0.5, rng),
    lambda rng: Joss(0.05, rng),
    lambda rng: Joss(0.15, rng),
    Prober,
    Tester,
    FirmButFair,
    Mistrust,
    Bully,
    WinStayLoseShift2,
]


@dataclass
class Result:
    a: str
    b: str
    score_a: int
    score_b: int


def play_match(a: Strategy, b: Strategy, rounds=100) -> Result:
    a.reset()
    b.reset()
    ah: list[int] = []
    bh: list[int] = []
    sa = sb = 0
    for t in range(rounds):
        x = a.act(t, ah, bh)
        y = b.act(t, bh, ah)
        ah.append(x)
        bh.append(y)
        sa += payoff(x, y)
        sb += payoff(y, x)
    return Result(a.name, b.name, sa, sb)


def tournament(population: Sequence[Strategy], rounds=100, include_self=False):
    n = len(population)
    scores = [0] * n
    matches = [0] * n
    matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j and not include_self:
                continue
            r = play_match(population[i], population[j], rounds)
            matrix[i][j] = (r.score_a, r.score_b)
            matrix[j][i] = (r.score_b, r.score_a)
            scores[i] += r.score_a
            scores[j] += r.score_b
            matches[i] += 1
            matches[j] += 1
    return scores, matches, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--reps", type=int, default=100,
                        help="number of random-population tournaments")
    args = parser.parse_args()

    # One of each strategy, for a quick smoke test.
    rng0 = random.Random(args.seed)
    pop = [f(rng0) for f in ALL_STRATEGIES]
    scores, matches, matrix = tournament(pop, args.rounds)
    order = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
    print(f"{'strategy':20} {'total':>7} {'avg':>8}")
    for i in order:
        print(f"{pop[i].name:20} {scores[i]:7d} {scores[i]/matches[i]:8.2f}")


if __name__ == "__main__":
    main()
