#!/usr/bin/env python3
"""Iterated prisoner's dilemma tournament simulator.

Scoring per round (row player):
    CC = 6, CD = 0, DC = 10, DD = 1
where the first letter is our move.

The simulator is deliberately dependency-free so it is easy to run locally.
Simple canonical opponents live in :mod:`ipd.core.opponents.basic`; this
module re-exports them for backwards compatibility and keeps the match/tournament
engine.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Sequence

from .primitives import C, D, Strategy, payoff
from .opponents.basic import (
    AllC, AllD, Bully, FirmButFair, GTFT, Grim, Grudge2, Joss, Mistrust,
    Pavlov, Prober, RandomMoves, SuspiciousTFT, TFT, TFT2, TFTLastD, Tester,
    WinStayLoseShift2,
)

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
