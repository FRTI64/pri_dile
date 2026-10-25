#!/usr/bin/env python3
"""Early-probe / exploit strategy for fields full of naive late harvesters.

This wraps the `ProbeBot` policy in a 1-based `move(day, mine, theirs)`
interface so it can be used like `Machine` in `ipd_strategy.py`.

Warning: it defects on day 1 to test the opponent.  This is excellent against
AllC, TFT2-like and "cooperate then harvest late" opponents, but it is very
bad against Grim/Mistrust-style permanent punishers.
"""
from __future__ import annotations

import random

from ipd_field import ProbeBot


class EarlyProbeMachine(ProbeBot):
    name = "EarlyProbe"

    def __init__(self, seed: int = 20261024, rng: random.Random | None = None):
        super().__init__(rng=rng if rng is not None else random.Random(seed))

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return super().act(day - 1, mine, theirs)


if __name__ == "__main__":
    from ipd_sim import AllC, AllD, Grim, TFT, play_match

    for opponent in (AllC, AllD, Grim, TFT):
        m = EarlyProbeMachine()
        r = play_match(m, opponent())
        print(opponent().name, r.score_a, r.score_b)
