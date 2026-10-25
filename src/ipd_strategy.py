#!/usr/bin/env python3
"""Final adaptive strategy for a 100-round prisoner's dilemma tournament.

The public interface is the ``Machine`` class:

    m = Machine()
    m.reset()                       # call at the start of every match
    action = m.move(day, mine, theirs)

``day`` is 1-based (1..100), ``mine`` and ``theirs`` contain the moves made in
all previous days.  Actions are ``C`` (cooperate) and ``D`` (defect).

The strategy is deliberately simple to port:

* Start by cooperating.
* Estimate two conditional defection probabilities:
      pC = P(opponent defects now | we cooperated on the previous day)
      pD = P(opponent defects now | we defected on the previous day)
  A large pD-pC means the opponent is responsive: it punishes defections
  and rewards cooperation.  We then cooperate, forgiving isolated noise.
  If pC and pD are both similar, the opponent is not responding to us
  (AllD, random, periodic, tester, ...), so we exploit it by defecting.
* Until there is enough evidence, forgive the first isolated defection with
  a substantial probability.  This avoids long punishment spirals against
  noisy reciprocators such as Joss or imperfect TFT.
* If a non-responsive opponent has been punished, make at most two short
  peace tests.  A reciprocal opponent returns to cooperation; an
  unconditional defector does not.
* If the opponent shows a long exact alternating pattern, defect.  This
  handles a pure alternator, against which defecting is worth 5.5 points
  per round.
* Defect on the final day: it is strictly dominant for the last round.

No knowledge of the opponent's source code or identity is used.
"""
from __future__ import annotations

import random

C = 0
D = 1


class Machine:
    # Tuning constants.  They are not very sensitive; see the report.
    P_UNPROVOKED = 0.25   # forgive a D after our C while still uncertain
    P_MUTUAL = 0.45       # forgive a D after our D while still uncertain
    P_RESPONSIVE = 0.95   # forgive when the opponent has proven reciprocal
    MIN_C = 5             # observations with our previous move = C
    MIN_D = 3             # observations with our previous move = D
    RESP_GAP = 0.38       # pD-pC needed to call an opponent responsive
    DEFECT_RATE = 0.18    # overall D rate needed to call it non-responsive
    PEACE_TESTS = 2
    PEACE_COOLDOWN = 8

    def __init__(self, seed: int = 20261024):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.mode = "nice"       # nice | punish | test1 | test2 | test3
        self.tests = 0
        self.last_test = -100
        self.pattern = False

    @staticmethod
    def _stats(mine: list[int], theirs: list[int]):
        n_c = n_d = d_c = d_d = 0
        for i in range(1, len(theirs)):
            if mine[i - 1] == C:
                n_c += 1
                d_c += theirs[i] == D
            else:
                n_d += 1
                d_d += theirs[i] == D
        return n_c, n_d, d_c, d_d

    @staticmethod
    def _alternating(moves: list[int], run: int = 10) -> bool:
        if len(moves) < run:
            return False
        return all(moves[i] != moves[i - 1]
                   for i in range(len(moves) - run + 1, len(moves)))

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        if day >= 100:                    # final round: D is dominant
            return D
        if day == 1:
            return C

        n_c, n_d, d_c, d_d = self._stats(mine, theirs)
        p_c = d_c / n_c if n_c else 0.0
        p_d = d_d / n_d if n_d else 0.0
        d_rate = sum(theirs) / len(theirs)

        # Detect a strictly alternating opponent.  The condition is cleared
        # as soon as the pattern disappears, so a false positive is cheap.
        if self.pattern:
            if self._alternating(theirs):
                return D
            self.pattern = False
        elif self._alternating(theirs):
            self.pattern = True
            return D

        # Two consecutive cooperative peace offers after a punishment phase.
        if self.mode == "test1":
            self.mode = "test2"
            return C
        if self.mode == "test2":
            self.mode = "test3"
            return C
        if self.mode == "test3":
            if theirs[-1] == C:
                self.mode = "nice"
                return C
            self.mode = "punish"
            self.last_test = day
            return D

        enough = n_c >= self.MIN_C and n_d >= self.MIN_D
        responsive = (enough and p_c <= 0.35 and p_d >= 0.65
                      and p_d - p_c >= self.RESP_GAP)
        nonresponsive = (enough and p_d - p_c < 0.35
                         and d_rate >= self.DEFECT_RATE)

        if self.mode == "nice":
            if nonresponsive:
                self.mode = "punish"
                return D

            if theirs[-1] == C:
                return C

            # The opponent just defected.
            if mine[-1] == C:
                p_forgive = self.P_RESPONSIVE if responsive else self.P_UNPROVOKED
            else:
                p_forgive = self.P_RESPONSIVE if responsive else self.P_MUTUAL
            return C if self.rng.random() < p_forgive else D

        # Punishment / exploitation mode.  Make a peace offer only when the
        # opponent has just shown the characteristic signature of a
        # reciprocator.  The cooldown prevents repeated useless tests.
        if (self.tests < self.PEACE_TESTS
                and day - self.last_test >= self.PEACE_COOLDOWN
                and enough and p_c <= 0.35 and p_d >= 0.65
                and p_d - p_c >= 0.40):
            self.tests += 1
            self.last_test = day
            self.mode = "test1"
            return C
        return D


# Convenience function for contest harnesses that pass only histories.
_default_machine = Machine()


def choose_action(day: int, mine: list[int], theirs: list[int]) -> int:
    if day == 1:
        _default_machine.reset()
    return _default_machine.move(day, mine, theirs)


if __name__ == "__main__":
    # Quick self-test against a few fixed sequences.
    m = Machine(1)
    print("first move:", m.move(1, [], []))
    # AlwaysD: C once, then D.
    m.reset()
    mine, theirs = [], []
    total = 0
    for day in range(1, 101):
        x = m.move(day, mine, theirs)
        y = D
        mine.append(x)
        theirs.append(y)
        total += 6 if (x, y) == (C, C) else (0 if x == C else (10 if y == C else 1))
    print("against AlwaysD:", total, "final mode:", m.mode)
    # AlwaysC: mutual cooperation, with a final defection.
    m.reset()
    mine, theirs = [], []
    total = 0
    for day in range(1, 101):
        x = m.move(day, mine, theirs)
        y = C
        mine.append(x)
        theirs.append(y)
        total += 6 if (x, y) == (C, C) else (0 if x == C else (10 if y == C else 1))
    print("against AlwaysC:", total)
