#!/usr/bin/env python3
"""Original state machine + EMA-weighted pC/pD + dynamic forgiveness.

No early probe and no scheduled harvest: the strategy never initiates a
defection before the final day except as a response to an observed defection.
Consequently it cannot preempt a late harvester before its first D.
"""
from __future__ import annotations

import random

from ipd.core.machine import Machine, C, D


class EMAForgiveMachine(Machine):
    name = "EMA-forgive"
    HALF_LIFE = 4.0
    G_PRIOR = 0.25          # forgiven probability before enough evidence
    G_SLOPE = 2.5           # forgiveness = slope * (pD-pC), clipped
    PATTERN_RUN = 8
    WEIGHT_FLOOR = 0.10     # keep a long-memory floor to avoid lock-in
    PERMANENT_PATTERN = True
    ENOUGH_D = 2.0

    def reset(self):
        super().reset()
        self.pattern_permanent = False
        self.last_pattern = False

    def act(self, t, mine, theirs):
        return self.move(t + 1, mine, theirs)

    def _ema(self, mine, theirs):
        n = len(theirs)
        wc = wd = dc = dd = dr = tw = 0.0
        for i in range(n):
            age = n - 1 - i
            w = max(self.WEIGHT_FLOOR, 0.5 ** (age / self.HALF_LIFE))
            tw += w
            dr += w * (theirs[i] == D)
            if i == 0:
                continue
            if mine[i - 1] == C:
                wc += w
                dc += w * (theirs[i] == D)
            else:
                wd += w
                dd += w * (theirs[i] == D)
        return wc, wd, dc, dd, (dr / tw if tw else 0.0)

    def move(self, day, mine, theirs):
        if day >= 100:
            return D
        if day == 1:
            return C

        wc, wd, dc, dd, rate = self._ema(mine, theirs)
        p_c = dc / wc if wc else 0.0
        p_d = dd / wd if wd else 0.0
        gap = p_d - p_c

        # Permanent alternation detector.
        if self.pattern_permanent:
            return D
        # Permanent alternation should only be inferred when the opponent
        # alternates *despite our continuous cooperation*.  Otherwise a
        # TFT/Joss two-cycle (where our own moves alternate) is mistaken for
        # a pure alternator.
        if (self.PERMANENT_PATTERN and len(theirs) >= self.PATTERN_RUN and
                all(theirs[i] != theirs[i - 1]
                    for i in range(len(theirs) - self.PATTERN_RUN + 1,
                                   len(theirs))) and
                all(x == C for x in mine[-self.PATTERN_RUN:])):
            self.pattern_permanent = True
            return D

        enough_c = wc >= 3.0
        enough_d = wd >= self.ENOUGH_D

        # Dynamic forgiveness.  With no evidence of retaliation (gap≈0),
        # forgive 0; with strong retaliation (gap→1), forgive up to 95%.
        if enough_c and enough_d:
            g = max(0.0, min(0.95, self.G_SLOPE * gap))
        else:
            g = self.G_PRIOR

        # Keep the original non-responsive exploit mode: if the opponent
        # defects regardless of what we do, exploit it instead of merely
        # copying it.
        nonresponsive = (enough_c and enough_d and gap < 0.35
                         and rate >= 0.18)

        if self.mode == "punish":
            # One short peace test only if the weighted evidence has turned
            # strongly reciprocal.
            if (self.tests < 2 and day - self.last_test >= 8
                    and enough_c and enough_d
                    and p_c <= 0.35 and p_d >= 0.65 and gap >= 0.40):
                self.tests += 1
                self.last_test = day
                self.mode = "test1"
                return C
            return D

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
            return D

        if nonresponsive:
            self.mode = "punish"
            return D

        if theirs[-1] == C:
            return C
        return C if self.rng.random() < g else D


if __name__ == "__main__":
    from ipd.core.simulator import AllC, AllD, Grim, TFT, play_match
    for opp in (AllC, AllD, Grim, TFT):
        m = EMAForgiveMachine()
        r = play_match(m, opp())
        print(opp().name, r.score_a, r.score_b)
