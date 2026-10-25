#!/usr/bin/env python3
"""Experimental EMA + endgame-threshold variant.

This keeps the no-early-probe philosophy: it only reacts to observed
defections, but recent observations get more weight and the classification
threshold becomes more aggressive near the end.
"""
from __future__ import annotations

import random

from ipd.core.machine import Machine, C, D


class EMAMachine(Machine):
    name = "EMA"
    HALF_LIFE = 12.0

    def reset(self):
        super().reset()

    def act(self, t, mine, theirs):
        return self.move(t + 1, mine, theirs)

    def _ema_stats(self, mine, theirs):
        n = len(theirs)
        wc = wd = dc = dd = dr = 0.0
        for i in range(n):
            w = 0.5 ** ((n - 1 - i) / self.HALF_LIFE)
            dr += w * (theirs[i] == D)
            if i == 0:
                continue
            if mine[i - 1] == C:
                wc += w
                dc += w * (theirs[i] == D)
            else:
                wd += w
                dd += w * (theirs[i] == D)
        total_w = sum(0.5 ** ((n - 1 - i) / self.HALF_LIFE)
                      for i in range(n))
        return wc, wd, dc, dd, dr / total_w if total_w else 0.0

    def move(self, day, mine, theirs):
        if day >= 100:
            return D
        if day == 1:
            return C

        n = len(theirs)
        wc, wd, dc, dd, rate = self._ema_stats(mine, theirs)
        p_c = dc / wc if wc else 0.0
        p_d = dd / wd if wd else 0.0
        remaining = 100 - day  # rounds after the current one

        # Effective sample sizes: the decay keeps older evidence relevant but
        # late observations count much more.
        enough_c = wc >= 3.0
        enough_d = wd >= 1.5
        # Near the end, require less evidence.
        if remaining <= 10:
            enough_c = wc >= 1.5
            enough_d = wd >= 1.0

        responsive = (enough_c and enough_d and p_c <= 0.35
                      and p_d >= 0.65 and p_d - p_c >= 0.38)
        # Dynamic non-responsive threshold: stricter (easier to trigger)
        # when little time remains.
        if remaining > 30:
            rate_thr = 0.18
        elif remaining > 15:
            rate_thr = 0.14
        elif remaining > 8:
            rate_thr = 0.10
        else:
            rate_thr = 0.07
        nonresponsive = (enough_c and enough_d and p_d - p_c < 0.35
                         and rate >= rate_thr)

        # Pattern detector (same as the base strategy).
        if self.pattern:
            if self._alternating(theirs):
                return D
            self.pattern = False
        elif self._alternating(theirs):
            self.pattern = True
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
            self.last_test = day
            return D

        if self.mode == "nice":
            if nonresponsive:
                self.mode = "punish"
                return D
            if theirs[-1] == C:
                return C
            if mine[-1] == C:
                p_forgive = self.P_RESPONSIVE if responsive else self.P_UNPROVOKED
            else:
                p_forgive = self.P_RESPONSIVE if responsive else self.P_MUTUAL
            return C if self.rng.random() < p_forgive else D

        if (self.tests < self.PEACE_TESTS
                and day - self.last_test >= self.PEACE_COOLDOWN
                and enough_c and enough_d and p_c <= 0.35 and p_d >= 0.65
                and p_d - p_c >= 0.40):
            self.tests += 1
            self.last_test = day
            self.mode = "test1"
            return C
        return D


if __name__ == "__main__":
    from ipd.core.simulator import AllC, AllD, Grim, TFT, play_match
    for opp in (AllC, AllD, Grim, TFT):
        m = EMAMachine()
        r = play_match(m, opp())
        print(opp().name, r.score_a, r.score_b)
