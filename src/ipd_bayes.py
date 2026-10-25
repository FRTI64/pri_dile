#!/usr/bin/env python3
"""Model-based opponent inference without extra proactive probing.

A bank of candidate opponent models is maintained and updated in a Bayesian
way.  The machine never defects before the opponent's first defection (except
on day 100), so this is a pure "more factors + inference" experiment.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from ipd_strategy import Machine, C, D
from ipd_sim import payoff


# Each model returns P(opponent defects now) given the history before now.
def p_allc(t, mine, theirs):
    return 0.0


def p_alld(t, mine, theirs):
    return 1.0


def p_tft(t, mine, theirs):
    return 0.0 if t == 0 else float(mine[t - 1] == D)


def p_stft(t, mine, theirs):
    return 1.0 if t == 0 else float(mine[t - 1] == D)


def p_grim(t, mine, theirs):
    return float(D in mine[:t])


def p_tft2(t, mine, theirs):
    if t == 0:
        return 0.0
    return float(t >= 2 and mine[t - 1] == D and mine[t - 2] == D)


def p_gtft(t, mine, theirs, g=0.1):
    if t == 0:
        return 0.0
    if mine[t - 1] == C:
        return 0.0
    return 1.0 - g


def p_joss(t, mine, theirs, p=0.05):
    if t == 0:
        return 0.0
    if mine[t - 1] == D:
        return 1.0
    return p


def p_random(t, mine, theirs, p=0.5):
    return p


def p_alternator(t, mine, theirs):
    return float(t % 2 == 1)


def p_pavlov(t, mine, theirs):
    if t == 0:
        return 0.0
    last_opp = theirs[t - 1]
    last_mine = mine[t - 1]
    opp_payoff = payoff(last_opp, last_mine)
    return float(last_opp if opp_payoff >= 3 else 1 - last_opp)


def p_lated(t, mine, theirs, k=10):
    return float(t >= 100 - k)


class BayesMachine(Machine):
    name = "Bayes"
    NONRESP_THRESHOLD = 0.40
    PUNITIVE_THRESHOLD = 0.55

    def reset(self):
        super().reset()
        # Prior for a late-harvest-heavy field.  AllC/TFT/GTFT/late and a few
        # noise/punitive types.  The prior only matters before the first D.
        self.models = []
        self._add("AllC", 8.0, p_allc)
        self._add("AllD", 4.0, p_alld)
        self._add("TFT", 18.0, p_tft)
        self._add("sTFT", 3.0, p_stft)
        self._add("Grim", 5.0, p_grim)
        self._add("TFT2", 8.0, p_tft2)
        self._add("GTFT.1", 8.0, lambda t, m, o: p_gtft(t, m, o, .1))
        self._add("GTFT.25", 3.0, lambda t, m, o: p_gtft(t, m, o, .25))
        self._add("Joss.05", 4.0, lambda t, m, o: p_joss(t, m, o, .05))
        self._add("Joss.15", 2.0, lambda t, m, o: p_joss(t, m, o, .15))
        self._add("Random.2", 3.0, lambda t, m, o: p_random(t, m, o, .2))
        self._add("Random.5", 3.0, lambda t, m, o: p_random(t, m, o, .5))
        self._add("Random.8", 1.0, lambda t, m, o: p_random(t, m, o, .8))
        self._add("Pavlov", 4.0, p_pavlov)
        self._add("Alternator", 1.0, p_alternator)
        for k in (5, 10, 15, 20, 25, 30, 40, 50):
            self._add(f"LateD{k}", 3.0, lambda t, m, o, k=k: p_lated(t, m, o, k))
        self._names = [x[0] for x in self.models]
        self._priors = [x[1] for x in self.models]
        self._fns = [x[2] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0

    def _add(self, name, prior, fn):
        self.models.append((name, prior, fn))

    def _posterior(self, mine, theirs):
        # Incremental update: add likelihoods for newly completed rounds only.
        for t in range(self._updated, len(theirs)):
            y = theirs[t]
            for i, fn in enumerate(self._fns):
                p = max(1e-6, min(1 - 1e-6, fn(t, mine, theirs)))
                self._logw[i] += math.log(p if y == D else 1.0 - p)
        self._updated = len(theirs)
        mx = max(self._logw)
        ws = [math.exp(x - mx) for x in self._logw]
        z = sum(ws)
        return {name: w / z for name, w in zip(self._names, ws)}

    def act(self, t, mine, theirs):
        return self.move(t + 1, mine, theirs)

    def move(self, day, mine, theirs):
        if day >= 100:
            return D
        if day == 1:
            return C
        # No proactive probe: with no observed D, stay cooperative.
        if D not in theirs:
            return C

        post = self._posterior(mine, theirs)
        resp = 0.0
        nonresp = 0.0
        punitive = 0.0
        for name, p in post.items():
            if name.startswith(("TFT", "sTFT", "GTFT", "Joss",
                                "NoisyTFT", "Pavlov")):
                resp += p
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                punitive += p
            else:
                nonresp += p

        # If the opponent looks like a one-off noisy reciprocator, forgive.
        # If it looks like a schedule/random/punitive type, defect.
        if nonresp >= self.NONRESP_THRESHOLD or punitive >= self.PUNITIVE_THRESHOLD:
            return D
        if resp >= 0.50:
            return C
        # Ambiguous: use the last move.  A second consecutive D is a strong
        # signal; otherwise make a cooperative test.
        if theirs[-1] == D and len(theirs) >= 2 and theirs[-2] == D:
            return D
        return C


if __name__ == "__main__":
    from ipd_sim import AllC, AllD, Grim, TFT, play_match
    for opp in (AllC, AllD, Grim, TFT):
        m = BayesMachine()
        r = play_match(m, opp())
        print(opp().name, r.score_a, r.score_b)
