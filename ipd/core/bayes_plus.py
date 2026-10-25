#!/usr/bin/env python3
"""Bayesian inference with an expanded model bank and richer features.

Adds, relative to ipd/core/bayes.py:
- all possible onset days for a late harvester (LateD k=1..99);
- TFT with 1/2/3-day delays and last-day defection;
- several GTFT and NoisyTFT noise levels;
- Grudger2/3, Mistrust, Periodic3/4, Tester/Prober;
- exact policy grouping by model family.

It still does not probe: before the opponent's first D it cooperates.
"""
from __future__ import annotations

import math

from ipd.core.bayes import (
    BayesMachine, p_allc, p_alld, p_tft, p_stft, p_grim, p_tft2,
    p_gtft, p_joss, p_random, p_alternator, p_pavlov, p_lated,
)
from ipd.core.simulator import C, D


def p_noisy_tft(t, mine, theirs, p=0.05):
    if t == 0:
        return 0.0
    base = mine[t - 1]
    return p if base == C else 1.0 - p


def p_tft_delay(t, mine, theirs, delay=2):
    if t < delay:
        return 0.0
    return float(mine[t - delay] == D)


def p_tft_lastd(t, mine, theirs, k=1):
    if t >= 100 - k:
        return 1.0
    return p_tft(t, mine, theirs)


def p_grudger(t, mine, theirs, k=1):
    return float(mine[:t].count(D) >= k)


def p_periodic(t, mine, theirs, pattern=(C, C, D)):
    return float(pattern[t % len(pattern)] == D)


def p_earlyd(t, mine, theirs, k=10):
    return float(t < k)


def p_allc_lastd(t, mine, theirs):
    return float(t >= 99)


def p_tester(t, mine, theirs):
    if t == 0:
        return 1.0
    if mine[0] == D:
        return float(mine[t - 1] == D)
    return float((t - 1) % 4 < 2)


def p_prober(t, mine, theirs):
    if t < 3:
        return (0.0, 0.0, 1.0)[t]
    if D in mine[:3]:
        return float(mine[t - 1] == D)
    return float((t - 3) % 4 < 2)


def p_remorseful_prober(t, mine, theirs):
    if t == 0:
        return 1.0
    # Replay the opponent's state machine using our observed actions.
    mode = "tft" if mine[0] == D else "exploit"
    action = C
    for s in range(1, t + 1):
        if mode == "exploit":
            if mine[s - 1] == D:
                mode = "tft"
                action = C
            else:
                action = D
        else:
            action = mine[s - 1]
    return float(action == D)


class BayesPlusMachine(BayesMachine):
    name = "BayesPlus"

    def reset(self):
        super().reset()
        # Increase the bank without changing the "no proactive probe" rule.
        for k in range(1, 100):
            self._add(f"LateDx{k}", 0.16,
                      lambda t, m, o, k=k: p_lated(t, m, o, k))
        for k in (5, 10, 20, 33, 50):
            self._add(f"GTFTx{k}", 1.5,
                      lambda t, m, o, k=k: p_gtft(t, m, o, k / 100.0))
        for p in (0.02, 0.10, 0.20):
            self._add(f"NoisyTFTx{p}", 1.5,
                      lambda t, m, o, p=p: p_noisy_tft(t, m, o, p))
        for delay in (1, 2, 3):
            self._add(f"TFTdelay{delay}", 1.5,
                      lambda t, m, o, d=delay: p_tft_delay(t, m, o, d))
        for k in (1, 2, 3, 5, 10):
            self._add(f"TFTlastD{k}", 1.0,
                      lambda t, m, o, k=k: p_tft_lastd(t, m, o, k))
        for k in (1, 2, 3):
            self._add(f"Grudger{k}", 2.0,
                      lambda t, m, o, k=k: p_grudger(t, m, o, k))
        self._add("Mistrust", 2.0, lambda t, m, o: p_grudger(t, m, o, 2))
        self._add("Periodic3", 1.0,
                  lambda t, m, o: p_periodic(t, m, o, (C, C, D)))
        self._add("Periodic4", 1.0,
                  lambda t, m, o: p_periodic(t, m, o, (C, D, D, C)))
        self._add("AlternatorD", 1.0,
                  lambda t, m, o: p_periodic(t, m, o, (D, C)))
        self._add("Tester", 2.0, p_tester)
        self._add("Prober", 2.0, p_prober)
        self._add("RemorsefulProber", 1.5, p_remorseful_prober)
        self._add("AllC-lastD", 1.0, p_allc_lastd)
        self._add("EarlyD10", 1.0,
                  lambda t, m, o: p_earlyd(t, m, o, 10))
        self._add("EarlyD25", 0.5,
                  lambda t, m, o: p_earlyd(t, m, o, 25))
        # Rebuild the cached parallel lists after appending.
        self._names = [x[0] for x in self.models]
        self._priors = [x[1] for x in self.models]
        self._fns = [x[2] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0


if __name__ == "__main__":
    from ipd.core.simulator import AllC, AllD, Grim, TFT, play_match
    for opp in (AllC, AllD, Grim, TFT):
        m = BayesPlusMachine()
        r = play_match(m, opp())
        print(opp().name, r.score_a, r.score_b)
