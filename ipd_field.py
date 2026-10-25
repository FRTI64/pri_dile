#!/usr/bin/env python3
"""Experiment driver: evaluate candidate strategies against synthetic fields."""
from __future__ import annotations

import argparse
import random
from collections import Counter
from typing import Callable

from ipd_sim import (
    C, D, Strategy, payoff,
    AllC, AllD, TFT, TFTLastD, SuspiciousTFT, Grim, GTFT, TFT2,
    Pavlov, RandomMoves, Joss, Prober, Tester, FirmButFair, Mistrust,
    Bully, WinStayLoseShift2,
)


class TFTProbe(Strategy):
    """TFT, with an optional one-shot probe of an always-cooperating opponent.

    Before the probe, play TFT.  At round `probe_round` (zero based), if the
    opponent has never defected, defect once.  If it still cooperates, keep
    defecting until it retaliates; then apologize for two rounds and return
    to TFT.  If the probe is immediately punished, apologize and return to
    TFT.
    """

    def __init__(self, probe_round=60, recover=2, last_d=True, rng=None):
        self.probe_round = probe_round
        self.recover = recover
        self.last_d = last_d
        super().__init__(rng)
        self.name = f"TFTProbe(p={probe_round},r={recover})"

    def reset(self):
        self.recovering = 0
        self.hunting = False
        self.probed = False

    def act(self, t, mine, theirs):
        if self.last_d and t == 99:
            return D
        if self.recovering:
            self.recovering -= 1
            return C

        # The opponent has never defected: probe or exploit.
        if not theirs or D not in theirs:
            if self.probed:
                # Continue exploiting an opponent that still cooperates.
                return D
            if t == self.probe_round:
                self.probed = True
                return D
            return C

        # A defection was seen while exploiting: stop and repair the relation.
        if self.probed:
            self.recovering = self.recover
            self.probed = False
            if self.recovering:
                self.recovering -= 1
                return C
            return C

        # Ordinary TFT.
        return theirs[-1]


class TFTProbeImmediate(TFTProbe):
    def __init__(self, probe_round=0, recover=2, last_d=True, rng=None):
        super().__init__(probe_round, recover, last_d, rng)
        self.name = f"TFTProbeFirst(r={recover})"


class AdaptiveProbe(Strategy):
    """A small model-based extension of TFTProbe.

    - Exploits any opponent that cooperates after being tested.
    - Repairs after accidental mutual-defection spirals.
    - Defects on the last day.
    """

    def __init__(self, probe_round=60, recover=2, last_d=True, rng=None):
        self.probe_round = probe_round
        self.recover = recover
        self.last_d = last_d
        super().__init__(rng)
        self.name = f"Adaptive(p={probe_round})"

    def reset(self):
        self.mode = "tft"       # tft, probe, hunt, repair
        self.repair_left = 0

    def act(self, t, mine, theirs):
        if self.last_d and t == 99:
            return D

        if self.mode == "repair":
            self.repair_left -= 1
            if self.repair_left <= 0:
                self.mode = "tft"
            return C

        # If we ever get locked into DD, make a contrite move once.
        if self.mode == "tft" and len(theirs) >= 2 and mine[-1] == D and theirs[-1] == D:
            self.mode = "repair"
            self.repair_left = self.recover - 1
            if self.repair_left <= 0:
                self.mode = "tft"
            return C

        if self.mode == "hunt":
            if theirs[-1] == C:
                return D
            # They finally retaliated.  Repair.
            self.mode = "repair"
            self.repair_left = self.recover
            self.repair_left -= 1
            if self.repair_left <= 0:
                self.mode = "tft"
            return C

        if self.mode == "probe":
            # We defected last round.  If they still cooperated, hunt.
            if theirs[-1] == C:
                self.mode = "hunt"
                return D
            self.mode = "repair"
            self.repair_left = self.recover
            self.repair_left -= 1
            if self.repair_left <= 0:
                self.mode = "tft"
            return C

        # tft mode
        if t == self.probe_round and D not in theirs:
            self.mode = "probe"
            return D
        if t == 0:
            return C
        return theirs[-1]


class GenerousTFT(Strategy):
    def __init__(self, g=0.1, last_d=False, rng=None):
        self.g = g
        self.last_d = last_d
        super().__init__(rng)
        self.name = f"GTFT(g={g:g}{',lastD' if last_d else ''})"

    def act(self, t, mine, theirs):
        if self.last_d and t == 99:
            return D
        if t == 0:
            return C
        if theirs[-1] == C:
            return C
        return C if self.rng.random() < self.g else D


class ContriteTFT(Strategy):
    name = "ContriteTFT"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if mine[-1] == D and theirs[-1] == D:
            return C
        return theirs[-1]


class Alternator(Strategy):
    name = "Alternator"

    def act(self, t, mine, theirs):
        return C if t % 2 == 0 else D


class LastKDefect(Strategy):
    def __init__(self, k=1, rng=None):
        self.k = k
        super().__init__(rng)
        self.name = f"TFT-last{self.k}D"

    def act(self, t, mine, theirs):
        if t >= 100 - self.k:
            return D
        return C if t == 0 else theirs[-1]


class NoisyTFT(Strategy):
    def __init__(self, p=0.05, rng=None):
        self.p = p
        super().__init__(rng)
        self.name = f"NoisyTFT({p:g})"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        x = theirs[-1]
        if self.rng.random() < self.p:
            x = 1 - x
        return x


class RandomThenTFT(Strategy):
    """Random for the first few moves to confuse an opponent, then TFT."""

    def __init__(self, k=3, p=0.5, rng=None):
        self.k = k
        self.p = p
        super().__init__(rng)
        self.name = f"Rand{k}->TFT"

    def act(self, t, mine, theirs):
        if t < self.k:
            return C if self.rng.random() < self.p else D
        return theirs[-1]


class Majority(Strategy):
    def __init__(self, k=3, rng=None):
        self.k = k
        super().__init__(rng)
        self.name = f"Majority{k}"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        recent = theirs[-self.k:]
        return D if recent.count(D) > len(recent) / 2 else C


class TFTForgiveOnce(Strategy):
    """Retaliate once, then cooperate even if the opponent is still hostile."""

    name = "TFTForgiveOnce"

    def reset(self):
        self.last_retaliated = False

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if theirs[-1] == D and not self.last_retaliated:
            self.last_retaliated = True
            return D
        self.last_retaliated = False
        return C


class GrimForgive(Strategy):
    def __init__(self, g=0.05, rng=None):
        self.g = g
        super().__init__(rng)
        self.name = f"GrimForgive({g:g})"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if theirs[-1] == D:
            return C if self.rng.random() < self.g else D
        return C if D not in theirs else D


class TFT2LastD(TFT2):
    name = "TFT2-lastD"

    def act(self, t, mine, theirs):
        return D if t == 99 else super().act(t, mine, theirs)


class ForgiveTFT(Strategy):
    """TFT that ignores isolated unprovoked defections.

    After an opponent defection it cooperates once if the opponent has not
    been defecting repeatedly.  A second consecutive defection is punished.
    A short, capped repair phase tries to escape mutual-defection locks.
    """

    def __init__(self, last_d=True, missed_limit=1, rng=None):
        self.last_d = last_d
        self.missed_limit = missed_limit
        super().__init__(rng)
        self.name = f"ForgiveTFT(m={missed_limit})"

    def reset(self):
        self.missed = 0
        self.repair_left = 0

    def act(self, t, mine, theirs):
        if self.last_d and t == 99:
            return D
        if t == 0:
            return C
        if self.repair_left:
            self.repair_left -= 1
            return C
        if theirs[-1] == C:
            self.missed = 0
            return C
        # opponent just defected
        if mine[-1] == C and self.missed < self.missed_limit:
            self.missed += 1
            return C
        # punish.  If the previous round was already DD, make one attempt to
        # restore cooperation after a short punishment.
        if mine[-1] == D and theirs[-1] == D:
            self.repair_left = 1
            return C
        return D


class TwoStrike(Strategy):
    """Cooperate until the opponent's second defection, then retaliate once."""

    def __init__(self, strikes=2, last_d=True, rng=None):
        self.strikes = strikes
        self.last_d = last_d
        super().__init__(rng)
        self.name = f"TwoStrike(s={strikes})"

    def reset(self):
        self.repair_left = 0
        self.punishing = 0

    def act(self, t, mine, theirs):
        if self.last_d and t == 99:
            return D
        if t == 0:
            return C
        if self.repair_left:
            self.repair_left -= 1
            return C
        if theirs[-1] == D and theirs.count(D) >= self.strikes:
            # retaliate, but not forever: after DD, offer one repair
            if mine[-1] == D:
                self.repair_left = 1
                return C
            return D
        return C


class Responsive(Strategy):
    """Cooperate with responsive opponents, exploit non-responsive ones.

    We estimate
        pC = P(opponent defects now | we cooperated last round)
        pD = P(opponent defects now | we defected last round)
    A large pD-pC means the opponent is reciprocating, so cooperation is
    worthwhile.  If both are similar and the opponent defects often enough,
    our action has little influence; then always defecting is profitable.
    """

    def __init__(self, g=0.2, min_c=4, min_d=3, resp=0.30,
                 rate=0.18, last_d=True, max_tests=2, rng=None):
        self.g = g
        self.min_c = min_c
        self.min_d = min_d
        self.resp = resp
        self.rate = rate
        self.last_d = last_d
        self.max_tests = max_tests
        super().__init__(rng)
        self.name = "Responsive"

    def reset(self):
        self.mode = "nice"   # nice, punish, test
        self.tests = 0

    def stats(self, mine, theirs):
        nC = nD = dC = dD = 0
        for i in range(1, len(theirs)):
            if mine[i - 1] == C:
                nC += 1
                dC += theirs[i] == D
            else:
                nD += 1
                dD += theirs[i] == D
        return nC, nD, dC, dD

    def act(self, t, mine, theirs):
        if self.last_d and t == 99:
            return D
        if t == 0:
            return C

        nC, nD, dC, dD = self.stats(mine, theirs)
        pC = dC / nC if nC else 0.0
        pD = dD / nD if nD else 0.0
        rate = sum(theirs) / len(theirs)

        if self.mode == "test":
            # Last round we cooperated after a punishment phase.  This is
            # the opponent's response to that peace offer.
            if theirs[-1] == C:
                self.mode = "nice"
                return C
            self.mode = "punish"
            return D

        if self.mode == "nice":
            enough = nD >= self.min_d and nC >= self.min_c
            if enough and rate >= self.rate and (pD - pC) < self.resp:
                self.mode = "punish"
                return D
            if theirs[-1] == D:
                # Forgive an occasional defect.  The first (usually
                # accidental) defection is forgiven almost always.
                gg = 0.9 if len(theirs) <= 2 else self.g
                return C if self.rng.random() < gg else D
            return C

        # Punishment / exploitation mode.  If the opponent has started to
        # respond strongly to our defections, make one peace offer.
        if (self.tests < self.max_tests and nC >= self.min_c and nD >= 2
                and pC <= 0.30 and pD >= 0.65 and pD - pC >= 0.45):
            self.tests += 1
            self.mode = "test"
            return C
        return D


class ResponsiveProbe(Responsive):
    """Responsive with one carefully timed test of an all-cooperator."""

    def __init__(self, g=0.2, probe=96, rng=None, **kw):
        self.probe = probe
        super().__init__(g=g, rng=rng, **kw)
        self.name = f"ResponsiveProbe(p={probe})"

    def reset(self):
        super().reset()
        self.probed = False

    def act(self, t, mine, theirs):
        if (t == self.probe and not self.probed and self.mode == "nice"
                and D not in theirs):
            self.probed = True
            self.mode = "probe"
            return D
        if getattr(self, "mode", "") == "probe":
            if theirs[-1] == C:
                self.mode = "punish"
                return D
            self.mode = "test"
            return C
        return super().act(t, mine, theirs)


class Responsive2(Responsive):
    """Responsive variant with strong forgiveness of an isolated defect.

    An unprovoked defection is almost always forgiven immediately, which is
    important against noisy reciprocators.  Repeated defections switch the
    policy into exploitation mode.
    """

    def __init__(self, g_unprov=1.0, g_mutual=0.10, strike=2,
                 min_c=4, min_d=2, resp=0.35, last_d=True,
                 max_tests=3, rng=None):
        self.g_unprov = g_unprov
        self.g_mutual = g_mutual
        self.strike = strike
        super().__init__(g=g_mutual, min_c=min_c, min_d=min_d,
                         resp=resp, last_d=last_d,
                         max_tests=max_tests, rng=rng)
        self.name = f"Resp2(u={g_unprov:g},m={g_mutual:g})"

    def act(self, t, mine, theirs):
        if self.last_d and t == 99:
            return D
        if t == 0:
            return C

        nC, nD, dC, dD = self.stats(mine, theirs)
        pC = dC / nC if nC else 0.0
        pD = dD / nD if nD else 0.0

        if self.mode == "test":
            if theirs[-1] == C:
                self.mode = "nice"
                return C
            self.mode = "punish"
            return D

        if self.mode == "nice":
            # Repeated unprovoked defections: exploit.
            run = 0
            for x in reversed(theirs):
                if x == D:
                    run += 1
                else:
                    break
            if run >= self.strike and mine[-1] == C:
                self.mode = "punish"
                return D
            # Strong evidence of a non-responsive defector.
            if nC >= 8 and sum(theirs) / len(theirs) >= 0.28:
                self.mode = "punish"
                return D
            if theirs[-1] == D:
                if mine[-1] == C:
                    if self.rng.random() < self.g_unprov:
                        return C
                    self.mode = "punish"
                    return D
                return C if self.rng.random() < self.g_mutual else D
            return C

        # Punish.  A peace offer only if the opponent now looks like a
        # reciprocator (defects a lot when we defect, but not when we
        # cooperate).
        if (self.tests < self.max_tests and nC >= self.min_c and nD >= 2
                and pC <= 0.35 and pD >= 0.65 and pD - pC >= 0.40):
            self.tests += 1
            self.mode = "test"
            return C
        return D


class Responsive3(Responsive):
    """Responsive with adaptive forgiveness.

    Once the opponent has shown that it retaliates when we defect (pD high)
    but cooperates when we cooperate (pC low), a defection is forgiven almost
    immediately.  If instead pD and pC are similar, the opponent is not
    responsive and is exploited.
    """

    def __init__(self, g_unprov=0.55, g_mutual=0.40, g_resp=0.95,
                 min_c=5, min_d=3, resp=0.38, rate=0.18,
                 last_d=True, max_tests=2, cooldown=8, rng=None):
        self.g_unprov = g_unprov
        self.g_mutual = g_mutual
        self.g_resp = g_resp
        self.cooldown = cooldown
        super().__init__(g=g_unprov, min_c=min_c, min_d=min_d,
                         resp=resp, last_d=last_d,
                         max_tests=max_tests, rng=rng)
        self.name = "Resp3"

    def reset(self):
        super().reset()
        self.last_test = -100
        self.pattern_mode = False

    def act(self, t, mine, theirs):
        if self.last_d and t == 99:
            return D
        if t == 0:
            return C

        nC, nD, dC, dD = self.stats(mine, theirs)
        pC = dC / nC if nC else 0.0
        pD = dD / nD if nD else 0.0
        rate = sum(theirs) / len(theirs)

        # A long strictly alternating history is an autonomous pattern.
        # Use a run of 10 to avoid false positives from a noisy reciprocator.
        if self.pattern_mode:
            if (len(theirs) < 10 or not all(theirs[i] != theirs[i - 1]
                    for i in range(len(theirs) - 9, len(theirs)))):
                self.pattern_mode = False
            else:
                return D
        elif len(theirs) >= 10 and all(theirs[i] != theirs[i - 1]
                for i in range(len(theirs) - 9, len(theirs))):
            self.pattern_mode = True
            return D

        if self.mode == "test1":
            # First peace move was last round.  Offer a second one, because
            # TFT-like opponents are still reacting to our earlier D.
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
            self.last_test = t
            return D

        if self.mode == "nice":
            enough = nC >= self.min_c and nD >= self.min_d
            responsive = (enough and pC <= 0.35 and pD >= 0.65
                          and pD - pC >= self.resp)
            nonresponsive = (enough and (pD - pC) < 0.35
                             and rate >= self.rate)
            if nonresponsive:
                self.mode = "punish"
                return D
            if theirs[-1] == D:
                if mine[-1] == C:
                    g = self.g_resp if responsive else self.g_unprov
                else:
                    g = self.g_resp if responsive else self.g_mutual
                return C if self.rng.random() < g else D
            return C

        # Punish mode.  Make a two-round peace offer only when fresh evidence
        # says that the opponent reciprocates.
        if (self.tests < self.max_tests and t - self.last_test >= self.cooldown
                and nC >= self.min_c and nD >= 2
                and pC <= 0.35 and pD >= 0.65 and pD - pC >= 0.40):
            self.tests += 1
            self.last_test = t
            self.mode = "test1"
            return C
        return D


class ProbeBot(Responsive3):
    """One early probe, then a type-specific response.

    The first two rounds are used to separate unconditional cooperators and
    TFT2-like opponents from ordinary reciprocators.  Against the former we
    play a best-response sequence; against everybody else the adaptive
    Responsive3 policy takes over.
    """

    def __init__(self, rng=None, **kw):
        super().__init__(g_unprov=0.4, g_mutual=0.3, g_resp=0.9, rng=rng, **kw)
        self.name = "ProbeBot"

    def reset(self):
        super().reset()
        self.prefix = "start"
        self.hostile_rounds = 0
        self.alt_phase = None

    def _normal(self, t, mine, theirs, first=C):
        self.prefix = "done"
        self.mode = "nice"
        return first

    def act(self, t, mine, theirs):
        if t == 99:
            return D

        p = self.prefix
        if p == "start":
            self.prefix = "after0"
            return D
        if p == "after0":
            # Response to the initial D is known.
            if theirs[0] == D:
                self.prefix = "hostile"
                self.hostile_rounds = 0
                return C
            self.prefix = "wait1"
            return C
        if p == "hostile":
            self.hostile_rounds += 1
            if self.hostile_rounds <= 2:
                return C
            # Two peace offers were enough to distinguish sTFT from AllD.
            if theirs[-1] == D and theirs[-2] == D:
                self.prefix = "done"
                self.mode = "punish"
                return D
            return self._normal(t, mine, theirs, C)
        if p == "wait1":
            if theirs[1] == D:
                self.prefix = "recip_test"
                return C
            self.prefix = "double1"
            return D
        if p == "recip_test":
            if theirs[2] == D:
                self.prefix = "done"
                self.mode = "punish"
                return D
            return self._normal(t, mine, theirs, C)
        if p == "double1":
            self.prefix = "double2"
            return D
        if p == "double2":
            if theirs[3] == D:
                # TFT2-like: alternate D,C so that it always sees at most one
                # of our defections in its two-round memory.
                self.prefix = "alt"
                self.alt_phase = t
                return C
            # Unconditional cooperator.
            self.prefix = "hunt"
            return D
        if p == "hunt":
            if theirs[-1] == D:
                return self._normal(t, mine, theirs, C)
            return D
        if p == "alt":
            if theirs[-1] == D:
                return self._normal(t, mine, theirs, C)
            return D if (t - self.alt_phase) % 2 == 1 else C

        return super().act(t, mine, theirs)


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


class DelayedTFT(Strategy):
    def __init__(self, delay=2, rng=None):
        self.delay = delay
        super().__init__(rng)
        self.name = f"TFT-delay{delay}"

    def act(self, t, mine, theirs):
        if t < self.delay:
            return C
        return theirs[-self.delay]


class RemorsefulProber(Strategy):
    """Defect once, apologise, then play TFT; exploit if no retaliation."""

    def reset(self):
        self.mode = "probe"

    name = "RemorsefulProber"

    def act(self, t, mine, theirs):
        if t == 0:
            self.mode = "probe"
            return D
        if self.mode == "probe":
            if theirs[0] == D:
                self.mode = "tft"
                return C
            self.mode = "exploit"
            return D
        if self.mode == "exploit":
            if theirs[-1] == D:
                self.mode = "tft"
                return C
            return D
        return theirs[-1]


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


class TFTNoisyForgive(Strategy):
    """TFT, but after a defection cooperate with probability 1/3."""

    def __init__(self, g=1/3, rng=None):
        self.g = g
        super().__init__(rng)
        self.name = f"GTFT({g:.2f})"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if theirs[-1] == C:
            return C
        return C if self.rng.random() < self.g else D


# Factories: (weight, name, factory)
FIELD_SPECS: list[tuple[float, str, Callable]] = [
    (7.0, "AllC", lambda r: AllC(r)),
    (7.0, "AllD", lambda r: AllD(r)),
    (18.0, "TFT", lambda r: TFT(r)),
    (7.0, "TFT-lastD", lambda r: TFTLastD(r)),
    (4.0, "sTFT", lambda r: SuspiciousTFT(r)),
    (7.0, "Grim", lambda r: Grim(r)),
    (6.0, "GTFT.05", lambda r: GTFT(0.05, r)),
    (4.0, "GTFT.15", lambda r: GTFT(0.15, r)),
    (5.0, "TFT2", lambda r: TFT2(r)),
    (7.0, "Pavlov", lambda r: Pavlov(r)),
    (4.0, "Random.5", lambda r: RandomMoves(0.5, r)),
    (3.0, "Random.2", lambda r: RandomMoves(0.2, r)),
    (3.0, "Joss.05", lambda r: Joss(0.05, r)),
    (2.0, "Joss.15", lambda r: Joss(0.15, r)),
    (2.0, "Prober", lambda r: Prober(r)),
    (2.0, "Tester", lambda r: Tester(r)),
    (2.0, "FirmButFair", lambda r: FirmButFair(r)),
    (3.0, "Mistrust", lambda r: Mistrust(r)),
    (2.0, "Bully", lambda r: Bully(r)),
    (2.0, "WSLS2", lambda r: WinStayLoseShift2(r)),
    (2.0, "ContriteTFT", lambda r: ContriteTFT(r)),
    (1.0, "Alternator", lambda r: Alternator(r)),
    (2.0, "TFT-last2D", lambda r: LastKDefect(2, r)),
    (2.0, "NoisyTFT", lambda r: NoisyTFT(0.05, r)),
    (2.0, "Rand3->TFT", lambda r: RandomThenTFT(3, 0.5, r)),
    (2.0, "Majority3", lambda r: Majority(3, r)),
    (2.0, "TFTForgiveOnce", lambda r: TFTForgiveOnce(r)),
    (1.0, "GrimForgive", lambda r: GrimForgive(0.05, r)),
    (2.0, "LateD10", lambda r: LateDefector(10, r)),
    (2.0, "LateD25", lambda r: LateDefector(25, r)),
    (1.0, "EarlyD10", lambda r: EarlyDefector(10, r)),
    (1.0, "Gradual", lambda r: Gradual(r)),
    (2.0, "TFT-delay2", lambda r: DelayedTFT(2, r)),
    (1.0, "RemorsefulProber", lambda r: RemorsefulProber(r)),
    (1.0, "Grudger2", lambda r: Grudger2(r)),
    (1.0, "Random.1", lambda r: RandomP(0.1, r)),
    (1.0, "Random.8", lambda r: RandomP(0.8, r)),
    (1.0, "GTFT.33", lambda r: TFTNoisyForgive(1/3, r)),
]

WEIGHT_TOTAL = sum(w for w, _, _ in FIELD_SPECS)


def make_field(n: int, rng: random.Random) -> list[Strategy]:
    specs = FIELD_SPECS
    weights = [w for w, _, _ in specs]
    out = []
    for _ in range(n):
        _, _, f = rng.choices(specs, weights=weights, k=1)[0]
        out.append(f(rng))
    return out


def score_against(factory: Callable, field: list[Strategy], rounds=100) -> tuple[int, dict]:
    me = factory(random.Random(1234567))
    by_name: dict[str, list[int]] = {}
    total = 0
    for opp in field:
        me.reset()
        opp.reset()
        mh: list[int] = []
        oh: list[int] = []
        s = 0
        for t in range(rounds):
            x = me.act(t, mh, oh)
            y = opp.act(t, oh, mh)
            mh.append(x)
            oh.append(y)
            s += payoff(x, y)
        total += s
        by_name.setdefault(opp.name, []).append(s)
    return total, {k: sum(v) / len(v) for k, v in by_name.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--fields", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=100)
    args = ap.parse_args()

    candidates = [
        ("TFT", lambda r: TFT(r)),
        ("TFT-lastD", lambda r: TFTLastD(r)),
        ("TFT2", lambda r: TFT2(r)),
        ("TFT2-lastD", lambda r: TFT2LastD(r)),
        ("TwoStrike2", lambda r: TwoStrike(2, True, r)),
        ("TwoStrike3", lambda r: TwoStrike(3, True, r)),
        ("ForgiveTFT1", lambda r: ForgiveTFT(True, 1, r)),
        ("ForgiveTFT2", lambda r: ForgiveTFT(True, 2, r)),
        ("GTFT.05-lastD", lambda r: GenerousTFT(0.05, True, r)),
        ("GTFT.1-lastD", lambda r: GenerousTFT(0.1, True, r)),
        ("GTFT.2-lastD", lambda r: GenerousTFT(0.2, True, r)),
        ("Responsive", lambda r: Responsive(g=0.4, rng=r)),
        ("Resp3", lambda r: Responsive3(rng=r)),
        ("ProbeBot", lambda r: ProbeBot(rng=r)),
        ("Resp2-u1", lambda r: Responsive2(1.0, 0.10, rng=r)),
        ("Resp2-u.85", lambda r: Responsive2(0.85, 0.10, rng=r)),
        ("Resp2-u.7", lambda r: Responsive2(0.7, 0.10, rng=r)),
        ("ContriteTFT", lambda r: ContriteTFT(r)),
        ("Probe-first", lambda r: TFTProbe(0, 2, True, r)),
        ("Probe-10", lambda r: TFTProbe(10, 2, True, r)),
        ("Probe-20", lambda r: TFTProbe(20, 2, True, r)),
        ("Probe-30", lambda r: TFTProbe(30, 2, True, r)),
        ("Probe-40", lambda r: TFTProbe(40, 2, True, r)),
        ("Probe-50", lambda r: TFTProbe(50, 2, True, r)),
        ("Probe-60", lambda r: TFTProbe(60, 2, True, r)),
        ("Probe-70", lambda r: TFTProbe(70, 2, True, r)),
        ("Probe-80", lambda r: TFTProbe(80, 2, True, r)),
        ("Probe-90", lambda r: TFTProbe(90, 2, True, r)),
        ("Probe-95", lambda r: TFTProbe(95, 2, True, r)),
        ("Adaptive-60", lambda r: AdaptiveProbe(60, 2, True, r)),
    ]

    master = random.Random(args.seed)
    totals = {name: [] for name, _ in candidates}
    per_type: dict[str, dict[str, list[int]]] = {name: {} for name, _ in candidates}
    for fi in range(args.fields):
        field = make_field(args.n, random.Random(master.random()))
        for name, fac in candidates:
            total, by = score_against(fac, field, args.rounds)
            totals[name].append(total)
            for k, v in by.items():
                per_type[name].setdefault(k, []).append(v)
        print(f"field {fi+1}/{args.fields} done", flush=True)

    print("\n=== mean totals over fields ===")
    for name, vals in sorted(totals.items(), key=lambda kv: sum(kv[1])/len(kv[1]), reverse=True):
        mean = sum(vals) / len(vals)
        lo, hi = min(vals), max(vals)
        print(f"{name:18} mean={mean:8.1f}  min={lo:5d} max={hi:5d}  avg/opp={mean/args.n:6.2f}")

    print("\n=== per-opponent-type averages (best few) ===")
    best_names = sorted(totals, key=lambda n: sum(totals[n]) / len(totals[n]), reverse=True)[:5]
    types = sorted(per_type[best_names[0]])
    print(f"{'type':18}" + "".join(f"{n:>14}" for n in best_names))
    for typ in types:
        row = f"{typ:18}"
        for n in best_names:
            vals = per_type[n].get(typ, [])
            row += f"{(sum(vals)/len(vals) if vals else float('nan')):14.1f}"
        print(row)


if __name__ == "__main__":
    main()
