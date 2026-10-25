#!/usr/bin/env python3
"""Extended opponent bank and mixture sampler for robustness tests.

The original `ipd_field.py` has 38 hand-written archetypes.  This module adds
parameterised variants, random memory-one/reactive/FSM opponents, zero-
determinant (extortion / generous) opponents, and a sampler that randomises
both the type weights and the number of copies of each type.

Only the standard library is used.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from ipd.core.simulator import (
    C, D, Strategy, payoff,
    AllC, AllD, TFT, TFTLastD, SuspiciousTFT, Grim, GTFT, TFT2,
    Pavlov, RandomMoves, Joss, Prober, Tester, FirmButFair, Mistrust,
    Bully, WinStayLoseShift2,
)
from ipd.core.opponents import (
    TFT2LastD, TFTNoisyForgive, GenerousTFT, ContriteTFT, Alternator,
    LastKDefect, NoisyTFT, RandomThenTFT, Majority, TFTForgiveOnce,
    GrimForgive, LateDefector, EarlyDefector, Gradual, DelayedTFT,
    RemorsefulProber, Grudger2, RandomP, ForgiveTFT, TwoStrike,
    Responsive, Responsive3, ProbeBot,
)

R, S, T, P = 6, 0, 10, 1


# ---------------------------------------------------------------------------
# Additional strategy implementations
# ---------------------------------------------------------------------------

class TwoTitsForTat(Strategy):
    name = "TTFT"

    def reset(self):
        self.punish = 0

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if self.punish:
            self.punish -= 1
            return D
        if theirs[-1] == D:
            self.punish = 1
            return D
        return C


class AllCWithLastD(Strategy):
    name = "AllC-lastD"

    def act(self, t, mine, theirs):
        return D if t == 99 else C


class GrudgerK(Strategy):
    def __init__(self, k=3, rng=None):
        self.k = k
        super().__init__(rng)
        self.name = f"Grudger{self.k}"

    def act(self, t, mine, theirs):
        return D if theirs.count(D) >= self.k else C


class OmegaTFTLite(Strategy):
    """TFT-like strategy with deadlock repair and alternation detection."""

    name = "OmegaTFT-lite"

    def reset(self):
        self.dead = False

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if self.dead:
            return D
        # Long alternating opponent history: exploit it.
        if (len(theirs) >= 8 and
                all(theirs[i] != theirs[i - 1]
                    for i in range(len(theirs) - 7, len(theirs)))):
            self.dead = True
            return D
        # Two consecutive DD rounds: make one cooperative repair attempt.
        if (len(theirs) >= 2 and mine[-1] == D and theirs[-1] == D and
                mine[-2] == D and theirs[-2] == D):
            return C
        return theirs[-1]


class Periodic(Strategy):
    def __init__(self, pattern=(C, D, C), start=0, rng=None):
        self.pattern = tuple(pattern)
        self.start = start
        super().__init__(rng)
        self.name = "Periodic" + "".join("C" if x == C else "D"
                                         for x in self.pattern)

    def act(self, t, mine, theirs):
        return self.pattern[(t + self.start) % len(self.pattern)]


class MemoryOne(Strategy):
    """General memory-one plan.

    p = (p_CC, p_CD, p_DC, p_DD), where the subscript is
    (my previous move, opponent previous move).
    """

    def __init__(self, p, first=C, name="MemOne", rng=None):
        self.p = tuple(float(x) for x in p)
        self.first = first
        super().__init__(rng)
        self.name = name

    def act(self, t, mine, theirs):
        if t == 0:
            return self.first
        key = (mine[-1], theirs[-1])
        q = self.p[(0 if key == (C, C) else
                    1 if key == (C, D) else
                    2 if key == (D, C) else 3)]
        return C if self.rng.random() < q else D


class Reactive(Strategy):
    def __init__(self, first=C, p_after_c=0.8, p_after_d=0.1,
                 name="Reactive", rng=None):
        self.first = first
        self.p_after_c = p_after_c
        self.p_after_d = p_after_d
        super().__init__(rng)
        self.name = name

    def act(self, t, mine, theirs):
        if t == 0:
            return self.first
        q = self.p_after_c if theirs[-1] == C else self.p_after_d
        return C if self.rng.random() < q else D


class RandomFSM(Strategy):
    """Random finite-state machine, transitions driven by opponent's last move."""

    def __init__(self, rng, k=3):
        self.k = k
        self.output = [C if rng.random() < 0.58 else D for _ in range(k)]
        self.next_c = [[rng.randrange(k) for _ in range(2)] for _ in range(k)]
        self.next_d = [[rng.randrange(k) for _ in range(2)] for _ in range(k)]
        self.state = 0
        super().__init__(rng)
        self.name = "FSM-rand"

    def reset(self):
        self.state = 0

    def act(self, t, mine, theirs):
        if t > 0:
            table = self.next_c if theirs[-1] == C else self.next_d
            self.state = table[self.state][self.rng.randrange(2)]
        return self.output[self.state]


class ExtortZD(Strategy):
    """Zero-determinant extortion strategy: s_X-P = chi*(s_Y-P)."""

    def __init__(self, chi=2.0, rng=None):
        self.chi = chi
        c = max(chi - 1.0, 1e-9)
        limits = [
            1.0 / (c * (R - P)),
            1.0 / (chi * (T - P) + P),
            1.0 / (T + c * P),
        ]
        lam = 0.95 * min(limits)
        p = (
            1.0 - lam * c * (R - P),
            1.0 - lam * (chi * (T - P) + P),
            lam * (T + c * P),
            0.0,
        )
        super().__init__(rng)
        self.p = tuple(max(0.0, min(1.0, x)) for x in p)
        self.name = f"Extort{chi:g}"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        key = (mine[-1], theirs[-1])
        q = self.p[(0 if key == (C, C) else
                    1 if key == (C, D) else
                    2 if key == (D, C) else 3)]
        return C if self.rng.random() < q else D


class GenerousZD(Strategy):
    """Zero-determinant generous strategy: s_X-R = kappa*(s_Y-R)."""

    def __init__(self, kappa=2.0, rng=None):
        self.kappa = kappa
        k = max(kappa, 1e-9)
        limits = [
            1.0 / (k * T - (k - 1.0) * R),
            1.0 / (T + (k - 1.0) * R),
            1.0 / ((k - 1.0) * (R - P)),
        ]
        lam = 0.95 * min(x for x in limits if x > 0)
        p = (
            1.0,
            1.0 - lam * (k * T - (k - 1.0) * R),
            lam * (T + (k - 1.0) * R),
            lam * (k - 1.0) * (R - P),
        )
        super().__init__(rng)
        self.p = tuple(max(0.0, min(1.0, x)) for x in p)
        self.name = f"GenZD{kappa:g}"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        key = (mine[-1], theirs[-1])
        q = self.p[(0 if key == (C, C) else
                    1 if key == (C, D) else
                    2 if key == (D, C) else 3)]
        return C if self.rng.random() < q else D


# ---------------------------------------------------------------------------
# Type bank: (name, family, base weight, factory)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypeSpec:
    name: str
    family: str
    weight: float
    factory: Callable[[random.Random], Strategy]


def _memone_wild(rng):
    if rng.random() < 0.65:
        p = (rng.uniform(0.75, 1.0), rng.uniform(0.0, 0.6),
             rng.uniform(0.4, 1.0), rng.uniform(0.0, 0.35))
    else:
        p = tuple(rng.random() for _ in range(4))
    return MemoryOne(p, first=C, name="MemOne-rand", rng=rng)


def _reactive_wild(rng):
    return Reactive(first=C if rng.random() < 0.9 else D,
                    p_after_c=rng.uniform(0.35, 1.0),
                    p_after_d=rng.uniform(0.0, 0.75),
                    name="Reactive-rand", rng=rng)


def _fsm_wild(rng):
    return RandomFSM(rng, k=rng.randrange(2, 5))


TYPE_BANK: list[TypeSpec] = [
    # Nice / reciprocal
    TypeSpec("AllC", "nice", 8, lambda r: AllC(r)),
    TypeSpec("AllC-lastD", "nice", 2, lambda r: AllCWithLastD(r)),
    TypeSpec("TFT", "nice", 22, lambda r: TFT(r)),
    TypeSpec("TFT-lastD", "nice", 7, lambda r: TFTLastD(r)),
    TypeSpec("TFT-last2D", "nice", 2, lambda r: LastKDefect(2, r)),
    TypeSpec("TFT-last5D", "nice", 1, lambda r: LastKDefect(5, r)),
    TypeSpec("sTFT", "nice", 5, lambda r: SuspiciousTFT(r)),
    TypeSpec("TFT2", "nice", 6, lambda r: TFT2(r)),
    TypeSpec("TFT2-lastD", "nice", 2, lambda r: TFT2LastD(r)),
    TypeSpec("ContriteTFT", "nice", 2, lambda r: ContriteTFT(r)),
    TypeSpec("TFTForgiveOnce", "nice", 2, lambda r: TFTForgiveOnce(r)),
    TypeSpec("TFT-delay2", "nice", 2, lambda r: DelayedTFT(2, r)),
    TypeSpec("FirmButFair", "nice", 2, lambda r: FirmButFair(r)),
    TypeSpec("TTFT", "nice", 1.5, lambda r: TwoTitsForTat(r)),
    TypeSpec("Pavlov", "nice", 6, lambda r: Pavlov(r)),
    TypeSpec("WSLS2", "nice", 2, lambda r: WinStayLoseShift2(r)),

    # Forgiving / noisy reciprocators
    TypeSpec("GTFT.01", "forgiving", 1.5, lambda r: GenerousTFT(0.01, True, r)),
    TypeSpec("GTFT.05", "forgiving", 3, lambda r: GenerousTFT(0.05, True, r)),
    TypeSpec("GTFT.10", "forgiving", 3, lambda r: GenerousTFT(0.10, True, r)),
    TypeSpec("GTFT.20", "forgiving", 2.5, lambda r: GenerousTFT(0.20, True, r)),
    TypeSpec("GTFT.33", "forgiving", 2, lambda r: TFTNoisyForgive(1/3, r)),
    TypeSpec("GTFT.50", "forgiving", 1.5, lambda r: GenerousTFT(0.5, True, r)),
    TypeSpec("NoisyTFT.02", "forgiving", 1.5, lambda r: NoisyTFT(0.02, r)),
    TypeSpec("NoisyTFT.05", "forgiving", 2.5, lambda r: NoisyTFT(0.05, r)),
    TypeSpec("NoisyTFT.10", "forgiving", 2, lambda r: NoisyTFT(0.10, r)),
    TypeSpec("NoisyTFT.20", "forgiving", 1.5, lambda r: NoisyTFT(0.20, r)),
    TypeSpec("Joss.02", "forgiving", 1.5, lambda r: Joss(0.02, r)),
    TypeSpec("Joss.05", "forgiving", 2.5, lambda r: Joss(0.05, r)),
    TypeSpec("Joss.10", "forgiving", 2, lambda r: Joss(0.10, r)),
    TypeSpec("Joss.20", "forgiving", 1.5, lambda r: Joss(0.20, r)),
    TypeSpec("ForgiveTFT1", "forgiving", 1.5, lambda r: ForgiveTFT(True, 1, r)),
    TypeSpec("ForgiveTFT2", "forgiving", 1, lambda r: ForgiveTFT(True, 2, r)),
    TypeSpec("TwoStrike2", "forgiving", 1, lambda r: TwoStrike(2, True, r)),
    TypeSpec("TwoStrike3", "forgiving", 1, lambda r: TwoStrike(3, True, r)),

    # Punitive / grudge
    TypeSpec("Grim", "punitive", 8, lambda r: Grim(r)),
    TypeSpec("GrimForgive.05", "punitive", 2, lambda r: GrimForgive(0.05, r)),
    TypeSpec("GrimForgive.15", "punitive", 1, lambda r: GrimForgive(0.15, r)),
    TypeSpec("Grudger2", "punitive", 2, lambda r: Grudger2(r)),
    TypeSpec("Grudger3", "punitive", 1.5, lambda r: GrudgerK(3, r)),
    TypeSpec("Mistrust", "punitive", 3, lambda r: Mistrust(r)),
    TypeSpec("Gradual", "punitive", 1, lambda r: Gradual(r)),
    TypeSpec("OmegaTFT-lite", "punitive", 2, lambda r: OmegaTFTLite(r)),
    TypeSpec("Bully", "punitive", 2, lambda r: Bully(r)),

    # Defectors / exploiters / endgame
    TypeSpec("AllD", "defect", 8, lambda r: AllD(r)),
    TypeSpec("EarlyD1", "defect", 1, lambda r: EarlyDefector(1, r)),
    TypeSpec("EarlyD10", "defect", 1.5, lambda r: EarlyDefector(10, r)),
    TypeSpec("EarlyD25", "defect", 1, lambda r: EarlyDefector(25, r)),
    TypeSpec("LateD5", "late", 2, lambda r: LateDefector(5, r)),
    TypeSpec("LateD10", "late", 3, lambda r: LateDefector(10, r)),
    TypeSpec("LateD15", "late", 2, lambda r: LateDefector(15, r)),
    TypeSpec("LateD20", "late", 2, lambda r: LateDefector(20, r)),
    TypeSpec("LateD25", "late", 3, lambda r: LateDefector(25, r)),
    TypeSpec("LateD30", "late", 2, lambda r: LateDefector(30, r)),
    TypeSpec("LateD40", "late", 2, lambda r: LateDefector(40, r)),
    TypeSpec("LateD50", "late", 2, lambda r: LateDefector(50, r)),
    TypeSpec("Tester", "defect", 2, lambda r: Tester(r)),
    TypeSpec("Prober", "defect", 2, lambda r: Prober(r)),
    TypeSpec("RemorsefulProber", "defect", 1.5, lambda r: RemorsefulProber(r)),

    # Autonomous / random / periodic
    TypeSpec("Random.05", "random", 1, lambda r: RandomP(0.05, r)),
    TypeSpec("Random.10", "random", 1.5, lambda r: RandomP(0.10, r)),
    TypeSpec("Random.20", "random", 2, lambda r: RandomP(0.20, r)),
    TypeSpec("Random.30", "random", 1.5, lambda r: RandomP(0.30, r)),
    TypeSpec("Random.50", "random", 4, lambda r: RandomMoves(0.5, r)),
    TypeSpec("Random.70", "random", 1.5, lambda r: RandomP(0.70, r)),
    TypeSpec("Random.80", "random", 1.5, lambda r: RandomP(0.80, r)),
    TypeSpec("Random.90", "random", 1, lambda r: RandomP(0.90, r)),
    TypeSpec("Alternator", "random", 1.5, lambda r: Alternator(r)),
    TypeSpec("AlternatorD", "random", 1, lambda r: Periodic((D, C), 0, r)),
    TypeSpec("Periodic3", "random", 1, lambda r: Periodic((C, C, D), 0, r)),
    TypeSpec("Periodic4", "random", 1, lambda r: Periodic((C, D, D, C), 0, r)),
    TypeSpec("Rand3->TFT", "random", 2, lambda r: RandomThenTFT(3, 0.5, r)),
    TypeSpec("MemOne-rand", "wild", 3, _memone_wild),
    TypeSpec("Reactive-rand", "wild", 3, _reactive_wild),
    TypeSpec("FSM-rand", "wild", 3, _fsm_wild),

    # Adaptive / model-based / advanced
    TypeSpec("Majority2", "adaptive", 1, lambda r: Majority(2, r)),
    TypeSpec("Majority3", "adaptive", 2, lambda r: Majority(3, r)),
    TypeSpec("Majority5", "adaptive", 1, lambda r: Majority(5, r)),
    TypeSpec("Responsive", "adaptive", 1.5, lambda r: Responsive(g=0.4, rng=r)),
    TypeSpec("Responsive3", "adaptive", 1.5, lambda r: Responsive3(rng=r)),
    TypeSpec("ProbeBot", "adaptive", 1, lambda r: ProbeBot(rng=r)),
    TypeSpec("Extort1.5", "adaptive", 1, lambda r: ExtortZD(1.5, r)),
    TypeSpec("Extort2", "adaptive", 1, lambda r: ExtortZD(2.0, r)),
    TypeSpec("Extort3", "adaptive", 0.5, lambda r: ExtortZD(3.0, r)),
    TypeSpec("GenZD2", "adaptive", 1, lambda r: GenerousZD(2.0, r)),
    TypeSpec("GenZD3", "adaptive", 0.5, lambda r: GenerousZD(3.0, r)),
]

TYPE_BY_NAME = {spec.name: spec for spec in TYPE_BANK}
FAMILIES = sorted({spec.family for spec in TYPE_BANK})


# ---------------------------------------------------------------------------
# Scenario families and random mixture sampling
# ---------------------------------------------------------------------------

FAMILY_MULTIPLIERS = {
    "balanced": {f: 1.0 for f in FAMILIES},
    "cooperative": {
        "nice": 3.0, "forgiving": 2.0, "punitive": 0.5,
        "defect": 0.3, "late": 0.5, "random": 0.5, "adaptive": 1.0, "wild": 0.4,
    },
    "defector": {
        "nice": 0.7, "forgiving": 0.7, "punitive": 1.5,
        "defect": 3.0, "late": 2.0, "random": 2.0, "adaptive": 0.7, "wild": 0.8,
    },
    "naive": {
        "nice": 3.0, "forgiving": 1.0, "punitive": 0.5,
        "defect": 1.0, "late": 2.5, "random": 1.0, "adaptive": 0.2, "wild": 0.3,
    },
    "sophisticated": {
        "nice": 1.0, "forgiving": 2.0, "punitive": 1.0,
        "defect": 0.7, "late": 1.0, "random": 0.7, "adaptive": 2.5, "wild": 1.0,
    },
    "noisy": {
        "nice": 1.0, "forgiving": 3.0, "punitive": 1.0,
        "defect": 0.7, "late": 0.8, "random": 2.0, "adaptive": 1.0, "wild": 1.5,
    },
    "punitive": {
        "nice": 1.0, "forgiving": 1.0, "punitive": 3.0,
        "defect": 1.0, "late": 0.8, "random": 1.0, "adaptive": 1.0, "wild": 1.0,
    },
    "wild": {
        "nice": 0.5, "forgiving": 0.8, "punitive": 0.6,
        "defect": 0.5, "late": 0.5, "random": 1.0, "adaptive": 0.5, "wild": 4.0,
    },
    "late_harvest": {
        "nice": 2.2, "forgiving": 1.0, "punitive": 0.5,
        "defect": 0.5, "late": 5.0, "random": 0.8, "adaptive": 0.3, "wild": 0.2,
    },
}

SCENARIOS = sorted(FAMILY_MULTIPLIERS)


def effective_weights(scenario: str) -> tuple[list[TypeSpec], list[float]]:
    mult = FAMILY_MULTIPLIERS[scenario]
    specs = TYPE_BANK
    weights = [spec.weight * mult[spec.family] for spec in specs]
    return list(specs), weights


def sample_mixture(rng: random.Random, scenario: str,
                   concentration: float | None = None):
    """Draw a random mixture over types around a scenario prior.

    Gamma(shape = weight * c) followed by normalisation gives a Dirichlet
    distribution.  Small c creates very uneven populations; large c stays
    close to the base weights.  We randomise c per field as well.
    """
    specs, base = effective_weights(scenario)
    if concentration is None:
        concentration = math.exp(rng.uniform(math.log(0.18), math.log(1.8)))
    gammas = [rng.gammavariate(max(0.05, w * concentration), 1.0)
              for w in base]
    total = sum(gammas)
    probs = [g / total for g in gammas]
    return specs, probs


def sample_population(n: int, rng: random.Random, scenario: str = "balanced"):
    specs, probs = sample_mixture(rng, scenario)
    # random.choices is fine even with ~80 categories.
    pop = []
    for _ in range(n):
        spec = rng.choices(specs, weights=probs, k=1)[0]
        obj = spec.factory(rng)
        obj.name = spec.name
        pop.append(obj)
    return pop


def random_scenario(rng: random.Random) -> str:
    """Sample a field prior; 'wild' gets some extra probability."""
    weights = {
        "balanced": 20, "cooperative": 6, "defector": 6, "naive": 7,
        "sophisticated": 20, "noisy": 10, "punitive": 8, "wild": 8,
        "late_harvest": 15,
    }
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def sample_mixed_population(n: int, rng: random.Random):
    scenario = random_scenario(rng)
    return sample_population(n, rng, scenario), scenario
