"""The 85-type opponent bank and mixture samplers."""
from __future__ import annotations

import math
import random

from ..primitives import C, D, Strategy
from .basic import (
    AllC, AllD, Bully, FirmButFair, Grim, Joss, Mistrust, Pavlov, Prober,
    RandomMoves, SuspiciousTFT, TFT, TFT2, TFTLastD, Tester,
    WinStayLoseShift2,
)
from .defectors import (
    Alternator, EarlyDefector, Gradual, Grudger2, LateDefector, RandomP,
)
from .memory import (
    AllCWithLastD, GrudgerK, MemoryOne, OmegaTFTLite, Periodic, RandomFSM,
    Reactive, TwoTitsForTat,
)
from .reciprocal import (
    ContriteTFT, DelayedTFT, ForgiveTFT, GenerousTFT, GrimForgive,
    LastKDefect, Majority, NoisyTFT, ProbeBot, RandomThenTFT, RemorsefulProber,
    Responsive, Responsive3, TFT2LastD, TFTForgiveOnce, TFTNoisyForgive,
    TwoStrike,
)
from .spec import TypeSpec
from .zero_determinant import ExtortZD, GenerousZD


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


__all__ = [
    "TYPE_BANK", "TYPE_BY_NAME", "FAMILIES", "FAMILY_MULTIPLIERS",
    "SCENARIOS", "effective_weights", "sample_mixture", "sample_population",
    "random_scenario", "sample_mixed_population",
]
