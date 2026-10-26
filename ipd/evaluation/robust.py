"""Domain-randomized opponent mixture for robust A60 fine-tuning (E1).

Unlike ``hard``/``hard2`` (fixed parameter points), every factory here samples
its parameters from the training ranges on every call.  The mixture keeps a
small anchor block of cooperative / punitive / always-defect opponents so the
policy cannot collapse into permanent defection.
"""
from __future__ import annotations

import random
from typing import Callable

from ipd.core.opponents import (
    AllC, AllD, Bully, GenerousTFT, Grim, Joss, LateDefector, NoisyTFT,
    Pavlov, RandomMoves, TFT,
)
from ipd.core.opponents.zero_determinant import ExtortZD
from ipd.core.primitives import Strategy

from .hard2 import DynamicWindow, HarvestFactor, Kamikaze

Factory = Callable[[random.Random], Strategy]


def _extort(rng: random.Random) -> Strategy:
    return ExtortZD(rng.uniform(1.5, 3.8), rng)


def _random(rng: random.Random) -> Strategy:
    return RandomMoves(rng.uniform(0.01, 0.95), rng)


def _random_repair(rng: random.Random) -> Strategy:
    # Narrower than ``robust``: avoid the near-AllC/near-AllD extremes when
    # repairing the random-family regression.
    return RandomMoves(rng.uniform(0.08, 0.92), rng)


def _gtft(rng: random.Random) -> Strategy:
    return GenerousTFT(rng.uniform(0.05, 0.35), True, rng)


def _late(rng: random.Random) -> Strategy:
    return LateDefector(rng.randint(20, 95), rng)


def _noisy(rng: random.Random) -> Strategy:
    return NoisyTFT(rng.uniform(0.01, 0.50), rng)


def _joss(rng: random.Random) -> Strategy:
    return Joss(rng.uniform(0.01, 0.50), rng)


def _dynamic(rng: random.Random) -> Strategy:
    return DynamicWindow(
        rng.randint(3, 10),
        rng.uniform(0.02, 0.60),
        rng.uniform(0.60, 1.00),
        rng.uniform(0.00, 0.80),
        rng=rng,
    )


def _kamikaze(rng: random.Random) -> Strategy:
    return Kamikaze(rng.randint(8, 30), rng.uniform(0.50, 1.00), rng)


def _harvest(rng: random.Random) -> Strategy:
    return HarvestFactor(
        rng.randint(20, 90), rng.random() < 0.5,
        p_defect=rng.uniform(0.70, 1.00), rng=rng,
    )


# (label, weight, factory).  Labels are for diagnostics; the concrete bot name
# carries the sampled parameters.
ROBUST_OPPONENTS: list[tuple[str, float, Factory]] = [
    ("Extort", 3.0, _extort),
    ("Random", 3.0, _random),
    ("LateD", 3.0, _late),
    ("NoisyTFT", 3.0, _noisy),
    ("Joss", 2.0, _joss),
    ("DynamicWindow", 2.0, _dynamic),
    ("Kamikaze", 2.0, _kamikaze),
    ("Harvest", 1.0, _harvest),
    ("TFT", 2.0, lambda rng: TFT(rng)),
    ("Grim", 1.0, lambda rng: Grim(rng)),
    ("AllD", 1.0, lambda rng: AllD(rng)),
    ("Bully", 1.0, lambda rng: Bully(rng)),
    ("AllC", 1.0, lambda rng: AllC(rng)),
]


# E1c repair mixture: keep some extort pressure, but strongly reweight the
# random/noisy/adaptive tail and add more cooperative anchors.  Starting point
# is the E1b checkpoint, not A60.
REPAIR_OPPONENTS: list[tuple[str, float, Factory]] = [
    ("Random", 5.0, _random_repair),
    ("NoisyTFT", 3.0, _noisy),
    ("Joss", 2.0, _joss),
    ("DynamicWindow", 2.0, _dynamic),
    ("LateD", 2.0, _late),
    ("Extort", 1.5, _extort),
    ("Kamikaze", 1.0, _kamikaze),
    ("Harvest", 1.0, _harvest),
    ("TFT", 3.0, lambda rng: TFT(rng)),
    ("GTFT", 2.0, _gtft),
    ("Pavlov", 2.0, lambda rng: Pavlov(rng)),
    ("AllC", 2.0, lambda rng: AllC(rng)),
    ("Grim", 1.0, lambda rng: Grim(rng)),
    ("AllD", 1.0, lambda rng: AllD(rng)),
    ("Bully", 1.0, lambda rng: Bully(rng)),
]


def _build_field(seed: int, n: int,
                 opponents: list[tuple[str, float, Factory]]
                 ) -> list[Strategy]:
    rng = random.Random(seed)
    names = [name for name, _, _ in opponents]
    weights = [weight for _, weight, _ in opponents]
    factories = [factory for _, _, factory in opponents]
    field = []
    for _ in range(n):
        index = rng.choices(range(len(names)), weights=weights, k=1)[0]
        bot = factories[index](rng)
        bot.name = f"{names[index]}:{getattr(bot, 'name', type(bot).__name__)}"
        field.append(bot)
    return field


def build_robust_field(seed: int, n: int) -> list[Strategy]:
    """Sample ``n`` opponents, randomizing every non-anchor parameter."""
    return _build_field(seed, n, ROBUST_OPPONENTS)


def build_repair_field(seed: int, n: int) -> list[Strategy]:
    """E1c repair mixture: random/noisy-heavy with cooperative anchors."""
    return _build_field(seed, n, REPAIR_OPPONENTS)


__all__ = [
    "ROBUST_OPPONENTS", "REPAIR_OPPONENTS", "build_robust_field",
    "build_repair_field",
]
