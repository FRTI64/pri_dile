"""Training-only hard-mining distribution for zero-determinant extortion.

The protocol catalog's train tier covers ``ExtortZD`` chi up to 3.95, but the
matchup scan shows that chi 4.5–6.5 still exploits the current policy.  This
distribution samples a continuous extortion range and keeps cooperative /
punitive anchors so the learner cannot collapse to permanent defection.
"""
from __future__ import annotations

import random
from typing import Callable

from ipd.core.opponents import (
    AllC, AllD, Bully, Grim, RandomMoves, TFT,
)
from ipd.core.opponents.zero_determinant import ExtortZD, GenerousZD
from ipd.core.primitives import Strategy

Factory = Callable[[random.Random], Strategy]


def _extort_strong(rng: random.Random) -> Strategy:
    return ExtortZD(rng.uniform(4.5, 7.0), rng)


def _extort_mid(rng: random.Random) -> Strategy:
    return ExtortZD(rng.uniform(2.5, 4.5), rng)


def _extort_mild(rng: random.Random) -> Strategy:
    return ExtortZD(rng.uniform(1.5, 2.5), rng)


def _generous(rng: random.Random) -> Strategy:
    return GenerousZD(rng.uniform(2.0, 5.0), rng)


def _random(rng: random.Random) -> Strategy:
    return RandomMoves(rng.uniform(0.05, 0.30), rng)


EXTORT_HARD_OPPONENTS: list[tuple[str, float, Factory]] = [
    ("ExtortStrong", 6.0, _extort_strong),
    ("ExtortMid", 4.0, _extort_mid),
    ("ExtortMild", 2.0, _extort_mild),
    ("GenerousZD", 1.0, _generous),
    ("Random", 1.0, _random),
    ("TFT", 1.5, lambda rng: TFT(rng)),
    ("Grim", 1.0, lambda rng: Grim(rng)),
    ("AllD", 0.5, lambda rng: AllD(rng)),
    ("AllC", 1.0, lambda rng: AllC(rng)),
    ("Bully", 0.5, lambda rng: Bully(rng)),
]


def build_extort_hard_field(seed: int, n: int) -> list[Strategy]:
    """Sample ``n`` extortion-hard opponents with deterministic seed."""
    rng = random.Random(seed)
    names = [name for name, _, _ in EXTORT_HARD_OPPONENTS]
    weights = [weight for _, weight, _ in EXTORT_HARD_OPPONENTS]
    factories = [factory for _, _, factory in EXTORT_HARD_OPPONENTS]
    field: list[Strategy] = []
    for _ in range(n):
        index = rng.choices(range(len(names)), weights=weights, k=1)[0]
        bot = factories[index](rng)
        bot.name = f"{names[index]}:{getattr(bot, 'name', type(bot).__name__)}"
        field.append(bot)
    return field


__all__ = ["EXTORT_HARD_OPPONENTS", "build_extort_hard_field"]
