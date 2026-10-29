"""Training-only hard-mining distribution for periodic / Kamikaze opponents.

The full opponent scan shows that deterministic periodic defection and
Kamikaze schedules still exploit RL-Wide160.  This distribution samples
periodic patterns, Kamikaze schedules, and DynamicWindow probes while keeping
cooperative / punitive anchors.
"""
from __future__ import annotations

import random
from typing import Callable

from ipd.core.opponents import (
    AllC, AllD, Bully, Grim, TFT,
)
from ipd.core.opponents.memory import Periodic, RandomFSM
from ipd.core.primitives import C, D, Strategy

from .hard2 import DynamicWindow, Kamikaze

Factory = Callable[[random.Random], Strategy]

_PATTERNS = (
    (D, C), (D, C, C), (D, C, C, C), (C, D, C, C),
    (D, C, D, C, C), (C, C, D, C, C), (D, D, C, C),
)


def _kamikaze(rng: random.Random) -> Strategy:
    return Kamikaze(rng.randint(4, 12), rng.choice((0.75, 1.0)), rng)


def _periodic(rng: random.Random) -> Strategy:
    return Periodic(rng.choice(_PATTERNS), rng.randint(0, 3), rng)


def _dynamic(rng: random.Random) -> Strategy:
    return DynamicWindow(
        rng.randint(5, 15), rng.uniform(0.20, 0.50),
        rng.uniform(0.70, 1.00), rng.uniform(0.10, 0.40),
        rng=rng,
    )


def _fsm(rng: random.Random) -> Strategy:
    return RandomFSM(rng, rng.randint(3, 5))


PERIODIC_HARD_OPPONENTS: list[tuple[str, float, Factory]] = [
    ("Kamikaze", 6.0, _kamikaze),
    ("Periodic", 4.0, _periodic),
    ("DynamicWindow", 2.0, _dynamic),
    ("RandomFSM", 1.0, _fsm),
    ("TFT", 1.5, lambda rng: TFT(rng)),
    ("Grim", 1.0, lambda rng: Grim(rng)),
    ("AllC", 1.0, lambda rng: AllC(rng)),
    ("AllD", 0.5, lambda rng: AllD(rng)),
    ("Bully", 0.5, lambda rng: Bully(rng)),
]


def build_periodic_hard_field(seed: int, n: int) -> list[Strategy]:
    rng = random.Random(seed)
    names = [name for name, _, _ in PERIODIC_HARD_OPPONENTS]
    weights = [weight for _, weight, _ in PERIODIC_HARD_OPPONENTS]
    factories = [factory for _, _, factory in PERIODIC_HARD_OPPONENTS]
    field: list[Strategy] = []
    for _ in range(n):
        index = rng.choices(range(len(names)), weights=weights, k=1)[0]
        bot = factories[index](rng)
        bot.name = f"{names[index]}:{getattr(bot, 'name', type(bot).__name__)}"
        field.append(bot)
    return field


__all__ = ["PERIODIC_HARD_OPPONENTS", "build_periodic_hard_field"]
