"""Tail-risk opponent mixture for RL-PPO fine-tuning.

This is a separate training-only distribution.  It deliberately does not
modify the official ``original/late/grim30/wide`` samplers or the existing
``hard`` mixture, so historical results stay comparable.

The parameter values below are disjoint from the held-out final-test values
where practical (e.g. LateD 50/60/70/80/90 instead of 55/68/72/88/94).
"""

from __future__ import annotations

import random
from typing import Callable

from ipd.core.opponents.basic import AllD, Bully, Grim, Joss, RandomMoves, TFT
from ipd.core.opponents.defectors import LateDefector
from ipd.core.opponents.reciprocal import NoisyTFT
from ipd.core.opponents.zero_determinant import ExtortZD
from ipd.core.primitives import C, D, Strategy


class DynamicWindow(Strategy):
    """Windowed reciprocity with random probing and forgiveness."""

    def __init__(self, window: int = 5, p_defect: float = 0.2,
                 retaliate: float = 0.85, forgiveness: float = 0.25,
                 last_d: bool = True, rng: random.Random | None = None):
        self.window = int(window)
        self.p_defect = float(p_defect)
        self.retaliate = float(retaliate)
        self.forgiveness = float(forgiveness)
        self.last_d = bool(last_d)
        super().__init__(rng)
        self.name = (f"DynamicW{self.window}"
                     f"(p={self.p_defect:g},r={self.retaliate:g},"
                     f"f={self.forgiveness:g})")

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if self.last_d and t == 99:
            return D
        if t == 0:
            return C
        window = theirs[-self.window:]
        rate = sum(1 for x in window if x == D) / max(1, len(window))
        if rate >= 0.5:
            return D
        if theirs[-1] == D:
            return C if self.rng.random() < self.forgiveness else D
        return D if self.rng.random() < self.p_defect else C


class Kamikaze(Strategy):
    """Steal once every K rounds; permanent retaliation if punished."""

    def __init__(self, period: int = 15, p_steal: float = 1.0,
                 rng: random.Random | None = None):
        self.period = int(period)
        self.p_steal = float(p_steal)
        super().__init__(rng)
        self.name = f"Kamikaze{self.period}"

    def reset(self) -> None:
        self.angry = False
        self.stole_last = False

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if self.angry:
            return D
        if self.stole_last and theirs[-1] == D:
            self.angry = True
            self.stole_last = False
            return D
        self.stole_last = False
        if t > 0 and t % self.period == 0 and self.rng.random() < self.p_steal:
            self.stole_last = True
            return D
        return C


class HarvestFactor(Strategy):
    """Cooperate for T rounds, then harvest; optionally repair."""

    def __init__(self, start: int = 50, repairable: bool = True,
                 p_defect: float = 1.0, last_d: bool = True,
                 rng: random.Random | None = None):
        self.start = int(start)
        self.repairable = bool(repairable)
        self.p_defect = float(p_defect)
        self.last_d = bool(last_d)
        super().__init__(rng)
        self.name = (f"Harvest{self.start}"
                     f"{'R' if self.repairable else 'H'}"
                     f"(p={self.p_defect:g})")

    def reset(self) -> None:
        self.repair_left = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if self.last_d and t == 99:
            return D
        if self.repair_left:
            self.repair_left -= 1
            return C
        if t < self.start:
            return C
        if (self.repairable and len(theirs) >= 2
                and theirs[-1] == D and theirs[-2] == D):
            self.repair_left = 2
            return C
        return D if self.rng.random() < self.p_defect else C


def _extort(chi: float) -> Callable[[random.Random], Strategy]:
    return lambda rng: ExtortZD(chi, rng)


def _random(p: float) -> Callable[[random.Random], Strategy]:
    return lambda rng: RandomMoves(p, rng)


def _late(k: int) -> Callable[[random.Random], Strategy]:
    return lambda rng: LateDefector(k, rng)


def _noisy(p: float) -> Callable[[random.Random], Strategy]:
    return lambda rng: NoisyTFT(p, rng)


def _joss(p: float) -> Callable[[random.Random], Strategy]:
    return lambda rng: Joss(p, rng)


def _dynamic(window: int, p_defect: float, retaliate: float,
             forgiveness: float) -> Callable[[random.Random], Strategy]:
    return lambda rng: DynamicWindow(window, p_defect, retaliate,
                                     forgiveness, rng=rng)


def _kamikaze(period: int) -> Callable[[random.Random], Strategy]:
    return lambda rng: Kamikaze(period, rng=rng)


def _harvest(start: int, repairable: bool) -> Callable[[random.Random], Strategy]:
    return lambda rng: HarvestFactor(start, repairable, rng=rng)


# (name, weight, factory)
HARD2_OPPONENTS: list[tuple[str, float, Callable[[random.Random], Strategy]]] = [
    # High extortion: the final test's weak spot.
    ("Extort2.2", 3.0, _extort(2.2)),
    ("Extort2.8", 3.0, _extort(2.8)),
    ("Extort3.2", 3.0, _extort(3.2)),
    ("Extort3.5", 3.0, _extort(3.5)),
    # Low-cooperation random: force early stop-loss instead of endless hope.
    ("Random0.04", 2.0, _random(0.04)),
    ("Random0.12", 2.0, _random(0.12)),
    ("Random0.22", 2.0, _random(0.22)),
    ("Random0.35", 2.0, _random(0.35)),
    # Late harvest, disjoint from final-test breakpoints.
    ("LateD50", 3.0, _late(50)),
    ("LateD60", 3.0, _late(60)),
    ("LateD70", 3.0, _late(70)),
    ("LateD80", 3.0, _late(80)),
    ("LateD90", 3.0, _late(90)),
    ("Harvest40R", 2.0, _harvest(40, True)),
    ("Harvest60H", 2.0, _harvest(60, False)),
    # High noise / Joss: keep the probabilistic damping.
    ("NoisyTFT0.25", 2.0, _noisy(0.25)),
    ("NoisyTFT0.32", 2.0, _noisy(0.32)),
    ("NoisyTFT0.45", 2.0, _noisy(0.45)),
    ("Joss0.40", 2.0, _joss(0.40)),
    # Dynamic-window / kamikaze adversaries.
    ("DynamicW4", 2.0, _dynamic(4, 0.22, 0.90, 0.15)),
    ("DynamicW6", 2.0, _dynamic(6, 0.30, 0.85, 0.25)),
    ("DynamicW8", 2.0, _dynamic(8, 0.18, 0.95, 0.10)),
    ("Kamikaze10", 2.0, _kamikaze(10)),
    ("Kamikaze18", 2.0, _kamikaze(18)),
    # A small anchor of strong/cooperative opponents so the policy does not
    # collapse into permanent defection.
    ("Bully", 3.0, lambda rng: Bully(rng)),
    ("AllD", 2.0, lambda rng: AllD(rng)),
    ("Grim", 2.0, lambda rng: Grim(rng)),
    ("TFT", 2.0, lambda rng: TFT(rng)),
]


def build_hard2_field(seed: int, n: int):
    """Sample ``n`` opponents from the hard2 mixture."""
    rng = random.Random(seed)
    names = [name for name, _, _ in HARD2_OPPONENTS]
    weights = [weight for _, weight, _ in HARD2_OPPONENTS]
    factories = [factory for _, _, factory in HARD2_OPPONENTS]
    field = []
    for _ in range(n):
        i = rng.choices(range(len(names)), weights=weights, k=1)[0]
        opponent = factories[i](rng)
        opponent.name = names[i]
        field.append(opponent)
    return field


__all__ = ["HARD2_OPPONENTS", "build_hard2_field"]
