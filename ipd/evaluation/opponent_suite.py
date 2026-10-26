"""Frozen opponent suites used by the E0 league harness.

``validation`` and ``final`` are re-exported from the self-play suite so the
historical numbers stay comparable.  ``heldout_v2`` is a new, frozen suite with
new parameter draws and three new behavioural families; it was created for E0
and must not be used for iterative model selection.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ipd.core.opponents import (
    AdaptiveProbe, AllC, AllD, Bully, ContriteTFT, DelayedTFT,
    FirmButFair, GenerousTFT, Grim, Grudge2, Grudger2, GrudgerK, Joss,
    LateDefector, Mistrust, NoisyTFT, OmegaTFTLite, Pavlov, Periodic,
    Prober, RandomMoves, Responsive2, Responsive3, Strategy, TFT, TFT2,
    TFTLastD, TFTNoisyForgive, TFTProbe, Tester, TwoStrike,
    WinStayLoseShift2,
)
from ipd.core.opponents.zero_determinant import ExtortZD
from ipd.core.primitives import C, D

Factory = Callable[[random.Random], object]


@dataclass(frozen=True)
class SuiteItem:
    name: str
    family: str
    factory: Factory

    def build(self, rng: random.Random):
        return self.factory(rng)


# ---------------------------------------------------------------------------
# New held-out behaviours.  These are evaluation-only opponents; training does
# not see them or their parameter values.
# ---------------------------------------------------------------------------


class WindowTFT(Strategy):
    """TFT over a recent window with tunable retaliation/forgiveness."""

    def __init__(self, window: int = 5, retaliate: float = 0.8,
                 forgive: float = 0.25, rng: random.Random | None = None):
        self.window = int(window)
        self.retaliate = float(retaliate)
        self.forgive = float(forgive)
        super().__init__(rng)
        self.name = f"WindowTFT{self.window}"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        if t == 99:
            return D
        recent = theirs[-self.window:]
        rate = sum(1 for x in recent if x == D) / max(1, len(recent))
        if rate >= self.retaliate:
            return D
        if theirs[-1] == D:
            return C if self.rng.random() < self.forgive else D
        return C


class ProbabilisticGrim(Strategy):
    """Punish after ``patience`` defections, but only with probability p."""

    def __init__(self, patience: int = 2, p_punish: float = 0.85,
                 rng: random.Random | None = None):
        self.patience = int(patience)
        self.p_punish = float(p_punish)
        super().__init__(rng)
        self.name = f"ProbGrim{self.patience}"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        if t == 99:
            return D
        if theirs.count(D) > self.patience:
            return D if self.rng.random() < self.p_punish else C
        return C


class ResetGrim(Strategy):
    """Punish for a fixed cooldown, then return to cooperation."""

    def __init__(self, patience: int = 2, cooldown: int = 12,
                 rng: random.Random | None = None):
        self.patience = int(patience)
        self.cooldown = int(cooldown)
        super().__init__(rng)
        self.name = f"ResetGrim{self.patience}"

    def reset(self) -> None:
        self.punish_left = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        if t == 99:
            return D
        if self.punish_left > 0:
            self.punish_left -= 1
            return D
        if theirs.count(D) > self.patience:
            self.punish_left = self.cooldown
            return D
        return C


class SpitefulGrim(Strategy):
    """Punish after ``patience`` defections, then defect forever."""

    def __init__(self, patience: int = 2, rng: random.Random | None = None):
        self.patience = int(patience)
        super().__init__(rng)
        self.name = f"SpitefulGrim{self.patience}"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        if t == 99:
            return D
        return D if theirs.count(D) > self.patience else C


class DelayedPunisher(Strategy):
    """Forgive a few isolated defections, then punish for a limited period."""

    def __init__(self, patience: int = 3, punish_rounds: int = 10,
                 rng: random.Random | None = None):
        self.patience = int(patience)
        self.punish_rounds = int(punish_rounds)
        super().__init__(rng)
        self.name = f"DelayedPunisher{self.patience}"

    def reset(self) -> None:
        self.punish_left = 0
        self.pending = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        if t == 99:
            return D
        if self.punish_left > 0:
            self.punish_left -= 1
            return D
        if theirs[-1] == D:
            self.pending += 1
            if self.pending > self.patience:
                self.pending = 0
                self.punish_left = self.punish_rounds - 1
                return D
        else:
            self.pending = 0
        return C


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _item(name: str, family: str, factory: Factory) -> SuiteItem:
    return SuiteItem(name=name, family=family, factory=factory)


def _bot(item: SuiteItem, rng: random.Random):
    bot = item.factory(rng)
    if hasattr(bot, "name"):
        bot.name = item.name
    return bot


def build_heldout_v2_suite() -> list[SuiteItem]:
    """New 73-opponent held-out suite with disjoint parameter draws."""
    suite: list[SuiteItem] = []

    # Reciprocal / conditional cooperators.
    reciprocal: list[SuiteItem] = [
        _item("TFT.v2", "reciprocal_v2", lambda r: TFT(r)),
        _item("TFT2.v2", "reciprocal_v2", lambda r: TFT2(r)),
        _item("TFT-lastD.v2", "reciprocal_v2", lambda r: TFTLastD(r)),
        _item("Pavlov.v2", "reciprocal_v2", lambda r: Pavlov(r)),
        _item("Contrite.v2", "reciprocal_v2", lambda r: ContriteTFT(r)),
        _item("WSLS2.v2", "reciprocal_v2", lambda r: WinStayLoseShift2(r)),
        _item("DelayedTFT.v2.5", "reciprocal_v2",
              lambda r: DelayedTFT(5, r)),
        _item("WindowTFT.v2.3", "reciprocal_v2",
              lambda r: WindowTFT(3, 0.75, 0.35, r)),
        _item("WindowTFT.v2.7", "reciprocal_v2",
              lambda r: WindowTFT(7, 0.90, 0.10, r)),
    ]
    for g in (0.015, 0.38, 0.45, 0.49):
        reciprocal.append(_item(
            f"GTFT.v2.{g:g}", "reciprocal_v2",
            lambda r, g=g: GenerousTFT(g, True, r)))
    suite += reciprocal

    # Punitive / trigger strategies.
    punitive: list[SuiteItem] = [
        _item("Grim.v2", "punitive_v2", lambda r: Grim(r)),
        _item("Grudge2.v2", "punitive_v2", lambda r: Grudge2(r)),
        _item("Grudger2.v2", "punitive_v2", lambda r: Grudger2(r)),
        _item("Grudger3.v2", "punitive_v2", lambda r: GrudgerK(3, r)),
        _item("TwoStrike.v2", "punitive_v2", lambda r: TwoStrike(2, True, r)),
        _item("FirmButFair.v2", "punitive_v2", lambda r: FirmButFair(r)),
        _item("ProbGrim.v2.2", "punitive_v2",
              lambda r: ProbabilisticGrim(2, 0.85, r)),
        _item("ResetGrim.v2.2", "punitive_v2",
              lambda r: ResetGrim(2, 12, r)),
    ]
    suite += punitive

    # Noisy reciprocators.
    noisy: list[SuiteItem] = []
    for p in (0.23, 0.27, 0.30, 0.33, 0.37, 0.43):
        noisy.append(_item(f"NoisyTFT.v2.{p:g}", "noisy_v2",
                           lambda r, p=p: NoisyTFT(p, r)))
    for p in (0.24, 0.29, 0.39):
        noisy.append(_item(f"Joss.v2.{p:g}", "noisy_v2",
                           lambda r, p=p: Joss(p, r)))
    suite += noisy

    # Extortion / probing.
    extort: list[SuiteItem] = [
        _item("Prober.v2", "extort_v2", lambda r: Prober(r)),
        _item("Bully.v2", "extort_v2", lambda r: Bully(r)),
    ]
    for chi in (1.05, 1.58, 2.15, 2.40, 3.10, 3.80):
        extort.append(_item(f"Extort.v2.{chi:g}", "extort_v2",
                            lambda r, chi=chi: ExtortZD(chi, r)))
    suite += extort

    # Random / periodic.
    random_suite: list[SuiteItem] = [
        _item("Periodic.v2", "random_v2",
              lambda r: Periodic((C, C, D, C, D), 1, r)),
    ]
    for p in (0.02, 0.11, 0.21, 0.33, 0.47, 0.66, 0.84):
        random_suite.append(_item(f"Random.v2.{p:g}", "random_v2",
                                  lambda r, p=p: RandomMoves(p, r)))
    suite += random_suite

    # Endgame / temporal opponents.
    temporal: list[SuiteItem] = []
    for k in (3, 7, 17, 27, 36, 44, 57, 63, 79, 91):
        temporal.append(_item(f"LateD.v2.{k}", "temporal_v2",
                              lambda r, k=k: LateDefector(k, r)))
    temporal += [
        _item("Harvest.v2.8R", "temporal_v2",
              lambda r: _Harvest(8, True, r)),
        _item("Harvest.v2.89H", "temporal_v2",
              lambda r: _Harvest(89, False, r)),
        _item("Harvest.v2.93R", "temporal_v2",
              lambda r: _Harvest(93, True, r)),
        _item("DynamicW.v2.11", "temporal_v2",
              lambda r: _DynamicWindow(11, 0.12, 0.95, 0.04, r)),
        _item("DynamicW.v2.13", "temporal_v2",
              lambda r: _DynamicWindow(13, 0.33, 0.70, 0.45, r)),
        _item("ProbePhase.v2.7", "temporal_v2",
              lambda r: _ProbePhase(7, 0.55, "harvest", 66, r)),
        _item("ProbePhase.v2.9", "temporal_v2",
              lambda r: _ProbePhase(9, 0.35, "tft", 46, r)),
        _item("Kamikaze.v2.6", "temporal_v2",
              lambda r: _Kamikaze(6, 0.65, r)),
        _item("Kamikaze.v2.26", "temporal_v2",
              lambda r: _Kamikaze(26, 0.95, r)),
        _item("DelayedPunisher.v2", "temporal_v2",
              lambda r: DelayedPunisher(3, 10, r)),
    ]
    suite += temporal

    # Adaptive / mixed strategies.
    adaptive: list[SuiteItem] = [
        _item("AdaptiveProbe.v2", "adaptive_v2",
              lambda r: AdaptiveProbe(72, 3, True, r)),
        _item("Resp2.v2", "adaptive_v2",
              lambda r: Responsive2(0.92, 0.14, 2, 5, 3, 0.31, True, 3, r)),
        _item("Resp3.v2", "adaptive_v2",
              lambda r: Responsive3(0.48, 0.33, 0.91, 6, 3, 0.34, 0.15,
                                    True, 2, 11, r)),
        _item("TFTNoisyForgive.v2", "adaptive_v2",
              lambda r: TFTNoisyForgive(0.41, r)),
        _item("TFTProbe.v2", "adaptive_v2",
              lambda r: TFTProbe(74, 3, True, r)),
        _item("OmegaTFT.v2", "adaptive_v2",
              lambda r: OmegaTFTLite(rng=r)),
        _item("Mistrust.v2", "adaptive_v2", lambda r: Mistrust(r)),
        _item("Tester.v2", "adaptive_v2", lambda r: Tester(r)),
    ]
    suite += adaptive
    return suite


def build_heldout_v3_suite() -> list[SuiteItem]:
    """Second fresh suite for one-shot E1 final testing.

    Family names, opponent names and parameter draws are disjoint from
    ``heldout_v2``.  Use it once after E1 selection; do not iterate on it.
    """
    suite: list[SuiteItem] = []

    reciprocal: list[SuiteItem] = [
        _item("TFT.v3", "reciprocal_v3", lambda r: TFT(r)),
        _item("TFT2.v3", "reciprocal_v3", lambda r: TFT2(r)),
        _item("TFT-lastD.v3", "reciprocal_v3", lambda r: TFTLastD(r)),
        _item("Pavlov.v3", "reciprocal_v3", lambda r: Pavlov(r)),
        _item("Contrite.v3", "reciprocal_v3", lambda r: ContriteTFT(r)),
        _item("WSLS2.v3", "reciprocal_v3", lambda r: WinStayLoseShift2(r)),
        _item("DelayedTFT.v3.7", "reciprocal_v3",
              lambda r: DelayedTFT(7, r)),
        _item("WindowTFT.v3.4", "reciprocal_v3",
              lambda r: WindowTFT(4, 0.82, 0.22, r)),
        _item("WindowTFT.v3.9", "reciprocal_v3",
              lambda r: WindowTFT(9, 0.88, 0.18, r)),
    ]
    for g in (0.06, 0.21, 0.34, 0.47):
        reciprocal.append(_item(
            f"GTFT.v3.{g:g}", "reciprocal_v3",
            lambda r, g=g: GenerousTFT(g, True, r)))
    suite += reciprocal

    punitive: list[SuiteItem] = [
        _item("Grim.v3", "punitive_v3", lambda r: Grim(r)),
        _item("Grudge2.v3", "punitive_v3", lambda r: Grudge2(r)),
        _item("Grudger2.v3", "punitive_v3", lambda r: Grudger2(r)),
        _item("GrudgerK2.v3", "punitive_v3", lambda r: GrudgerK(2, r)),
        _item("GrudgerK4.v3", "punitive_v3", lambda r: GrudgerK(4, r)),
        _item("TwoStrike.v3", "punitive_v3",
              lambda r: TwoStrike(3, True, r)),
        _item("FirmButFair.v3", "punitive_v3", lambda r: FirmButFair(r)),
        _item("ProbGrim.v3.3", "punitive_v3",
              lambda r: ProbabilisticGrim(3, 0.90, r)),
        _item("ResetGrim.v3.1", "punitive_v3",
              lambda r: ResetGrim(1, 16, r)),
        _item("SpitefulGrim.v3", "punitive_v3",
              lambda r: SpitefulGrim(1, r)),
    ]
    suite += punitive

    noisy: list[SuiteItem] = []
    for p in (0.025, 0.07, 0.16, 0.28, 0.36, 0.49):
        noisy.append(_item(f"NoisyTFT.v3.{p:g}", "noisy_v3",
                           lambda r, p=p: NoisyTFT(p, r)))
    for p in (0.03, 0.18, 0.34, 0.47):
        noisy.append(_item(f"Joss.v3.{p:g}", "noisy_v3",
                           lambda r, p=p: Joss(p, r)))
    suite += noisy

    extort: list[SuiteItem] = [
        _item("Prober.v3", "extort_v3", lambda r: Prober(r)),
        _item("Bully.v3", "extort_v3", lambda r: Bully(r)),
    ]
    for chi in (0.95, 1.32, 1.68, 2.25, 2.55, 2.95, 3.62):
        extort.append(_item(f"Extort.v3.{chi:g}", "extort_v3",
                            lambda r, chi=chi: ExtortZD(chi, r)))
    suite += extort

    random_suite: list[SuiteItem] = [
        _item("Periodic.v3", "random_v3",
              lambda r: Periodic((D, C, C, D, C), 2, r)),
    ]
    for p in (0.015, 0.09, 0.19, 0.29, 0.41, 0.55, 0.69, 0.81, 0.94):
        random_suite.append(_item(f"Random.v3.{p:g}", "random_v3",
                                  lambda r, p=p: RandomMoves(p, r)))
    suite += random_suite

    temporal: list[SuiteItem] = []
    for k in (6, 11, 23, 31, 39, 53, 66, 71, 83, 96):
        temporal.append(_item(f"LateD.v3.{k}", "temporal_v3",
                              lambda r, k=k: LateDefector(k, r)))
    temporal += [
        _item("Harvest.v3.14R", "temporal_v3",
              lambda r: _Harvest(14, True, r)),
        _item("Harvest.v3.57H", "temporal_v3",
              lambda r: _Harvest(57, False, r)),
        _item("Harvest.v3.81R", "temporal_v3",
              lambda r: _Harvest(81, True, r)),
        _item("DynamicW.v3.4", "temporal_v3",
              lambda r: _DynamicWindow(4, 0.25, 0.85, 0.30, r)),
        _item("DynamicW.v3.12", "temporal_v3",
              lambda r: _DynamicWindow(12, 0.08, 0.90, 0.55, r)),
        _item("ProbePhase.v3.5", "temporal_v3",
              lambda r: _ProbePhase(5, 0.60, "harvest", 52, r)),
        _item("ProbePhase.v3.10", "temporal_v3",
              lambda r: _ProbePhase(10, 0.40, "tft", 72, r)),
        _item("Kamikaze.v3.14", "temporal_v3",
              lambda r: _Kamikaze(14, 0.80, r)),
        _item("Kamikaze.v3.29", "temporal_v3",
              lambda r: _Kamikaze(29, 0.55, r)),
        _item("DelayedPunisher.v3", "temporal_v3",
              lambda r: DelayedPunisher(2, 14, r)),
    ]
    suite += temporal

    adaptive: list[SuiteItem] = [
        _item("AdaptiveProbe.v3", "adaptive_v3",
              lambda r: AdaptiveProbe(66, 2, True, r)),
        _item("Resp2.v3", "adaptive_v3",
              lambda r: Responsive2(0.78, 0.22, 3, 4, 2, 0.27, True, 2, r)),
        _item("Resp3.v3", "adaptive_v3",
              lambda r: Responsive3(0.41, 0.29, 0.88, 4, 4, 0.41, 0.22,
                                    True, 3, 7, r)),
        _item("TFTNoisyForgive.v3", "adaptive_v3",
              lambda r: TFTNoisyForgive(0.27, r)),
        _item("TFTProbe.v3", "adaptive_v3",
              lambda r: TFTProbe(82, 4, True, r)),
        _item("OmegaTFT.v3", "adaptive_v3",
              lambda r: OmegaTFTLite(rng=r)),
        _item("Mistrust.v3", "adaptive_v3", lambda r: Mistrust(r)),
        _item("Tester.v3", "adaptive_v3", lambda r: Tester(r)),
    ]
    suite += adaptive

    suite += [
        _item("AllC.v3", "anchor_v3", lambda r: AllC(r)),
        _item("AllD.v3", "anchor_v3", lambda r: AllD(r)),
        _item("RandomMid.v3", "anchor_v3", lambda r: RandomMoves(0.50, r)),
    ]
    return suite


def build_heldout_v4_suite() -> list[SuiteItem]:
    """Third fresh suite, reserved for the next post-E1c final test."""
    suite: list[SuiteItem] = []

    reciprocal: list[SuiteItem] = [
        _item("TFT.v4", "reciprocal_v4", lambda r: TFT(r)),
        _item("TFT2.v4", "reciprocal_v4", lambda r: TFT2(r)),
        _item("TFT-lastD.v4", "reciprocal_v4", lambda r: TFTLastD(r)),
        _item("Pavlov.v4", "reciprocal_v4", lambda r: Pavlov(r)),
        _item("Contrite.v4", "reciprocal_v4", lambda r: ContriteTFT(r)),
        _item("WSLS2.v4", "reciprocal_v4", lambda r: WinStayLoseShift2(r)),
        _item("DelayedTFT.v4.3", "reciprocal_v4",
              lambda r: DelayedTFT(3, r)),
        _item("WindowTFT.v4.6", "reciprocal_v4",
              lambda r: WindowTFT(6, 0.86, 0.28, r)),
        _item("WindowTFT.v4.10", "reciprocal_v4",
              lambda r: WindowTFT(10, 0.78, 0.40, r)),
    ]
    for g in (0.12, 0.25, 0.41, 0.48):
        reciprocal.append(_item(
            f"GTFT.v4.{g:g}", "reciprocal_v4",
            lambda r, g=g: GenerousTFT(g, True, r)))
    suite += reciprocal

    punitive: list[SuiteItem] = [
        _item("Grim.v4", "punitive_v4", lambda r: Grim(r)),
        _item("Grudge2.v4", "punitive_v4", lambda r: Grudge2(r)),
        _item("Grudger2.v4", "punitive_v4", lambda r: Grudger2(r)),
        _item("GrudgerK1.v4", "punitive_v4", lambda r: GrudgerK(1, r)),
        _item("GrudgerK5.v4", "punitive_v4", lambda r: GrudgerK(5, r)),
        _item("TwoStrike.v4", "punitive_v4",
              lambda r: TwoStrike(4, True, r)),
        _item("FirmButFair.v4", "punitive_v4", lambda r: FirmButFair(r)),
        _item("ProbGrim.v4.1", "punitive_v4",
              lambda r: ProbabilisticGrim(1, 0.75, r)),
        _item("ResetGrim.v4.3", "punitive_v4",
              lambda r: ResetGrim(3, 9, r)),
        _item("SpitefulGrim.v4", "punitive_v4",
              lambda r: SpitefulGrim(2, r)),
    ]
    suite += punitive

    noisy: list[SuiteItem] = []
    for p in (0.06, 0.12, 0.21, 0.32, 0.44, 0.47):
        noisy.append(_item(f"NoisyTFT.v4.{p:g}", "noisy_v4",
                           lambda r, p=p: NoisyTFT(p, r)))
    for p in (0.11, 0.26, 0.36, 0.45):
        noisy.append(_item(f"Joss.v4.{p:g}", "noisy_v4",
                           lambda r, p=p: Joss(p, r)))
    suite += noisy

    extort: list[SuiteItem] = [
        _item("Prober.v4", "extort_v4", lambda r: Prober(r)),
        _item("Bully.v4", "extort_v4", lambda r: Bully(r)),
    ]
    for chi in (1.12, 1.42, 1.85, 2.12, 2.62, 3.18, 3.72):
        extort.append(_item(f"Extort.v4.{chi:g}", "extort_v4",
                            lambda r, chi=chi: ExtortZD(chi, r)))
    suite += extort

    random_suite: list[SuiteItem] = [
        _item("Periodic.v4", "random_v4",
              lambda r: Periodic((C, D, C, D, C), 3, r)),
    ]
    for p in (0.04, 0.13, 0.26, 0.35, 0.44, 0.61, 0.76, 0.87, 0.97):
        random_suite.append(_item(f"Random.v4.{p:g}", "random_v4",
                                  lambda r, p=p: RandomMoves(p, r)))
    suite += random_suite

    temporal: list[SuiteItem] = []
    for k in (5, 9, 19, 25, 34, 47, 59, 69, 81, 93):
        temporal.append(_item(f"LateD.v4.{k}", "temporal_v4",
                              lambda r, k=k: LateDefector(k, r)))
    temporal += [
        _item("Harvest.v4.11R", "temporal_v4",
              lambda r: _Harvest(11, True, r)),
        _item("Harvest.v4.63H", "temporal_v4",
              lambda r: _Harvest(63, False, r)),
        _item("Harvest.v4.87R", "temporal_v4",
              lambda r: _Harvest(87, True, r)),
        _item("DynamicW.v4.6", "temporal_v4",
              lambda r: _DynamicWindow(6, 0.18, 0.92, 0.12, r)),
        _item("DynamicW.v4.14", "temporal_v4",
              lambda r: _DynamicWindow(14, 0.30, 0.75, 0.38, r)),
        _item("ProbePhase.v4.3", "temporal_v4",
              lambda r: _ProbePhase(3, 0.45, "tft", 58, r)),
        _item("ProbePhase.v4.11", "temporal_v4",
              lambda r: _ProbePhase(11, 0.65, "harvest", 76, r)),
        _item("Kamikaze.v4.9", "temporal_v4",
              lambda r: _Kamikaze(9, 0.90, r)),
        _item("Kamikaze.v4.21", "temporal_v4",
              lambda r: _Kamikaze(21, 0.70, r)),
        _item("DelayedPunisher.v4", "temporal_v4",
              lambda r: DelayedPunisher(4, 7, r)),
    ]
    suite += temporal

    adaptive: list[SuiteItem] = [
        _item("AdaptiveProbe.v4", "adaptive_v4",
              lambda r: AdaptiveProbe(58, 4, True, r)),
        _item("Resp2.v4", "adaptive_v4",
              lambda r: Responsive2(0.85, 0.18, 2, 6, 2, 0.29, True, 4, r)),
        _item("Resp3.v4", "adaptive_v4",
              lambda r: Responsive3(0.52, 0.36, 0.93, 5, 3, 0.37, 0.19,
                                    True, 2, 9, r)),
        _item("TFTNoisyForgive.v4", "adaptive_v4",
              lambda r: TFTNoisyForgive(0.35, r)),
        _item("TFTProbe.v4", "adaptive_v4",
              lambda r: TFTProbe(70, 2, True, r)),
        _item("OmegaTFT.v4", "adaptive_v4",
              lambda r: OmegaTFTLite(rng=r)),
        _item("Mistrust.v4", "adaptive_v4", lambda r: Mistrust(r)),
        _item("Tester.v4", "adaptive_v4", lambda r: Tester(r)),
    ]
    suite += adaptive

    suite += [
        _item("AllC.v4", "anchor_v4", lambda r: AllC(r)),
        _item("AllD.v4", "anchor_v4", lambda r: AllD(r)),
        _item("RandomMid.v4", "anchor_v4", lambda r: RandomMoves(0.50, r)),
    ]
    return suite


# Small local versions of the training temporal opponents.  Keeping them here
# makes the evaluation suite independent of the training module.
class _Harvest(Strategy):
    def __init__(self, start: int, repairable: bool, rng):
        self.start = int(start)
        self.repairable = bool(repairable)
        super().__init__(rng)

    def reset(self) -> None:
        self.repair_left = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
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
        return D


class _DynamicWindow(Strategy):
    def __init__(self, window: int, p_defect: float, retaliate: float,
                 forgiveness: float, rng):
        self.window = int(window)
        self.p_defect = float(p_defect)
        self.retaliate = float(retaliate)
        self.forgiveness = float(forgiveness)
        super().__init__(rng)

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        if t == 99:
            return D
        recent = theirs[-self.window:]
        rate = sum(1 for x in recent if x == D) / max(1, len(recent))
        if rate >= self.retaliate:
            return D
        if theirs[-1] == D:
            return C if self.rng.random() < self.forgiveness else D
        return D if self.rng.random() < self.p_defect else C


class _ProbePhase(Strategy):
    def __init__(self, probe: int, p_probe: float, phase: str,
                 harvest_start: int, rng):
        self.probe = int(probe)
        self.p_probe = float(p_probe)
        self.phase = str(phase)
        self.harvest_start = int(harvest_start)
        super().__init__(rng)

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t < self.probe:
            return C if self.rng.random() < self.p_probe else D
        if self.phase == "harvest":
            return D if t >= self.harvest_start else C
        return theirs[-1]


class _Kamikaze(Strategy):
    def __init__(self, period: int, p_steal: float, rng):
        self.period = int(period)
        self.p_steal = float(p_steal)
        super().__init__(rng)

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


def _convert(items: Iterable[object]) -> list[SuiteItem]:
    return [SuiteItem(name=item.name, family=item.family,
                      factory=item.factory) for item in items]


def build_validation_suite() -> list[SuiteItem]:
    from ipd.training.selfplay.opponents import build_validation_suite as build

    return _convert(build())


def build_final_suite() -> list[SuiteItem]:
    from ipd.training.selfplay.opponents import build_final_suite as build

    return _convert(build())


_BUILDERS: dict[str, Callable[[], list[SuiteItem]]] = {
    "validation": build_validation_suite,
    "final": build_final_suite,
    "heldout_v2": build_heldout_v2_suite,
    "heldout_v3": build_heldout_v3_suite,
    "heldout_v4": build_heldout_v4_suite,
}


def suite_names() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def build_suite(name: str) -> list[SuiteItem]:
    if name not in _BUILDERS:
        raise ValueError(f"unknown suite {name!r}; choose from {suite_names()}")
    return _BUILDERS[name]()


def materialize(items: Sequence[SuiteItem], seeds: Sequence[int]
                 ) -> list[tuple[SuiteItem, random.Random]]:
    """Pair every suite item with one RNG per seed."""
    return [(item, random.Random(int(seed))) for item in items for seed in seeds]


__all__ = [
    "SuiteItem", "WindowTFT", "ProbabilisticGrim", "ResetGrim",
    "SpitefulGrim", "DelayedPunisher", "build_heldout_v2_suite",
    "build_heldout_v3_suite", "build_heldout_v4_suite",
    "build_validation_suite", "build_final_suite", "build_suite",
    "suite_names", "materialize",
]
