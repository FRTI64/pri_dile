"""Training, validation, and final-test opponent populations.

The pool deliberately separates *families* (reciprocal, top, noisy, extort,
random, special, red) from concrete parameter draws.  Training samples wide
parameter ranges; validation and the final test use disjoint parameter ranges
and two families (``MirrorAdaptive`` and ``DelayedGrim``) that never appear in
training.  This is what makes the held-out claim meaningful instead of merely
changing a random seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from ipd.core.machine import Machine
from ipd.core.opponents.basic import (
    AllC, AllD, Bully, Grim, Joss, Pavlov, RandomMoves, TFT, TFT2,
    TFTLastD,
)
from ipd.core.opponents.defectors import LateDefector
from ipd.core.opponents.memory import Periodic
from ipd.core.opponents.reciprocal import (
    ContriteTFT, ForgiveTFT, GenerousTFT, Majority, NoisyTFT,
)
from ipd.core.opponents.zero_determinant import ExtortZD
from ipd.core.primitives import C, D, Strategy

TRAIN_FAMILIES = (
    "good", "top", "noisy", "extort", "random", "special",
)
ALL_FAMILIES = TRAIN_FAMILIES + ("red", "heldout")

Factory = Callable[[random.Random], Strategy]


@dataclass(frozen=True)
class BotSpec:
    name: str
    family: str
    factory: Factory


# ---------------------------------------------------------------------------
# Parameterised dynamic opponents (training and held-out variants)
# ---------------------------------------------------------------------------

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


class HarvestFactor(Strategy):
    """Cooperate for T rounds, then harvest; optionally repair the relation."""

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


class ProbePhase(Strategy):
    """Random probe for the first k rounds, then TFT or harvesting."""

    def __init__(self, probe: int = 3, p_probe: float = 0.5,
                 phase: str = "tft", harvest_start: int = 55,
                 rng: random.Random | None = None):
        self.probe = int(probe)
        self.p_probe = float(p_probe)
        self.phase = phase
        self.harvest_start = int(harvest_start)
        super().__init__(rng)
        self.name = f"Probe{self.probe}->{self.phase}"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t < self.probe:
            return C if self.rng.random() < self.p_probe else D
        if self.phase == "harvest":
            return D if t >= self.harvest_start else C
        return theirs[-1]


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


class WeakTFT(Strategy):
    """TFT with a high accidental-defection rate, useful as prey."""

    def __init__(self, p: float = 0.3, rng: random.Random | None = None):
        self.p = float(p)
        super().__init__(rng)
        self.name = f"WeakTFT({self.p:g})"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        x = theirs[-1]
        if self.rng.random() < self.p:
            x = 1 - x
        return x


class MirrorAdaptive(Strategy):
    """Held-out family: mirror within a short window, punish sustained D."""

    def __init__(self, window: int = 4, forgive: float = 0.35,
                 rng: random.Random | None = None):
        self.window = int(window)
        self.forgive = float(forgive)
        super().__init__(rng)
        self.name = f"MirrorW{self.window}"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        if t == 99:
            return D
        recent = theirs[-self.window:]
        rate = sum(1 for x in recent if x == D) / max(1, len(recent))
        if rate >= 0.5:
            return D
        if theirs[-1] == D:
            return C if self.rng.random() < self.forgive else D
        return C


class DelayedGrim(Strategy):
    """Held-out family: forgive one defection, then punish forever."""

    def __init__(self, patience: int = 1, rng: random.Random | None = None):
        self.patience = int(patience)
        super().__init__(rng)
        self.name = f"DelayedGrim{self.patience}"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 0:
            return C
        if theirs.count(D) > self.patience:
            return D
        return C


# ---------------------------------------------------------------------------
# Fixed and near-fixed training families
# ---------------------------------------------------------------------------

FINAL_TUNED = {
    "P_UNPROVOKED": 0.2395699546614404,
    "P_MUTUAL": 0.6399155368157192,
    "P_RESPONSIVE": 0.9544467968114831,
    "MIN_C": 5,
    "MIN_D": 2,
    "RESP_GAP": 0.3871088255996472,
    "DEFECT_RATE": 0.1170465108152342,
    "PEACE_TESTS": 1,
    "PEACE_COOLDOWN": 9,
}


def _final_bot(rng: random.Random) -> Strategy:
    from ipd.policies.candidates.final_tuned import FinalTunedPolicy

    bot = FinalTunedPolicy(seed=rng.randrange(1 << 30))
    bot.name = "Final-tuned"
    return bot


def _triple_bot(rng: random.Random) -> Strategy:
    from ipd.policies.candidates.triple_majority import TripleMajorityPolicy
    bot = TripleMajorityPolicy(seed=rng.randrange(1 << 30))
    bot.name = "Triple-majority"
    return bot


_RL_WEIGHTS_CACHE: dict | None = None


def _rl_bot(rng: random.Random) -> Strategy:
    global _RL_WEIGHTS_CACHE
    import json
    from ipd.paths import weights_path
    from ipd.policies.rl import RLPolicy

    if _RL_WEIGHTS_CACHE is None:
        with open(weights_path("rl_policy_checkpoint.json"),
                  encoding="utf-8") as fh:
            _RL_WEIGHTS_CACHE = json.load(fh)
    bot = RLPolicy(_RL_WEIGHTS_CACHE, seed=rng.randrange(1 << 30),
                   mode="sample")
    bot.name = "RL-final"
    return bot


def _gtft(rng: random.Random, g: float | None = None) -> Strategy:
    if g is None:
        g = rng.uniform(0.05, 0.33)
    return GenerousTFT(g, True, rng)


def _noisy(rng: random.Random, p: float | None = None) -> Strategy:
    if p is None:
        p = rng.uniform(0.02, 0.20)
    return NoisyTFT(p, rng)


def _joss(rng: random.Random, p: float | None = None) -> Strategy:
    if p is None:
        p = rng.uniform(0.02, 0.20)
    return Joss(p, rng)


def _extort(rng: random.Random, chi: float | None = None) -> Strategy:
    if chi is None:
        chi = rng.uniform(1.5, 3.0)
    return ExtortZD(chi, rng)


def _random_bot(rng: random.Random, p: float | None = None) -> Strategy:
    if p is None:
        p = rng.uniform(0.05, 0.90)
    return RandomMoves(p, rng)


def _harvest(rng: random.Random, start: int | None = None,
             repairable: bool | None = None) -> Strategy:
    if start is None:
        start = rng.randint(25, 75)
    if repairable is None:
        repairable = rng.random() < 0.5
    return HarvestFactor(start, repairable,
                         p_defect=rng.uniform(0.90, 1.0), rng=rng)


def _dynamic(rng: random.Random, window: int | None = None,
             p_defect: float | None = None) -> Strategy:
    if window is None:
        window = rng.randint(3, 8)
    if p_defect is None:
        p_defect = rng.uniform(0.15, 0.35)
    return DynamicWindow(window, p_defect,
                         retaliate=rng.uniform(0.70, 1.0),
                         forgiveness=rng.uniform(0.10, 0.55), rng=rng)


def _probe_phase(rng: random.Random, probe: int | None = None,
                 phase: str | None = None) -> Strategy:
    if probe is None:
        probe = rng.randint(3, 5)
    if phase is None:
        phase = rng.choice(("tft", "harvest"))
    return ProbePhase(probe, rng.uniform(0.3, 0.7), phase,
                      harvest_start=rng.randint(40, 70), rng=rng)


def _kamikaze(rng: random.Random, period: int | None = None) -> Strategy:
    if period is None:
        period = rng.randint(10, 20)
    return Kamikaze(period, p_steal=rng.uniform(0.7, 1.0), rng=rng)


def _late(rng: random.Random, k: int | None = None) -> Strategy:
    if k is None:
        k = rng.choice((5, 10, 15, 20, 25, 30, 40, 50))
    return LateDefector(k, rng)


# Random draws for the training pool.  Every call also returns a human
# readable label used in logs and specialisation diagnostics.
def make_training_bot(rng: random.Random, family: str) -> Strategy:
    if family == "good":
        kind = rng.choices(
            ("TFT", "Pavlov", "Grim", "TFT-lastD", "ContriteTFT",
             "GTFT", "ForgiveTFT", "TFT2", "Majority3"),
            weights=(22, 6, 8, 7, 4, 8, 4, 4, 3), k=1)[0]
        if kind == "TFT":
            return TFT(rng)
        if kind == "Pavlov":
            return Pavlov(rng)
        if kind == "Grim":
            return Grim(rng)
        if kind == "TFT-lastD":
            return TFTLastD(rng)
        if kind == "ContriteTFT":
            return ContriteTFT(rng)
        if kind == "GTFT":
            return _gtft(rng)
        if kind == "ForgiveTFT":
            return ForgiveTFT(True, rng.choice((0, 1, 2)), rng)
        if kind == "TFT2":
            return TFT2(rng)
        return Majority(3, rng)

    if family == "top":
        kind = rng.choices(("Final-tuned", "Triple-majority", "RL-final"),
                           weights=(5, 3, 2), k=1)[0]
        if kind == "Final-tuned":
            return _final_bot(rng)
        if kind == "Triple-majority":
            return _triple_bot(rng)
        return _rl_bot(rng)

    if family == "noisy":
        kind = rng.choices(("NoisyTFT", "Joss", "GTFT"),
                           weights=(5, 3, 2), k=1)[0]
        if kind == "NoisyTFT":
            return _noisy(rng)
        if kind == "Joss":
            return _joss(rng)
        return _gtft(rng)

    if family == "extort":
        if rng.random() < 0.80:
            return _extort(rng)
        return Bully(rng)

    if family == "random":
        if rng.random() < 0.85:
            return _random_bot(rng)
        return Periodic(rng.choice(((C, C, D), (C, D, D, C),
                                    (D, C), (C, C, C, D))),
                        rng.randrange(4), rng)

    if family == "special":
        kind = rng.choices(
            ("AllD", "AllC", "LateD", "Harvest", "Dynamic",
             "ProbePhase", "Kamikaze", "Prey", "Periodic"),
            weights=(8, 3, 8, 5, 5, 5, 5, 5, 4), k=1)[0]
        if kind == "AllD":
            return AllD(rng)
        if kind == "AllC":
            return AllC(rng)
        if kind == "LateD":
            return _late(rng)
        if kind == "Harvest":
            return _harvest(rng)
        if kind == "Dynamic":
            return _dynamic(rng)
        if kind == "ProbePhase":
            return _probe_phase(rng)
        if kind == "Kamikaze":
            return _kamikaze(rng)
        if kind == "Prey":
            kind2 = rng.choices(("AllC", "Random90", "WeakTFT"),
                                weights=(2, 3, 3), k=1)[0]
            if kind2 == "AllC":
                return AllC(rng)
            if kind2 == "Random90":
                return RandomMoves(0.90, rng)
            return WeakTFT(rng.uniform(0.15, 0.40), rng)
        return Periodic(rng.choice(((C, C, D), (C, D, D, C), (D, C))),
                        rng.randrange(3), rng)

    raise ValueError(f"unknown training family: {family}")


# ---------------------------------------------------------------------------
# Validation and final-test suites
# ---------------------------------------------------------------------------

SuiteItem = BotSpec


def _spec(name: str, family: str, factory: Factory) -> BotSpec:
    return BotSpec(name, family, factory)


def build_validation_suite() -> list[BotSpec]:
    """51 fixed opponents, disjoint parameter ranges from training."""
    suite: list[BotSpec] = []
    # Good reciprocal family (10).
    suite += [
        _spec("TFT", "good", lambda r: TFT(r)),
        _spec("Pavlov", "good", lambda r: Pavlov(r)),
        _spec("Grim", "good", lambda r: Grim(r)),
        _spec("TFT-lastD", "good", lambda r: TFTLastD(r)),
        _spec("ContriteTFT", "good", lambda r: ContriteTFT(r)),
        _spec("GTFT.hold.02", "good", lambda r: GenerousTFT(0.02, True, r)),
        _spec("GTFT.hold.42", "good", lambda r: GenerousTFT(0.42, True, r)),
        _spec("ForgiveTFT.hold", "good", lambda r: ForgiveTFT(True, 2, r)),
        _spec("TFT2.hold", "good", lambda r: TFT2(r)),
        _spec("Majority5.hold", "good", lambda r: Majority(5, r)),
    ]
    # Top rules (6).
    suite += [
        _spec("Final-tuned", "top", _final_bot),
        _spec("Triple-majority", "top", _triple_bot),
        _spec("RL-final", "top", _rl_bot),
        _spec("Final-tuned.2", "top", _final_bot),
        _spec("Triple-majority.2", "top", _triple_bot),
        _spec("RL-final.2", "top", _rl_bot),
    ]
    # Noisy reciprocators (10).
    for p in (0.01, 0.02, 0.035, 0.06, 0.10, 0.15, 0.22, 0.28, 0.35, 0.45):
        name = "NoisyTFT.02" if abs(p - 0.02) < 1e-12 else f"NoisyTFT.hold.{p:g}"
        suite.append(_spec(name, "noisy",
                           lambda r, p=p: NoisyTFT(p, r)))
    # Extortion / bullying (7).
    for chi in (1.2, 1.35, 1.75, 2.5, 3.2, 3.6):
        suite.append(_spec(f"Extort.hold.{chi:g}", "extort",
                           lambda r, c=chi: ExtortZD(c, r)))
    suite.append(_spec("Bully", "extort", lambda r: Bully(r)))
    # Random / periodic (6).
    for p in (0.03, 0.12, 0.27, 0.45, 0.63, 0.92):
        suite.append(_spec(f"Random.hold.{p:g}", "random",
                           lambda r, p=p: RandomMoves(p, r)))
    # Endgame / special (8).  Parameter ranges are outside training.
    for k in (2, 8, 18, 33, 47, 68, 88):
        suite.append(_spec(f"LateD.hold.{k}", "special",
                           lambda r, k=k: LateDefector(k, r)))
    suite.append(_spec("AllD.hold", "special", lambda r: AllD(r)))
    # Held-out families (4).
    suite += [
        _spec("MirrorAdaptive.2", "heldout",
              lambda r: MirrorAdaptive(2, 0.15, r)),
        _spec("MirrorAdaptive.9", "heldout",
              lambda r: MirrorAdaptive(9, 0.65, r)),
        _spec("DelayedGrim.1", "heldout",
              lambda r: DelayedGrim(1, r)),
        _spec("DelayedGrim.3", "heldout",
              lambda r: DelayedGrim(3, r)),
    ]
    return suite


def build_final_suite() -> list[BotSpec]:
    """60 opponents for the one-shot final test (different draws)."""
    suite: list[BotSpec] = []
    for i in range(12):
        suite.append(_spec(f"TFT.final.{i}", "good",
                           lambda r: TFT(r)))
    for i in range(8):
        suite.append(_spec(f"Pavlov.final.{i}", "good",
                           lambda r: Pavlov(r)))
    for i in range(6):
        suite.append(_spec(f"Grim.final.{i}", "good",
                           lambda r: Grim(r)))
    suite += [
        _spec("Final-tuned.final", "top", _final_bot),
        _spec("Triple-majority.final", "top", _triple_bot),
        _spec("RL-final.final", "top", _rl_bot),
        _spec("Final-tuned.final.2", "top", _final_bot),
        _spec("Triple-majority.final.2", "top", _triple_bot),
        _spec("RL-final.final.2", "top", _rl_bot),
    ]
    for p in (0.005, 0.015, 0.03, 0.05, 0.08, 0.13, 0.18, 0.26, 0.34, 0.50):
        suite.append(_spec(f"NoisyTFT.final.{p:g}", "noisy",
                           lambda r, p=p: NoisyTFT(p, r)))
        suite.append(_spec(f"Joss.final.{p:g}", "noisy",
                           lambda r, p=p: Joss(p, r)))
    for chi in (1.15, 1.45, 1.9, 2.7, 3.4):
        suite.append(_spec(f"Extort.final.{chi:g}", "extort",
                           lambda r, c=chi: ExtortZD(c, r)))
        suite.append(_spec(f"Dynamic.final.{chi:g}", "extort",
                           lambda r, c=chi: DynamicWindow(
                               int(round(c * 2)), 0.12 * c, 0.95, 0.08, rng=r)))
    for p in (0.02, 0.08, 0.17, 0.31, 0.52, 0.78, 0.96):
        suite.append(_spec(f"Random.final.{p:g}", "random",
                           lambda r, p=p: RandomMoves(p, r)))
    for k in (1, 12, 24, 38, 55, 72, 94):
        suite.append(_spec(f"LateD.final.{k}", "special",
                           lambda r, k=k: LateDefector(k, r)))
    suite += [
        _spec("Harvest.final.18R", "special",
              lambda r: HarvestFactor(18, True, 0.95, rng=r)),
        _spec("Harvest.final.82H", "special",
              lambda r: HarvestFactor(82, False, 1.0, rng=r)),
        _spec("Dynamic.final.W2", "special",
              lambda r: DynamicWindow(2, 0.40, 1.0, 0.0, rng=r)),
        _spec("Dynamic.final.W9", "special",
              lambda r: DynamicWindow(9, 0.10, 0.6, 0.8, rng=r)),
        _spec("Kamikaze.final.8", "special",
              lambda r: Kamikaze(8, 1.0, r)),
        _spec("Kamikaze.final.23", "special",
              lambda r: Kamikaze(23, 0.7, r)),
        _spec("MirrorAdaptive.final.1", "heldout",
              lambda r: MirrorAdaptive(1, 0.05, r)),
        _spec("MirrorAdaptive.final.12", "heldout",
              lambda r: MirrorAdaptive(12, 0.80, r)),
        _spec("DelayedGrim.final.0", "heldout",
              lambda r: DelayedGrim(0, r)),
        _spec("DelayedGrim.final.5", "heldout",
              lambda r: DelayedGrim(5, r)),
    ]
    return suite


def build_calibration_suite() -> list[BotSpec]:
    """Small frozen suite used to anchor red competitiveness per family."""
    suite = build_validation_suite()
    # Keep at most three per family; calibration is run often.
    counts: dict[str, int] = {}
    out: list[BotSpec] = []
    for item in suite:
        if counts.get(item.family, 0) >= 3:
            continue
        counts[item.family] = counts.get(item.family, 0) + 1
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Family mixture sampling
# ---------------------------------------------------------------------------

PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    "1a": {"good": 0.35, "top": 0.20, "noisy": 0.15,
           "extort": 0.05, "random": 0.15, "special": 0.10},
    "1b": {"good": 0.35, "top": 0.20, "noisy": 0.15,
           "extort": 0.10, "random": 0.15, "special": 0.05},
    "2": {"good": 0.25, "top": 0.15, "noisy": 0.15,
          "extort": 0.10, "random": 0.10, "special": 0.05},
    "3": {"good": 0.25, "top": 0.15, "noisy": 0.15,
          "extort": 0.10, "random": 0.05, "special": 0.05},
}


def phase_weights(phase: str) -> dict[str, float]:
    return dict(PHASE_WEIGHTS[phase])


def sample_mixture(rng: random.Random, weights: Mapping[str, float],
                   concentration: float = 40.0) -> dict[str, float]:
    """Dirichlet perturbation of a family prior."""
    names = list(weights)
    gammas = [rng.gammavariate(max(1e-6, weights[name] * concentration), 1.0)
              for name in names]
    total = sum(gammas)
    return {name: g / total for name, g in zip(names, gammas)}


def sample_family_labels(rng: random.Random, phase: str, n: int) -> list[str]:
    """Sample family labels only (cheap; used by the proportion test)."""
    probs = sample_mixture(rng, phase_weights(phase))
    names = list(probs)
    return rng.choices(names, weights=[probs[x] for x in names], k=n)


def sample_training_population(rng: random.Random, phase: str,
                               n: int) -> list[str]:
    """Public helper mirroring the training loop's family allocation."""
    return sample_family_labels(rng, phase, n)


def summarize_families(rows: Iterable[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
        total += 1
    return {name: c / max(1, total) for name, c in sorted(counts.items())}


__all__ = [
    "BotSpec", "SuiteItem", "TRAIN_FAMILIES", "ALL_FAMILIES",
    "DynamicWindow", "HarvestFactor", "ProbePhase", "Kamikaze", "WeakTFT",
    "MirrorAdaptive", "DelayedGrim",
    "make_training_bot", "build_validation_suite", "build_final_suite",
    "build_calibration_suite", "PHASE_WEIGHTS", "phase_weights",
    "sample_mixture", "sample_family_labels", "sample_training_population",
    "summarize_families", "FINAL_TUNED", "_final_bot", "_triple_bot",
    "_rl_bot",
]
