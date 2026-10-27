"""Protocol v2: versioned train / dev / test opponent splits.

The old evaluation stack mixes training and reporting distributions: the four
``official`` distributions are also used by RL training, ``blind`` only changes
the field seed, and ``heldout_v2`` was reused as validation for several
experiments.  This module makes the split explicit and auditable.

Design
------
* An opponent *config* is a family, a concrete strategy type, and a parameter
  tuple.  A catalog of configs is generated deterministically.
* Parameterised families are split 70/15/15 into ``train`` / ``dev`` / ``test``
  with a deterministic, family-local shuffle.  Every tier therefore covers the
  full parameter range, but no concrete config is shared.
* ``dev_family`` and ``test_only`` families are held out by family, not just by
  parameter draw.
* ``test_v5`` is a weighted field: 60% seen-family test configs, 25% test-only
  families, 15% fixed anchors.  It is guarded by a usage ledger and must be
  consumed once per candidate family.
* Fixed anchors (AllC, AllD, TFT, Grim, Random.50) appear in every tier so the
  score remains interpretable; they are excluded from leakage checks because
  their behaviour is identical in all tiers.

The module is deliberately pure standard library so the audit and suite
builders work without PyTorch.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..core.opponents import (
    AdaptiveProbe, AllC, AllD, Bully, ContriteTFT, EarlyDefector,
    FirmButFair, GenerousTFT, Grim, Grudger2, GrudgerK, Joss,
    LateDefector, MemoryOne, Mistrust, NoisyTFT, OmegaTFTLite, Pavlov,
    Periodic, ProbeBot, Prober, RandomMoves, Reactive, Responsive3,
    Tester, TFT, TFT2, TFTLastD, TFTProbe, TwoStrike,
    WinStayLoseShift2,
)
from ..core.opponents.zero_determinant import ExtortZD, GenerousZD
from ..core.primitives import C, D, Strategy
from ..paths import CONFIG_DIR, RESULTS_DIR, ensure_parent
from .hard2 import DynamicWindow, HarvestFactor, Kamikaze
from .opponent_suite import (
    DelayedPunisher, ProbabilisticGrim, ResetGrim, SpitefulGrim, WindowTFT,
)

PROTOCOL_NAME = "protocol_v2"
PROTOCOL_VERSION = "2"
TIER_NAMES = ("train", "dev_param", "dev_family", "test_v5", "dev_red",
              "test_red", "legacy")
DEV_TIERS = ("dev_param", "dev_family", "dev_red")
TEST_TIERS = ("test_v5", "test_red")
SPLIT_TIERS = ("train", "dev_param", "dev_family", "test_v5")

# Group membership used by the field builder.  The values are mixture weights;
# a group may contain several families.
TIER_GROUPS: dict[str, dict[str, float]] = {
    "train": {"train": 0.95, "anchor": 0.05},
    "dev_param": {"dev": 0.95, "anchor": 0.05},
    "dev_family": {"dev_family": 0.85, "anchor": 0.15},
    "test_v5": {"test": 0.60, "test_only": 0.25, "anchor": 0.15},
}

# Relative family weights inside a group.  Counts are normalised per family so
# a large parameter grid does not dominate a small but important family.
FAMILY_WEIGHTS: dict[str, float] = {
    "nice": 2.0,
    "forgiving": 2.0,
    "punitive": 2.0,
    "defect": 1.5,
    "late": 2.0,
    "random": 1.5,
    "wild": 1.0,
    "adaptive": 1.5,
    "extort": 1.5,
    "temporal": 2.0,
    "noisy": 2.0,
    "anchor": 1.0,
    "early_harvest": 1.0,
    "frequent_kamikaze": 1.0,
    "probe_harvest": 1.0,
    "hyper_extort": 1.0,
    "early_probe": 1.0,
    "adaptive_probe": 1.0,
    "mirror_adaptive": 1.0,
    "delayed_grim": 1.0,
}

ANCHORS = ("AllC", "AllD", "TFT", "Grim", "Random.50")
DEV_ONLY_FAMILIES = ("mirror_adaptive", "delayed_grim")
TEST_ONLY_FAMILIES = ("early_harvest", "frequent_kamikaze",
                      "probe_harvest", "hyper_extort", "early_probe",
                      "adaptive_probe")

MANIFEST_PATH = CONFIG_DIR / "splits" / PROTOCOL_NAME / "manifest.json"
USAGE_PATH = RESULTS_DIR / PROTOCOL_NAME / "usage.json"

# Red policy pools.  These are the P3 held-out reds; they are disjoint between
# dev and test but are legacy artifacts, so the audit reports a warning until a
# fresh protocol-specific red pair is trained.
DEV_RED_POOL: tuple[dict[str, Any], ...] = (
    {"name": "Red-v5", "path": "weights/red_pool/red_rl_v5_heldout.json"},
)
TEST_RED_POOL: tuple[dict[str, Any], ...] = (
    {"name": "Red-v6", "path": "weights/red_pool/red_rl_v6_heldout.json"},
)


# ---------------------------------------------------------------------------
# Held-out opponent classes that are not part of the core bank
# ---------------------------------------------------------------------------


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


class ProbePhase(Strategy):
    """Test-only family: random early probe, then harvest or TFT."""

    def __init__(self, probe: int = 3, p_probe: float = 0.7,
                 phase: str = "harvest", harvest_start: int = 50,
                 rng: random.Random | None = None):
        self.probe = int(probe)
        self.p_probe = float(p_probe)
        self.phase = str(phase)
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


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _b_allc(rng: random.Random) -> Strategy:
    return AllC(rng)


def _b_alld(rng: random.Random) -> Strategy:
    return AllD(rng)


def _b_tft(rng: random.Random) -> Strategy:
    return TFT(rng)


def _b_tft2(rng: random.Random) -> Strategy:
    return TFT2(rng)


def _b_tftlastd(rng: random.Random) -> Strategy:
    return TFTLastD(rng)


def _b_pavlov(rng: random.Random) -> Strategy:
    return Pavlov(rng)


def _b_contrite(rng: random.Random) -> Strategy:
    return ContriteTFT(rng)


def _b_wsls2(rng: random.Random) -> Strategy:
    return WinStayLoseShift2(rng)


def _b_gtft(rng: random.Random, g: float) -> Strategy:
    return GenerousTFT(float(g), True, rng)


def _b_noisy(rng: random.Random, p: float) -> Strategy:
    return NoisyTFT(float(p), rng)


def _b_joss(rng: random.Random, p: float) -> Strategy:
    return Joss(float(p), rng)


def _b_two_strike(rng: random.Random, strikes: int) -> Strategy:
    return TwoStrike(int(strikes), True, rng)


def _b_grim(rng: random.Random) -> Strategy:
    return Grim(rng)


def _b_grudger2(rng: random.Random) -> Strategy:
    return Grudger2(rng)


def _b_grudgerk(rng: random.Random, k: int) -> Strategy:
    return GrudgerK(int(k), rng)


def _b_firm(rng: random.Random) -> Strategy:
    return FirmButFair(rng)


def _b_mistrust(rng: random.Random) -> Strategy:
    return Mistrust(rng)


def _b_bully(rng: random.Random) -> Strategy:
    return Bully(rng)


def _b_prober(rng: random.Random) -> Strategy:
    return Prober(rng)


def _b_tester(rng: random.Random) -> Strategy:
    return Tester(rng)


def _b_early(rng: random.Random, k: int) -> Strategy:
    return EarlyDefector(int(k), rng)


def _b_late(rng: random.Random, k: int) -> Strategy:
    return LateDefector(int(k), rng)


def _b_random(rng: random.Random, p: float) -> Strategy:
    return RandomMoves(float(p), rng)


def _b_periodic(rng: random.Random, pattern: Sequence[int],
                start: int) -> Strategy:
    return Periodic(tuple(int(x) for x in pattern), int(start), rng)


def _b_memone(rng: random.Random, p: Sequence[float]) -> Strategy:
    return MemoryOne(tuple(float(x) for x in p), first=C, rng=rng)


def _b_reactive(rng: random.Random, first: int, p_after_c: float,
                p_after_d: float) -> Strategy:
    return Reactive(int(first), float(p_after_c), float(p_after_d), rng=rng)


def _b_resp3(rng: random.Random, g_unprov: float, g_mutual: float,
             g_resp: float, min_c: int, min_d: int, resp: float,
             rate: float, max_tests: int, cooldown: int) -> Strategy:
    return Responsive3(
        float(g_unprov), float(g_mutual), float(g_resp), int(min_c),
        int(min_d), float(resp), float(rate), True, int(max_tests),
        int(cooldown), rng,
    )


def _b_probebot(rng: random.Random) -> Strategy:
    return ProbeBot(rng=rng)


def _b_omega(rng: random.Random) -> Strategy:
    return OmegaTFTLite(rng=rng)


def _b_extort(rng: random.Random, chi: float) -> Strategy:
    return ExtortZD(float(chi), rng)


def _b_generous_zd(rng: random.Random, chi: float) -> Strategy:
    return GenerousZD(float(chi), rng)


def _b_harvest(rng: random.Random, start: int, repairable: bool,
               p_defect: float) -> Strategy:
    return HarvestFactor(int(start), bool(repairable),
                         p_defect=float(p_defect), rng=rng)


def _b_dynamic(rng: random.Random, window: int, p_defect: float,
               retaliate: float, forgiveness: float) -> Strategy:
    return DynamicWindow(int(window), float(p_defect), float(retaliate),
                         float(forgiveness), rng=rng)


def _b_kamikaze(rng: random.Random, period: int, p_steal: float) -> Strategy:
    return Kamikaze(int(period), float(p_steal), rng)


def _b_prob_grim(rng: random.Random, patience: int,
                 p_punish: float) -> Strategy:
    return ProbabilisticGrim(int(patience), float(p_punish), rng)


def _b_reset_grim(rng: random.Random, patience: int,
                  cooldown: int) -> Strategy:
    return ResetGrim(int(patience), int(cooldown), rng)


def _b_spiteful_grim(rng: random.Random, patience: int) -> Strategy:
    return SpitefulGrim(int(patience), rng)


def _b_delayed_punisher(rng: random.Random, patience: int,
                        punish_rounds: int) -> Strategy:
    return DelayedPunisher(int(patience), int(punish_rounds), rng)


def _b_window_tft(rng: random.Random, window: int, retaliate: float,
                  forgive: float) -> Strategy:
    return WindowTFT(int(window), float(retaliate), float(forgive), rng)


def _b_probe_phase(rng: random.Random, probe: int, p_probe: float,
                   phase: str, harvest_start: int) -> Strategy:
    return ProbePhase(int(probe), float(p_probe), str(phase),
                      int(harvest_start), rng)


def _b_tft_probe(rng: random.Random, probe_round: int,
                 recover: int) -> Strategy:
    return TFTProbe(int(probe_round), int(recover), True, rng)


def _b_adaptive_probe(rng: random.Random, probe_round: int,
                      recover: int) -> Strategy:
    return AdaptiveProbe(int(probe_round), int(recover), True, rng)


def _b_mirror(rng: random.Random, window: int, forgive: float) -> Strategy:
    return MirrorAdaptive(int(window), float(forgive), rng)


def _b_delayed_grim(rng: random.Random, patience: int) -> Strategy:
    return DelayedGrim(int(patience), rng)


_BUILDERS: dict[str, Callable[..., Strategy]] = {
    "AllC": _b_allc,
    "AllD": _b_alld,
    "TFT": _b_tft,
    "TFT2": _b_tft2,
    "TFTLastD": _b_tftlastd,
    "Pavlov": _b_pavlov,
    "ContriteTFT": _b_contrite,
    "WSLS2": _b_wsls2,
    "GenerousTFT": _b_gtft,
    "NoisyTFT": _b_noisy,
    "Joss": _b_joss,
    "TwoStrike": _b_two_strike,
    "Grim": _b_grim,
    "Grudger2": _b_grudger2,
    "GrudgerK": _b_grudgerk,
    "FirmButFair": _b_firm,
    "Mistrust": _b_mistrust,
    "Bully": _b_bully,
    "Prober": _b_prober,
    "Tester": _b_tester,
    "EarlyDefector": _b_early,
    "LateDefector": _b_late,
    "RandomMoves": _b_random,
    "Periodic": _b_periodic,
    "MemoryOne": _b_memone,
    "Reactive": _b_reactive,
    "Responsive3": _b_resp3,
    "ProbeBot": _b_probebot,
    "OmegaTFTLite": _b_omega,
    "ExtortZD": _b_extort,
    "GenerousZD": _b_generous_zd,
    "HarvestFactor": _b_harvest,
    "DynamicWindow": _b_dynamic,
    "Kamikaze": _b_kamikaze,
    "ProbabilisticGrim": _b_prob_grim,
    "ResetGrim": _b_reset_grim,
    "SpitefulGrim": _b_spiteful_grim,
    "DelayedPunisher": _b_delayed_punisher,
    "WindowTFT": _b_window_tft,
    "ProbePhase": _b_probe_phase,
    "TFTProbe": _b_tft_probe,
    "AdaptiveProbe": _b_adaptive_probe,
    "MirrorAdaptive": _b_mirror,
    "DelayedGrim": _b_delayed_grim,
}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpponentConfig:
    """One concrete opponent configuration in the frozen catalog."""

    key: str
    family: str
    type_name: str
    params: tuple[tuple[str, Any], ...] = ()
    tier: str = "train"
    weight: float = 1.0
    anchor: bool = False

    @property
    def group(self) -> str:
        return "anchor" if self.anchor else self.tier

    def build(self, rng: random.Random) -> Strategy:
        if self.type_name not in _BUILDERS:
            raise KeyError(f"no builder for {self.type_name!r}")
        return _BUILDERS[self.type_name](rng, **dict(self.params))

    def signature(self) -> tuple[str, str]:
        return (self.type_name,
                json.dumps(dict(self.params), sort_keys=True, default=str))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "family": self.family,
            "type": self.type_name,
            "params": dict(self.params),
            "tier": self.tier,
            "weight": self.weight,
            "anchor": self.anchor,
        }


def _short_params(params: Mapping[str, Any]) -> str:
    if not params:
        return ""
    return "|".join(f"{k}={v}" for k, v in sorted(params.items()))


def _add(specs: list[dict[str, Any]], family: str, type_name: str,
         **params: Any) -> None:
    if type_name not in _BUILDERS:
        raise KeyError(f"unknown builder {type_name!r}")
    text = _short_params(params)
    key = f"{family}:{type_name}" + (f":{text}" if text else "")
    if any(item["key"] == key for item in specs):
        return
    specs.append({"key": key, "family": family, "type_name": type_name,
                  "params": tuple(sorted(params.items()))})


def _catalog_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    # Reciprocal / nice.
    for type_name in ("TFT2", "TFTLastD", "Pavlov", "ContriteTFT", "WSLS2"):
        _add(specs, "nice", type_name)
    for g in (0.02, 0.05, 0.09, 0.13, 0.18, 0.24, 0.31, 0.38, 0.44, 0.49):
        _add(specs, "forgiving", "GenerousTFT", g=g)
    _add(specs, "forgiving", "TwoStrike", strikes=2)
    _add(specs, "forgiving", "TwoStrike", strikes=3)

    # Punitive.
    for type_name in ("Grim", "Grudger2", "FirmButFair", "Mistrust"):
        _add(specs, "punitive", type_name)
    for k in (1, 2, 4, 5):
        _add(specs, "punitive", "GrudgerK", k=k)

    # Defectors / probing.
    for type_name in ("Bully", "Prober", "Tester"):
        _add(specs, "defect", type_name)
    for k in (2, 5, 12, 22):
        _add(specs, "defect", "EarlyDefector", k=k)

    # Endgame.
    for k in (9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79,
              84, 89, 94):
        _add(specs, "late", "LateDefector", k=k)

    # Random / periodic.
    for p in (0.02, 0.04, 0.07, 0.11, 0.16, 0.22, 0.27, 0.33, 0.38, 0.43,
              0.48, 0.53, 0.58, 0.63, 0.68, 0.73, 0.78, 0.83, 0.88, 0.93):
        _add(specs, "random", "RandomMoves", p=p)
    _add(specs, "random", "Periodic", pattern=(C, C, D, C, D), start=1)
    _add(specs, "random", "Periodic", pattern=(C, D, D, C), start=2)

    # Wild memory-one / reactive.
    for p in ((0.85, 0.10, 0.55, 0.05),
              (0.70, 0.25, 0.40, 0.10),
              (0.95, 0.40, 0.75, 0.02),
              (0.60, 0.05, 0.20, 0.30)):
        _add(specs, "wild", "MemoryOne", p=p)
    _add(specs, "wild", "Reactive", first=C, p_after_c=0.95, p_after_d=0.05)
    _add(specs, "wild", "Reactive", first=D, p_after_c=0.70, p_after_d=0.30)

    # Adaptive.
    _add(specs, "adaptive", "Responsive3", g_unprov=0.55, g_mutual=0.40,
         g_resp=0.95, min_c=5, min_d=3, resp=0.38, rate=0.18,
         max_tests=2, cooldown=8)
    _add(specs, "adaptive", "Responsive3", g_unprov=0.40, g_mutual=0.25,
         g_resp=0.88, min_c=4, min_d=4, resp=0.45, rate=0.22,
         max_tests=3, cooldown=12)
    _add(specs, "adaptive", "ProbeBot")
    _add(specs, "adaptive", "OmegaTFTLite")

    # Extortion / generous ZD.
    for chi in (1.1, 1.2, 1.35, 1.45, 1.65, 1.75, 1.85, 1.95, 2.05, 2.35,
                2.45, 2.65, 2.75, 2.85, 3.05, 3.15, 3.25, 3.45, 3.65, 3.85):
        _add(specs, "extort", "ExtortZD", chi=chi)
    for chi in (1.5, 2.0, 2.5):
        _add(specs, "extort", "GenerousZD", chi=chi)

    # Temporal.
    for start in (13, 17, 23, 29, 37, 43, 49, 53, 59, 67, 73, 79, 83, 91):
        _add(specs, "temporal", "HarvestFactor", start=start,
             repairable=start % 2 == 0, p_defect=0.95)
    for window in (2, 3, 5, 7, 8, 9, 10, 15, 16, 18):
        _add(specs, "temporal", "DynamicWindow", window=window, p_defect=0.18,
             retaliate=0.85, forgiveness=0.25)
    for period in (7, 10, 12, 16, 18, 20, 22, 24, 28, 31, 33):
        _add(specs, "temporal", "Kamikaze", period=period, p_steal=0.75)

    # Noisy reciprocators.
    for p in (0.04, 0.07, 0.11, 0.15, 0.19, 0.23, 0.27, 0.31, 0.36, 0.41,
              0.46, 0.49):
        _add(specs, "noisy", "NoisyTFT", p=p)
        _add(specs, "noisy", "Joss", p=p)

    # Test-only families: adversarial behaviours that initiate exploitation
    # rather than waiting to be provoked.  They are intentionally harder than
    # the train pool and are never used for training or model selection.
    for start in (4, 6, 8, 10, 12):
        _add(specs, "early_harvest", "HarvestFactor", start=start,
             repairable=False, p_defect=1.0)
    for period in (4, 5, 6, 7, 8):
        _add(specs, "frequent_kamikaze", "Kamikaze", period=period,
             p_steal=1.0)
    for probe, start in ((2, 35), (3, 40), (4, 45), (5, 50), (6, 55)):
        _add(specs, "probe_harvest", "ProbePhase", probe=probe,
             p_probe=0.8, phase="harvest", harvest_start=start)
    for chi in (4.0, 4.5, 5.0, 5.5, 6.0):
        _add(specs, "hyper_extort", "ExtortZD", chi=chi)
    for probe in (2, 4, 6, 8, 10):
        _add(specs, "early_probe", "TFTProbe", probe_round=probe, recover=2)
    for probe in (5, 10, 15, 20, 25):
        _add(specs, "adaptive_probe", "AdaptiveProbe", probe_round=probe,
             recover=2)

    # Dev-only families (held out from train and test).
    for window, forgive in ((3, 0.25), (5, 0.45), (7, 0.55), (10, 0.70),
                            (11, 0.15), (12, 0.60)):
        _add(specs, "mirror_adaptive", "MirrorAdaptive", window=window,
             forgive=forgive)
    for patience in (2, 4, 6, 7):
        _add(specs, "delayed_grim", "DelayedGrim", patience=patience)

    return specs


def _split_counts(n: int) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError("a split family needs at least three configs")
    n_train = max(1, int(round(0.70 * n)))
    n_dev = max(1, int(round(0.15 * n)))
    n_test = n - n_train - n_dev
    while n_test < 1:
        if n_train > n_dev:
            n_train -= 1
        else:
            n_dev -= 1
        n_test = n - n_train - n_dev
    return n_train, n_dev, n_test


def _family_tiers(family: str, n: int) -> list[str]:
    if family in TEST_ONLY_FAMILIES:
        return ["test_only"] * n
    if family in DEV_ONLY_FAMILIES:
        return ["dev_family"] * n
    n_train, n_dev, n_test = _split_counts(n)
    tiers = ["train"] * n_train + ["dev"] * n_dev + ["test"] * n_test
    rng = random.Random(int(
        hashlib.sha256(family.encode("utf-8")).hexdigest()[:16], 16))
    rng.shuffle(tiers)
    return tiers


@lru_cache(maxsize=1)
def build_catalog() -> tuple[OpponentConfig, ...]:
    """Build the deterministic, frozen opponent catalog."""
    specs = _catalog_specs()

    by_family: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        by_family.setdefault(spec["family"], []).append(spec)

    entries: list[OpponentConfig] = []
    for family, rows in sorted(by_family.items()):
        tiers = _family_tiers(family, len(rows))
        counts: dict[str, int] = {}
        for tier in tiers:
            counts[tier] = counts.get(tier, 0) + 1
        for row, tier in zip(rows, tiers):
            base = FAMILY_WEIGHTS.get(family, 1.0) / max(1, counts[tier])
            entries.append(OpponentConfig(
                key=row["key"], family=family, type_name=row["type_name"],
                params=row["params"], tier=tier, weight=base,
            ))

    # Fixed anchors are duplicated per tier; their behaviour is identical, so
    # they are exempt from the config-overlap audit.
    anchor_specs = {
        "AllC": ("AllC", ()),
        "AllD": ("AllD", ()),
        "TFT": ("TFT", ()),
        "Grim": ("Grim", ()),
        "Random.50": ("RandomMoves", (("p", 0.50),)),
    }
    for tier in ("train", "dev", "dev_family", "test"):
        for name in ANCHORS:
            type_name, params = anchor_specs[name]
            entries.append(OpponentConfig(
                key=f"anchor:{name}:{tier}", family="anchor",
                type_name=type_name, params=params, tier=tier,
                anchor=True, weight=1.0,
            ))

    entries.sort(key=lambda e: (e.tier, e.family, e.key))
    return tuple(entries)


def catalog_sha256(catalog: Sequence[OpponentConfig] | None = None) -> str:
    rows = catalog if catalog is not None else build_catalog()
    payload = json.dumps([entry.to_dict() for entry in rows],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def catalog_counts(catalog: Sequence[OpponentConfig] | None = None
                   ) -> dict[str, dict[str, int]]:
    rows = catalog if catalog is not None else build_catalog()
    out: dict[str, dict[str, int]] = {}
    for entry in rows:
        group = entry.group
        out.setdefault(group, {})
        out[group][entry.family] = out[group].get(entry.family, 0) + 1
    return out


def protocol_manifest() -> dict[str, Any]:
    catalog = build_catalog()
    return {
        "schema": "ipd-protocol-v2",
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "catalog_sha256": catalog_sha256(catalog),
        "catalog_size": len(catalog),
        "tiers": list(SPLIT_TIERS),
        "dev_only_families": list(DEV_ONLY_FAMILIES),
        "test_only_families": list(TEST_ONLY_FAMILIES),
        "anchors": list(ANCHORS),
        "group_weights": {k: dict(v) for k, v in TIER_GROUPS.items()},
        "red_pools": {
            "dev_red": [dict(entry) for entry in DEV_RED_POOL],
            "test_red": [dict(entry) for entry in TEST_RED_POOL],
        },
        "counts": catalog_counts(catalog),
        "entries": [entry.to_dict() for entry in catalog],
    }


def write_manifest(path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else MANIFEST_PATH
    ensure_parent(target)
    target.write_text(json.dumps(protocol_manifest(), ensure_ascii=False,
                                 indent=2), encoding="utf-8")
    return target


def read_manifest(path: str | Path | None = None) -> dict[str, Any] | None:
    target = Path(path) if path is not None else MANIFEST_PATH
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def verify_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest = read_manifest(path)
    current = catalog_sha256()
    if manifest is None:
        return {"ok": False, "reason": "manifest missing", "path": str(
            Path(path) if path is not None else MANIFEST_PATH),
            "catalog_sha256": current}
    expected = str(manifest.get("catalog_sha256", ""))
    return {
        "ok": expected == current,
        "reason": "ok" if expected == current else "catalog hash mismatch",
        "path": str(Path(path) if path is not None else MANIFEST_PATH),
        "manifest_sha256": expected,
        "catalog_sha256": current,
    }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _tier_families(catalog: Sequence[OpponentConfig],
                   tier: str) -> set[str]:
    return {e.family for e in catalog if e.tier == tier and not e.anchor}


def audit_protocol(catalog: Sequence[OpponentConfig] | None = None
                   ) -> dict[str, Any]:
    """Return a leakage / coverage report for the current catalog."""
    rows = catalog if catalog is not None else build_catalog()
    errors: list[str] = []
    warnings: list[str] = []

    group_names = ("train", "dev", "dev_family", "test", "test_only",
                   "anchor")
    counts = catalog_counts(rows)
    for group in group_names:
        if group not in counts:
            errors.append(f"empty group: {group}")

    # Exact concrete-config overlap (anchors exempt).
    seen: dict[tuple[str, str], OpponentConfig] = {}
    for entry in rows:
        if entry.anchor:
            continue
        sig = entry.signature()
        if sig in seen and seen[sig].tier != entry.tier:
            errors.append(
                f"config shared across tiers: {entry.key} "
                f"({seen[sig].tier} vs {entry.tier})")
        seen[sig] = entry

    train_families = _tier_families(rows, "train")
    dev_families = _tier_families(rows, "dev")
    dev_family_families = _tier_families(rows, "dev_family")
    test_families = _tier_families(rows, "test")
    test_only_families = _tier_families(rows, "test_only")

    for family in DEV_ONLY_FAMILIES:
        if family not in dev_family_families:
            errors.append(f"dev-only family missing from dev_family: {family}")
        if family in train_families or family in dev_families \
                or family in test_families or family in test_only_families:
            errors.append(f"dev-only family leaked: {family}")
    for family in TEST_ONLY_FAMILIES:
        if family not in test_only_families:
            errors.append(f"test-only family missing from test_only: {family}")
        if family in train_families or family in dev_families \
                or family in dev_family_families or family in test_families:
            errors.append(f"test-only family leaked: {family}")

    # Test must not silently inherit a dev-only family.
    leaked_dev = dev_family_families & test_only_families
    if leaked_dev:
        errors.append(f"dev/test family overlap: {sorted(leaked_dev)}")

    # Red pool separation.  The P3 reds are frozen artifacts, so they are
    # accepted but flagged until a protocol-specific pair is trained.
    from ..paths import ROOT
    dev_red = {str(entry["path"]) for entry in DEV_RED_POOL}
    test_red = {str(entry["path"]) for entry in TEST_RED_POOL}
    red_overlap = dev_red & test_red
    if red_overlap:
        errors.append(f"red pool overlap: {sorted(red_overlap)}")
    for path in sorted(dev_red | test_red):
        if not (ROOT / path).exists():
            errors.append(f"red policy missing: {path}")
    if dev_red and test_red:
        warnings.append("red pools are legacy P3 artifacts; retrain before final")

    # Coverage checks.
    for group in ("train", "dev", "dev_family", "test_only"):
        n = sum(count for count in counts.get(group, {}).values())
        if n < 5:
            warnings.append(f"small group {group!r}: {n} configs")

    manifest = read_manifest()
    manifest_check = verify_manifest()
    if manifest is not None and not manifest_check["ok"]:
        errors.append(manifest_check["reason"])
    elif manifest is None:
        warnings.append("manifest missing; run: python3 -m ipd protocol manifest")

    return {
        "ok": not errors,
        "protocol": PROTOCOL_NAME,
        "catalog_sha256": catalog_sha256(rows),
        "catalog_size": len(rows),
        "counts": counts,
        "tier_families": {
            "train": sorted(train_families),
            "dev_param": sorted(dev_families),
            "dev_family": sorted(dev_family_families),
            "test_seen": sorted(test_families),
            "test_only": sorted(test_only_families),
        },
        "manifest": manifest_check,
        "red_pools": {
            "dev_red": [entry["name"] for entry in DEV_RED_POOL],
            "test_red": [entry["name"] for entry in TEST_RED_POOL],
        },
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Field builder
# ---------------------------------------------------------------------------


def _weighted_entries(tier: str) -> tuple[list[OpponentConfig], list[float]]:
    if tier not in TIER_GROUPS:
        raise ValueError(f"unknown protocol tier {tier!r}; "
                         f"choose from {tuple(TIER_GROUPS)}")
    groups = TIER_GROUPS[tier]
    rows = [e for e in build_catalog() if e.group in groups]
    if not rows:
        raise ValueError(f"tier {tier!r} has no configs")
    counts: dict[tuple[str, str], int] = {}
    group_family_totals: dict[str, float] = {}
    for entry in rows:
        key = (entry.group, entry.family)
        counts[key] = counts.get(key, 0) + 1
        if entry.group not in group_family_totals:
            group_family_totals[entry.group] = 0.0
    for group, families in {
        group: {e.family for e in rows if e.group == group}
        for group in groups
    }.items():
        group_family_totals[group] = sum(
            FAMILY_WEIGHTS.get(family, 1.0) for family in families)
    weights: list[float] = []
    for entry in rows:
        family_count = max(1, counts[(entry.group, entry.family)])
        group_total = max(1e-12, group_family_totals[entry.group])
        weights.append(
            groups[entry.group]
            * (FAMILY_WEIGHTS.get(entry.family, 1.0) / group_total)
            / family_count
        )
    return rows, weights


def build_protocol_field(tier: str, seed: int, n: int = 299) -> list[Strategy]:
    """Build one deterministic field for a protocol tier.

    ``tier`` is one of ``train``, ``dev_param``, ``dev_family`` or
    ``test_v5``.  Field membership is fixed by the catalog; ``seed`` only
    changes the mixture and the per-opponent behaviour RNG.
    """
    rows, weights = _weighted_entries(tier)
    rng = random.Random(int(seed))
    chosen = rng.choices(rows, weights=weights, k=int(n))
    field: list[Strategy] = []
    for entry in chosen:
        bot = entry.build(random.Random(rng.randrange(1 << 30)))
        bot.name = f"{entry.family}:{entry.key}"
        field.append(bot)
    return field


def red_pool(tier: str) -> tuple[dict[str, Any], ...]:
    if tier == "dev_red":
        return DEV_RED_POOL
    if tier == "test_red":
        return TEST_RED_POOL
    raise ValueError(f"unknown red tier {tier!r}")


def build_protocol_red_field(tier: str, seed: int, n: int) -> list:
    """Build a field of frozen red policies for a red tier."""
    from ..paths import ROOT
    from .policy_pool import build_policy_field

    entries = [
        {"name": entry["name"],
         "path": str(ROOT / str(entry["path"])),
         "weight": float(entry.get("weight", 1.0))}
        for entry in red_pool(tier)
    ]
    return build_policy_field(entries, seed=seed, n=n, mode="sample")


# ---------------------------------------------------------------------------
# Usage ledger (test consumption)
# ---------------------------------------------------------------------------


def _usage_target(path: str | Path | None) -> Path:
    return Path(path) if path is not None else USAGE_PATH


def load_usage(path: str | Path | None = None) -> dict[str, Any]:
    target = _usage_target(path)
    if not target.exists():
        return {"schema": "ipd-protocol-usage-v1", "protocol": PROTOCOL_NAME,
                "catalog_sha256": catalog_sha256(), "entries": []}
    return json.loads(target.read_text(encoding="utf-8"))


def _save_usage(data: Mapping[str, Any], path: str | Path | None) -> Path:
    target = _usage_target(path)
    ensure_parent(target)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def guard_test_usage(candidate: str, *, path: str | Path | None = None,
                     force: bool = False,
                     tier: str = "test_v5") -> dict[str, Any]:
    """Return the prior test entry, or raise if ``candidate`` already used it."""
    data = load_usage(path)
    prior = [e for e in data.get("entries", [])
             if e.get("tier") == tier
             and e.get("candidate") == candidate
             and not e.get("contaminated")]
    if prior and not force:
        raise RuntimeError(
            f"{candidate!r} already consumed test_v5; pass --force to rerun "
            "and mark the result contaminated")
    return data


def record_usage(tier: str, candidate: str, result_path: str, *,
                 path: str | Path | None = None, force: bool = False,
                 note: str = "") -> Path:
    """Append one usage entry; a forced rerun is marked contaminated."""
    data = load_usage(path)
    contaminated = bool(force and any(
        e.get("tier") == tier and e.get("candidate") == candidate
        for e in data.get("entries", [])))
    data.setdefault("entries", []).append({
        "tier": tier,
        "candidate": candidate,
        "result": str(result_path),
        "catalog_sha256": catalog_sha256(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "contaminated": contaminated,
        "note": note,
    })
    data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return _save_usage(data, path)


__all__ = [
    "PROTOCOL_NAME", "PROTOCOL_VERSION", "TIER_NAMES", "DEV_TIERS",
    "TEST_TIERS", "SPLIT_TIERS", "TIER_GROUPS", "FAMILY_WEIGHTS",
    "ANCHORS", "DEV_ONLY_FAMILIES", "TEST_ONLY_FAMILIES",
    "MANIFEST_PATH", "USAGE_PATH", "OpponentConfig", "MirrorAdaptive",
    "DelayedGrim", "build_catalog", "catalog_sha256", "catalog_counts",
    "protocol_manifest", "write_manifest", "read_manifest", "verify_manifest",
    "audit_protocol", "build_protocol_field", "build_protocol_red_field",
    "red_pool", "load_usage", "guard_test_usage", "record_usage",
]
