"""Protocol v4: versioned train / dev / test opponent splits.

The old evaluation stack mixes training and reporting distributions: the four
``official`` distributions are also used by RL training, ``blind`` only changes
the field seed, and ``heldout_v2`` was reused as validation for several
experiments.  This module makes the split explicit and auditable.

Design
------
* An opponent *config* is a family, a concrete strategy type, and a parameter
  tuple.  A catalog of configs is generated deterministically.
* Standard and complex families are split 70/15/15 into disjoint
  train/dev/test configs with a family-local shuffle.  Every tier covers the
  full parameter range, but no concrete config is shared.
* ``dev_family`` and ``test_only`` families are held out by family, not just
  by parameter draw.  ``dev_complex`` validates the complex families and
  ``dev_ood`` is a deliberately shifted validation mixture.
* ``test_v7`` is a weighted field: 30% seen-family test configs, 20%
  test-only adversarial families, 15% complex families, 10% policy opponents,
  15% Axelrod opponents and 10% anchors.  It is guarded by a usage ledger and
  by ``test_policy.json``; Protocol v4 locks consumption, so CLI test runs are
  provisional unless the policy is explicitly changed.
* Training profiles (``train_v3`` / ``train_easy`` / ``train_hard`` /
  ``train_complex`` / ``train_policy``) provide different curricula over the
  same frozen catalog.
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
import zlib
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..core.opponents import (
    AdaptiveProbe, AllC, AllD, Bully, ContriteTFT, EarlyDefector,
    FirmButFair, ForgiveTFT, GenerousTFT, Gradual, Grim, GrimForgive, Grudger2,
    GrudgerK, Joss, LastKDefect, LateDefector, Majority, MemoryOne,
    Mistrust, NoisyTFT, OmegaTFTLite, Pavlov, Periodic, ProbeBot,
    Prober, RandomFSM, RandomMoves, RandomThenTFT, Reactive,
    RemorsefulProber, Responsive, Responsive2, Responsive3,
    SuspiciousTFT, Tester, TFT, TFT2, TFT2LastD, TFTForgiveOnce,
    TFTLastD, TFTNoisyForgive, TFTProbe, TFTProbeImmediate,
    TwoStrike, TwoTitsForTat, WinStayLoseShift2,
)
from ..core.opponents.zero_determinant import ExtortZD, GenerousZD
from ..core.primitives import C, D, Strategy
from ..paths import CONFIG_DIR, RESULTS_DIR, ROOT, ensure_parent
from .hard2 import DynamicWindow, HarvestFactor, Kamikaze
from .opponent_suite import (
    DelayedPunisher, ProbabilisticGrim, ResetGrim, SpitefulGrim, WindowTFT,
)

PROTOCOL_NAME = "protocol_v4"
PROTOCOL_VERSION = "4"
TEST_TIER = "test_v7"
TIER_NAMES = ("train", "dev_param", "dev_family", "dev_complex",
              "dev_axelrod", "dev_ood", TEST_TIER, "dev_red", "test_red",
              "legacy")
DEV_TIERS = ("dev_param", "dev_family", "dev_complex", "dev_axelrod",
             "dev_ood", "dev_red")
TEST_TIERS = (TEST_TIER, "test_red")
LEVEL_TIERS = tuple(f"dev_level{i}" for i in range(5))
COMPLEXITY_TIERS = tuple(f"dev_complexity{i}" for i in range(5))
SPLIT_TIERS = ("train", "dev_param", "dev_family", "dev_complex",
               "dev_axelrod", "dev_ood", TEST_TIER) \
    + LEVEL_TIERS + COMPLEXITY_TIERS

# Larger default match sizes for Protocol v4.  The old v3 baseline used
# 299/20/40; these constants make the new scale explicit for CLI and docs.
# 2048 opponents per field gives a 2000+ capacity while keeping field sizes
# close to a power of two for deterministic sampling.
PROTOCOL_N_PLAYERS = 2048
DEV_FIELDS = 30
TEST_FIELDS = 50

# Group membership used by the field builder.  The values are mixture weights;
# a group may contain several families.  Protocol v4 adds complex, policy and
# Axelrod groups, and validation mixtures (dev_axelrod / dev_ood).
TIER_GROUPS: dict[str, dict[str, float]] = {
    "train": {"train": 0.70, "train_complex": 0.12,
              "train_policy": 0.05, "train_axelrod": 0.10,
              "anchor": 0.03},
    "dev_param": {"dev": 0.80, "dev_axelrod": 0.15, "anchor": 0.05},
    "dev_family": {"dev_family": 0.85, "anchor": 0.15},
    "dev_complex": {"dev_complex": 0.90, "anchor": 0.10},
    "dev_axelrod": {"dev_axelrod": 0.90, "anchor": 0.10},
    "dev_ood": {"dev": 0.30, "dev_complex": 0.25,
                "dev_family": 0.10, "dev_axelrod": 0.15,
                "anchor": 0.20},
    TEST_TIER: {"test": 0.30, "test_only": 0.20,
                "test_complex": 0.15, "test_policy": 0.10,
                "test_axelrod": 0.15, "anchor": 0.10},
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
    "wild": 1.5,
    "adaptive": 2.0,
    "extort": 1.8,
    "temporal": 2.2,
    "noisy": 2.0,
    "phase_switch": 1.2,
    "adaptive_mix": 1.2,
    "probe_exploit": 1.0,
    "delayed_extort": 1.0,
    "memory_two": 1.0,
    "meta_mix": 1.0,
    "noise_adaptive": 1.1,
    "reputation_grim": 1.0,
    "contrition_cycle": 1.0,
    "endgame_hunter": 1.1,
    "pattern_breaker": 1.0,
    "conditional_extort": 1.1,
    "policy": 1.5,
    "axelrod": 1.5,
    "axelrod_extra": 1.2,
    "axelrod_mix": 1.2,
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
COMPLEX_FAMILIES = ("phase_switch", "adaptive_mix", "probe_exploit",
                    "delayed_extort", "memory_two", "meta_mix",
                    "noise_adaptive", "reputation_grim",
                    "contrition_cycle", "endgame_hunter",
                    "pattern_breaker", "conditional_extort",
                    "lookahead", "portfolio")
POLICY_FAMILIES = ("policy",)
AXELROD_FAMILIES = ("axelrod", "axelrod_extra", "axelrod_mix")

# Curated Axelrod 4.14.0 classics plus modern/adaptive/ZD strategies.  These
# names are resolved lazily, so building the catalog does not require Axelrod;
# building a field does.
AXELROD_STRATEGIES = (
    "TitForTat", "TitFor2Tats", "TwoTitsForTat", "Grudger", "Forgiver",
    "ForgivingTitForTat", "GTFT", "ContriteTitForTat", "FirmButFair",
    "WinStayLoseShift", "GoByMajority", "HardGoByMajority", "SoftGrudger",
    "Grumpy", "Prober", "RemorsefulProber", "NaiveProber", "HardProber",
    "Adaptive", "AdaptiveTitForTat", "BackStabber", "DoubleCrosser",
    "Bully", "Predator", "Ripoff", "SneakyTitForTat", "TrickyCooperator",
    "TrickyDefector", "Alternator", "CyclerCCD", "CyclerCCCD",
    "RandomTitForTat", "Random", "Defector", "Cooperator", "ZDExtort2",
    "ZDExtort3", "ZDExtortion", "ZDGTFT2", "MetaHunter", "EvolvedFSM16",
    "EvolvedANN", "PSOGambler1_1_1", "Calculator", "OmegaTFT", "Stalker",
    "Thumper", "LimitedRetaliate",
    # Fast high-level additions (no long-run Meta players).
    "Winner12", "Winner21", "EvolvedHMM5", "EvolvedFSM16Noise05",
    "EvolvedANNNoise05", "PSOGambler2_2_2", "PSOGamblerMem1",
    "KnowledgeableWorseAndWorse", "ThueMorse", "ThueMorseInverse",
    "LimitedRetaliate3", "Retaliate2", "Retaliate3", "HardTitFor2Tats",
    "SlowTitForTwoTats2", "DynamicTwoTitsForTat", "MEM2", "ShortMem",
)

# Remaining Axelrod 4.14.0 fast bank (158 strategies), kept as a separate
# family so the curated core remains lightweight.  Long-run Meta players are
# intentionally excluded.
AXELROD_EXTRA = (
    "ALLCorALLD", "AON2", "APavlov2006", "APavlov2011", "AdaptorBrief",
    "AdaptorLong", "Aggravater", "Alexei", "AlternatorHunter", "AntiCycler",
    "AntiTitForTat", "Appeaser", "ArrogantQLearner", "AverageCopier",
    "BetterAndBetter", "BurnBothEnds", "BushMosteller", "Capri",
    "CautiousQLearner", "CollectiveStrategy", "CooperatorHunter",
    "CycleHunter", "CyclerCCCCCD", "CyclerCCCDCD", "CyclerDC", "CyclerDDC",
    "DefectorHunter", "DelayedAON1", "Desperate", "Detective",
    "DoubleResurrection", "Doubler", "EasyGo", "EugineNier",
    "EventualCycleHunter", "EvolvedANN5", "EvolvedFSM4", "EvolvedFSM6",
    "EvolvedLookerUp1_1_1", "EvolvedLookerUp2_2_2", "FirstByAnonymous",
    "FirstByDavis", "FirstByDowning", "FirstByFeld", "FirstByGraaskamp",
    "FirstByGrofman", "FirstByJoss", "FirstByNydegger", "FirstByShubik",
    "FirstBySteinAndRapoport", "FirstByTidemanAndChieruzzi", "FirstByTullock",
    "FoolMeOnce", "ForgetfulFoolMeOnce", "ForgetfulGrudger", "Fortress3",
    "Fortress4", "FrequencyAnalyzer", "GeneralSoftGrudger",
    "GoByMajority10", "GoByMajority20", "GoByMajority40", "GoByMajority5",
    "Golden", "Gradual", "GradualKiller", "GrudgerAlternator", "Handshake",
    "HardGoByMajority10", "HardGoByMajority20", "HardGoByMajority40",
    "HardGoByMajority5", "HardTitForTat", "HesitantQLearner", "Hopeless",
    "Inverse", "InversePunisher", "LevelPunisher", "LimitedRetaliate2",
    "MathConstantHunter", "Michaelos", "Momentum", "NTitsForMTats",
    "Negation", "NiceAverageCopier", "OnceBitten", "OppositeGrudger",
    "OriginalGradual", "PSOGambler2_2_2_Noise05", "Pi", "Prober2", "Prober3",
    "Prober4", "Pun1", "Punisher", "Raider", "RandomHunter", "Resurrection",
    "Retaliate", "RevisedDowning", "RiskyQLearner", "SecondByAppold",
    "SecondByBlack", "SecondByBorufsen", "SecondByCave", "SecondByChampion",
    "SecondByColbert", "SecondByEatherley", "SecondByGetzler",
    "SecondByGladstein", "SecondByGraaskampKatzen", "SecondByGrofman",
    "SecondByHarrington", "SecondByKluepfel", "SecondByLeyvraz",
    "SecondByMikkelson", "SecondByRichardHufford", "SecondByRowsam",
    "SecondByTester", "SecondByTidemanAndChieruzzi", "SecondByTranquilizer",
    "SecondByWeiner", "SecondByWhite", "SecondByWmAdams", "SecondByYamachi",
    "SelfSteem", "SoftJoss", "SolutionB1", "SolutionB5", "SpitefulTitForTat",
    "SpitefulCC", "StochasticCooperator", "StochasticWSLS",
    "SuspiciousTitForTat", "TF1", "TF2", "TF3", "TrickyLevelPunisher",
    "UsuallyCooperates", "UsuallyDefects", "VeryBad", "Willing",
    "WinShiftLoseStay", "WorseAndWorse", "WorseAndWorse2", "WorseAndWorse3",
    "ZDExtort2v2", "ZDExtort4", "ZDGen2", "ZDMem2", "ZDMischief",
    "ZDSet2", "e", "MemoryDecay", "MetaHunterAggressive",
    "MetaMajorityMemoryOne", "MetaWinnerMemoryOne", "NMWEMemoryOne",
)

AXELROD_MIXES = (
    ("TitForTat", "Grudger", "Adaptive"),
    ("TitForTat", "BackStabber", "Prober"),
    ("GTFT", "FirmButFair", "Random"),
    ("GoByMajority", "HardGoByMajority", "SoftGrudger"),
    ("ZDExtort2", "ZDExtort3", "ZDExtortion"),
    ("Adaptive", "EvolvedFSM16", "Winner12"),
    ("TitForTat", "DoubleCrosser", "RemorsefulProber"),
    ("Forgiver", "Grumpy", "Predator"),
    ("Winner21", "MetaHunter", "AdaptiveTitForTat"),
    ("Cooperator", "Defector", "Alternator"),
    ("TitFor2Tats", "TwoTitsForTat", "LimitedRetaliate"),
    ("RandomTitForTat", "SneakyTitForTat", "TrickyCooperator"),
    ("Calculator", "Stalker", "Thumper"),
    ("ZDGTFT2", "OmegaTFT", "PSOGambler1_1_1"),
    ("TitForTat", "Grudger", "Forgiver", "Random"),
)

# Thinking-level ladder: 0 = non-strategic, 1 = fixed heuristic, 2 = reactive /
# adaptive, 3 = meta / opponent-aware, 4 = search / ensemble / evolved.
FAMILY_LEVELS: dict[str, int] = {
    "anchor": 0,
    "random": 0,
    "nice": 1,
    "forgiving": 1,
    "punitive": 1,
    "defect": 1,
    "late": 1,
    "noisy": 1,
    "delayed_grim": 1,
    "wild": 2,
    "extort": 2,
    "temporal": 2,
    "memory_two": 2,
    "noise_adaptive": 2,
    "contrition_cycle": 2,
    "adaptive": 3,
    "phase_switch": 3,
    "adaptive_mix": 3,
    "probe_exploit": 3,
    "delayed_extort": 3,
    "reputation_grim": 3,
    "endgame_hunter": 3,
    "pattern_breaker": 3,
    "conditional_extort": 3,
    "lookahead": 4,
    "portfolio": 4,
    "axelrod": 2,
    "axelrod_mix": 3,
    "meta_mix": 4,
    "policy": 4,
    "early_harvest": 2,
    "frequent_kamikaze": 2,
    "probe_harvest": 3,
    "hyper_extort": 2,
    "early_probe": 2,
    "adaptive_probe": 3,
    "mirror_adaptive": 2,
}

# Strategy complexity is related but not identical to thinking level.
FAMILY_COMPLEXITY: dict[str, int] = {
    "anchor": 0,
    "random": 0,
    "nice": 1,
    "forgiving": 1,
    "punitive": 1,
    "defect": 1,
    "late": 1,
    "noisy": 1,
    "delayed_grim": 1,
    "wild": 2,
    "extort": 2,
    "temporal": 2,
    "memory_two": 2,
    "noise_adaptive": 3,
    "contrition_cycle": 3,
    "adaptive": 3,
    "phase_switch": 3,
    "adaptive_mix": 4,
    "probe_exploit": 3,
    "delayed_extort": 2,
    "reputation_grim": 3,
    "endgame_hunter": 3,
    "pattern_breaker": 4,
    "conditional_extort": 4,
    "lookahead": 4,
    "portfolio": 4,
    "axelrod": 2,
    "axelrod_mix": 3,
    "meta_mix": 4,
    "policy": 4,
    "early_harvest": 2,
    "frequent_kamikaze": 2,
    "probe_harvest": 3,
    "hyper_extort": 2,
    "early_probe": 2,
    "adaptive_probe": 3,
    "mirror_adaptive": 2,
}

AXELROD_LEVELS: dict[str, int] = {
    "Cooperator": 0, "Defector": 0, "Random": 0, "Alternator": 0,
    "CyclerCCD": 0, "CyclerCCCD": 0,
    "TitForTat": 1, "TitFor2Tats": 1, "TwoTitsForTat": 1, "Grudger": 1,
    "Forgiver": 1, "ForgivingTitForTat": 1, "GTFT": 1,
    "ContriteTitForTat": 1, "FirmButFair": 1, "WinStayLoseShift": 1,
    "GoByMajority": 1, "HardGoByMajority": 1, "SoftGrudger": 1,
    "Grumpy": 1, "LimitedRetaliate": 1, "Retaliate2": 1, "Retaliate3": 1,
    "HardTitFor2Tats": 1, "SlowTitForTwoTats2": 1,
    "DynamicTwoTitsForTat": 1, "MEM2": 1, "ShortMem": 1,
    "ThueMorse": 1, "ThueMorseInverse": 1,
    "Prober": 2, "RemorsefulProber": 2, "NaiveProber": 2, "HardProber": 2,
    "Adaptive": 2, "AdaptiveTitForTat": 2, "BackStabber": 2,
    "DoubleCrosser": 2, "Bully": 2, "Predator": 2, "Ripoff": 2,
    "SneakyTitForTat": 2, "TrickyCooperator": 2, "TrickyDefector": 2,
    "RandomTitForTat": 2, "ZDExtort2": 2, "ZDExtort3": 2,
    "ZDExtortion": 2, "ZDGTFT2": 2, "Calculator": 2, "OmegaTFT": 2,
    "Stalker": 2, "Thumper": 2,
    "MetaHunter": 3, "Winner12": 3, "Winner21": 3, "EvolvedFSM16": 3,
    "EvolvedANN": 3, "KnowledgeableWorseAndWorse": 3,
    "PSOGambler1_1_1": 3, "PSOGambler2_2_2": 3, "PSOGamblerMem1": 3,
    "EvolvedHMM5": 4, "EvolvedFSM16Noise05": 4,
    "EvolvedANNNoise05": 4,
}

MANIFEST_PATH = CONFIG_DIR / "splits" / PROTOCOL_NAME / "manifest.json"
USAGE_PATH = RESULTS_DIR / PROTOCOL_NAME / "usage.json"
TEST_POLICY_PATH = CONFIG_DIR / "splits" / PROTOCOL_NAME / "test_policy.json"

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


class PhaseSwitcher(Strategy):
    """Complex family: cooperate/TFT for a while, then switch behaviour."""

    def __init__(self, first: str = "tft", switch: int = 40,
                 second: str = "extort", chi: float = 3.0,
                 rng: random.Random | None = None):
        self.first = str(first)
        self.switch = int(switch)
        self.second = str(second)
        self.chi = float(chi)
        self._ext = ExtortZD(self.chi, rng)
        self._harvest = HarvestFactor(self.switch, False, 1.0, rng=rng)
        super().__init__(rng)
        self.name = f"Phase:{first}->{second}@{switch}"

    def reset(self) -> None:
        self._ext.reset()
        self._harvest.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t < self.switch:
            if self.first == "coop":
                return C
            if self.first == "probe":
                return D if t > 0 and t % 6 == 0 else C
            return theirs[-1] if t > 0 else C
        if self.second == "extort":
            return self._ext.act(t, mine, theirs)
        if self.second == "harvest":
            return self._harvest.act(t, mine, theirs)
        if self.second == "grim":
            return D if any(x == D for x in theirs) else C
        if self.second == "alld":
            return D
        if self.second == "tft":
            return theirs[-1] if theirs else C
        return C


class AdaptiveMix(Strategy):
    """Complex family: reciprocate if exploited, otherwise exploit."""

    def __init__(self, window: int = 10, threshold: float = 0.35,
                 exploit: str = "extort", chi: float = 3.0,
                 rng: random.Random | None = None):
        self.window = int(window)
        self.threshold = float(threshold)
        self.exploit = str(exploit)
        self._ext = ExtortZD(chi, rng)
        self._harvest = HarvestFactor(25, False, 1.0, rng=rng)
        super().__init__(rng)
        self.name = f"AdaptiveMix:{exploit}@{window}"

    def reset(self) -> None:
        self._ext.reset()
        self._harvest.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t == 0:
            return C
        recent = theirs[-self.window:]
        rate = sum(1 for x in recent if x == D) / max(1, len(recent))
        if rate >= self.threshold:
            if theirs[-1] == D:
                return D
            return C if rate < 0.65 else D
        if self.exploit == "extort":
            return self._ext.act(t, mine, theirs)
        if self.exploit == "harvest":
            return self._harvest.act(t, mine, theirs)
        return D


class ProbeExploit(Strategy):
    """Complex family: probe once, then TFT if punished or exploit if not."""

    def __init__(self, probe: int = 4, mode: str = "extort",
                 chi: float = 3.0, rng: random.Random | None = None):
        self.probe = int(probe)
        self.mode = str(mode)
        self._ext = ExtortZD(chi, rng)
        self._harvest = HarvestFactor(20, False, 1.0, rng=rng)
        super().__init__(rng)
        self.name = f"ProbeExploit:{mode}@{probe}"

    def reset(self) -> None:
        self.probed = False
        self.retaliated = False
        self._ext.reset()
        self._harvest.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if not self.probed:
            if t < self.probe - 1:
                return C
            if t == self.probe - 1:
                return D
            self.probed = True
            self.retaliated = bool(theirs and theirs[-1] == D)
        if self.retaliated:
            return theirs[-1] if t > 0 else C
        if self.mode == "extort":
            return self._ext.act(t, mine, theirs)
        if self.mode == "harvest":
            return self._harvest.act(t, mine, theirs)
        return D


class DelayedExtort(Strategy):
    """Complex family: cooperate for N rounds, then run ZD extortion."""

    def __init__(self, delay: int = 30, chi: float = 3.0,
                 rng: random.Random | None = None):
        self.delay = int(delay)
        self._ext = ExtortZD(chi, rng)
        super().__init__(rng)
        self.name = f"DelayedExtort@{delay}"

    def reset(self) -> None:
        self._ext.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t < self.delay:
            return C
        return self._ext.act(t, mine, theirs)


class MemoryTwo(Strategy):
    """Complex family: punish two consecutive defections, then forgive."""

    def __init__(self, punish: int = 8, forgive: bool = True,
                 rng: random.Random | None = None):
        self.punish = int(punish)
        self.forgive = bool(forgive)
        super().__init__(rng)
        self.name = f"MemoryTwo:{punish}"

    def reset(self) -> None:
        self.punish_left = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if self.punish_left > 0:
            self.punish_left -= 1
            return D
        if len(theirs) >= 2 and theirs[-1] == D and theirs[-2] == D:
            self.punish_left = self.punish - 1
            return D
        if theirs and theirs[-1] == D and not self.forgive:
            return D
        return C


class MetaMix(Strategy):
    """Complex family: one of several behaviours, chosen per match."""

    def __init__(self, modes: Sequence[str] = ("tft", "grim", "extort",
                                               "harvest", "random"),
                 rng: random.Random | None = None):
        self.modes = tuple(str(x) for x in modes)
        self.mode = rng.choice(self.modes) if rng is not None else "tft"
        self._ext = ExtortZD(3.0, rng)
        self._harvest = HarvestFactor(25, True, 1.0, rng=rng)
        super().__init__(rng)
        self.name = f"Meta:{self.mode}"

    def reset(self) -> None:
        self._ext.reset()
        self._harvest.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if self.mode == "tft":
            return theirs[-1] if t > 0 else C
        if self.mode == "grim":
            return D if any(x == D for x in theirs) else C
        if self.mode == "extort":
            return self._ext.act(t, mine, theirs)
        if self.mode == "harvest":
            return self._harvest.act(t, mine, theirs)
        if self.mode == "coop":
            return C
        return C if self.rng.random() < 0.5 else D


class NoiseAdaptive(Strategy):
    """New style: forgive isolated noise, punish sustained defection."""

    def __init__(self, window: int = 10, threshold: float = 0.25,
                 forgive: float = 0.7, punish: int = 6,
                 rng: random.Random | None = None):
        self.window = int(window)
        self.threshold = float(threshold)
        self.forgive = float(forgive)
        self.punish = int(punish)
        super().__init__(rng)
        self.name = f"NoiseAdaptive:{window}"

    def reset(self) -> None:
        self.punish_left = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t == 0:
            return C
        if self.punish_left > 0:
            self.punish_left -= 1
            return D
        recent = theirs[-self.window:]
        rate = sum(1 for x in recent if x == D) / max(1, len(recent))
        if rate >= self.threshold:
            self.punish_left = self.punish - 1
            return D
        if theirs[-1] == D:
            return C if self.rng.random() < self.forgive else D
        return C


class ReputationGrim(Strategy):
    """New style: track cumulative reputation, turn grim below threshold."""

    def __init__(self, threshold: float = 0.55, forgive_after: int = 0,
                 rng: random.Random | None = None):
        self.threshold = float(threshold)
        self.forgive_after = int(forgive_after)
        super().__init__(rng)
        self.name = f"ReputationGrim:{threshold:g}"

    def reset(self) -> None:
        self.grim = False
        self.cooldown = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if self.grim:
            if self.forgive_after and theirs and theirs[-1] == C:
                self.cooldown += 1
                if self.cooldown >= self.forgive_after:
                    self.grim = False
                    self.cooldown = 0
                    return C
            return D
        if len(theirs) >= 6:
            cooperation = 1.0 - sum(1 for x in theirs if x == D) / len(theirs)
            if cooperation < self.threshold:
                self.grim = True
                return D
        return C


class ContritionCycle(Strategy):
    """New style: apologise after own defection and repair the relation."""

    def __init__(self, repair: int = 2, patience: int = 3,
                 rng: random.Random | None = None):
        self.repair = int(repair)
        self.patience = int(patience)
        super().__init__(rng)
        self.name = f"Contrition:{repair}/{patience}"

    def reset(self) -> None:
        self.repair_left = 0
        self.pending = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t == 0:
            return C
        if self.repair_left > 0:
            self.repair_left -= 1
            return C
        if mine[-1] == D:
            self.repair_left = self.repair
            return C
        if theirs[-1] == D:
            self.pending += 1
            if self.pending >= self.patience:
                self.pending = 0
                self.repair_left = self.repair
                return D
            return D
        self.pending = 0
        return C


class EndgameHunter(Strategy):
    """New style: TFT until the endgame, then preempt or harvest."""

    def __init__(self, start: int = 60, preempt: int = 8,
                 rng: random.Random | None = None):
        self.start = int(start)
        self.preempt = int(preempt)
        super().__init__(rng)
        self.name = f"EndgameHunter:{start}"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t >= self.start:
            return D
        if t >= max(0, self.start - self.preempt) and theirs and theirs[-1] == D:
            return D
        return theirs[-1] if t > 0 else C


class PatternBreaker(Strategy):
    """New style: detect short cycles and play the counter-cycle."""

    def __init__(self, window: int = 8, threshold: float = 0.75,
                 rng: random.Random | None = None):
        self.window = int(window)
        self.threshold = float(threshold)
        super().__init__(rng)
        self.name = f"PatternBreaker:{window}"

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t == 0:
            return C
        recent = theirs[-self.window:]
        if len(recent) < self.window:
            return recent[-1]
        best_rate = 0.0
        best_period = 0
        for period in (2, 3, 4):
            matches = sum(1 for i in range(period, len(recent))
                          if recent[i] == recent[i - period])
            rate = matches / max(1, len(recent) - period)
            if rate > best_rate:
                best_rate = rate
                best_period = period
        if best_period and best_rate >= self.threshold:
            return 1 - recent[-best_period]
        return recent[-1]


class ConditionalExtort(Strategy):
    """New style: exploit cooperative opponents, otherwise play TFT."""

    def __init__(self, window: int = 12, coop_threshold: float = 0.65,
                 chi: float = 3.0, rng: random.Random | None = None):
        self.window = int(window)
        self.coop_threshold = float(coop_threshold)
        self._ext = ExtortZD(float(chi), rng)
        super().__init__(rng)
        self.name = f"ConditionalExtort:{window}"

    def reset(self) -> None:
        self._ext.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t == 0:
            return C
        recent = theirs[-self.window:]
        cooperation = 1.0 - sum(1 for x in recent if x == D) / max(1, len(recent))
        if cooperation >= self.coop_threshold:
            return self._ext.act(t, mine, theirs)
        return theirs[-1]


class LookaheadOpponent(Strategy):
    """High-level style: model opponent cooperation and look ahead."""

    def __init__(self, window: int = 12, gamma: float = 0.9,
                 horizon: int = 2, retaliation: float = 0.35,
                 rng: random.Random | None = None):
        self.window = int(window)
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.retaliation = float(retaliation)
        super().__init__(rng)
        self.name = f"Lookahead:{horizon}"

    def reset(self) -> None:
        self.p_coop = 0.5

    def _future(self, p: float, depth: int) -> float:
        if depth <= 0:
            return 0.0
        p_c = min(1.0, p + 0.10)
        p_d = max(0.0, p - self.retaliation)
        v_c = 6.0 * p + self.gamma * self._future(p_c, depth - 1)
        v_d = 10.0 * p + 1.0 * (1.0 - p) \
            + self.gamma * self._future(p_d, depth - 1)
        return max(v_c, v_d)

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        if t == 0:
            return C
        recent = theirs[-self.window:]
        observed = 1.0 - sum(1 for x in recent if x == D) / max(1, len(recent))
        self.p_coop = 0.7 * self.p_coop + 0.3 * observed
        p = self.p_coop
        p_c = min(1.0, p + 0.10)
        p_d = max(0.0, p - self.retaliation)
        v_c = 6.0 * p + self.gamma * self._future(p_c, self.horizon - 1)
        v_d = 10.0 * p + 1.0 * (1.0 - p) \
            + self.gamma * self._future(p_d, self.horizon - 1)
        return C if v_c >= v_d else D


class VotingPortfolio(Strategy):
    """High-level style: vote over a portfolio of simple strategies."""

    def __init__(self, members: Sequence[str] = ("tft", "grim", "extort",
                                                 "adaptive"),
                 tie: str = "D", rng: random.Random | None = None):
        self.members = tuple(str(x) for x in members)
        self.tie = str(tie)
        self.bots = []
        super().__init__(rng)
        self.name = "Portfolio:" + "/".join(self.members)

    def _make_bot(self, mode: str):
        if mode == "tft":
            return TFT(self.rng)
        if mode == "grim":
            return Grim(self.rng)
        if mode == "extort":
            return ExtortZD(3.0, self.rng)
        if mode == "harvest":
            return HarvestFactor(30, True, 1.0, rng=self.rng)
        if mode == "adaptive":
            return Responsive3(rng=self.rng)
        if mode == "probe":
            return TFTProbe(6, 2, True, self.rng)
        if mode == "random":
            return RandomMoves(0.5, self.rng)
        if mode == "coop":
            return AllC(self.rng)
        return AllD(self.rng)

    def reset(self) -> None:
        self.bots = [self._make_bot(mode) for mode in self.members]
        for bot in self.bots:
            bot.reset()

    def set_rounds(self, rounds: int) -> None:
        for bot in self.bots:
            if hasattr(bot, "set_rounds"):
                bot.set_rounds(rounds)

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t == 99:
            return D
        votes = [bot.act(t, mine, theirs) for bot in self.bots]
        c = votes.count(C)
        d = votes.count(D)
        if c > d:
            return C
        if d > c:
            return D
        return C if self.tie == "C" else D


class AxelrodMix(Strategy):
    """Complex family: one Axelrod strategy drawn per match."""

    def __init__(self, strategies: Sequence[str], rng: random.Random | None = None):
        self.strategies = tuple(str(x) for x in strategies)
        self._bot = None
        super().__init__(rng)
        self.name = "AxelrodMix"

    def reset(self) -> None:
        from ..core.opponents.axelrod_bank import axelrod_by_name

        bank = axelrod_by_name(include_long_run=False)
        name = self.rng.choice(self.strategies)
        spec = bank.get(name)
        if spec is None:
            raise KeyError(f"Axelrod strategy not available: {name}")
        self._bot = spec.factory(self.rng)
        self._bot.reset()
        self.name = f"AxelrodMix:{name}"

    def set_rounds(self, rounds: int) -> None:
        if self._bot is not None and hasattr(self._bot, "set_rounds"):
            self._bot.set_rounds(rounds)

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if self._bot is None:
            self.reset()
        return self._bot.act(t, mine, theirs)


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


def _b_forgive_tft(rng: random.Random, missed_limit: int) -> Strategy:
    return ForgiveTFT(True, int(missed_limit), rng)


def _b_tft_forgive_once(rng: random.Random) -> Strategy:
    return TFTForgiveOnce(rng=rng)


def _b_tft2lastd(rng: random.Random) -> Strategy:
    return TFT2LastD(rng)


def _b_majority(rng: random.Random, k: int) -> Strategy:
    return Majority(int(k), rng)


def _b_ttft(rng: random.Random) -> Strategy:
    return TwoTitsForTat(rng)


def _b_suspicious(rng: random.Random) -> Strategy:
    return SuspiciousTFT(rng)


def _b_grim_forgive(rng: random.Random, g: float) -> Strategy:
    return GrimForgive(float(g), rng)


def _b_last_k(rng: random.Random, k: int) -> Strategy:
    return LastKDefect(int(k), rng)


def _b_random_then_tft(rng: random.Random, k: int, p: float) -> Strategy:
    return RandomThenTFT(int(k), float(p), rng)


def _b_remorseful(rng: random.Random) -> Strategy:
    return RemorsefulProber(rng=rng)


def _b_tft_probe_immediate(rng: random.Random, probe: int,
                           recover: int) -> Strategy:
    return TFTProbeImmediate(int(probe), int(recover), True, rng)


def _b_responsive(rng: random.Random, g: float, min_c: int, min_d: int,
                  resp: float, rate: float, max_tests: int) -> Strategy:
    return Responsive(float(g), int(min_c), int(min_d), float(resp),
                      float(rate), True, int(max_tests), rng)


def _b_responsive2(rng: random.Random, g_unprov: float, g_mutual: float,
                   strike: int, min_c: int, min_d: int, resp: float,
                   max_tests: int) -> Strategy:
    return Responsive2(float(g_unprov), float(g_mutual), int(strike),
                       int(min_c), int(min_d), float(resp), True,
                       int(max_tests), rng)


def _b_random_fsm(rng: random.Random, k: int) -> Strategy:
    return RandomFSM(rng, int(k))


def _b_tft_noisy_forgive(rng: random.Random, g: float) -> Strategy:
    return TFTNoisyForgive(float(g), rng)


def _b_phase_switch(rng: random.Random, first: str, switch: int,
                    second: str, chi: float) -> Strategy:
    return PhaseSwitcher(first, int(switch), second, float(chi), rng)


def _b_adaptive_mix(rng: random.Random, window: int, threshold: float,
                    exploit: str, chi: float) -> Strategy:
    return AdaptiveMix(int(window), float(threshold), exploit, float(chi), rng)


def _b_probe_exploit(rng: random.Random, probe: int, mode: str,
                     chi: float) -> Strategy:
    return ProbeExploit(int(probe), mode, float(chi), rng)


def _b_delayed_extort(rng: random.Random, delay: int, chi: float) -> Strategy:
    return DelayedExtort(int(delay), float(chi), rng)


def _b_memory_two(rng: random.Random, punish: int,
                  forgive: bool) -> Strategy:
    return MemoryTwo(int(punish), bool(forgive), rng)


def _b_meta_mix(rng: random.Random, modes: Sequence[str]) -> Strategy:
    return MetaMix(tuple(str(x) for x in modes), rng)


def _b_noise_adaptive(rng: random.Random, window: int, threshold: float,
                      forgive: float, punish: int) -> Strategy:
    return NoiseAdaptive(int(window), float(threshold), float(forgive),
                         int(punish), rng)


def _b_reputation_grim(rng: random.Random, threshold: float,
                       forgive_after: int) -> Strategy:
    return ReputationGrim(float(threshold), int(forgive_after), rng)


def _b_contrition(rng: random.Random, repair: int,
                  patience: int) -> Strategy:
    return ContritionCycle(int(repair), int(patience), rng)


def _b_endgame_hunter(rng: random.Random, start: int,
                      preempt: int) -> Strategy:
    return EndgameHunter(int(start), int(preempt), rng)


def _b_pattern_breaker(rng: random.Random, window: int,
                       threshold: float) -> Strategy:
    return PatternBreaker(int(window), float(threshold), rng)


def _b_conditional_extort(rng: random.Random, window: int,
                          coop_threshold: float, chi: float) -> Strategy:
    return ConditionalExtort(int(window), float(coop_threshold),
                             float(chi), rng)


def _b_gradual(rng: random.Random) -> Strategy:
    return Gradual(rng)


def _b_lookahead(rng: random.Random, window: int, gamma: float,
                 horizon: int, retaliation: float) -> Strategy:
    return LookaheadOpponent(int(window), float(gamma), int(horizon),
                             float(retaliation), rng)


def _b_portfolio(rng: random.Random, members: Sequence[str],
                 tie: str) -> Strategy:
    return VotingPortfolio(tuple(str(x) for x in members), str(tie), rng)


def _b_axelrod(rng: random.Random, strategy: str) -> Strategy:
    from ..core.opponents.axelrod_bank import axelrod_by_name

    bank = axelrod_by_name(include_long_run=False)
    spec = bank.get(str(strategy))
    if spec is None:
        raise KeyError(f"Axelrod strategy not available: {strategy}")
    bot = spec.factory(rng)
    bot.name = f"Axelrod:{strategy}"
    return bot


def _b_axelrod_mix(rng: random.Random, strategies: Sequence[str]) -> Strategy:
    return AxelrodMix(tuple(str(x) for x in strategies), rng)


def _policy_path(weights: str) -> str:
    path = Path(str(weights))
    return str(path if path.is_absolute() else ROOT / path)


def _b_policy_final(rng: random.Random, **_: object) -> Strategy:
    from ipd.policies.candidates.final_tuned import FinalTunedPolicy

    return FinalTunedPolicy(seed=rng.randrange(1 << 30))


def _b_policy_triple(rng: random.Random, **_: object) -> Strategy:
    from ipd.policies.candidates.triple_majority import TripleMajorityPolicy

    return TripleMajorityPolicy(seed=rng.randrange(1 << 30))


def _b_policy_rl(rng: random.Random, weights: str,
                 mode: str = "sample") -> Strategy:
    from ipd.policies.rl import RLPolicy

    path = _policy_path(weights)
    policy = RLPolicy(path, seed=rng.randrange(1 << 30), mode=mode)
    policy._gpu_weights_path = path
    return policy


def _b_policy_neural(rng: random.Random, weights: str,
                     mode: str = "sample") -> Strategy:
    from ipd.policies.neural import NeuralPolicy

    path = _policy_path(weights)
    policy = NeuralPolicy(path, seed=rng.randrange(1 << 30), mode=mode)
    policy._gpu_weights_path = path
    return policy


def _b_policy_gru(rng: random.Random, weights: str,
                  mode: str = "sample") -> Strategy:
    from ipd.policies.gru import GRUPolicy

    path = _policy_path(weights)
    policy = GRUPolicy(path, seed=rng.randrange(1 << 30), mode=mode)
    policy._gpu_weights_path = path
    return policy


def _b_policy_selfplay(rng: random.Random, weights: str,
                       mode: str = "sample") -> Strategy:
    from ipd.policies.selfplay_gru import SelfPlayGRUPolicy

    path = _policy_path(weights)
    policy = SelfPlayGRUPolicy(path, seed=rng.randrange(1 << 30), mode=mode)
    policy._gpu_weights_path = path
    return policy


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
    "ForgiveTFT": _b_forgive_tft,
    "TFTForgiveOnce": _b_tft_forgive_once,
    "TFT2LastD": _b_tft2lastd,
    "Majority": _b_majority,
    "TwoTitsForTat": _b_ttft,
    "SuspiciousTFT": _b_suspicious,
    "GrimForgive": _b_grim_forgive,
    "LastKDefect": _b_last_k,
    "RandomThenTFT": _b_random_then_tft,
    "RemorsefulProber": _b_remorseful,
    "TFTProbeImmediate": _b_tft_probe_immediate,
    "Responsive": _b_responsive,
    "Responsive2": _b_responsive2,
    "RandomFSM": _b_random_fsm,
    "TFTNoisyForgive": _b_tft_noisy_forgive,
    "PhaseSwitcher": _b_phase_switch,
    "AdaptiveMix": _b_adaptive_mix,
    "ProbeExploit": _b_probe_exploit,
    "DelayedExtort": _b_delayed_extort,
    "MemoryTwo": _b_memory_two,
    "MetaMix": _b_meta_mix,
    "NoiseAdaptive": _b_noise_adaptive,
    "ReputationGrim": _b_reputation_grim,
    "ContritionCycle": _b_contrition,
    "EndgameHunter": _b_endgame_hunter,
    "PatternBreaker": _b_pattern_breaker,
    "ConditionalExtort": _b_conditional_extort,
    "Gradual": _b_gradual,
    "LookaheadOpponent": _b_lookahead,
    "VotingPortfolio": _b_portfolio,
    "Axelrod": _b_axelrod,
    "AxelrodMix": _b_axelrod_mix,
    "PolicyFinal": _b_policy_final,
    "PolicyTriple": _b_policy_triple,
    "PolicyRL": _b_policy_rl,
    "PolicyNeural": _b_policy_neural,
    "PolicyGRU": _b_policy_gru,
    "PolicySelfPlay": _b_policy_selfplay,
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
    thinking_level: int = 1
    complexity: int = 1

    @property
    def group(self) -> str:
        return "anchor" if self.anchor else self.tier

    def build(self, rng: random.Random) -> Strategy:
        if self.type_name not in _BUILDERS:
            raise KeyError(f"no builder for {self.type_name!r}")
        bot = _BUILDERS[self.type_name](rng, **dict(self.params))
        # Metadata for the Plan 3 fast rule bank.  Unsupported types
        # (Axelrod, learned policies, RandomFSM, ProbeBot) are ignored by
        # attach_fast_spec and keep their original implementation.
        try:
            from .fast.spec import attach_fast_spec

            attach_fast_spec(bot, self.type_name, family=self.family,
                             name=f"{self.family}:{self.key}",
                             **dict(self.params))
        except Exception:
            pass
        return bot

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
            "thinking_level": self.thinking_level,
            "complexity": self.complexity,
        }


def _short_params(params: Mapping[str, Any]) -> str:
    if not params:
        return ""
    return "|".join(f"{k}={v}" for k, v in sorted(params.items()))


def _add(specs: list[dict[str, Any]], family: str, type_name: str,
         tier: str | None = None, **params: Any) -> None:
    if type_name not in _BUILDERS:
        raise KeyError(f"unknown builder {type_name!r}")
    text = _short_params(params)
    key = f"{family}:{type_name}" + (f":{text}" if text else "")
    if any(item["key"] == key for item in specs):
        return
    row: dict[str, Any] = {
        "key": key, "family": family, "type_name": type_name,
        "params": tuple(sorted(params.items())),
    }
    if tier is not None:
        row["tier"] = tier
    specs.append(row)


def _catalog_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Standard families (seen in train/dev/test with disjoint configs)
    # ------------------------------------------------------------------
    for type_name in ("TFT2", "TFTLastD", "Pavlov", "ContriteTFT", "WSLS2",
                      "TFTForgiveOnce", "TFT2LastD", "TwoTitsForTat",
                      "SuspiciousTFT"):
        _add(specs, "nice", type_name)
    for k in (2, 3, 4, 5, 6):
        _add(specs, "nice", "Majority", k=k)
    for g in (0.01, 0.02, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22,
              0.26, 0.31, 0.36, 0.42, 0.49):
        _add(specs, "forgiving", "GenerousTFT", g=g)
    for strikes in (2, 3, 4):
        _add(specs, "forgiving", "TwoStrike", strikes=strikes)
    for missed in (1, 2, 3, 4):
        _add(specs, "forgiving", "ForgiveTFT", missed_limit=missed)
    for g in (0.08, 0.16, 0.24, 0.33, 0.42, 0.50):
        _add(specs, "forgiving", "TFTNoisyForgive", g=g)
    for chi in (1.35, 1.80):
        _add(specs, "forgiving", "GenerousZD", chi=chi)

    for type_name in ("Grim", "Grudger2", "FirmButFair", "Mistrust",
                      "Gradual"):
        _add(specs, "punitive", type_name)
    for k in (1, 2, 3, 4, 5, 6):
        _add(specs, "punitive", "GrudgerK", k=k)
    for g in (0.03, 0.08, 0.15, 0.25, 0.40):
        _add(specs, "punitive", "GrimForgive", g=g)
    for patience in (2, 4):
        _add(specs, "punitive", "ProbabilisticGrim", patience=patience,
             p_punish=0.8)
    for patience, cooldown in ((2, 10), (3, 12)):
        _add(specs, "punitive", "ResetGrim", patience=patience,
             cooldown=cooldown)
    for patience in (1, 3):
        _add(specs, "punitive", "SpitefulGrim", patience=patience)
    for patience, punish in ((2, 8), (3, 12)):
        _add(specs, "punitive", "DelayedPunisher", patience=patience,
             punish_rounds=punish)

    for type_name in ("Bully", "Prober", "Tester", "RemorsefulProber"):
        _add(specs, "defect", type_name)
    for k in (1, 2, 5, 10, 15, 25, 30, 40):
        _add(specs, "defect", "EarlyDefector", k=k)
    for k in (4, 8):
        _add(specs, "defect", "RandomThenTFT", k=k, p=0.5)

    for k in (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
              48, 51, 54, 57, 60, 65, 70, 75, 80, 85, 90, 94, 97, 98, 99):
        _add(specs, "late", "LateDefector", k=k)

    for p in (0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22,
              0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62,
              0.66, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.93, 0.96, 0.98):
        _add(specs, "random", "RandomMoves", p=p)
    for pattern, start in (
        ((C, C, D), 0), ((C, D), 1), ((C, D, D, C), 2),
        ((C, C, C, D), 3), ((D, C, C, C), 1), ((C, C, D, D, C), 4),
        ((C, C, D, D), 0), ((D, D, C, C), 1), ((C, C, C, C, D), 2),
        ((D, C, D, C, C), 3),
    ):
        _add(specs, "random", "Periodic", pattern=pattern, start=start)

    memone_params = (
        (0.95, 0.02, 0.70, 0.02), (0.90, 0.08, 0.55, 0.05),
        (0.85, 0.15, 0.45, 0.10), (0.80, 0.22, 0.35, 0.15),
        (0.75, 0.30, 0.30, 0.20), (0.70, 0.40, 0.25, 0.25),
        (0.65, 0.05, 0.60, 0.35), (0.60, 0.10, 0.50, 0.40),
        (0.55, 0.20, 0.40, 0.45), (0.50, 0.30, 0.30, 0.50),
        (0.45, 0.10, 0.20, 0.60), (0.40, 0.25, 0.15, 0.70),
        (0.98, 0.05, 0.90, 0.01), (0.35, 0.45, 0.10, 0.80),
    )
    for p in memone_params:
        _add(specs, "wild", "MemoryOne", p=p)
    reactive_params = (
        (C, 0.98, 0.02), (C, 0.90, 0.10), (C, 0.80, 0.20),
        (C, 0.70, 0.30), (C, 0.60, 0.40), (D, 0.85, 0.05),
        (D, 0.70, 0.20), (D, 0.55, 0.35), (D, 0.40, 0.50),
        (C, 0.50, 0.50), (D, 0.25, 0.65),
    )
    for first, pc, pd in reactive_params:
        _add(specs, "wild", "Reactive", first=first,
             p_after_c=pc, p_after_d=pd)
    for k in (2, 3, 4, 5, 6, 8, 10, 12):
        _add(specs, "wild", "RandomFSM", k=k)

    resp3_params = (
        (0.55, 0.40, 0.95, 5, 3, 0.38, 0.18, 2, 8),
        (0.40, 0.25, 0.88, 4, 4, 0.45, 0.22, 3, 12),
        (0.70, 0.50, 0.97, 6, 2, 0.32, 0.15, 1, 6),
        (0.30, 0.20, 0.85, 3, 5, 0.50, 0.25, 4, 14),
        (0.60, 0.35, 0.92, 5, 3, 0.40, 0.20, 2, 10),
        (0.45, 0.30, 0.90, 4, 2, 0.42, 0.18, 3, 7),
        (0.50, 0.40, 0.86, 6, 4, 0.36, 0.24, 2, 11),
        (0.35, 0.15, 0.80, 3, 3, 0.55, 0.30, 4, 16),
    )
    for params in resp3_params:
        _add(specs, "adaptive", "Responsive3",
             g_unprov=params[0], g_mutual=params[1], g_resp=params[2],
             min_c=params[3], min_d=params[4], resp=params[5],
             rate=params[6], max_tests=params[7], cooldown=params[8])
    for g, min_c, min_d, resp, rate, tests in (
        (0.20, 4, 3, 0.30, 0.18, 2),
        (0.30, 5, 2, 0.40, 0.15, 3),
        (0.15, 3, 4, 0.25, 0.22, 1),
        (0.40, 6, 3, 0.45, 0.12, 4),
    ):
        _add(specs, "adaptive", "Responsive", g=g, min_c=min_c,
             min_d=min_d, resp=resp, rate=rate, max_tests=tests)
    for g_unprov, g_mutual, strike, min_c, min_d, resp, tests in (
        (1.00, 0.10, 2, 4, 2, 0.35, 3),
        (0.80, 0.20, 3, 5, 2, 0.30, 2),
        (0.60, 0.30, 2, 3, 3, 0.45, 4),
        (0.90, 0.15, 1, 4, 4, 0.25, 2),
    ):
        _add(specs, "adaptive", "Responsive2", g_unprov=g_unprov,
             g_mutual=g_mutual, strike=strike, min_c=min_c, min_d=min_d,
             resp=resp, max_tests=tests)
    _add(specs, "adaptive", "ProbeBot")
    _add(specs, "adaptive", "OmegaTFTLite")
    for probe in (2, 4, 6, 8, 10, 12, 15, 20):
        _add(specs, "adaptive", "TFTProbe", probe_round=probe, recover=2)
    for probe in (4, 8, 12, 16, 20, 25, 30, 40):
        _add(specs, "adaptive", "AdaptiveProbe", probe_round=probe, recover=2)
    for probe in (1, 2, 3):
        _add(specs, "adaptive", "TFTProbeImmediate", probe=probe, recover=2)
    for g in (0.10, 0.20, 0.30, 0.40):
        _add(specs, "adaptive", "TFTNoisyForgive", g=g)

    for chi in (1.0, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75,
                1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65,
                2.75, 2.85, 2.95, 3.05, 3.15, 3.25, 3.35, 3.45, 3.55,
                3.65, 3.75, 3.85, 3.95):
        _add(specs, "extort", "ExtortZD", chi=chi)
    for chi in (1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6):
        _add(specs, "extort", "GenerousZD", chi=chi)

    for start in (8, 11, 14, 17, 20, 23, 26, 29, 32, 36, 40, 44, 48, 53,
                  58, 63, 68, 73, 78, 83, 88, 93, 96, 98):
        _add(specs, "temporal", "HarvestFactor", start=start,
             repairable=start % 2 == 0, p_defect=0.95)
    for window in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25):
        for p_defect in (0.08, 0.20, 0.38):
            _add(specs, "temporal", "DynamicWindow", window=window,
                 p_defect=p_defect, retaliate=0.85, forgiveness=0.25)
    for period in (4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30,
                   35, 40):
        _add(specs, "temporal", "Kamikaze", period=period, p_steal=0.75)
    for probe in (2, 3, 4, 5, 6, 8, 10, 12, 14, 16):
        _add(specs, "temporal", "ProbePhase", probe=probe, p_probe=0.75,
             phase="harvest", harvest_start=40 + 5 * probe)
    for window in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16):
        _add(specs, "temporal", "WindowTFT", window=window,
             retaliate=0.82, forgive=0.30)

    for p in (0.01, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.22, 0.25,
              0.28, 0.31, 0.34, 0.37, 0.40, 0.43, 0.46, 0.49, 0.52,
              0.55):
        _add(specs, "noisy", "NoisyTFT", p=p)
        _add(specs, "noisy", "Joss", p=p)

    # ------------------------------------------------------------------
    # Complex families (train/dev/test with disjoint configs)
    # ------------------------------------------------------------------
    for first in ("tft", "coop", "probe"):
        for switch in (25, 40, 55, 70):
            for second in ("extort", "harvest", "alld"):
                _add(specs, "phase_switch", "PhaseSwitcher",
                     first=first, switch=switch, second=second, chi=3.0)
    for window in (5, 10, 20):
        for threshold in (0.20, 0.35, 0.50):
            for exploit in ("extort", "harvest"):
                _add(specs, "adaptive_mix", "AdaptiveMix",
                     window=window, threshold=threshold, exploit=exploit,
                     chi=3.0)
    for probe in (2, 4, 6, 8):
        for mode in ("extort", "harvest", "alld"):
            _add(specs, "probe_exploit", "ProbeExploit",
                 probe=probe, mode=mode, chi=3.0)
    for delay in (15, 25, 35, 45, 55, 65):
        for chi in (2.0, 3.0, 4.0):
            _add(specs, "delayed_extort", "DelayedExtort",
                 delay=delay, chi=chi)
    for punish in (3, 6, 9, 12, 16):
        for forgive in (True, False):
            _add(specs, "memory_two", "MemoryTwo",
                 punish=punish, forgive=forgive)
    meta_modes = (
        ("tft", "grim", "extort", "harvest", "random"),
        ("tft", "extort", "harvest"),
        ("grim", "extort", "alld"),
        ("tft", "grim", "random"),
        ("extort", "harvest", "random"),
        ("tft", "harvest", "alld"),
        ("coop", "grim", "extort"),
        ("tft", "extort", "alld"),
        ("grim", "harvest", "random"),
        ("coop", "tft", "extort", "random"),
        ("tft", "grim", "extort", "alld"),
        ("coop", "harvest", "extort", "random"),
    )
    for modes in meta_modes:
        _add(specs, "meta_mix", "MetaMix", modes=modes)

    # New style sets: each is a distinct behavioural family, not just a
    # parameter variant of an existing one.
    for window in (5, 10, 20):
        for threshold in (0.20, 0.35, 0.50):
            for forgive in (0.4, 0.7):
                for punish in (4, 8):
                    _add(specs, "noise_adaptive", "NoiseAdaptive",
                         window=window, threshold=threshold,
                         forgive=forgive, punish=punish)
    for threshold in (0.40, 0.50, 0.60, 0.70):
        for forgive_after in (0, 5, 10):
            _add(specs, "reputation_grim", "ReputationGrim",
                 threshold=threshold, forgive_after=forgive_after)
    for repair in (1, 2, 3):
        for patience in (2, 3, 4):
            _add(specs, "contrition_cycle", "ContritionCycle",
                 repair=repair, patience=patience)
    for start in (45, 55, 65, 75):
        for preempt in (4, 8, 12):
            _add(specs, "endgame_hunter", "EndgameHunter",
                 start=start, preempt=preempt)
    for window in (6, 8, 10):
        for threshold in (0.60, 0.75, 0.90):
            _add(specs, "pattern_breaker", "PatternBreaker",
                 window=window, threshold=threshold)
    for window in (8, 12, 20):
        for coop in (0.55, 0.65, 0.75):
            for chi in (2.5, 3.5):
                _add(specs, "conditional_extort", "ConditionalExtort",
                     window=window, coop_threshold=coop, chi=chi)
    for window in (8, 12, 20):
        for gamma in (0.80, 0.92, 0.98):
            for horizon in (2, 3):
                for retaliation in (0.20, 0.40):
                    _add(specs, "lookahead", "LookaheadOpponent",
                         window=window, gamma=gamma, horizon=horizon,
                         retaliation=retaliation)
    portfolio_members = (
        ("tft", "grim", "extort"),
        ("tft", "extort", "adaptive"),
        ("grim", "extort", "harvest"),
        ("tft", "grim", "adaptive"),
        ("extort", "harvest", "random"),
        ("tft", "probe", "extort"),
        ("coop", "grim", "extort"),
        ("tft", "grim", "extort", "adaptive"),
        ("tft", "harvest", "adaptive"),
        ("grim", "harvest", "alld"),
    )
    for members in portfolio_members:
        for tie in ("C", "D"):
            _add(specs, "portfolio", "VotingPortfolio",
                 members=members, tie=tie)

    # ------------------------------------------------------------------
    # Axelrod strategies and custom combinations
    # ------------------------------------------------------------------
    for name in AXELROD_STRATEGIES:
        _add(specs, "axelrod", "Axelrod", strategy=name)
    for name in AXELROD_EXTRA:
        _add(specs, "axelrod_extra", "Axelrod", strategy=name)
    for mix in AXELROD_MIXES:
        _add(specs, "axelrod_mix", "AxelrodMix", strategies=mix)

    # ------------------------------------------------------------------
    # Policy opponents (complex, but not parameterised rules)
    # ------------------------------------------------------------------
    for name, type_name, weights in (
        ("rl-bc", "PolicyRL", "weights/rl_policy_checkpoint_bc.json"),
        ("selfplay-bc", "PolicySelfPlay", "weights/selfplay_bc_policy.json"),
        ("neural-main", "PolicyNeural", "weights/neural_policy_weights.json"),
        ("gru-main", "PolicyGRU", "weights/gru_policy_weights.json"),
    ):
        _add(specs, "policy", type_name, tier="train_policy",
             weights=weights, mode="sample")
    for name, type_name, weights in (
        ("final-tuned", "PolicyFinal", ""),
        ("triple-majority", "PolicyTriple", ""),
        ("rl-scratch", "PolicyRL",
         "weights/rl_policy_checkpoint_scratch_base.json"),
        ("selfplay-main", "PolicySelfPlay", "weights/selfplay_gru_policy.json"),
        ("neural-retrain", "PolicyNeural",
         "weights/retrain_20261025/neural_policy_weights.json"),
        ("gru-retrain", "PolicyGRU",
         "weights/retrain_20261025/gru_policy_weights.json"),
    ):
        kwargs = {"mode": "sample"}
        if weights:
            kwargs["weights"] = weights
        _add(specs, "policy", type_name, tier="test_policy", **kwargs)

    # ------------------------------------------------------------------
    # Test-only simple adversarial families
    # ------------------------------------------------------------------
    for start in (4, 6, 8, 10, 12, 14):
        _add(specs, "early_harvest", "HarvestFactor", start=start,
             repairable=False, p_defect=1.0)
    for period in (4, 5, 6, 7, 8, 9):
        _add(specs, "frequent_kamikaze", "Kamikaze", period=period,
             p_steal=1.0)
    for probe, start in ((2, 35), (3, 40), (4, 45), (5, 50), (6, 55),
                         (7, 60)):
        _add(specs, "probe_harvest", "ProbePhase", probe=probe,
             p_probe=0.8, phase="harvest", harvest_start=start)
    for chi in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5):
        _add(specs, "hyper_extort", "ExtortZD", chi=chi)
    for probe in (3, 5, 7, 9, 11, 13):
        _add(specs, "early_probe", "TFTProbe", probe_round=probe, recover=2)
    for probe in (7, 11, 17, 23, 29, 35):
        _add(specs, "adaptive_probe", "AdaptiveProbe", probe_round=probe,
             recover=2)

    # ------------------------------------------------------------------
    # Dev-only families (held out from train and test)
    # ------------------------------------------------------------------
    for window, forgive in ((3, 0.25), (5, 0.45), (7, 0.55), (10, 0.70),
                            (11, 0.15), (12, 0.60), (15, 0.50), (18, 0.40)):
        _add(specs, "mirror_adaptive", "MirrorAdaptive", window=window,
             forgive=forgive)
    for patience in (1, 2, 4, 5, 6, 7):
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
    if family in COMPLEX_FAMILIES:
        names = ("train_complex", "dev_complex", "test_complex")
    elif family in AXELROD_FAMILIES:
        names = ("train_axelrod", "dev_axelrod", "test_axelrod")
    else:
        names = ("train", "dev", "test")
    tiers = ([names[0]] * n_train + [names[1]] * n_dev
             + [names[2]] * n_test)
    rng = random.Random(int(
        hashlib.sha256(family.encode("utf-8")).hexdigest()[:16], 16))
    rng.shuffle(tiers)
    return tiers


def _axelrod_level(name: str) -> int:
    if name in AXELROD_LEVELS:
        return AXELROD_LEVELS[name]
    lower = name.lower()
    if any(key in lower for key in (
            "meta", "winner", "nmwe", "evolved", "pso", "knowledgeable",
            "qlearner")):
        return 4 if any(key in lower for key in (
            "evolved", "meta", "nmwe", "pso")) else 3
    if any(key in lower for key in (
            "adaptive", "prober", "hunter", "cycler", "zd", "extort",
            "grudger", "punisher", "retaliate", "fortress", "detective",
            "frequency", "predict", "worse", "better", "downing")):
        return 2
    if any(key in lower for key in (
            "random", "cooperator", "defector", "alternator", "allc",
            "alld", "periodic")):
        return 0
    return 1


def _entry_level(family: str, type_name: str) -> int:
    if family in ("axelrod", "axelrod_extra"):
        return _axelrod_level(type_name)
    return FAMILY_LEVELS.get(family, 1)


def _entry_complexity(family: str, type_name: str) -> int:
    if family in ("axelrod", "axelrod_extra"):
        return min(4, _axelrod_level(type_name) + 1)
    return FAMILY_COMPLEXITY.get(family, 1)


@lru_cache(maxsize=1)
def build_catalog() -> tuple[OpponentConfig, ...]:
    """Build the deterministic, frozen opponent catalog."""
    specs = _catalog_specs()

    by_family: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        by_family.setdefault(spec["family"], []).append(spec)

    entries: list[OpponentConfig] = []
    for family, rows in sorted(by_family.items()):
        explicit = all("tier" in row for row in rows)
        tiers = ([row["tier"] for row in rows] if explicit
                 else _family_tiers(family, len(rows)))
        counts: dict[str, int] = {}
        for tier in tiers:
            counts[tier] = counts.get(tier, 0) + 1
        for row, tier in zip(rows, tiers):
            base = FAMILY_WEIGHTS.get(family, 1.0) / max(1, counts[tier])
            entries.append(OpponentConfig(
                key=row["key"], family=family, type_name=row["type_name"],
                params=row["params"], tier=tier, weight=base,
                thinking_level=_entry_level(family, row["type_name"]),
                complexity=_entry_complexity(family, row["type_name"]),
            ))

    # Fixed anchors appear in every tier; their behaviour is identical, so
    # they are exempt from the config-overlap audit.
    anchor_specs = {
        "AllC": ("AllC", ()),
        "AllD": ("AllD", ()),
        "TFT": ("TFT", ()),
        "Grim": ("Grim", ()),
        "Random.50": ("RandomMoves", (("p", 0.50),)),
    }
    for name in ANCHORS:
        type_name, params = anchor_specs[name]
        entries.append(OpponentConfig(
            key=f"anchor:{name}", family="anchor",
            type_name=type_name, params=params, tier="anchor",
            anchor=True, weight=1.0, thinking_level=0, complexity=0,
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
        "schema": "ipd-protocol-v4",
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "catalog_sha256": catalog_sha256(catalog),
        "catalog_size": len(catalog),
        "tiers": list(SPLIT_TIERS),
        "complex_families": list(COMPLEX_FAMILIES),
        "policy_families": list(POLICY_FAMILIES),
        "axelrod_families": list(AXELROD_FAMILIES),
        "axelrod_strategies": list(AXELROD_STRATEGIES),
        "axelrod_extra": list(AXELROD_EXTRA),
        "axelrod_mixes": [list(x) for x in AXELROD_MIXES],
        "dev_only_families": list(DEV_ONLY_FAMILIES),
        "test_only_families": list(TEST_ONLY_FAMILIES),
        "anchors": list(ANCHORS),
        "group_weights": {k: dict(v) for k, v in TIER_GROUPS.items()},
        "red_pools": {
            "dev_red": [dict(entry) for entry in DEV_RED_POOL],
            "test_red": [dict(entry) for entry in TEST_RED_POOL],
        },
        "counts": catalog_counts(catalog),
        "thinking_level_counts": {
            str(level): sum(1 for entry in catalog
                            if entry.thinking_level == level)
            for level in range(5)
        },
        "complexity_counts": {
            str(level): sum(1 for entry in catalog
                            if entry.complexity == level)
            for level in range(5)
        },
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

    group_names = ("train", "train_complex", "train_policy",
                   "train_axelrod", "dev", "dev_family", "dev_complex",
                   "dev_axelrod", "test", "test_complex", "test_policy",
                   "test_axelrod", "test_only", "anchor")
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
    train_complex = _tier_families(rows, "train_complex")
    train_policy = _tier_families(rows, "train_policy")
    train_axelrod = _tier_families(rows, "train_axelrod")
    dev_families = _tier_families(rows, "dev")
    dev_family_families = _tier_families(rows, "dev_family")
    dev_complex = _tier_families(rows, "dev_complex")
    dev_axelrod = _tier_families(rows, "dev_axelrod")
    test_families = _tier_families(rows, "test")
    test_complex = _tier_families(rows, "test_complex")
    test_policy = _tier_families(rows, "test_policy")
    test_axelrod = _tier_families(rows, "test_axelrod")
    test_only_families = _tier_families(rows, "test_only")

    for family in DEV_ONLY_FAMILIES:
        if family not in dev_family_families:
            errors.append(f"dev-only family missing from dev_family: {family}")
        if family in train_families or family in dev_families \
                or family in test_families or family in test_only_families \
                or family in train_complex or family in dev_complex \
                or family in test_complex or family in train_axelrod \
                or family in dev_axelrod or family in test_axelrod:
            errors.append(f"dev-only family leaked: {family}")
    for family in TEST_ONLY_FAMILIES:
        if family not in test_only_families:
            errors.append(f"test-only family missing from test_only: {family}")
        if family in train_families or family in dev_families \
                or family in dev_family_families or family in test_families \
                or family in train_complex or family in dev_complex \
                or family in test_complex or family in train_axelrod \
                or family in dev_axelrod or family in test_axelrod:
            errors.append(f"test-only family leaked: {family}")

    # Complex families must appear in all three complex tiers.
    for family in COMPLEX_FAMILIES:
        if family not in train_complex or family not in dev_complex \
                or family not in test_complex:
            errors.append(f"complex family missing from split: {family}")
    for family in AXELROD_FAMILIES:
        if family not in train_axelrod or family not in dev_axelrod \
                or family not in test_axelrod:
            errors.append(f"Axelrod family missing from split: {family}")

    # Policy train/test pools are checked by the exact-config audit above.

    # Test must not silently inherit a dev-only family.
    leaked_dev = dev_family_families & test_only_families
    if leaked_dev:
        errors.append(f"dev/test family overlap: {sorted(leaked_dev)}")

    # Axelrod is optional; the catalog can still be audited without it, but
    # field evaluation needs the package installed.
    try:
        from ..core.opponents.axelrod_bank import require_axelrod

        require_axelrod()
    except RuntimeError:
        warnings.append(
            "Axelrod missing: install requirements-axelrod.txt before "
            "evaluating Protocol v4 tiers")

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
    for group in ("train", "train_complex", "train_policy",
                  "train_axelrod", "dev", "dev_family", "dev_complex",
                  "dev_axelrod", "test", "test_complex", "test_policy",
                  "test_axelrod", "test_only"):
        n = sum(count for count in counts.get(group, {}).values())
        if n < 3:
            warnings.append(f"small group {group!r}: {n} configs")

    for level in range(5):
        if not any(entry.thinking_level == level and not entry.anchor
                   for entry in rows):
            errors.append(f"empty thinking level: {level}")
    for level in range(5):
        if not any(entry.complexity == level and not entry.anchor
                   for entry in rows):
            errors.append(f"empty complexity level: {level}")

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
            "train_complex": sorted(train_complex),
            "train_policy": sorted(train_policy),
            "train_axelrod": sorted(train_axelrod),
            "dev_param": sorted(dev_families),
            "dev_family": sorted(dev_family_families),
            "dev_complex": sorted(dev_complex),
            "dev_axelrod": sorted(dev_axelrod),
            "test_seen": sorted(test_families),
            "test_complex": sorted(test_complex),
            "test_policy": sorted(test_policy),
            "test_axelrod": sorted(test_axelrod),
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


def _family_weights(rows: Sequence[OpponentConfig]) -> list[float]:
    counts: dict[str, int] = {}
    for entry in rows:
        counts[entry.family] = counts.get(entry.family, 0) + 1
    return [
        FAMILY_WEIGHTS.get(entry.family, 1.0)
        / max(1, counts[entry.family])
        for entry in rows
    ]


def _sample_from_entries(rows: Sequence[OpponentConfig], seed: int,
                         n: int) -> list[Strategy]:
    weights = _family_weights(rows)
    rng = random.Random(int(seed))
    chosen = rng.choices(rows, weights=weights, k=int(n))
    field: list[Strategy] = []
    for entry in chosen:
        bot = entry.build(random.Random(rng.randrange(1 << 30)))
        bot.name = f"{entry.family}:{entry.key}"
        bot.thinking_level = entry.thinking_level
        bot.complexity = entry.complexity
        field.append(bot)
    return field


def _derived_entries(tier: str) -> list[OpponentConfig] | None:
    if tier in LEVEL_TIERS:
        level = int(tier.rsplit("level", 1)[1])
        return [entry for entry in build_catalog()
                if entry.thinking_level == level
                and entry.tier.startswith("dev") and not entry.anchor]
    if tier in COMPLEXITY_TIERS:
        level = int(tier.rsplit("complexity", 1)[1])
        return [entry for entry in build_catalog()
                if entry.complexity == level
                and entry.tier.startswith("dev") and not entry.anchor]
    return None


def _weighted_entries(tier: str) -> tuple[list[OpponentConfig], list[float]]:
    derived = _derived_entries(tier)
    if derived is not None:
        if not derived:
            raise ValueError(f"derived tier {tier!r} has no configs")
        return derived, _family_weights(derived)
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


def build_protocol_field(tier: str, seed: int,
                         n: int = PROTOCOL_N_PLAYERS) -> list[Strategy]:
    """Build one deterministic field for a protocol tier.

    ``tier`` is one of ``train``, ``dev_param``, ``dev_family``,
    ``dev_complex``, ``dev_axelrod``, ``dev_ood`` or ``test_v7``.  Field
    membership is fixed
    by the catalog; ``seed`` only changes the mixture and the per-opponent
    behaviour RNG.
    """
    rows, weights = _weighted_entries(tier)
    rng = random.Random(int(seed))
    chosen = rng.choices(rows, weights=weights, k=int(n))
    field: list[Strategy] = []
    for entry in chosen:
        bot = entry.build(random.Random(rng.randrange(1 << 30)))
        bot.name = f"{entry.family}:{entry.key}"
        bot.thinking_level = entry.thinking_level
        bot.complexity = entry.complexity
        field.append(bot)
    return field


# Field objects may be reused across candidates only when their whole mutable
# RNG state can be restored.  Axelrod's cheating bank (Darwin) keeps state on
# the class instead of the player instance, so it must always be rebuilt.
FIELD_REUSE_UNSAFE_TYPES = frozenset({"Darwin", "Geller", "MindReader"})


def _is_policy_like(value: Any) -> bool:
    """True for protocol strategies and policy opponents with an RNG."""
    if isinstance(value, (Strategy, random.Random)):
        return True
    return (hasattr(value, "__dict__") and hasattr(value, "rng")
            and callable(getattr(value, "act", None)))


def _container_has_state(value: Any) -> bool:
    items = value.values() if isinstance(value, dict) else value
    return any(_is_policy_like(item) for item in items)


def _iter_field_rngs(value: Any, seen: set[int] | None = None) -> Iterable[
        random.Random]:
    """Yield random generators reachable from one field object graph.

    Hand-written strategies subclass :class:`Strategy`, but learned policy
    opponents (Neural/GRU/SelfPlay) are plain objects.  Walk both kinds while
    pruning large weight tensors: containers are only descended into when
    they directly hold a strategy, policy or RNG.
    """
    if isinstance(value, random.Random):
        yield value
        return
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (list, tuple, dict)):
        if _container_has_state(value):
            items = value.values() if isinstance(value, dict) else value
            for item in items:
                if isinstance(item, (random.Random, list, tuple, dict)) \
                        or _is_policy_like(item):
                    yield from _iter_field_rngs(item, seen)
        return
    if not _is_policy_like(value):
        return
    for item in vars(value).values():
        if isinstance(item, (random.Random, list, tuple, dict)) \
                or _is_policy_like(item):
            yield from _iter_field_rngs(item, seen)


def snapshot_field_rng(field: Sequence[Any]
                       ) -> list[tuple[random.Random, object]]:
    """Capture the RNG state of every opponent in a freshly built field.

    ``reset()`` does not rewind an opponent's Mersenne Twister: several
    opponents explicitly keep their stream across matches.  A field can only
    be reused for another candidate after restoring these states, because the
    first candidate advances every opponent RNG.
    """
    snapshot: list[tuple[random.Random, object]] = []
    seen: set[int] = set()
    for rng in _iter_field_rngs(field):
        if id(rng) in seen:
            continue
        seen.add(id(rng))
        snapshot.append((rng, rng.getstate()))
    return snapshot


def restore_field_rng(field: Sequence[Any],
                      snapshot: Sequence[tuple[random.Random, object]]) -> None:
    """Restore states captured by :func:`snapshot_field_rng`."""
    for rng, state in snapshot:
        rng.setstate(state)


def protocol_match_seed(base_seed: int, field_seed: int,
                        candidate: str, match_index: int) -> int:
    """Deterministic per-match seed for the opt-in Protocol v5 RNG contract.

    Protocol v4 uses one sequential Mersenne Twister stream per candidate and
    field.  That stream cannot be batched across matches.  The v5 contract
    gives every match its own stream, derived from the run seed, field seed,
    candidate name and match index, so CPU and GPU can reproduce it exactly.
    """
    payload = (f"{int(base_seed)}|{int(field_seed)}|{candidate}|"
               f"{int(match_index)}")
    return zlib.crc32(payload.encode("utf-8")) & 0x7FFFFFFF


def sample_uniforms(seed: int, rounds: int) -> list[float]:
    """Draw the per-round uniforms consumed by one v5 match."""
    rng = random.Random(int(seed))
    return [rng.random() for _ in range(int(rounds))]


def _opponent_type_names(opponent: Any) -> set[str]:
    names = {type(opponent).__name__}
    player_cls = getattr(opponent, "player_cls", None)
    if isinstance(player_cls, type):
        names.add(player_cls.__name__)
    nested = getattr(opponent, "_bot", None)
    if nested is not None and nested is not opponent:
        names.add(type(nested).__name__)
    return names


def field_reuse_safe(field: Sequence[Strategy]) -> bool:
    """Return whether every opponent supports object reuse after reset()."""
    for opponent in field:
        if bool(getattr(opponent, "reuse_unsafe", False)):
            return False
        if _opponent_type_names(opponent) & FIELD_REUSE_UNSAFE_TYPES:
            return False
    return True


TRAIN_PROFILES: dict[str, tuple[str, ...] | None] = {
    "train_v4": None,
    "train_v3": None,  # legacy alias for the same full mixture
    "train_easy": ("nice", "forgiving", "punitive", "defect", "late",
                   "anchor"),
    "train_hard": ("noisy", "random", "extort", "temporal", "wild",
                   "adaptive", "anchor"),
    "train_complex": tuple(COMPLEX_FAMILIES) + ("anchor",),
    # Targeted hard-mining profiles for families where the current policy has
    # a large gap.  They contain only training-tier entries, so dev/test
    # parameters stay held out.
    "train_pattern_breaker": ("pattern_breaker",),
    "train_weak_complex": ("pattern_breaker", "adaptive_mix",
                           "probe_exploit", "conditional_extort", "lookahead"),
    "train_policy": ("policy",),
    "train_axelrod": ("axelrod", "axelrod_mix"),
}


def build_training_field(profile: str, seed: int,
                         n: int = PROTOCOL_N_PLAYERS
                         ) -> list[Strategy]:
    """Build a training profile (easy / hard / complex / policy mixture)."""
    if profile.startswith("train_level"):
        try:
            level = int(profile.rsplit("level", 1)[1])
        except ValueError as exc:
            raise ValueError(f"bad training profile: {profile!r}") from exc
        rows = [entry for entry in build_catalog()
                if entry.thinking_level == level
                and entry.tier in ("train", "train_complex",
                                   "train_policy", "train_axelrod")]
        if not rows:
            raise ValueError(f"training profile {profile!r} has no configs")
        return _sample_from_entries(rows, seed, n)
    if profile.startswith("train_complexity"):
        try:
            level = int(profile.rsplit("complexity", 1)[1])
        except ValueError as exc:
            raise ValueError(f"bad training profile: {profile!r}") from exc
        rows = [entry for entry in build_catalog()
                if entry.complexity == level
                and entry.tier in ("train", "train_complex",
                                   "train_policy", "train_axelrod")]
        if not rows:
            raise ValueError(f"training profile {profile!r} has no configs")
        return _sample_from_entries(rows, seed, n)
    if profile not in TRAIN_PROFILES:
        raise ValueError(f"unknown training profile {profile!r}; "
                         f"choose from {tuple(TRAIN_PROFILES)}")
    if profile in ("train_v3", "train_v4"):
        return build_protocol_field("train", seed, n)
    allowed = set(TRAIN_PROFILES[profile] or ())
    rows = [
        entry for entry in build_catalog()
        if entry.family in allowed
        and (entry.tier in ("train", "train_complex", "train_policy",
                            "train_axelrod")
             or entry.anchor)
    ]
    if profile in ("train_policy", "train_axelrod"):
        rows = [entry for entry in rows if entry.tier ==
                ("train_policy" if profile == "train_policy"
                 else "train_axelrod")]
    if not rows:
        raise ValueError(f"training profile {profile!r} has no configs")
    return _sample_from_entries(rows, seed, n)


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


def load_test_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load the test-consumption policy, failing closed.

    The policy is deliberately separate from the CLI so a test run cannot
    consume a held-out tier just because a flag was forgotten.  Missing,
    unreadable or malformed policy files all mean "locked".
    """
    target = Path(path) if path is not None else TEST_POLICY_PATH
    locked = {
        "schema": "ipd-protocol-test-policy-v1",
        "protocol": PROTOCOL_NAME,
        "consume_test": False,
        "reason": f"test policy {target} is missing or unreadable; "
                  "test consumption is locked",
    }
    if not target.exists():
        return locked
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return locked
    if not isinstance(data, dict):
        return locked
    data.setdefault("schema", "ipd-protocol-test-policy-v1")
    data.setdefault("protocol", PROTOCOL_NAME)
    data.setdefault("consume_test", False)
    return data


def test_consumption_allowed(path: str | Path | None = None) -> bool:
    """Return True only if the policy explicitly unlocks test consumption."""
    return load_test_policy(path).get("consume_test") is True


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
            f"{candidate!r} already consumed {tier}; pass --force to rerun "
            "and mark the result contaminated")
    return data


def record_usage(tier: str, candidate: str, result_path: str, *,
                 path: str | Path | None = None, force: bool = False,
                 note: str = "", allow_locked: bool = False) -> Path:
    """Append one usage entry; a forced rerun is marked contaminated.

    The test policy is enforced here as well as in the CLI so direct callers
    cannot consume a held-out tier while the lock is active.  Low-level
    tooling/tests can pass ``allow_locked=True`` explicitly.
    """
    if not allow_locked and not test_consumption_allowed():
        raise RuntimeError(
            f"test consumption is locked by {TEST_POLICY_PATH}; refusing to "
            "write the usage ledger")
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
    "PROTOCOL_NAME", "PROTOCOL_VERSION", "TEST_TIER", "TIER_NAMES",
    "DEV_TIERS", "TEST_TIERS", "SPLIT_TIERS", "LEVEL_TIERS",
    "COMPLEXITY_TIERS", "TIER_GROUPS", "FAMILY_WEIGHTS",
    "FAMILY_LEVELS", "FAMILY_COMPLEXITY", "AXELROD_LEVELS",
    "TRAIN_PROFILES", "ANCHORS", "COMPLEX_FAMILIES",
    "POLICY_FAMILIES", "AXELROD_FAMILIES", "AXELROD_STRATEGIES",
    "AXELROD_EXTRA", "AXELROD_MIXES", "PROTOCOL_N_PLAYERS", "DEV_FIELDS",
    "TEST_FIELDS",
    "DEV_ONLY_FAMILIES", "TEST_ONLY_FAMILIES", "MANIFEST_PATH",
    "USAGE_PATH", "TEST_POLICY_PATH", "load_test_policy",
    "test_consumption_allowed", "OpponentConfig", "MirrorAdaptive",
    "DelayedGrim",
    "PhaseSwitcher", "AdaptiveMix", "ProbeExploit", "DelayedExtort",
    "MemoryTwo", "MetaMix", "NoiseAdaptive", "ReputationGrim",
    "ContritionCycle", "EndgameHunter", "PatternBreaker",
    "ConditionalExtort", "LookaheadOpponent", "VotingPortfolio",
    "AxelrodMix", "build_catalog",
    "catalog_sha256", "catalog_counts", "protocol_manifest",
    "write_manifest", "read_manifest", "verify_manifest", "audit_protocol",
    "build_protocol_field", "build_training_field",
    "build_protocol_red_field", "red_pool", "load_usage",
    "guard_test_usage", "record_usage",
    "FIELD_REUSE_UNSAFE_TYPES", "snapshot_field_rng", "restore_field_rng",
    "field_reuse_safe",
]
