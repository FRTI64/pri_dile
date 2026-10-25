"""Core simulation primitives and opponent banks."""
from __future__ import annotations

from .simulator import (
    ALL_STRATEGIES, C, D, AllC, AllD, Bully, FirmButFair, GTFT, Grim,
    Grudge2, Joss, Mistrust, Pavlov, Prober, RandomMoves, Result, Strategy,
    SuspiciousTFT, TFT, TFT2, TFTLastD, Tester, WinStayLoseShift2,
    payoff, play_match, tournament,
)
from .machine import Machine, choose_action
from .harvest import HarvestMachine
from .early import EarlyProbeMachine
from .opponents import (
    FIELD_SPECS, WEIGHT_TOTAL, AdaptiveProbe, Alternator, ContriteTFT,
    DelayedTFT, EarlyDefector, ForgiveTFT, GenerousTFT, Gradual,
    GrimForgive, LastKDefect, LateDefector, Majority, NoisyTFT, ProbeBot,
    RandomThenTFT, RandomP, RemorsefulProber, Responsive, Responsive2,
    Responsive3, TFT2LastD, TFTForgiveOnce, TFTNoisyForgive, TFTProbe,
    TwoStrike, make_field, score_against,
)
from .opponents_extended import (
    SCENARIOS, TYPE_BANK, TYPE_BY_NAME,
    ExtortZD, GenerousZD, MemoryOne, RandomFSM, Reactive, TypeSpec,
    effective_weights, sample_mixed_population, sample_mixture,
    sample_population, random_scenario,
)
from .opponents_plus import ALL_SPECS as EXTENDED_SPECS
from .bayes import BayesMachine
from .bayes_plus import BayesPlusMachine
from .ema import EMAMachine
from .ema_forgive import EMAForgiveMachine

__all__ = [
    "C", "D", "Result", "Strategy", "ALL_STRATEGIES", "payoff",
    "play_match", "tournament", "AllC", "AllD", "Bully", "FirmButFair",
    "GTFT", "Grim", "Grudge2", "Joss", "Mistrust", "Pavlov", "Prober",
    "RandomMoves", "SuspiciousTFT", "TFT", "TFT2", "TFTLastD", "Tester",
    "WinStayLoseShift2", "Machine", "choose_action",
    "HarvestMachine", "EarlyProbeMachine",
    "Responsive", "Responsive2", "Responsive3", "ProbeBot",
    "GenerousTFT", "TFT2LastD", "TFTProbe", "AdaptiveProbe",
    "ContriteTFT", "Alternator", "LastKDefect", "NoisyTFT",
    "RandomThenTFT", "Majority", "TFTForgiveOnce", "GrimForgive",
    "ForgiveTFT", "TwoStrike", "LateDefector", "EarlyDefector",
    "Gradual", "DelayedTFT", "RemorsefulProber", "RandomP",
    "TFTNoisyForgive", "make_field", "score_against",
    "FIELD_SPECS", "WEIGHT_TOTAL",
    "TypeSpec", "MemoryOne", "Reactive", "RandomFSM", "ExtortZD",
    "GenerousZD", "TYPE_BANK", "TYPE_BY_NAME", "SCENARIOS",
    "EXTENDED_SPECS", "effective_weights", "sample_mixture",
    "sample_population", "sample_mixed_population", "random_scenario",
    "BayesMachine", "BayesPlusMachine", "EMAMachine", "EMAForgiveMachine",
]
