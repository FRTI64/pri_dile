"""Opponent bank, split by strategy family.

The package keeps the historical import surface (``ipd.core.opponents``) while
moving each opponent family into its own module:

* :mod:`.basic` -- canonical simple opponents (AllC, TFT, Grim, ...)
* :mod:`.reciprocal` -- reciprocal, forgiving, and adaptive opponents
* :mod:`.defectors` -- late/early defectors and autonomous policies
* :mod:`.memory` -- memory-one, reactive, periodic, and FSM opponents
* :mod:`.zero_determinant` -- ZD extortion / generous opponents
* :mod:`.bank` -- the 85-type bank and mixture samplers
* :mod:`.plus` -- extra parameterised types
* :mod:`.legacy` -- the old fixed-weight field helper
"""
from __future__ import annotations

from ..primitives import C, D, Strategy, payoff
from .bank import (
    FAMILIES, FAMILY_MULTIPLIERS, SCENARIOS, TYPE_BANK, TYPE_BY_NAME,
    effective_weights, random_scenario, sample_mixed_population,
    sample_mixture, sample_population,
)
from .basic import (
    AllC, AllD, Bully, FirmButFair, GTFT, Grim, Grudge2, Joss, Mistrust,
    Pavlov, Prober, RandomMoves, SuspiciousTFT, TFT, TFT2, TFTLastD, Tester,
    WinStayLoseShift2,
)
from .defectors import (
    Alternator, EarlyDefector, Gradual, Grudger2, LateDefector, RandomP,
)
from .legacy import (
    FIELD_SPECS, WEIGHT_TOTAL, make_field, score_against,
)
from .memory import (
    AllCWithLastD, GrudgerK, MemoryOne, OmegaTFTLite, Periodic, RandomFSM,
    Reactive, TwoTitsForTat,
)
from .plus import ALL_SPECS as PLUS_SPECS
from .plus import PLUS
from .reciprocal import (
    AdaptiveProbe, ContriteTFT, DelayedTFT, ForgiveTFT, GenerousTFT,
    GrimForgive, LastKDefect, Majority, NoisyTFT, ProbeBot, RandomThenTFT,
    RemorsefulProber, Responsive, Responsive2, Responsive3, TFT2LastD,
    TFTForgiveOnce, TFTNoisyForgive, TFTProbe, TFTProbeImmediate, TwoStrike,
)
from .spec import TypeSpec
from .zero_determinant import ExtortZD, GenerousZD

__all__ = [
    "C", "D", "Strategy", "payoff",
    "AllC", "AllD", "Bully", "FirmButFair", "GTFT", "Grim", "Grudge2",
    "Joss", "Mistrust", "Pavlov", "Prober", "RandomMoves", "SuspiciousTFT",
    "TFT", "TFT2", "TFTLastD", "Tester", "WinStayLoseShift2",
    "Alternator", "EarlyDefector", "Gradual", "Grudger2", "LateDefector",
    "RandomP",
    "AllCWithLastD", "GrudgerK", "MemoryOne", "OmegaTFTLite", "Periodic",
    "RandomFSM", "Reactive", "TwoTitsForTat",
    "AdaptiveProbe", "ContriteTFT", "DelayedTFT", "ForgiveTFT",
    "GenerousTFT", "GrimForgive", "LastKDefect", "Majority", "NoisyTFT",
    "ProbeBot", "RandomThenTFT", "RemorsefulProber", "Responsive",
    "Responsive2", "Responsive3", "TFT2LastD", "TFTForgiveOnce",
    "TFTNoisyForgive", "TFTProbe", "TFTProbeImmediate", "TwoStrike",
    "ExtortZD", "GenerousZD", "TypeSpec",
    "TYPE_BANK", "TYPE_BY_NAME", "FAMILIES", "FAMILY_MULTIPLIERS",
    "SCENARIOS", "effective_weights", "sample_mixture", "sample_population",
    "sample_mixed_population", "random_scenario",
    "PLUS", "PLUS_SPECS",
    "FIELD_SPECS", "WEIGHT_TOTAL", "make_field", "score_against",
]
