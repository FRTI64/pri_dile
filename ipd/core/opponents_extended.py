"""Compatibility shim for the historical extended-opponent module.

The implementation now lives in the :mod:`ipd.core.opponents` package; this
module keeps the old import path working.
"""
from __future__ import annotations

from .opponents import (
    FAMILIES, FAMILY_MULTIPLIERS, SCENARIOS, TYPE_BANK, TYPE_BY_NAME,
    AllCWithLastD, ExtortZD, GenerousZD, GrudgerK, MemoryOne, OmegaTFTLite,
    Periodic, RandomFSM, Reactive, TwoTitsForTat, TypeSpec, effective_weights,
    random_scenario, sample_mixed_population, sample_mixture, sample_population,
)
from .opponents.zero_determinant import R, S, T, P

__all__ = [
    "R", "S", "T", "P", "TwoTitsForTat", "AllCWithLastD", "GrudgerK",
    "OmegaTFTLite", "Periodic", "MemoryOne", "Reactive", "RandomFSM",
    "ExtortZD", "GenerousZD", "TypeSpec", "TYPE_BANK", "TYPE_BY_NAME",
    "FAMILIES", "FAMILY_MULTIPLIERS", "SCENARIOS", "effective_weights",
    "sample_mixture", "sample_population", "sample_mixed_population",
    "random_scenario",
]
