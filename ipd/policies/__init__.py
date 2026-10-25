"""Tournament policies: unified interface, concrete implementations, registry."""
from __future__ import annotations

from .adversarial import AdversarialPolicy
from .base import Policy, PolicyAdapter, boundaries_ok, ensure_policy
from .gru import GRUNetPolicy, GRUPolicy
from .neural import NeuralMachine, NeuralPolicy, extract_features
from .registry import (
    ALL_NAMES, BASE_NAMES, CLASSIC_NAMES, CORE_NAMES, CandidateSpec, DISPLAY,
    LEARNED_NAMES, LEGACY_NAMES, SUITE_NAMES, candidate_specs,
    canonical_name, legacy_arch_names,
    make_policy, resolve_names,
)
from .rl import RLPolicy
from .rules import (
    GapStartPolicy, HarvestPolicy, Hybrid, ParamBayes, ParamBayesPrior,
    ParamHarvest, ParamPolicy, ParamResp, Triple,
)
from .tuned import TunedConfigs, load_tuned_configs

__all__ = [
    "Policy", "PolicyAdapter", "ensure_policy", "boundaries_ok",
    "NeuralPolicy", "NeuralMachine", "extract_features",
    "GRUPolicy", "GRUNetPolicy", "RLPolicy", "AdversarialPolicy",
    "ParamPolicy", "ParamResp", "ParamBayes", "ParamBayesPrior",
    "ParamHarvest", "HarvestPolicy", "Hybrid", "Triple", "GapStartPolicy",
    "TunedConfigs", "load_tuned_configs",
    "CandidateSpec", "make_policy", "candidate_specs", "resolve_names",
    "canonical_name", "legacy_arch_names", "DISPLAY",
    "CORE_NAMES", "BASE_NAMES", "LEGACY_NAMES", "CLASSIC_NAMES",
    "SUITE_NAMES", "LEARNED_NAMES", "ALL_NAMES",
]
