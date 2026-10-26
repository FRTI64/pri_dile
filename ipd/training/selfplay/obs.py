"""NumPy wrapper around the frozen 21-dimensional observation contract.

The contract itself lives in :mod:`ipd.policies.selfplay_gru_obs` so training
and dependency-free inference share exactly one implementation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ipd.policies.selfplay_gru_obs import (
    N_FEATURES, OBSERVATION_VERSION, feature_names, features, outcome_index,
)

from . import N_FEATURES as _TRAIN_N_FEATURES

if N_FEATURES != _TRAIN_N_FEATURES:  # pragma: no cover - import guard
    raise RuntimeError("observation feature count drift")

__all__ = [
    "N_FEATURES", "OBSERVATION_VERSION", "features", "features_batch",
    "feature_names", "outcome_index",
]


def features_batch(envs: Sequence[object]) -> np.ndarray:
    """Stack observations for objects exposing ``t``, ``mine`` and ``theirs``."""
    rows = [features(env.t, env.mine, env.theirs) for env in envs]
    return np.asarray(rows, dtype=np.float32)
