"""Pure-standard-library 21-dimensional observation used at inference.

This mirrors :mod:`ipd.training.selfplay.obs`; keeping a dependency-free copy
under ``policies`` lets the submitted wrapper run without NumPy or PyTorch.
"""
from __future__ import annotations

import math
from typing import Sequence

C = 0
D = 1
OBSERVATION_VERSION = "selfplay-21-v1"
N_FEATURES = 21


def outcome_index(mine: int, theirs: int) -> int:
    if mine == C:
        return 0 if theirs == C else 1
    return 2 if theirs == C else 3


def feature_names() -> tuple[str, ...]:
    return (
        "t_over_100", "remaining_over_100",
        "last_CC", "last_CD", "last_DC", "last_DD",
        "lag1_mine", "lag1_theirs",
        "lag2_mine", "lag2_theirs",
        "lag3_mine", "lag3_theirs",
        "lag4_mine", "lag4_theirs",
        "opp_defect_rate", "own_defect_rate",
        "opp_defect_rate_10", "defect_rate_delta",
        "cc_rate", "dd_rate", "endgame",
    )


def features_v2(t: int, mine: Sequence[int], theirs: Sequence[int]) -> list[float]:
    """Experimental v2 contract: delta = pD - pC response gap."""
    if len(mine) != len(theirs):
        raise ValueError("mine and theirs must have the same length")
    if int(t) != len(mine):
        raise ValueError("t must match the pre-step history length")
    t = int(t)
    out = [0.0] * N_FEATURES
    out[0] = t / 100.0
    out[1] = (100 - t) / 100.0
    # Missing micro-history days are explicit rather than looking like CC.
    for lag in range(1, 5):
        j = 6 + 2 * (lag - 1)
        out[j] = -1.0
        out[j + 1] = -1.0
    if t:
        out[2 + outcome_index(mine[-1], theirs[-1])] = 1.0
        for lag in range(1, 5):
            j = 6 + 2 * (lag - 1)
            if t >= lag:
                out[j] = float(mine[-lag])
                out[j + 1] = float(theirs[-lag])
        denom = float(max(1, t))
        opp_def = sum(1 for x in theirs if x == D) / denom
        own_def = sum(1 for x in mine if x == D) / denom
        recent = theirs[-10:]
        opp_def_10 = (sum(1 for x in recent if x == D) / len(recent)
                      if recent else 0.0)
        cc = sum(1 for a, b in zip(mine, theirs) if a == C and b == C)
        dd = sum(1 for a, b in zip(mine, theirs) if a == D and b == D)
        # Response gap: P(opponent defects next | we cooperated) versus
        # P(opponent defects next | we defected).  Positive values identify
        # reciprocal/punitive opponents; near zero identifies autonomous
        # opponents (Random, AllD, Bully), which can be exploited.
        n_c = n_d = d_c = d_d = 0
        for i in range(t - 1):
            if mine[i] == C:
                n_c += 1
                d_c += theirs[i + 1] == D
            else:
                n_d += 1
                d_d += theirs[i + 1] == D
        p_c = d_c / n_c if n_c else 0.0
        p_d = d_d / n_d if n_d else p_c
        out[14] = opp_def
        out[15] = own_def
        out[16] = opp_def_10
        out[17] = p_d - p_c
        out[18] = cc / denom
        out[19] = dd / denom
    out[20] = 1.0 if t >= 90 else 0.0
    if any(not math.isfinite(x) for x in out):
        raise ValueError("non-finite observation")
    return out


def features(t: int, mine: Sequence[int], theirs: Sequence[int]) -> list[float]:
    """Canonical v1 contract: delta = own_defect_rate - opp_defect_rate."""
    if len(mine) != len(theirs):
        raise ValueError("mine and theirs must have the same length")
    if int(t) != len(mine):
        raise ValueError("t must match the pre-step history length")
    t = int(t)
    out = [0.0] * N_FEATURES
    out[0] = t / 100.0
    out[1] = (100 - t) / 100.0
    for lag in range(1, 5):
        j = 6 + 2 * (lag - 1)
        out[j] = -1.0
        out[j + 1] = -1.0
    if t:
        out[2 + outcome_index(mine[-1], theirs[-1])] = 1.0
        for lag in range(1, 5):
            j = 6 + 2 * (lag - 1)
            if t >= lag:
                out[j] = float(mine[-lag])
                out[j + 1] = float(theirs[-lag])
        denom = float(max(1, t))
        opp_def = sum(1 for x in theirs if x == D) / denom
        own_def = sum(1 for x in mine if x == D) / denom
        recent = theirs[-10:]
        out[14] = opp_def
        out[15] = own_def
        out[16] = (sum(1 for x in recent if x == D) / len(recent)
                   if recent else 0.0)
        out[17] = own_def - opp_def
        out[18] = sum(1 for a, b in zip(mine, theirs)
                      if a == C and b == C) / denom
        out[19] = sum(1 for a, b in zip(mine, theirs)
                      if a == D and b == D) / denom
    out[20] = 1.0 if t >= 90 else 0.0
    return out


__all__ = [
    "C", "D", "N_FEATURES", "OBSERVATION_VERSION", "feature_names",
    "outcome_index", "features", "features_v2",
]
