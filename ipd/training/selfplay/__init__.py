"""Asymmetric self-play PPO for 100-round IPD.

This package implements the plan in ``docs/research/selfplay-plan.md``:

* a frozen 21-dimensional public-history observation,
* a 32-environment lock-step simulator,
* parameterised bot populations with train/validation/test splits,
* independent recurrent actor-critics for blue and red,
* a red league and constrained red reward, and
* a three-phase curriculum with fast/full validation and a held-out final test.
"""
from __future__ import annotations

OBSERVATION_VERSION = "selfplay-21-v1"
N_FEATURES = 21

__all__ = ["OBSERVATION_VERSION", "N_FEATURES"]
