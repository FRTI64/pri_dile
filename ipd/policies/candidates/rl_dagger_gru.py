"""Candidate: single-model GRU distilled from the RL-Switch teacher.

This policy is trained with DAgger to imitate the RL-Switch rule
(RL-Wide160 until a narrow early-switch signal, then RL-Repair).  Unlike
``rl-switch`` it keeps one weight file and one recurrent policy, so it does
not need the two-expert switch at inference time.  The GRU hidden state
carries the match-level switch bit.

It is an experimental candidate: dev parity with RL-Switch on both official
seeds, but not promoted over the main weights.  See
``docs/research/dagger-gru.md``.
"""
from __future__ import annotations

from ...paths import weights_path
from ..gru import GRUPolicy

KIND = "rl-dagger-gru"
DISPLAY = "GRU-DAgger"
DEFAULT_WEIGHTS = "rl_dagger_gru_weights.json"


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> GRUPolicy:
    policy = GRUPolicy(
        weights or weights_path(DEFAULT_WEIGHTS), seed=seed, mode=mode)
    policy.name = name or DISPLAY
    return policy


__all__ = ["KIND", "DISPLAY", "DEFAULT_WEIGHTS", "build"]
