"""Candidate: RL-Wide160 with a narrow early switch to RL-Repair.

The meta-policy stays on RL-Wide160 for the normal safe path and switches to
RL-Repair only when the first six rounds show mutual defection followed by
opponent cooperation.  This targets the extortion / noisy tail found in the
full opponent scan; it is Python-only for now.
"""
from __future__ import annotations

from ...paths import weights_path
from ..meta import RLSwitchPolicy

KIND = "rl-switch"
DISPLAY = "RL-Switch"
BASE_WEIGHTS = "rl_policy_weights_wide160.json"
ALT_WEIGHTS = "rl_policy_weights_repair.json"


def build(*, seed: int = 20261024, mode: str = "sample",
          name: str | None = None, weights: str | None = None,
          **_: object) -> object:
    policy = RLSwitchPolicy(
        weights or weights_path(BASE_WEIGHTS),
        weights_path(ALT_WEIGHTS),
        seed=seed, mode=mode, name=name or DISPLAY)
    return policy


__all__ = ["KIND", "DISPLAY", "BASE_WEIGHTS", "ALT_WEIGHTS", "build"]
