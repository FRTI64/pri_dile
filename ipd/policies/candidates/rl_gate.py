"""Candidate: learned match-level gate between RL-Wide160 and RL-Repair.

The gate observes the first 10 rounds (played by RL-Wide160) and then chooses
one expert for the rest of the match.  The default gate is the cost-sensitive
logistic model trained on train-tier counterfactual rollouts; see
``results/meta_20261040/train_cost_sensitive_gate.py`` and
``docs/research/rl-gate.md``.  Python-only for now.
"""
from __future__ import annotations

from ...paths import weights_path
from ..meta import LearnedSwitchPolicy

KIND = "rl-gate"
DISPLAY = "RL-Gate"
BASE_WEIGHTS = "rl_policy_weights_wide160.json"
ALT_WEIGHTS = "rl_policy_weights_repair.json"
GATE_WEIGHTS = "meta_gate_cs_wide160_repair.json"


def build(*, seed: int = 20261024, mode: str = "sample",
          name: str | None = None, weights: str | None = None,
          **_: object) -> object:
    policy = LearnedSwitchPolicy(
        weights or weights_path(BASE_WEIGHTS),
        weights_path(ALT_WEIGHTS),
        weights_path(GATE_WEIGHTS),
        seed=seed, mode=mode, name=name or DISPLAY)
    return policy


__all__ = ["KIND", "DISPLAY", "BASE_WEIGHTS", "ALT_WEIGHTS",
           "GATE_WEIGHTS", "build"]
