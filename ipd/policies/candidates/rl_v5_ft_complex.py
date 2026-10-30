"""Candidate: RL-V5-FT with extra ``train_complex`` fine-tuning.

Second-stage fine-tune from RL-V5-FT on ``train_v5 + train_complex``
(weights 1:1), 10 iterations, lr 5e-5, selecting iter 6.  It improves
dev_complex / dev_axelrod / dev_ood over RL-V5-FT at flat dev_param and
slightly improves test_v8 provisional.  Experimental, not promoted.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "rl-v5-ft-complex"
DISPLAY = "RL-V5-FT-Complex"
DEFAULT_WEIGHTS = "rl_policy_weights_v5_ft_complex.json"


def _load_base(weights: str, seed: int, mode: str):
    with open(weights, encoding="utf-8") as fh:
        fmt = json.load(fh).get("format")
    cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
    return cls(weights, seed=seed, mode=mode)


def build(*, seed: int = 20261024, weights: str | None = None,
          mode: str = "sample", name: str | None = None,
          **_: object) -> object:
    policy = _load_base(weights or weights_path(DEFAULT_WEIGHTS), seed, mode)
    policy.name = name or DISPLAY
    return policy


__all__ = ["KIND", "DISPLAY", "DEFAULT_WEIGHTS", "build"]
