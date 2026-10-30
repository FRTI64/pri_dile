"""Candidate: RL-Wide160 fine-tuned on the Protocol v5 ``train_v5`` profile.

Starts from the 160/128 RL-Wide160 checkpoint and runs 12 short PPO
iterations on the full v5 training mixture (standard + complex + policy +
Axelrod + anchors), with lr 1e-4 and a KL anchor to the initial policy.
Iteration 8 is selected on the five v5 dev tiers.

The candidate keeps the base safety rules unchanged.  It improves
``dev_complex`` / ``dev_axelrod`` / ``dev_ood`` over RL-Wide160 in two dev
seeds and in ``test_v8`` provisional, but ``probe_exploit`` remains the
bottleneck; it is an experimental candidate, not a promoted main weight.
"""
from __future__ import annotations

import json

from ...paths import weights_path
from ..neural import NeuralPolicy
from ..rl import RLPolicy

KIND = "rl-v5-ft"
DISPLAY = "RL-V5-FT"
DEFAULT_WEIGHTS = "rl_policy_weights_v5_ft.json"


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
