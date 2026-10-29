"""Tests for net2wider MLP expansion (no PyTorch required)."""
from __future__ import annotations

import copy
import random
import unittest

from ipd.policies.neural import NeuralPolicy
from ipd.training.expand import (
    expand_actor_critic, expand_deployable, _mapping,
)

N_FEATURES = 85


def _layer(rng: random.Random, out: int, inp: int) -> dict:
    return {
        "w": [[rng.uniform(-0.2, 0.2) for _ in range(inp)]
              for _ in range(out)],
        "b": [rng.uniform(-0.1, 0.1) for _ in range(out)],
    }


def _deployable() -> dict:
    rng = random.Random(1)
    return {
        "format": "ipd-neural-policy-v1",
        "feature_version": 3,
        "n_features": N_FEATURES,
        "activation": "tanh",
        "layers": [
            _layer(rng, 64, N_FEATURES),
            _layer(rng, 32, 64),
            _layer(rng, 1, 32),
        ],
        "feature_mean": [0.0] * N_FEATURES,
        "feature_std": [1.0] * N_FEATURES,
        "temperature": 1.0,
        "safe": True,
    }


def _actor_critic() -> dict:
    rng = random.Random(2)
    return {
        "format": "ipd-rl-policy-v1",
        "feature_version": 3,
        "n_features": N_FEATURES,
        "activation": "tanh",
        "trunk": [_layer(rng, 64, N_FEATURES), _layer(rng, 32, 64)],
        "value_trunk": [_layer(rng, 64, N_FEATURES), _layer(rng, 32, 64)],
        "policy_head": _layer(rng, 1, 32),
        "value_head": _layer(rng, 1, 32),
        "feature_mean": [0.0] * N_FEATURES,
        "feature_std": [1.0] * N_FEATURES,
        "temperature": 1.0,
        "safe": True,
        "meta": {},
    }


class ExpandTests(unittest.TestCase):
    def test_mapping_is_balanced(self) -> None:
        mapping = _mapping(3, 8)
        self.assertEqual(len(mapping), 8)
        self.assertEqual(sorted(set(mapping)), [0, 1, 2])
        self.assertEqual(mapping.count(0), 3)
        self.assertEqual(mapping.count(1), 3)
        self.assertEqual(mapping.count(2), 2)

    def test_deployable_expansion_shapes_and_function(self) -> None:
        src = _deployable()
        out = expand_deployable(copy.deepcopy(src), [128, 64], seed=7)
        self.assertEqual([len(layer["w"]) for layer in out["layers"]],
                         [128, 64, 1])
        self.assertEqual(len(out["layers"][1]["w"][0]), 128)
        self.assertEqual(len(out["layers"][2]["w"][0]), 64)
        base = NeuralPolicy(copy.deepcopy(src))
        wide = NeuralPolicy(copy.deepcopy(out))
        features = [random.Random(3).uniform(-1.0, 1.0)
                    for _ in range(N_FEATURES)]
        # net2wider plus 1% symmetry-breaking noise: the function is close,
        # not identical.
        self.assertAlmostEqual(base._logit(features), wide._logit(features),
                               delta=0.05)

    def test_actor_critic_expansion_shapes(self) -> None:
        out = expand_actor_critic(copy.deepcopy(_actor_critic()),
                                  [128, 64], seed=9)
        self.assertEqual([len(layer["w"]) for layer in out["trunk"]],
                         [128, 64])
        self.assertEqual([len(layer["w"]) for layer in out["value_trunk"]],
                         [128, 64])
        self.assertEqual(len(out["policy_head"]["w"][0]), 64)
        self.assertEqual(len(out["value_head"]["w"][0]), 64)
        self.assertEqual(out["meta"]["expanded_to_hidden"], [128, 64])

    def test_deployable_expansion_is_deterministic(self) -> None:
        a = expand_deployable(copy.deepcopy(_deployable()), [128, 64],
                              seed=11)
        b = expand_deployable(copy.deepcopy(_deployable()), [128, 64],
                              seed=11)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
