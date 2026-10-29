"""Tests for net2wider MLP expansion (no PyTorch required)."""
from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from ipd.policies.neural import (
    FEATURE_COUNT_V2, FEATURE_VERSION_V2, NeuralPolicy,
)
from ipd.tools.export_cpp import export_mlp, fmt
from ipd.training.expand import (
    expand_actor_critic, expand_deployable, expand_input_features, _mapping,
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

    def test_input_feature_expansion_pads_first_layer(self) -> None:
        src = _actor_critic()
        out = expand_input_features(copy.deepcopy(src), seed=5)
        self.assertEqual(out["n_features"], FEATURE_COUNT_V2)
        self.assertEqual(out["feature_version"], FEATURE_VERSION_V2)
        self.assertEqual(len(out["trunk"][0]["w"][0]), FEATURE_COUNT_V2)
        self.assertEqual(len(out["value_trunk"][0]["w"][0]),
                         FEATURE_COUNT_V2)
        self.assertEqual(out["trunk"][0]["w"][0][:85],
                         src["trunk"][0]["w"][0])
        self.assertEqual(len(out["feature_mean"]), FEATURE_COUNT_V2)

    def test_deployable_v2_policy_uses_v2_features(self) -> None:
        out = expand_input_features(copy.deepcopy(_deployable()), seed=6)
        policy = NeuralPolicy(copy.deepcopy(out))
        self.assertEqual(policy.n_features, FEATURE_COUNT_V2)
        self.assertEqual(policy.feature_version, FEATURE_VERSION_V2)
        self.assertIn(policy.act(0, [], []), (0, 1))

    def test_deployable_expansion_is_deterministic(self) -> None:
        a = expand_deployable(copy.deepcopy(_deployable()), [128, 64],
                              seed=11)
        b = expand_deployable(copy.deepcopy(_deployable()), [128, 64],
                              seed=11)
        self.assertEqual(a, b)


class ExportCppTests(unittest.TestCase):
    def test_float_literal_always_has_decimal_point(self) -> None:
        self.assertEqual(fmt(0.0), "0.0f")
        self.assertEqual(fmt(1.0), "1.0f")
        self.assertEqual(fmt(0.5), "0.5f")
        self.assertIn("e", fmt(1e-9).lower())

    def test_v2_export_sets_feature_macro(self) -> None:
        data = expand_input_features(copy.deepcopy(_deployable()), seed=8)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "weights.json"
            out = Path(tmp) / "weights.hpp"
            src.write_text(json.dumps(data), encoding="utf-8")
            export_mlp(str(src), str(out))
            text = out.read_text(encoding="utf-8")
        self.assertIn("#define IPD_FEATURES_V2 1", text)
        self.assertIn("constexpr int N_FEATURES = 97;", text)


if __name__ == "__main__":
    unittest.main()
