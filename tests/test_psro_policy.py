"""Deployable PSRO meta-policy: sampling, reset and delegation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ipd.policies.neural import NeuralPolicy
from ipd.policies.psro import PSROMetaPolicy, load_meta_strategy
from ipd.training.psro import make_scratch_policy


def _meta(path: Path, weights: dict[str, float]) -> Path:
    path.write_text(json.dumps({
        "schema": "ipd-psro-report-v1",
        "name": "test",
        "population": [
            {"name": "Final-tuned", "role": "blue",
             "entry": {"kind": "final-tuned"}},
            {"name": "Triple-majority", "role": "blue",
             "entry": {"kind": "triple-majority"}},
        ],
        "blue_meta": {"weights_by_name": weights},
    }), encoding="utf-8")
    return path


class ScratchPolicyTests(unittest.TestCase):
    def test_random_policy_is_loadable_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scratch.json"
            data = make_scratch_policy(path, seed=11, hidden=(8, 4))
            self.assertEqual(data["format"], "ipd-neural-policy-v1")
            self.assertEqual([len(layer["w"]) for layer in data["layers"]],
                             [8, 4, 1])
            policy = NeuralPolicy(str(path), seed=3)
            policy.reset()
            self.assertEqual(policy.act(0, [], []), 0)
            self.assertEqual(policy.act(99, [0] * 99, [0] * 99), 1)

    def test_deterministic_for_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make_scratch_policy(Path(tmp) / "a.json", seed=5,
                                    hidden=(4,))
            b = make_scratch_policy(Path(tmp) / "b.json", seed=5,
                                    hidden=(4,))
            self.assertEqual(a["layers"], b["layers"])


class PSROMetaPolicyTests(unittest.TestCase):
    def test_samples_only_weighted_members_and_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _meta(Path(tmp) / "meta.json",
                         {"Final-tuned": 1.0, "Triple-majority": 0.0})
            policy = PSROMetaPolicy(path, seed=7)
            for _ in range(5):
                policy.reset()
                self.assertEqual(policy.current_name, "Final-tuned")
                self.assertEqual(policy.act(0, [], []), 0)
                self.assertEqual(policy.act(99, [0] * 99, [0] * 99), 1)

    def test_uniform_fallback_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _meta(Path(tmp) / "meta.json", {})
            policy = PSROMetaPolicy(path, seed=3)
            names = set()
            for _ in range(20):
                policy.reset()
                names.add(policy.current_name)
            self.assertEqual(names, {"Final-tuned", "Triple-majority"})
            self.assertEqual(len(policy._members), 2)

    def test_loads_real_default_meta_strategy(self):
        from ipd.paths import weights_path

        path = weights_path("psro/meta_strategy.json")
        if not Path(path).is_file():
            self.skipTest("PSRO meta-strategy not generated yet")
        data = load_meta_strategy(path)
        self.assertIn("blue_meta", data)
        self.assertIn("population", data)


if __name__ == "__main__":
    unittest.main()
