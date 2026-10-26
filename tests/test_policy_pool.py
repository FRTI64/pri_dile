"""Frozen blue-policy pool used by P3 red best-response training."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from .helpers import ROOT
except ImportError:  # direct execution
    from helpers import ROOT  # noqa: F401

from ipd.evaluation.policy_pool import (
    build_policy_entry, build_policy_field, load_policy, load_policy_data,
)


class PolicyPoolTests(unittest.TestCase):
    def test_loads_supported_formats(self):
        paths = [
            ROOT / "weights" / "rl_policy_weights.json",
            ROOT / "weights" / "rl_policy_checkpoint.json",
            ROOT / "weights" / "gru_policy_weights.json",
            ROOT / "weights" / "selfplay_gru_policy.json",
        ]
        for path in paths:
            policy = load_policy(str(path), seed=7, mode="sample")
            self.assertTrue(policy.name)
            policy.reset()
            self.assertIn(policy.act(0, [], []), (0, 1))

    def test_field_is_deterministic(self):
        paths = [
            str(ROOT / "weights" / "rl_policy_weights.json"),
            str(ROOT / "weights" / "selfplay_gru_policy.json"),
        ]
        a = build_policy_field(paths, seed=42, n=4)
        b = build_policy_field(paths, seed=42, n=4)
        self.assertEqual([x.name for x in a], [x.name for x in b])
        self.assertEqual([type(x).__name__ for x in a],
                         [type(x).__name__ for x in b])
        for x, y in zip(a, b):
            x.reset()
            y.reset()
            self.assertEqual(x.act(0, [], []), y.act(0, [], []))

    def test_empty_and_unknown(self):
        self.assertEqual(build_policy_field([], seed=1, n=3), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"format": "not-a-policy"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(str(path))
            with self.assertRaises(ValueError):
                load_policy_data(str(path))

    def test_field_uses_requested_count(self):
        paths = [str(ROOT / "weights" / "rl_policy_weights.json")]
        field = build_policy_field(paths, seed=1, n=5)
        self.assertEqual(len(field), 5)
        self.assertEqual(len({x.name for x in field}), 5)

    def test_weighted_sampling_is_deterministic(self):
        paths = [
            str(ROOT / "weights" / "rl_policy_weights.json"),
            str(ROOT / "weights" / "selfplay_gru_policy.json"),
        ]
        weights = [0.9, 0.1]
        a = build_policy_field(paths, seed=42, n=40, weights=weights)
        b = build_policy_field(paths, seed=42, n=40, weights=weights)
        self.assertEqual([x.name for x in a], [x.name for x in b])
        self.assertEqual([type(x).__name__ for x in a],
                         [type(x).__name__ for x in b])
        rl = sum(1 for x in a if x.name.startswith("rl_policy_weights:"))
        self.assertGreater(rl, 20)

    def test_zero_weight_entry_is_never_sampled(self):
        paths = [
            str(ROOT / "weights" / "rl_policy_weights.json"),
            str(ROOT / "weights" / "selfplay_gru_policy.json"),
        ]
        field = build_policy_field(paths, seed=5, n=12, weights=[1.0, 0.0])
        self.assertTrue(all(x.name.startswith("rl_policy_weights:")
                            for x in field))

    def test_weight_length_must_match(self):
        paths = [str(ROOT / "weights" / "rl_policy_weights.json")]
        with self.assertRaises(ValueError):
            build_policy_field(paths, seed=1, n=2, weights=[1.0, 2.0])

    def test_candidate_spec_entries(self):
        entries = [
            {"kind": "final-tuned", "name": "Final-tuned", "weight": 1.0},
            {"kind": "triple-majority", "name": "Triple-majority",
             "weight": 1.0},
        ]
        a = build_policy_field(entries, seed=7, n=6)
        b = build_policy_field(entries, seed=7, n=6)
        self.assertEqual([x.name for x in a], [x.name for x in b])
        rule = build_policy_entry(entries[0], seed=1)
        self.assertTrue(callable(getattr(rule, "act", None)))
        self.assertIn(rule.act(0, [], []), (0, 1))


if __name__ == "__main__":
    unittest.main()
