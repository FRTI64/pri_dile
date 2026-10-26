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
    build_policy_field, load_policy, load_policy_data,
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


if __name__ == "__main__":
    unittest.main()
