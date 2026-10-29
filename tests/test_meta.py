"""Tests for the narrow RL-Switch meta policy."""
from __future__ import annotations

import unittest

from ipd.policies.meta import GATE_FEATURES, gate_features
from ipd.policies.registry import make_policy


class RLSwitchTests(unittest.TestCase):
    def test_switch_requires_early_mutual_defection(self) -> None:
        policy = make_policy("rl-switch", mode="argmax")
        policy.reset()
        mine = [1, 1, 0, 0, 0, 0]
        theirs = [1, 1, 0, 0, 0, 0]
        policy.act(6, mine, theirs)
        self.assertTrue(policy.switched)

    def test_cooperative_start_does_not_switch(self) -> None:
        policy = make_policy("rl-switch", mode="argmax")
        policy.reset()
        mine = [0, 0, 0, 0, 0, 0]
        theirs = [0, 0, 0, 0, 0, 0]
        policy.act(6, mine, theirs)
        self.assertFalse(policy.switched)

    def test_reset_clears_switch(self) -> None:
        policy = make_policy("rl-switch", mode="argmax")
        policy.reset()
        policy.act(6, [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0])
        self.assertTrue(policy.switched)
        policy.reset()
        self.assertFalse(policy.switched)


class LearnedGateTests(unittest.TestCase):
    def test_gate_features_shape(self) -> None:
        x = gate_features([0, 1, 0, 0], [0, 1, 1, 0])
        self.assertEqual(len(x), GATE_FEATURES)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in x[12:14]))

    def test_candidate_loads_and_decides(self) -> None:
        policy = make_policy("rl-gate", mode="argmax")
        policy.reset()
        mine = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        theirs = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        policy.act(10, mine, theirs)
        self.assertTrue(policy.decided)


if __name__ == "__main__":
    unittest.main()
