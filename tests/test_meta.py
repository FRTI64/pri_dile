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


class ProbeGuardTests(unittest.TestCase):
    def test_early_unprovoked_defection_triggers(self) -> None:
        policy = make_policy("rl-probe-guard", mode="argmax")
        policy.reset()
        mine = [0, 0, 0, 0, 0, 0]
        theirs = [0, 0, 0, 0, 0, 1]
        policy.act(6, mine, theirs)  # first observed D
        mine.append(0)
        theirs.append(1)
        self.assertEqual(policy.act(7, mine, theirs), 0)
        self.assertEqual(policy.state, "observe")

    def test_late_first_defection_does_not_trigger(self) -> None:
        policy = make_policy("rl-probe-guard", mode="argmax")
        policy.reset()
        mine = [0] * 9
        theirs = [0] * 8 + [1]
        policy.act(9, mine, theirs)
        mine.append(0)
        theirs.append(1)
        policy.act(10, mine, theirs)
        self.assertEqual(policy.state, "base")

    def test_defection_before_opponent_does_not_trigger(self) -> None:
        policy = make_policy("rl-probe-guard", mode="argmax")
        policy.reset()
        mine = [0, 0, 0, 0, 1, 0]
        theirs = [0, 0, 0, 0, 0, 1]
        policy.act(6, mine, theirs)
        mine.append(0)
        theirs.append(1)
        policy.act(7, mine, theirs)
        self.assertEqual(policy.state, "base")

    def test_cooperative_reply_releases(self) -> None:
        policy = make_policy("rl-probe-guard", mode="argmax")
        policy.reset()
        mine = [0, 0, 0, 0, 0, 0, 0]
        theirs = [0, 0, 0, 0, 0, 1, 1]
        policy.act(7, mine, theirs)
        self.assertEqual(policy.state, "observe")
        mine.append(0)
        theirs.append(0)
        policy.act(8, mine, theirs)
        self.assertEqual(policy.state, "base")

    def test_continued_defection_punishes(self) -> None:
        policy = make_policy("rl-probe-guard", mode="argmax")
        policy.reset()
        mine = [0, 0, 0, 0, 0, 0, 0]
        theirs = [0, 0, 0, 0, 0, 1, 1]
        policy.act(7, mine, theirs)
        for _ in range(3):
            mine.append(0)
            theirs.append(1)
            policy.act(len(theirs), mine, theirs)
        self.assertEqual(policy.state, "punish")
        mine.append(0)
        theirs.append(1)
        self.assertEqual(policy.act(len(theirs), mine, theirs), 1)


class DefectionGuardTests(unittest.TestCase):
    def test_late_first_defection_triggers(self) -> None:
        policy = make_policy("rl-defection-guard", mode="argmax")
        policy.reset()
        n = policy.late_confirm
        mine = [0] * (10 + n)
        theirs = [0] * 10 + [1] * n
        self.assertEqual(policy.act(10 + n, mine, theirs), 1)
        self.assertEqual(policy.state, "observe")

    def test_early_probe_is_not_handled(self) -> None:
        policy = make_policy("rl-defection-guard", mode="argmax")
        policy.reset()
        mine = [0] * 6
        theirs = [0] * 5 + [1]
        policy.act(6, mine, theirs)
        mine.append(0)
        theirs.append(1)
        policy.act(7, mine, theirs)
        self.assertEqual(policy.state, "base")


class FullGuardTests(unittest.TestCase):
    def test_probe_window_triggers(self) -> None:
        policy = make_policy("rl-full-guard", mode="argmax")
        policy.reset()
        mine = [0, 0, 0, 0, 0, 0]
        theirs = [0, 0, 0, 0, 0, 1]
        policy.act(6, mine, theirs)
        mine.append(0)
        theirs.append(1)
        policy.act(7, mine, theirs)
        self.assertEqual(policy.state, "base")
        mine.append(0)
        theirs.append(1)
        policy.act(8, mine, theirs)
        self.assertEqual(policy.state, "observe")

    def test_late_window_triggers(self) -> None:
        policy = make_policy("rl-full-guard", mode="argmax")
        policy.reset()
        n = policy.late_confirm
        mine = [0] * (10 + n)
        theirs = [0] * 10 + [1] * n
        policy.act(10 + n, mine, theirs)
        self.assertEqual(policy.state, "observe")

    def test_mid_window_does_not_trigger(self) -> None:
        policy = make_policy("rl-full-guard", mode="argmax")
        policy.reset()
        mine = [0] * 9
        theirs = [0] * 8 + [1]
        policy.act(9, mine, theirs)
        mine.append(0)
        theirs.append(1)
        policy.act(10, mine, theirs)
        self.assertEqual(policy.state, "base")


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
