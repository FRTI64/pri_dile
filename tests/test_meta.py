"""Tests for the narrow RL-Switch meta policy."""
from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
