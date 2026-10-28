"""Unified policy interface and registry construction."""
from __future__ import annotations

import unittest

try:
    from .helpers import ROOT
except ImportError:  # direct execution
    from helpers import ROOT  # noqa: F401

import ipd.paths as paths
from ipd.core.simulator import C, D, TFT
from ipd.policies.base import PolicyAdapter, ensure_policy
from ipd.policies.neural import NeuralPolicy
from ipd.policies.registry import (
    ALL_NAMES, CORE_NAMES, DISPLAY, candidate_specs, canonical_name,
    legacy_arch_names, make_policy, resolve_names,
)
from ipd.policies.rl import RLPolicy
from ipd.policies.tuned import DEFAULT_FINAL_TUNED, load_tuned_configs


class MoveOnly:
    name = "MoveOnly"

    def __init__(self):
        self.reset()

    def reset(self):
        self.day = 0

    def move(self, day, mine, theirs):
        self.day = day
        return C if day < 100 else D


class ActOnly:
    name = "ActOnly"

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = 0

    def act(self, t, mine, theirs):
        self.t = t
        return C if t < 99 else D


class BadAction:
    name = "Bad"

    def reset(self):
        pass

    def act(self, t, mine, theirs):
        return 7


class PolicyInterfaceTests(unittest.TestCase):
    def test_move_only_is_normalised(self):
        policy = ensure_policy(MoveOnly())
        self.assertIsInstance(policy, PolicyAdapter)
        self.assertEqual(policy.act(0, [], []), C)
        self.assertEqual(policy.act(99, [], []), D)
        self.assertEqual(policy.move(100, [], []), D)

    def test_act_only_is_normalised(self):
        policy = ensure_policy(ActOnly())
        self.assertEqual(policy.act(0, [], []), C)
        self.assertEqual(policy.act(99, [], []), D)

    def test_invalid_action_is_rejected(self):
        policy = ensure_policy(BadAction())
        with self.assertRaises(ValueError):
            policy.act(0, [], [])

    def test_ensure_policy_is_idempotent(self):
        policy = ensure_policy(ActOnly())
        self.assertIs(ensure_policy(policy), policy)
        self.assertIs(ensure_policy(policy, name="renamed"), policy)
        self.assertEqual(policy.name, "renamed")


class RegistryTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(canonical_name("Final-tuned"), "final-tuned")
        self.assertEqual(canonical_name("resp3_tuned"), "resp3-tuned")
        self.assertEqual(canonical_name("Final+Resp3"), "hybrid")
        self.assertEqual(canonical_name("RL-PPO"), "rl")
        self.assertEqual(canonical_name("pattern"), "rl-pattern")
        self.assertEqual(canonical_name("rl-pb"), "rl-pattern")
        self.assertEqual(legacy_arch_names("rl"), ["rl"])
        with self.assertRaises(ValueError):
            canonical_name("does-not-exist")

    def test_sets_and_resolution(self):
        self.assertIn("final-tuned", CORE_NAMES)
        self.assertEqual(resolve_names("core,neural"),
                         [*CORE_NAMES, "neural"])
        self.assertEqual(len(resolve_names("all")), len(ALL_NAMES))
        specs = candidate_specs("core")
        self.assertEqual([s.name for s in specs],
                         [DISPLAY[x] for x in CORE_NAMES])

    def test_all_candidates_build_and_act(self):
        for name in ALL_NAMES:
            policy = make_policy(name)
            self.assertEqual(policy.name, DISPLAY[name])
            policy.reset()
            self.assertIn(policy.act(0, [], []), (C, D), name)

    def test_submission_boundaries(self):
        for name in ALL_NAMES:
            if name == "gap-start":
                continue  # experimental search skeleton, probes immediately
            policy = make_policy(name)
            policy.reset()
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(100):
                mine.append(policy.act(t, mine, theirs))
                theirs.append(C)
            self.assertEqual(mine[0], C, name)
            self.assertEqual(mine[-1], D, name)

    def test_tuned_defaults_are_available(self):
        tuned = load_tuned_configs()
        self.assertEqual(tuned.final["MIN_C"],
                         DEFAULT_FINAL_TUNED["MIN_C"])
        self.assertIn("coop_scale", tuned.bayes)

    def test_rl_export_matches_actor_critic(self):
        checkpoint = paths.weights_path("rl_policy_checkpoint.json")
        policy = RLPolicy(checkpoint, mode="argmax")
        exported = NeuralPolicy(policy.export_policy_weights(), mode="argmax")
        for mine, theirs in (([], []), ([C, C], [C, C]),
                             ([C, D, C], [C, D, D]), ([D] * 10, [D] * 10)):
            self.assertAlmostEqual(policy.predict_proba(mine, theirs),
                                   exported.predict_proba(mine, theirs),
                                   places=7)

    def test_stateful_gru_reset(self):
        policy = make_policy("gru")
        policy.reset()
        mine: list[int] = []
        theirs: list[int] = []
        for t in range(5):
            mine.append(policy.act(t, mine, theirs))
            theirs.append(TFT().act(t, theirs, mine) if t else C)
        first = list(policy.hidden_state[0])
        policy.reset()
        self.assertNotEqual(first, policy.hidden_state[0])


if __name__ == "__main__":
    unittest.main()
