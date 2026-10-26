"""P2 C1: Bayesian belief planner offline contract and behaviour."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from ipd.core.opponents.basic import AllC, AllD
from ipd.core.primitives import C, D, payoff
from ipd.evaluation.planner_eval import compare, evaluate_policy
from ipd.policies.belief_planner import BeliefPlanner, CURATED_NAMES


def _run(planner, opponent, rounds=100):
    planner.reset()
    opponent.reset()
    mine: list[int] = []
    theirs: list[int] = []
    score = 0
    for t in range(rounds):
        action = planner.act(t, mine, theirs)
        reply = opponent.act(t, theirs, mine)
        mine.append(action)
        theirs.append(reply)
        score += payoff(action, reply)
    return score, mine, theirs


class PlannerContractTests(unittest.TestCase):
    def test_boundaries_and_reset(self):
        planner = BeliefPlanner(AllC(), horizon=2, top_k=3,
                                max_hypotheses=6, seed=1)
        self.assertEqual(planner.act(0, [], []), C)
        mine = [C] * 99
        theirs = [C] * 99
        self.assertEqual(planner.act(99, mine, theirs), D)
        planner.reset()
        self.assertEqual(planner.act(0, [], []), C)
        self.assertGreater(len(planner.particles), 0)

    def test_fallback_confidence_returns_base(self):
        base = AllC()
        planner = BeliefPlanner(base, horizon=2, top_k=3,
                                max_hypotheses=6,
                                fallback_confidence=1.01, seed=2)
        score, actions, _ = _run(planner, AllD(), rounds=99)
        self.assertEqual(score, 0)  # AllC vs AllD
        self.assertTrue(all(action == C for action in actions))
        self.assertGreater(planner.stats.fallback_low_confidence, 0)

    def test_belief_concentrates_on_all_d(self):
        planner = BeliefPlanner(AllC(), horizon=2, top_k=3,
                                max_hypotheses=20,
                                fallback_confidence=0.5, seed=3)
        _run(planner, AllD(), rounds=30)
        self.assertGreater(planner.predicted_action_probability(), 0.75)
        diagnostics = planner.diagnostics()
        self.assertEqual(diagnostics["rounds"], 30)
        self.assertAlmostEqual(
            sum(p.weight for p in planner.particles), 1.0, places=9)

    def test_planner_exploits_all_d(self):
        # AllC scores 0 against AllD; a confident planner should defect.
        planner = BeliefPlanner(AllC(), horizon=3, top_k=6,
                                max_hypotheses=24,
                                fallback_confidence=0.6, seed=4)
        score, actions, _ = _run(planner, AllD(), rounds=100)
        self.assertGreater(score, 50)
        self.assertIn(D, actions)
        self.assertGreater(planner.stats.overrides, 0)

    def test_deterministic(self):
        a = BeliefPlanner(AllC(), horizon=3, top_k=5,
                          max_hypotheses=20, seed=5)
        b = BeliefPlanner(AllC(), horizon=3, top_k=5,
                          max_hypotheses=20, seed=5)
        sa, aa, _ = _run(a, AllD(), rounds=40)
        sb, ab, _ = _run(b, AllD(), rounds=40)
        self.assertEqual(sa, sb)
        self.assertEqual(aa, ab)
        self.assertEqual(a.diagnostics()["overrides"],
                         b.diagnostics()["overrides"])


class ModelLibraryTests(unittest.TestCase):
    def test_curated_library_covers_key_families(self):
        self.assertIn("AllD", CURATED_NAMES)
        self.assertIn("TFT", CURATED_NAMES)
        self.assertIn("Grim", CURATED_NAMES)
        self.assertIn("Extort2", CURATED_NAMES)
        self.assertIn("Random.50", CURATED_NAMES)
        self.assertGreaterEqual(len(set(CURATED_NAMES)), 30)


class PlannerEvalTests(unittest.TestCase):
    def test_compare_and_evaluate(self):
        item = SimpleNamespace(
            name="AllC", family="nice", factory=lambda rng: AllC(rng))
        base = evaluate_policy(AllC(), [item], [1], rounds=10)
        planner = BeliefPlanner(
            AllC(), horizon=2, top_k=3, max_hypotheses=6,
            fallback_confidence=1.01, seed=7)
        result = evaluate_policy(
            planner, [item], [1], rounds=10, collect_diagnostics=True)
        compare(result, base)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["mean"], 60.0)
        self.assertAlmostEqual(result["min_norm"], 1.0)
        self.assertAlmostEqual(result["mean_delta"], 0.0)
        self.assertEqual(result["diagnostics"]["matches"], 1)


if __name__ == "__main__":
    unittest.main()
