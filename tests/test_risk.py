"""Risk-sensitive PPO helpers: CVaR episode weights and Dirichlet mixtures."""
from __future__ import annotations

import random
import unittest

from ipd.training.risk import (
    competitive_reward, cvar_episode_weights, dirichlet_weights,
    dro_update_weights,
)


class CVaRWeightTests(unittest.TestCase):
    def test_worst_quantile_gets_extra_weight(self):
        returns = [float(100 * i) for i in range(1, 11)]
        weights = cvar_episode_weights(returns, quantile=0.2, coef=1.0)
        self.assertEqual(len(weights), len(returns))
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0, places=9)
        boosted = [i for i, w in enumerate(weights) if w > 1.0]
        self.assertEqual(boosted, [0, 1])
        self.assertLess(weights[-1], weights[0])

    def test_zero_coef_is_uniform(self):
        weights = cvar_episode_weights([1, 2, 3, 4], quantile=0.25, coef=0.0)
        self.assertTrue(all(abs(w - 1.0) < 1e-12 for w in weights))

    def test_equal_returns_are_uniform(self):
        weights = cvar_episode_weights([500.0] * 8, quantile=0.2, coef=3.0)
        self.assertTrue(all(abs(w - 1.0) < 1e-12 for w in weights))

    def test_empty_and_single(self):
        self.assertEqual(cvar_episode_weights([], 0.2, 1.0), [])
        self.assertEqual(cvar_episode_weights([123.0], 0.2, 1.0), [1.0])


class DirichletTests(unittest.TestCase):
    def test_weights_sum_to_one_and_are_deterministic(self):
        base = {"a": 1.0, "b": 1.0, "c": 2.0}
        a = dirichlet_weights(base, random.Random(7), concentration=20.0)
        b = dirichlet_weights(base, random.Random(7), concentration=20.0)
        self.assertEqual(set(a), set(base))
        self.assertAlmostEqual(sum(a.values()), 1.0, places=9)
        self.assertEqual(a, b)
        self.assertTrue(all(v > 0 for v in a.values()))

    def test_high_concentration_is_near_base(self):
        base = {"a": 1.0, "b": 1.0, "c": 2.0}
        weights = dirichlet_weights(base, random.Random(3), concentration=5000.0)
        self.assertAlmostEqual(weights["c"], 0.5, delta=0.03)
        self.assertAlmostEqual(weights["a"], 0.25, delta=0.03)

    def test_zero_concentration_returns_base(self):
        base = {"a": 1.0, "b": 3.0}
        weights = dirichlet_weights(base, random.Random(1), concentration=0.0)
        self.assertAlmostEqual(weights["a"], 0.25)
        self.assertAlmostEqual(weights["b"], 0.75)


class CompetitiveRewardTests(unittest.TestCase):
    def test_zero_penalty_is_own_payoff(self):
        self.assertEqual(competitive_reward(6, 6, 0.0), 6)
        self.assertEqual(competitive_reward(10, 0, 0.0), 10)

    def test_penalty_subtracts_opponent_payoff(self):
        self.assertEqual(competitive_reward(6, 6, 1.0), 0)
        self.assertEqual(competitive_reward(10, 0, 1.0), 10)
        self.assertEqual(competitive_reward(0, 10, 1.0), -10)
        self.assertEqual(competitive_reward(0, 10, 0.5), -5)


class DROWeightTests(unittest.TestCase):
    def test_upweights_worst_distribution(self):
        base = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        means = {"a": 600.0, "b": 500.0, "c": 550.0, "d": 580.0}
        weights = dro_update_weights(base, means, eta=1.0, quantile=0.25,
                                     blend=0.5, floor=0.001)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        self.assertGreater(weights["b"], weights["a"])
        self.assertGreater(weights["b"], weights["c"])
        self.assertLess(weights["a"], weights["d"])

    def test_floor_prevents_zero_weight(self):
        base = {"a": 1.0, "b": 1.0, "c": 1.0}
        means = {"a": 1000.0, "b": 900.0, "c": 0.0}
        weights = dro_update_weights(base, means, eta=20.0, quantile=1.0,
                                     blend=1.0, floor=0.2)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        self.assertTrue(all(v >= 0.2 - 1e-12 for v in weights.values()))

    def test_zero_update_is_base(self):
        base = {"a": 1.0, "b": 3.0}
        means = {"a": 600.0, "b": 500.0}
        weights = dro_update_weights(base, means, eta=0.0, quantile=0.5,
                                     blend=0.0)
        self.assertAlmostEqual(weights["a"], 0.25)
        self.assertAlmostEqual(weights["b"], 0.75)

    def test_empty_and_single(self):
        self.assertEqual(dro_update_weights({}, {}), {})
        self.assertEqual(dro_update_weights({"a": 2.0}, {"a": 1.0}), {"a": 1.0})


if __name__ == "__main__":
    unittest.main()
