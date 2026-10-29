"""Domain-randomized training distribution for robust A60 fine-tuning."""
from __future__ import annotations

import random
import unittest

from ipd.evaluation.distributions import TRAIN_DISTS, build_train_dist_field
from ipd.evaluation.extort import (
    EXTORT_HARD_OPPONENTS, build_extort_hard_field,
)
from ipd.evaluation.robust import (
    REPAIR_OPPONENTS, ROBUST_OPPONENTS, build_repair_field,
    build_robust_field,
)


class RobustDistributionTests(unittest.TestCase):
    def test_registered_as_training_distribution(self):
        self.assertIn("robust", TRAIN_DISTS)
        field = build_train_dist_field("robust", 12345, 20)
        self.assertEqual(len(field), 20)

    def test_field_is_deterministic(self):
        a = build_robust_field(999, 40)
        b = build_robust_field(999, 40)
        self.assertEqual([type(x).__name__ for x in a],
                         [type(x).__name__ for x in b])
        self.assertEqual([getattr(x, "name", "") for x in a],
                         [getattr(x, "name", "") for x in b])

    def test_parameter_ranges_are_domain_randomized(self):
        rng = random.Random(20261026)
        bots = [factory(rng) for _name, _weight, factory in ROBUST_OPPONENTS
                for _ in range(25)]
        seen = set()
        for bot in bots:
            cls = type(bot).__name__
            seen.add(cls)
            if cls == "ExtortZD":
                self.assertGreaterEqual(bot.chi, 1.5)
                self.assertLessEqual(bot.chi, 3.8)
            elif cls == "RandomMoves":
                self.assertGreaterEqual(bot.p, 0.01)
                self.assertLessEqual(bot.p, 0.95)
            elif cls == "LateDefector":
                self.assertGreaterEqual(bot.k, 20)
                self.assertLessEqual(bot.k, 95)
            elif cls == "NoisyTFT":
                self.assertGreaterEqual(bot.p, 0.01)
                self.assertLessEqual(bot.p, 0.5)
            elif cls == "Joss":
                self.assertGreaterEqual(bot.p, 0.01)
                self.assertLessEqual(bot.p, 0.5)
            elif cls == "DynamicWindow":
                self.assertGreaterEqual(bot.window, 3)
                self.assertLessEqual(bot.window, 10)
            elif cls == "Kamikaze":
                self.assertGreaterEqual(bot.period, 8)
                self.assertLessEqual(bot.period, 30)
        self.assertTrue({"ExtortZD", "RandomMoves", "LateDefector",
                         "NoisyTFT", "Joss", "DynamicWindow",
                         "Kamikaze"} <= seen)

    def test_anchors_keep_cooperative_and_defecting_types(self):
        names = {name for name, _weight, _factory in ROBUST_OPPONENTS}
        self.assertIn("TFT", names)
        self.assertIn("Grim", names)
        self.assertIn("AllD", names)
        self.assertIn("AllC", names)

    def test_repair_distribution_is_registered_and_balanced(self):
        self.assertIn("repair", TRAIN_DISTS)
        field = build_train_dist_field("repair", 54321, 25)
        self.assertEqual(len(field), 25)
        weights = {name: weight for name, weight, _ in REPAIR_OPPONENTS}
        self.assertGreater(weights["Random"], weights["Extort"])
        for anchor in ("TFT", "GTFT", "Pavlov", "AllC", "Grim", "AllD"):
            self.assertIn(anchor, weights)

    def test_repair_field_is_deterministic(self):
        a = build_repair_field(777, 40)
        b = build_repair_field(777, 40)
        self.assertEqual([type(x).__name__ for x in a],
                         [type(x).__name__ for x in b])
        self.assertEqual([getattr(x, "name", "") for x in a],
                         [getattr(x, "name", "") for x in b])


class ExtortHardTests(unittest.TestCase):
    def test_registered_and_deterministic(self) -> None:
        self.assertIn("extort_hard", TRAIN_DISTS)
        a = build_extort_hard_field(20261036, 80)
        b = build_extort_hard_field(20261036, 80)
        self.assertEqual(len(a), 80)
        self.assertEqual([x.name for x in a], [x.name for x in b])

    def test_includes_strong_extortion_and_anchors(self) -> None:
        names = {name for name, _weight, _factory in EXTORT_HARD_OPPONENTS}
        self.assertIn("ExtortStrong", names)
        self.assertIn("TFT", names)
        self.assertIn("Grim", names)
        self.assertIn("AllD", names)
        field = build_extort_hard_field(7, 200)
        kinds = {type(x).__name__ for x in field}
        self.assertIn("ExtortZD", kinds)
        self.assertTrue(any(getattr(x, "chi", 0.0) >= 4.5
                            for x in field if type(x).__name__ == "ExtortZD"))


if __name__ == "__main__":
    unittest.main()
