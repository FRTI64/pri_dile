"""Core simulation, opponent banks and distributions."""
from __future__ import annotations

import random
import unittest

try:
    from .helpers import ROOT
except ImportError:  # direct execution
    from helpers import ROOT  # noqa: F401  (sets sys.path)

from ipd.core import (
    TYPE_BANK, TYPE_BY_NAME, C, D, Grim, TFT, Machine, payoff, play_match,
)
from ipd.core.opponents.plus import PLUS
from ipd.core.features import FEATURES, feature_vector
from ipd.evaluation.distributions import (
    DISTS, FAMILY_MULT, build_dist_field, build_field, build_train_dist_field,
    sample_blind_field, sample_suite_field, sample_wide,
)


class SimulatorTests(unittest.TestCase):
    def test_payoff_matrix(self):
        self.assertEqual(payoff(C, C), 6)
        self.assertEqual(payoff(C, D), 0)
        self.assertEqual(payoff(D, C), 10)
        self.assertEqual(payoff(D, D), 1)

    def test_play_match_is_symmetric(self):
        result = play_match(TFT(), Grim())
        reverse = play_match(Grim(), TFT())
        self.assertEqual(result.score_a, reverse.score_b)
        self.assertEqual(result.score_b, reverse.score_a)

    def test_machine_boundaries(self):
        machine = Machine()
        machine.reset()
        mine: list[int] = []
        theirs: list[int] = []
        for t in range(100):
            mine.append(machine.move(t + 1, mine, theirs))
            theirs.append(C)
        self.assertEqual(mine[0], C)
        self.assertEqual(mine[-1], D)

    def test_opponent_bank(self):
        self.assertGreaterEqual(len(TYPE_BANK), 80)
        self.assertEqual(len(TYPE_BY_NAME), len(TYPE_BANK))
        for name in ("TFT", "Grim", "ProbeBot", "LateD10", "Extort3"):
            self.assertIn(name, TYPE_BY_NAME)

    def test_rule_feature_vector(self):
        self.assertEqual(len(feature_vector(1, [], [])), len(FEATURES))
        values = feature_vector(50, [C, D, C], [C, D, D])
        self.assertTrue(all(0.0 <= x <= 1.0 for x in values))


class DistributionTests(unittest.TestCase):
    def test_four_distributions_are_deterministic(self):
        for dist in DISTS:
            a = build_dist_field(dist, 12345, 20)
            b = build_dist_field(dist, 12345, 20)
            self.assertEqual([x.name for x in a], [x.name for x in b], dist)
            self.assertEqual(len(a), 20)

    def test_hard_training_distribution(self):
        field = build_train_dist_field("hard", 12345, 20)
        self.assertEqual(len(field), 20)
        self.assertTrue(all(x.name for x in field))

    def test_wide_and_blind_helpers(self):
        field, scenario, share = sample_wide(7, 30, 0.15)
        self.assertEqual(len(field), 30)
        self.assertIsInstance(scenario, str)
        self.assertAlmostEqual(share, 0.15)
        blind = sample_blind_field(7, 30)
        self.assertEqual([x.name for x in blind], [x.name for x in field])

    def test_suite_names(self):
        self.assertEqual(len(FAMILY_MULT), 9)
        for suite in FAMILY_MULT:
            field = sample_suite_field(suite, 3, 10)
            self.assertEqual(len(field), 10)

    def test_legacy_field_builder(self):
        for kind in ("original", "late", "grim30"):
            field, scenario = build_field(kind, 5, 12)
            self.assertEqual(len(field), 12)
            self.assertIsInstance(scenario, str)


class OpponentFactoryTests(unittest.TestCase):
    def test_reactive_factory_parameters_are_order_independent(self):
        specs = [spec for spec in PLUS
                 if spec.name.startswith("Reactive.F")]
        self.assertEqual(len(specs), 8)
        for spec in specs:
            with self.subTest(spec=spec.name):
                first = spec.factory(random.Random(1))
                second = spec.factory(random.Random(2))
                self.assertEqual(
                    (first.p_after_c, first.p_after_d),
                    (second.p_after_c, second.p_after_d))


if __name__ == "__main__":
    unittest.main()
