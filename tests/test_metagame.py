"""Meta-game utilities: Nash averaging, Elo and exploitability."""
from __future__ import annotations

import unittest

from ipd.evaluation import metagame as mg


RPS = [
    [0.0, -1.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 1.0, 0.0],
]


class MatrixTests(unittest.TestCase):
    def test_antisymmetrize(self):
        matrix = [[1.0, 2.0], [4.0, 5.0]]
        out = mg.antisymmetrize(matrix)
        self.assertAlmostEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[0][1], -1.0)
        self.assertAlmostEqual(out[1][0], 1.0)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(out[i][j], -out[j][i])

    def test_rps_meta_strategy(self):
        result = mg.fictitious_play(RPS, iterations=50000, seed=7)
        weights = result["weights"]
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertTrue(all(w >= 0.0 for w in weights))
        self.assertLess(max(weights) - min(weights), 0.05)
        self.assertLess(result["exploitability"], 0.05)

    def test_exploitability_of_pure_strategy(self):
        result = mg.exploitability(RPS, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(result["gap"], 1.0)
        self.assertEqual(result["best_response"], 1)

    def test_nash_average_deterministic(self):
        a = mg.nash_average(RPS, iterations=5000, seed=3)
        b = mg.nash_average(RPS, iterations=5000, seed=3)
        self.assertEqual(a["weights"], b["weights"])
        self.assertAlmostEqual(a["exploitability"], b["exploitability"])

    def test_elo_transitive_order(self):
        # expected score matrix: row beats column with the given probability
        scores = [
            [0.5, 0.9, 0.9],
            [0.1, 0.5, 0.9],
            [0.1, 0.1, 0.5],
        ]
        elo = mg.elo_ratings(scores, iterations=500)
        self.assertGreater(elo[0], elo[1])
        self.assertGreater(elo[1], elo[2])

    def test_elo_symmetric_is_equal(self):
        scores = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        elo = mg.elo_ratings(scores, iterations=200)
        self.assertLess(max(elo) - min(elo), 1e-6)

    def test_maximin_rps_is_uniform(self):
        result = mg.maximin_strategy(RPS, iterations=20000, seed=5)
        weights = result["weights"]
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertLess(max(weights) - min(weights), 0.05)
        self.assertAlmostEqual(result["worst_case"], 0.0, places=1)

    def test_maximin_dominated_row_gets_no_weight(self):
        matrix = [
            [1.0, 0.0],
            [0.0, 1.0],
            [-2.0, -2.0],
        ]
        result = mg.maximin_strategy(matrix, iterations=20000, seed=2)
        self.assertAlmostEqual(sum(result["weights"]), 1.0, places=6)
        self.assertLess(result["weights"][2], 1e-6)
        self.assertAlmostEqual(result["worst_case"], 0.5, places=2)

    def test_maximin_deterministic_and_rectangular(self):
        matrix = [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]
        a = mg.maximin_strategy(matrix, iterations=5000, seed=1)
        b = mg.maximin_strategy(matrix, iterations=5000, seed=1)
        self.assertEqual(a["weights"], b["weights"])
        self.assertEqual(a["best_column"], b["best_column"])
        self.assertEqual(len(a["weights"]), 2)


if __name__ == "__main__":
    unittest.main()
