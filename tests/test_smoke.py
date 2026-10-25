"""Fast structural and import smoke tests.

Run from the repository root::

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402
from adversarial_policy import AdversarialPolicy  # noqa: E402
from gru_policy import GRUPolicy  # noqa: E402
from ipd_sim import C, D, TFT  # noqa: E402
from ipd_strategy import Machine  # noqa: E402
from neural_policy import NeuralPolicy  # noqa: E402


def play(policy, opponent=TFT(), rounds: int = 100):
    """Play one match and return both action histories."""
    policy.reset()
    mine: list[int] = []
    theirs: list[int] = []
    opp = opponent
    for day in range(1, rounds + 1):
        a = policy.move(day, mine, theirs)
        if not isinstance(a, int) or a not in (C, D):
            raise AssertionError(f"invalid action {a!r} on day {day}")
        b = opp.act(day - 1, theirs, mine) if hasattr(opp, "act") else opp.move(
            day, theirs, mine
        )
        mine.append(a)
        theirs.append(b)
    return mine, theirs


class LayoutTests(unittest.TestCase):
    def test_top_level_directories(self):
        for name in ("src", "cpp", "configs", "weights", "results", "docs"):
            self.assertTrue((paths.ROOT / name).is_dir(), name)

    def test_path_helpers_are_absolute(self):
        for path in (
            paths.config_path("x.json"),
            paths.weights_path("x.json"),
            paths.result_path("x.json"),
            paths.cpp_path("x.cpp"),
        ):
            self.assertTrue(Path(path).is_absolute())
            self.assertTrue(path.startswith(str(paths.ROOT)))

    def test_required_artifacts(self):
        expected = [
            paths.config_path("adversarial_policy_config.json"),
            paths.config_path("adversarial_policy_config_aggressive.json"),
            paths.config_path("adversarial_gru_policy_config.json"),
            paths.config_path("adversarial_gru_policy_config_aggressive.json"),
            paths.weights_path("neural_policy_weights.json"),
            paths.weights_path("gru_policy_weights.json"),
            paths.result_path("train_multiobj_results.json"),
            paths.result_path("train_bayes_priors_results.json"),
            paths.result_path("train_joint_results.json"),
            paths.cpp_path("ipd_features.hpp"),
            paths.cpp_path("neural_policy_weights.hpp"),
            paths.cpp_path("gru_policy_weights.hpp"),
        ]
        for path in expected:
            self.assertTrue(Path(path).is_file(), path)


class PolicyTests(unittest.TestCase):
    def assert_boundaries(self, policy):
        mine, _ = play(policy)
        self.assertEqual(mine[0], C)
        self.assertEqual(mine[-1], D)

    def test_final_machine(self):
        self.assert_boundaries(Machine())

    def test_neural_policy(self):
        self.assert_boundaries(
            NeuralPolicy(paths.weights_path("neural_policy_weights.json"))
        )

    def test_gru_policy(self):
        self.assert_boundaries(
            GRUPolicy(paths.weights_path("gru_policy_weights.json"))
        )

    def test_adversarial_config_resolution(self):
        for config in (
            "adversarial_policy_config.json",
            "adversarial_policy_config_aggressive.json",
            "adversarial_gru_policy_config.json",
            "adversarial_gru_policy_config_aggressive.json",
        ):
            policy = AdversarialPolicy.from_config(paths.config_path(config))
            self.assert_boundaries(policy)


if __name__ == "__main__":
    unittest.main()
