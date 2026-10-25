"""Directory layout, path helpers and required submission artifacts."""
from __future__ import annotations

import unittest
from pathlib import Path

try:
    from .helpers import ROOT
except ImportError:  # direct execution
    from helpers import ROOT

import ipd.paths as paths


class LayoutTests(unittest.TestCase):
    def test_top_level_directories(self):
        for name in ("ipd", "src", "cpp", "configs", "weights", "results",
                     "docs", "tests"):
            self.assertTrue((paths.ROOT / name).is_dir(), name)
        self.assertEqual(paths.ROOT, ROOT)

    def test_package_layers_exist(self):
        package = paths.ROOT / "ipd"
        for name in ("core", "policies", "evaluation", "training", "tools"):
            self.assertTrue((package / name / "__init__.py").is_file(), name)

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
            paths.config_path("adversarial_rl_policy_config.json"),
            paths.config_path("adversarial_rl_policy_config_aggressive.json"),
            paths.weights_path("neural_policy_weights.json"),
            paths.weights_path("gru_policy_weights.json"),
            paths.weights_path("rl_policy_checkpoint.json"),
            paths.weights_path("rl_policy_weights.json"),
            paths.weights_path("rl_policy_checkpoint_bc.json"),
            paths.weights_path("rl_policy_weights_bc.json"),
            paths.result_path("train_multiobj_results.json"),
            paths.result_path("train_bayes_priors_results.json"),
            paths.result_path("train_joint_results.json"),
            paths.cpp_path("ipd_features.hpp"),
            paths.cpp_path("neural_policy.hpp"),
            paths.cpp_path("neural_policy.cpp"),
            paths.cpp_path("adversarial_rl.cpp"),
        ]
        for path in expected:
            self.assertTrue(Path(path).is_file(), path)

    def test_pyproject_declares_package(self):
        text = (paths.ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "ipd-tournament"', text)
        self.assertIn('ipd = "ipd.cli:main"', text)


if __name__ == "__main__":
    unittest.main()
