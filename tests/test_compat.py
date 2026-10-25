"""Legacy ``src/*.py`` shims still expose the canonical objects."""
from __future__ import annotations

import importlib
import unittest

try:
    from .helpers import ROOT
except ImportError:  # direct execution
    from helpers import ROOT

import ipd.core.machine
import ipd.core.opponents
import ipd.core.simulator
import ipd.policies.adversarial
import ipd.policies.gru
import ipd.policies.neural
import ipd.policies.registry
import ipd.policies.rl
import ipd.policies.rules
import ipd.evaluation.distributions
import ipd.training.search.multiobj


class LegacyImportTests(unittest.TestCase):
    def test_old_modules_import(self):
        for path in sorted((ROOT / "src").glob("*.py")):
            importlib.import_module(path.stem)

    def test_core_reexports(self):
        import ipd_sim
        import ipd_field
        import ipd_strategy
        import neural_policy
        import gru_policy
        import rl_policy
        import adversarial_policy
        import triple_hybrid
        import search_common
        import train_multiobj
        import train_bayes_priors
        import train_joint_final_resp
        import train_wide
        import p2_skeleton_search

        self.assertIs(ipd_sim.TFT, ipd.core.simulator.TFT)
        self.assertIs(ipd_field.Responsive3, ipd.core.opponents.Responsive3)
        self.assertIs(ipd_strategy.Machine, ipd.core.machine.Machine)
        self.assertIs(neural_policy.NeuralPolicy,
                      ipd.policies.neural.NeuralPolicy)
        self.assertIs(gru_policy.GRUPolicy, ipd.policies.gru.GRUPolicy)
        self.assertIs(rl_policy.RLPolicy, ipd.policies.rl.RLPolicy)
        self.assertIs(adversarial_policy.AdversarialPolicy,
                      ipd.policies.adversarial.AdversarialPolicy)
        self.assertIs(triple_hybrid.Triple, ipd.policies.rules.Triple)
        self.assertIs(search_common.ParamPolicy,
                      ipd.policies.rules.ParamPolicy)
        self.assertIs(train_multiobj.ParamResp, ipd.policies.rules.ParamResp)
        self.assertIs(train_bayes_priors.ParamBayesPrior,
                      ipd.policies.rules.ParamBayesPrior)
        self.assertIs(train_joint_final_resp.Hybrid,
                      ipd.policies.rules.Hybrid)
        self.assertIs(train_wide.sample_wide,
                      ipd.evaluation.distributions.sample_wide)
        self.assertIs(p2_skeleton_search.GapStartPolicy,
                      ipd.policies.rules.GapStartPolicy)

    def test_legacy_paths(self):
        import paths
        import ipd.paths as canonical
        self.assertEqual(paths.ROOT, canonical.ROOT)
        self.assertEqual(paths.weights_path("x"), canonical.weights_path("x"))


if __name__ == "__main__":
    unittest.main()
