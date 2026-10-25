"""Candidate files are independent, and match the historical implementations."""
from __future__ import annotations

import ast
import importlib
import random
import unittest
from pathlib import Path

try:
    from .helpers import ROOT
except ImportError:  # direct execution
    from helpers import ROOT  # noqa: F401

from ipd.core.opponents import LateDefector, ProbeBot
from ipd.core.simulator import AllD, Grim, RandomMoves, TFT
from ipd.policies.registry import make_policy
from ipd.policies.rules import (
    GapStartPolicy, Hybrid, ParamBayesPrior, ParamHarvest, ParamPolicy,
    ParamResp, Triple,
)
from ipd.policies.tuned import (
    DEFAULT_BAYES_BASE, load_tuned_configs,
)

CANDIDATE_DIR = ROOT / "ipd" / "policies" / "candidates"


def _opponents():
    return {
        "tft": lambda: TFT(),
        "grim": lambda: Grim(),
        "alld": lambda: AllD(),
        "rand": lambda: RandomMoves(0.3, random.Random(5)),
        "late": lambda: LateDefector(25),
        "probe": lambda: ProbeBot(rng=random.Random(7)),
    }


def _actions(policy, opponent) -> list[int]:
    policy.reset()
    opponent.reset()
    mine: list[int] = []
    theirs: list[int] = []
    out: list[int] = []
    for t in range(100):
        x = policy.act(t, mine, theirs)
        y = opponent.act(t, theirs, mine)
        mine.append(x)
        theirs.append(y)
        out.append(x)
    return out


class CandidateIndependenceTests(unittest.TestCase):
    def test_no_candidate_imports_another_candidate(self):
        for path in sorted(CANDIDATE_DIR.glob("*.py")):
            if path.name == "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), path.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    # level 1 is a sibling import inside candidates/.
                    self.assertNotEqual(
                        node.level, 1,
                        f"{path.name} imports a sibling candidate: {node.module}")
                    module = node.module or ""
                    self.assertNotIn("ipd.policies.candidates", module,
                                     f"{path.name} imports candidates package")
                    self.assertNotIn("ipd.policies.rules", module,
                                     f"{path.name} reuses the old rule classes")
                    self.assertNotIn("ipd.policies.adversarial", module,
                                     f"{path.name} reuses the old adv wrapper")
                    self.assertNotEqual(
                        module, "rules",
                        f"{path.name} reuses the old rule classes")
                    self.assertFalse(
                        module.startswith("candidates"),
                        f"{path.name} imports the candidates package")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn("ipd.policies.rules", alias.name)
                        self.assertNotIn("ipd.policies.adversarial", alias.name)

    def test_registry_loads_one_module_per_candidate(self):
        modules = importlib.import_module("ipd.policies.candidates").MODULES
        self.assertEqual(len(set(modules.values())), len(modules))
        for kind, module_name in modules.items():
            module = importlib.import_module(
                f"ipd.policies.candidates.{module_name}")
            self.assertEqual(module.KIND, kind)
            self.assertTrue(module.DISPLAY)


class CandidateEquivalenceTests(unittest.TestCase):
    def test_rule_candidates_match_historical_classes(self):
        tuned = load_tuned_configs()
        refs = {
            "final-base": lambda: ParamPolicy(seed=20261024),
            "final-tuned": lambda: ParamPolicy(seed=20261024, **tuned.final),
            "resp3-base": lambda: ParamResp(seed=20261024),
            "resp3-tuned": lambda: ParamResp(seed=20261024, **tuned.resp),
            "bayes-base": lambda: ParamBayesPrior(DEFAULT_BAYES_BASE),
            "bayes-tuned": lambda: ParamBayesPrior(tuned.bayes),
            "harvest45": lambda: ParamHarvest(
                seed=20261024, HARVEST_START=45),
            "harvest-tuned": lambda: ParamHarvest(
                seed=20261024, **tuned.harvest),
            "hybrid": lambda: Hybrid(
                tuned.hybrid.get("final", tuned.final),
                tuned.hybrid.get("resp", {}),
                tuned.hybrid.get("p", 0.0),
            ),
            "triple-majority": lambda: Triple(
                tuned.final, tuned.resp, tuned.bayes, mode="majority"),
            "gap-start": lambda: GapStartPolicy(1, 0.5),
        }
        for name, factory in refs.items():
            for opp_kind, opp_factory in _opponents().items():
                expected = _actions(factory(), opp_factory())
                actual = _actions(make_policy(name), opp_factory())
                self.assertEqual(actual, expected, f"{name} vs {opp_kind}")

    def test_triple_modes_match_historical_classes(self):
        tuned = load_tuned_configs()
        for mode in ("or", "and", "majority"):
            expected = _actions(
                Triple(tuned.final, tuned.resp, tuned.bayes, mode=mode),
                TFT())
            actual = _actions(make_policy(f"triple-{mode}"), TFT())
            self.assertEqual(actual, expected, mode)

    def test_adv_candidates_match_historical_wrapper(self):
        from ipd.paths import config_path
        from ipd.policies.adversarial import AdversarialPolicy

        configs = {
            "adv-mlp": "adversarial_policy_config.json",
            "adv-gru": "adversarial_gru_policy_config.json",
            "adv-rl": "adversarial_rl_policy_config_aggressive.json",
        }
        for name, config_name in configs.items():
            config = config_path(config_name)
            for opp_kind, opp_factory in _opponents().items():
                expected = _actions(
                    AdversarialPolicy.from_config(
                        config, seed=20261024, mode="sample"),
                    opp_factory())
                actual = _actions(
                    make_policy(name, config=config, seed=20261024,
                                mode="sample"),
                    opp_factory())
                self.assertEqual(actual, expected, f"{name} vs {opp_kind}")


class CandidateParameterTests(unittest.TestCase):
    def test_embedded_tuned_values_match_tuned_configs(self):
        tuned = load_tuned_configs()
        final = importlib.import_module("ipd.policies.candidates.final_tuned")
        for key, value in tuned.final.items():
            self.assertAlmostEqual(getattr(final, key), value, places=12, msg=key)

        resp = importlib.import_module("ipd.policies.candidates.resp3_tuned")
        resp_keys = {
            "G_UNPROV": "g_unprov", "G_MUTUAL": "g_mutual",
            "G_RESP": "g_resp", "MIN_C": "min_c", "MIN_D": "min_d",
            "RESP": "resp", "RATE": "rate", "MAX_TESTS": "max_tests",
            "COOLDOWN": "cooldown",
        }
        for attr, key in resp_keys.items():
            self.assertAlmostEqual(getattr(resp, attr), tuned.resp[key],
                                   places=12, msg=key)

        bayes = importlib.import_module("ipd.policies.candidates.bayes_tuned")
        bayes_keys = {
            "COOP_SCALE": "coop_scale", "PUNITIVE_SCALE": "punitive_scale",
            "LATE_SCALE": "late_scale", "NOISE_SCALE": "noise_scale",
            "PROBE_SCALE": "probe_scale", "NONRESP_THRESHOLD": "nonresp",
            "PUNITIVE_THRESHOLD": "punitive_thr",
        }
        for attr, key in bayes_keys.items():
            self.assertAlmostEqual(getattr(bayes, attr), tuned.bayes[key],
                                   places=12, msg=key)

        harvest = importlib.import_module(
            "ipd.policies.candidates.harvest_tuned")
        for key, value in tuned.harvest.items():
            self.assertAlmostEqual(getattr(harvest, key), value,
                                   places=12, msg=key)

        hybrid = importlib.import_module("ipd.policies.candidates.hybrid")
        for key, value in tuned.hybrid["final"].items():
            self.assertAlmostEqual(hybrid.FINAL_TUNED[key], value,
                                   places=12, msg=key)

        triple = importlib.import_module(
            "ipd.policies.candidates.triple_majority")
        for key, value in tuned.final.items():
            self.assertAlmostEqual(triple.FINAL_TUNED[key], value,
                                   places=12, msg=key)
        for key, value in tuned.resp.items():
            self.assertAlmostEqual(triple.RESP_TUNED[key], value,
                                   places=12, msg=key)
        for key, value in tuned.bayes.items():
            self.assertAlmostEqual(triple.BAYES_TUNED[key], value,
                                   places=12, msg=key)


if __name__ == "__main__":
    unittest.main()
