"""Optional Axelrod-library opponent bank integration."""
from __future__ import annotations

import importlib.util
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path

from ipd.core.opponents.axelrod_bank import (
    axelrod_by_name, axelrod_cheating_by_name, axelrod_cheating_specs,
    axelrod_specs, sample_axelrod_cheating_field, sample_axelrod_field,
)
from ipd.core.simulator import C, D, TFT
from ipd.evaluation.benchmark import main as benchmark_main
from ipd.evaluation.common import run_match
from ipd.evaluation.distributions import build_dist_field
from ipd.evaluation.matchup import type_task
from ipd.policies.registry import candidate_specs

HAS_AXELROD = importlib.util.find_spec("axelrod") is not None


@unittest.skipUnless(HAS_AXELROD, "optional axelrod package not installed")
class AxelrodBankTests(unittest.TestCase):
    def test_bank_sizes_and_names(self):
        all_specs = axelrod_specs(include_long_run=True)
        fast_specs = axelrod_specs(include_long_run=False)
        self.assertEqual(len(all_specs), 243)
        self.assertEqual(len(fast_specs), 224)
        all_names = {spec.name for spec in all_specs}
        fast_names = {spec.name for spec in fast_specs}
        self.assertEqual(len(all_names), 243)
        self.assertTrue(fast_names < all_names)
        self.assertNotIn("Darwin", all_names)  # cheating strategy
        self.assertIn("TitForTat", fast_names)
        self.assertIn("MetaMajority", all_names - fast_names)

        cheating = axelrod_cheating_specs()
        self.assertEqual([spec.name for spec in cheating], ["Darwin"])
        self.assertTrue(all(spec.family == "axelrod-cheating"
                            for spec in cheating))

    def test_specs_are_picklable(self):
        specs = axelrod_specs(include_long_run=False)[:3]
        restored = pickle.loads(pickle.dumps(specs))
        self.assertEqual([spec.name for spec in restored],
                         [spec.name for spec in specs])

    def test_adapter_play_and_reset(self):
        spec = axelrod_by_name(include_long_run=True)["TitForTat"]
        opponent = spec.factory(random.Random(1))
        first = run_match(TFT(), opponent, rounds=20)
        self.assertEqual(len(first.mine), 20)
        self.assertEqual(first.opponent_name, "TitForTat")
        self.assertEqual(first.theirs, [C] * 20)

        # reset() must restart the wrapped player with the same seed.
        second = run_match(TFT(), opponent, rounds=20)
        self.assertEqual(second.mine, first.mine)
        self.assertEqual(second.theirs, first.theirs)

    def test_stochastic_adapter_is_reproducible(self):
        spec = axelrod_by_name(include_long_run=True)["Random"]
        opponent = spec.factory(random.Random(7))
        first = run_match(TFT(), opponent, rounds=50)
        second = run_match(TFT(), opponent, rounds=50)
        self.assertEqual(first.theirs, second.theirs)
        self.assertTrue(set(first.theirs) <= {C, D})

    def test_cheating_adapter_plays(self):
        import axelrod as axl

        axl.Darwin.reset_genome()
        spec = axelrod_cheating_by_name()["Darwin"]
        opponent = spec.factory(random.Random(3))
        result = run_match(TFT(), opponent, rounds=50)
        self.assertEqual(result.opponent_name, "Darwin")
        self.assertEqual(len(result.theirs), 50)
        self.assertTrue(set(result.theirs) <= {C, D})

    def test_set_rounds_propagates_to_player(self):
        spec = axelrod_by_name(include_long_run=True)["TitForTat"]
        opponent = spec.factory(random.Random(1))
        opponent.reset()
        opponent.set_rounds(37)
        self.assertEqual(opponent._player.match_attributes["length"], 37)

    def test_sample_field_and_all_names(self):
        field = sample_axelrod_field(123, 243, include_long_run=True)
        self.assertEqual(len(field), 243)
        self.assertEqual({opponent.name for opponent in field},
                         {spec.name for spec in axelrod_specs(True)})

        fast = sample_axelrod_field(123, 10, include_long_run=False)
        self.assertEqual(len(fast), 10)
        self.assertTrue({opponent.name for opponent in fast} <=
                        {spec.name for spec in axelrod_specs(False)})

        cheating = sample_axelrod_cheating_field(123, 3)
        self.assertEqual([opponent.name for opponent in cheating],
                         ["Darwin", "Darwin", "Darwin"])

    def test_build_dist_field(self):
        field = build_dist_field("axelrod-fast", 3, 12)
        self.assertEqual(len(field), 12)
        allowed = {spec.name for spec in axelrod_specs(False)}
        self.assertTrue(all(opponent.name in allowed for opponent in field))

        cheating = build_dist_field("axelrod-cheating", 3, 2)
        self.assertEqual([opponent.name for opponent in cheating],
                         ["Darwin", "Darwin"])

    def test_matchup_selection(self):
        from ipd.evaluation.matchup import _selected_specs

        self.assertEqual(len(_selected_specs("axelrod")), 243)
        self.assertEqual(len(_selected_specs("axelrod-fast")), 224)
        self.assertEqual([spec.name for spec in _selected_specs(
            "axelrod-cheating")], ["Darwin"])

    def test_matchup_worker_can_build_axelrod_opponent(self):
        spec = candidate_specs(["final-tuned"])[0]
        rows = type_task({
            "type": "TitForTat",
            "opponent_set": "axelrod",
            "reps": 1,
            "rounds": 5,
            "seed": 1,
            "candidates": [spec.to_dict()],
        })
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:3], ("Final-tuned", "TitForTat", "axelrod"))

        rows = type_task({
            "type": "Darwin",
            "opponent_set": "axelrod-cheating",
            "reps": 1,
            "rounds": 5,
            "seed": 1,
            "candidates": [spec.to_dict()],
        })
        self.assertEqual(rows[0][:3], ("Final-tuned", "Darwin",
                                       "axelrod-cheating"))

    def test_benchmark_cli_axelrod_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "axelrod_bench.json"
            code = benchmark_main([
                "--candidates", "final-tuned",
                "--dists", "axelrod-fast",
                "--fields", "1", "--n", "4", "--workers", "1",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["dists"], ["axelrod-fast"])
            self.assertIn("Final-tuned", payload["summary"])
            values = payload["summary"]["Final-tuned"]["dists"]["axelrod-fast"]
            self.assertEqual(len(values["values"]), 1)

            out = Path(tmp) / "axelrod_cheating_bench.json"
            code = benchmark_main([
                "--candidates", "final-tuned",
                "--dists", "axelrod-cheating",
                "--fields", "1", "--n", "1", "--workers", "1",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["dists"], ["axelrod-cheating"])
            self.assertEqual(payload["n"], 1)

    def test_all_fast_strategies_play(self):
        # The fast bank excludes Meta players, so this is a cheap smoke test
        # across all 224 strategies.
        for spec in axelrod_specs(include_long_run=False):
            opponent = spec.factory(random.Random(11))
            result = run_match(TFT(), opponent, rounds=10)
            self.assertEqual(len(result.theirs), 10, spec.name)
            self.assertTrue(set(result.theirs) <= {C, D}, spec.name)

    def test_long_running_meta_strategy_plays(self):
        spec = axelrod_by_name(include_long_run=True)["MetaMajority"]
        opponent = spec.factory(random.Random(11))
        result = run_match(TFT(), opponent, rounds=3)
        self.assertEqual(len(result.theirs), 3)
        self.assertTrue(set(result.theirs) <= {C, D})


if __name__ == "__main__":
    unittest.main()
