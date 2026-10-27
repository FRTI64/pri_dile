"""PSRO meta-game, population, oracle command and end-to-end smoke tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ipd.training.psro import (
    MetaGame, OracleConfig, PopulationEntry, PSROConfig, build_markdown,
    choose_blue_deploy, choose_blue_init, load_config, load_population,
    oracle_command, run_psro, save_population, solve_blue_red, split_roles,
    two_population_report,
)


def _entry(name: str, kind: str = "final-tuned", role: str = "blue",
           generation: int = 0) -> PopulationEntry:
    return PopulationEntry(
        name=name, entry={"kind": kind}, role=role, generation=generation)


class PopulationEntryTests(unittest.TestCase):
    def test_accepts_spec_path_and_nested_entry(self):
        spec = PopulationEntry.from_dict(
            {"name": "rule", "kind": "final-tuned"})
        self.assertEqual(spec.entry, {"kind": "final-tuned"})
        path = PopulationEntry.from_dict(
            {"name": "rl", "path": "weights/rl_policy_weights.json"})
        self.assertEqual(path.deploy_path, "weights/rl_policy_weights.json")
        nested = PopulationEntry.from_dict({
            "name": "nested", "role": "red", "generation": 2,
            "entry": {"kind": "rl", "params": {"weights": "w.json"}},
        })
        self.assertEqual(nested.role, "red")
        self.assertEqual(nested.generation, 2)
        self.assertEqual(nested.deploy_path, "w.json")

    def test_serialised_paths_are_repo_relative(self):
        from ipd.paths import ROOT

        entry = PopulationEntry(
            "x", {"path": str(ROOT / "weights" / "x.json")})
        self.assertEqual(entry.to_dict()["entry"]["path"],
                         "weights/x.json")
        self.assertEqual(entry.to_pool_entry()["path"], "weights/x.json")

    def test_rejects_missing_name_or_policy(self):
        with self.assertRaises(ValueError):
            PopulationEntry.from_dict({"kind": "final-tuned"})
        with self.assertRaises(ValueError):
            PopulationEntry.from_dict({"name": "x"})

    def test_population_roundtrip_and_duplicate_check(self):
        entries = [_entry("A"), _entry("B", role="red", generation=2)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "population.json"
            save_population(path, entries, meta={"round": 1})
            loaded = load_population(path)
            self.assertEqual([x.name for x in loaded], ["A", "B"])
            self.assertEqual(loaded[1].generation, 2)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["entries"].append(data["entries"][0])
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_population(path)


class MetaGameTests(unittest.TestCase):
    def _synthetic(self) -> MetaGame:
        game = MetaGame([_entry("A"), _entry("B"), _entry("C")],
                        seeds=[1], workers=1)
        # Rock-paper-scissors score differences: A > B > C > A.
        rows = [
            {"i": 0, "j": 1, "diffs": [10, 10],
             "i_scores": [600, 600], "j_scores": [590, 590]},
            {"i": 0, "j": 2, "diffs": [-10, -10],
             "i_scores": [590, 590], "j_scores": [600, 600]},
            {"i": 1, "j": 2, "diffs": [10, 10],
             "i_scores": [600, 600], "j_scores": [590, 590]},
        ]
        for row in rows:
            game._absorb(row)
        return game

    def test_matrix_is_antisymmetric_and_payoff_is_recorded(self):
        game = self._synthetic()
        matrix = game.matrix()
        self.assertEqual(matrix, [[0, 10, -10], [-10, 0, 10], [10, -10, 0]])
        payoff = game.payoff()
        self.assertAlmostEqual(payoff[0][1], 600.0)
        self.assertAlmostEqual(game.opponent_payoff()[0][1], 590.0)
        self.assertEqual(game.matches()[0][1], 2)

    def test_uniform_nash_has_zero_exploitability(self):
        game = self._synthetic()
        meta = game.solve(iterations=500, seed=3, uniform_mix=0.0)
        self.assertAlmostEqual(sum(meta["weights"]), 1.0)
        for value in meta["weights"]:
            self.assertAlmostEqual(value, 1 / 3, delta=0.05)
        self.assertLess(meta["exploitability"], 1.0)

    def test_add_evaluates_new_row_and_column(self):
        game = self._synthetic()
        with patch("ipd.training.psro._run_tasks") as run:
            run.return_value = [{
                "i": 3, "j": 0, "diffs": [5],
                "i_scores": [605], "j_scores": [600],
            }, {
                "i": 3, "j": 1, "diffs": [-5],
                "i_scores": [595], "j_scores": [600],
            }, {
                "i": 3, "j": 2, "diffs": [5],
                "i_scores": [605], "j_scores": [600],
            }]
            game.add(_entry("D"))
        self.assertEqual(len(game), 4)
        self.assertEqual(game.matrix()[3][0], 5)
        self.assertEqual(game.matrix()[0][3], -5)

    def test_best_init_uses_deployable_blue_policy(self):
        entries = [
            PopulationEntry("rule", {"kind": "final-tuned"},
                            role="blue", generation=0),
            PopulationEntry("blue", {"kind": "rl", "params": {
                "weights": "blue.json"}}, role="blue", generation=1),
            PopulationEntry("red", {"kind": "rl", "params": {
                "weights": "red.json"}}, role="red", generation=2),
        ]
        game = MetaGame(entries, seeds=[1], workers=1)
        with patch("ipd.training.psro._run_tasks", return_value=[]):
            game.evaluate()
        self.assertEqual(game.best_init([0.0, 1.0, 0.0]), "blue.json")


class BlueRedMetaGameTests(unittest.TestCase):
    def _game(self) -> MetaGame:
        entries = [
            PopulationEntry("A", {"kind": "rl", "params": {
                "weights": "a.json"}}, role="blue"),
            PopulationEntry("B", {"kind": "rl", "params": {
                "weights": "b.json"}}, role="blue"),
            PopulationEntry("X", {"kind": "rl", "params": {
                "weights": "x.json"}}, role="red"),
            PopulationEntry("Y", {"kind": "rl", "params": {
                "weights": "y.json"}}, role="red"),
        ]
        game = MetaGame(entries, seeds=[1], workers=1)
        for i, j, diff in ((0, 2, -10), (0, 3, -20),
                           (1, 2, -5), (1, 3, -30)):
            game._absorb({
                "i": i, "j": j, "diffs": [diff],
                "i_scores": [600 + diff], "j_scores": [600],
            })
        return game

    def test_split_roles_and_rectangular_solution(self):
        game = self._game()
        blue, red = split_roles(game.entries)
        self.assertEqual(blue, [0, 1])
        self.assertEqual(red, [2, 3])
        meta = solve_blue_red(game, blue, red, iterations=2000, seed=3)
        self.assertEqual(meta["blue_meta"]["weights_by_name"]["A"], 1.0)
        self.assertEqual(meta["red_meta"]["weights_by_name"]["Y"], 1.0)
        self.assertAlmostEqual(meta["blue_worst_case"], -20.0)
        self.assertAlmostEqual(meta["exploitability"], 20.0)
        self.assertEqual(meta["best_red"], "Y")
        self.assertEqual(meta["best_blue"], "A")

    def test_uniform_mix_only_changes_training_weights(self):
        game = self._game()
        blue, red = split_roles(game.entries)
        meta = solve_blue_red(game, blue, red, iterations=2000, seed=3,
                              uniform_mix=1.0)
        self.assertAlmostEqual(
            meta["red_meta"]["training_weights_by_name"]["X"], 0.5)
        self.assertAlmostEqual(
            meta["red_meta"]["training_weights_by_name"]["Y"], 0.5)
        self.assertEqual(meta["red_meta"]["weights_by_name"]["Y"], 1.0)

    def test_chooses_deployable_blue(self):
        game = self._game()
        blue, red = split_roles(game.entries)
        meta = solve_blue_red(game, blue, red, iterations=2000, seed=3)
        self.assertEqual(choose_blue_init(
            game, blue, red, meta["red_meta"]["training_weights"]),
            "a.json")
        deploy = choose_blue_deploy(game, blue, meta, "meta_best")
        self.assertEqual(deploy.name, "A")
        last = choose_blue_deploy(game, blue, meta, "last")
        self.assertEqual(last.name, "B")

    def test_rejects_missing_side(self):
        with self.assertRaises(ValueError):
            split_roles([_entry("A")])
        with self.assertRaises(ValueError):
            split_roles([_entry("X", role="red")])

    def test_report_contains_both_meta_strategies(self):
        game = self._game()
        blue, red = split_roles(game.entries)
        report = two_population_report(game, blue, red, iterations=500,
                                       seed=1)
        self.assertEqual(report["blue"], ["A", "B"])
        self.assertEqual(report["red"], ["X", "Y"])
        self.assertIn("blue_meta", report)
        self.assertIn("red_meta", report)


class MetaSolveTests(unittest.TestCase):
    def test_dominant_strategy_gets_weight_and_zero_exploitability(self):
        game = MetaGame([_entry("A"), _entry("B")], seeds=[1], workers=1)
        game._absorb({"i": 0, "j": 1, "diffs": [100],
                      "i_scores": [700], "j_scores": [600]})
        meta = game.solve(iterations=2000, seed=5)
        self.assertGreater(meta["weights_by_name"]["A"], 0.9)
        self.assertGreaterEqual(meta["exploitability"], 0.0)


class ConfigTests(unittest.TestCase):
    def _write(self, root: Path, **extra) -> Path:
        data = {
            "name": "smoke",
            "population": [
                {"name": "Final", "kind": "final-tuned"},
                {"name": "RL", "kind": "rl", "params": {
                    "weights": "weights/rl_policy_weights.json"}},
            ],
            "rounds": 2,
            "cross_play": {"seeds": [3, 4], "workers": 1, "rounds": 100},
            "meta": {"iterations": 500, "uniform_mix": 0.2},
            "output_dir": str(root / "out"),
            "deploy_path": str(root / "deploy.json"),
            "oracle": {
                "init_weights": "weights/rl_policy_checkpoint_repair.json",
                "iterations": 3,
                "fields_per_dist": 1,
                "n": 8,
                "workers": 1,
                "device": "cpu",
                "value_bias": 0.0,
                "train_dists": ["original", "repair"],
                "dist_weights": [1, 2],
            },
        }
        data.update(extra)
        path = root / "psro.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loads_paths_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write(root)
            cfg = load_config(path)
            self.assertEqual([x.name for x in cfg.population], ["Final", "RL"])
            self.assertEqual(cfg.seeds, (3, 4))
            self.assertEqual(cfg.rounds, 2)
            self.assertEqual(cfg.oracle.iterations, 3)
            self.assertTrue(cfg.oracle.init_weights.endswith(
                "rl_policy_checkpoint_repair.json"))
            self.assertEqual(Path(cfg.output_dir), root / "out")

    def test_oracle_command_contains_pool_and_reward_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(self._write(Path(tmp)))
            cmd = oracle_command(
                cfg, manifest="pool.json", out="out.json",
                checkpoint="ckpt.json", history="history.json", seed=7,
                init_weights="init.json")
            self.assertIn("--opponent-pool", cmd)
            self.assertIn("--opponent-penalty", cmd)
            self.assertIn("--no-validation", cmd)
            self.assertIn("--init-weights", cmd)
            self.assertEqual(cmd[cmd.index("--train-dists") + 1],
                             "original,repair")
            self.assertEqual(cmd[cmd.index("--dist-weights") + 1], "1,2")
            self.assertEqual(cmd[cmd.index("--value-bias") + 1], "0.0")
            self.assertEqual(cmd[cmd.index("--seed") + 1], "7")
            self.assertNotIn("--from-scratch", cmd)


class RunPSROTests(unittest.TestCase):
    def _config(self, root: Path, *, dry_run_path: bool = False) -> PSROConfig:
        return PSROConfig(
            name="smoke",
            population=(_entry("Final"), _entry("Red", role="red"),
                        _entry("Triple")),
            output_dir=str(root / "out"),
            config_path="",
            rounds=1,
            seeds=(1, 2),
            workers=1,
            nash_iterations=300,
            nash_seed=7,
            uniform_mix=0.0,
            deploy_path=str(root / "deploy.json"),
            deploy_policy="last",
            oracle=OracleConfig(iterations=1, fields_per_dist=1, n=2,
                                workers=1),
        )

    def test_fake_oracle_roundtrip_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._config(root)

            def fake(context):
                self.assertEqual(context.round_index, 1)
                self.assertTrue(Path(context.manifest_path).is_file())
                pool = json.loads(Path(context.manifest_path).read_text(
                    encoding="utf-8"))
                self.assertEqual(len(pool["entries"]), 1)
                self.assertEqual(pool["entries"][0]["name"], "Red")
                self.assertAlmostEqual(
                    sum(x["weight"] for x in pool["entries"]), 1.0)
                return _entry("PSRO-BR-01")

            report = run_psro(cfg, oracle=fake)
            self.assertEqual(report["schema"], "ipd-psro-report-v1")
            self.assertEqual(len(report["population"]), 4)
            self.assertEqual(report["rounds"][0]["entry"]["name"],
                             "PSRO-BR-01")
            self.assertIn("exploitability", report["final"])
            self.assertIn("blue_meta", report["final"])
            self.assertIn("red_meta", report["final"])
            out = root / "out"
            for name in ("report.json", "README.md", "population.json",
                         "meta_initial.json", "round_01.json",
                         "oracle_manifest_01.json"):
                self.assertTrue((out / name).is_file(), name)
            self.assertEqual(report["deploy"], {})
            self.assertIn("PSRO", build_markdown(report))

    def test_dry_run_does_not_train_or_deploy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._config(root)
            called = []

            def fake(context):
                called.append(context)
                return _entry("unused")

            report = run_psro(cfg, oracle=fake, dry_run=True)
            self.assertTrue(report["dry_run"])
            self.assertEqual(called, [])
            self.assertFalse((root / "deploy.json").exists())
            self.assertTrue((root / "out" / "meta_initial.json").is_file())


if __name__ == "__main__":
    unittest.main()
