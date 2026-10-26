"""E0 league harness: opponent suites, cross-play, summaries and CLI."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ipd.evaluation.opponent_suite import build_suite, suite_names
from ipd.evaluation.league import (
    _suite_tasks, build_cross_play_report, cross_play_task,
    load_league_config, main as league_main, red_task, summarize_rows,
)
from ipd.policies.registry import CandidateSpec


class OpponentSuiteTests(unittest.TestCase):
    def test_heldout_v2_is_fresh_and_structured(self):
        suite = build_suite("heldout_v2")
        self.assertGreaterEqual(len(suite), 50)
        names = [item.name for item in suite]
        self.assertEqual(len(names), len(set(names)))
        families = {item.family for item in suite}
        self.assertGreaterEqual(len(families), 5)
        self.assertNotIn("good", families)
        self.assertNotIn("top", families)
        self.assertNotIn("heldout", families)

    def test_known_suites_resolve(self):
        self.assertIn("validation", suite_names())
        self.assertIn("final", suite_names())
        self.assertIn("heldout_v2", suite_names())
        self.assertIn("heldout_v3", suite_names())
        self.assertIn("heldout_v4", suite_names())
        self.assertGreater(len(build_suite("validation")), 0)

    def test_heldout_v3_is_disjoint_from_v2(self):
        v2 = build_suite("heldout_v2")
        v3 = build_suite("heldout_v3")
        self.assertGreaterEqual(len(v3), 50)
        names2 = {item.name for item in v2}
        names3 = [item.name for item in v3]
        self.assertEqual(len(names3), len(set(names3)))
        self.assertFalse(names2 & set(names3))
        families2 = {item.family for item in v2}
        families3 = {item.family for item in v3}
        self.assertFalse(families2 & families3)
        self.assertGreaterEqual(len(families3), 5)

    def test_heldout_v4_is_disjoint_from_v3(self):
        v3 = build_suite("heldout_v3")
        v4 = build_suite("heldout_v4")
        self.assertGreaterEqual(len(v4), 50)
        names3 = {item.name for item in v3}
        names4 = [item.name for item in v4]
        self.assertEqual(len(names4), len(set(names4)))
        self.assertFalse(names3 & set(names4))
        families3 = {item.family for item in v3}
        families4 = {item.family for item in v4}
        self.assertFalse(families3 & families4)
        self.assertGreaterEqual(len(families4), 5)

    def test_factory_is_seed_deterministic(self):
        item = build_suite("heldout_v2")[0]
        import random

        bot_a = item.factory(random.Random(123))
        bot_b = item.factory(random.Random(123))
        mine: list[int] = []
        theirs: list[int] = []
        for t in range(30):
            a = bot_a.act(t, mine, theirs)
            b = bot_b.act(t, mine, theirs)
            self.assertEqual(a, b)
            mine.append(a)
            theirs.append(b)


class LeaguePrimitiveTests(unittest.TestCase):
    def test_cross_play_task_and_matrices(self):
        a = CandidateSpec("final-tuned", "Final-tuned")
        b = CandidateSpec("triple-majority", "Triple-majority")
        rows = cross_play_task({
            "row_spec": a.to_dict(), "col_spec": b.to_dict(),
            "seeds": [1, 2], "rounds": 100,
            "row_seed": 11, "col_seed": 12,
        })
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["row"], "Final-tuned")
            self.assertEqual(row["col"], "Triple-majority")
            self.assertGreater(row["row_score"], 0)
        report = build_cross_play_report(
            rows + [
                {**row, "row": "Triple-majority", "col": "Final-tuned",
                 "row_score": row["col_score"], "col_score": row["row_score"]}
                for row in rows
            ],
            ["Final-tuned", "Triple-majority"],
        )
        self.assertEqual(len(report["payoff"]), 2)
        self.assertIn("elo", report)
        self.assertIn("nash", report)
        self.assertEqual(report["matches"][0][1], 2)
        self.assertIn("pool_exploitability", report)

    def test_summarize_rows_min_norm_and_cvar(self):
        rows = []
        for family, values in (("a", [600, 610]), ("b", [400, 420]),
                               ("c", [500, 510])):
            for index, score in enumerate(values):
                rows.append({
                    "suite": "heldout_v2", "family": family,
                    "opponent": f"{family}.{index}", "score": score,
                    "opponent_score": 100,
                })
        baseline = {"heldout_v2:a": 600.0, "heldout_v2:b": 500.0,
                    "heldout_v2:c": 500.0}
        summary = summarize_rows(rows, baseline)
        self.assertAlmostEqual(summary["mean"], 506.6666667, places=4)
        self.assertAlmostEqual(summary["families"]["heldout_v2:a"], 605.0)
        self.assertAlmostEqual(summary["min_norm_family"], 0.82)
        self.assertAlmostEqual(summary["cvar10_family"], 410.0)
        self.assertEqual(summary["argmin_family"], "heldout_v2:b")

    def test_red_report_normalizes_against_baseline(self):
        from ipd.evaluation.league import build_red_report

        rows = [
            {"blue": "A", "red": "R", "blue_score": 300, "red_score": 600},
            {"blue": "A", "red": "R", "blue_score": 300, "red_score": 600},
            {"blue": "B", "red": "R", "blue_score": 200, "red_score": 700},
            {"blue": "B", "red": "R", "blue_score": 200, "red_score": 700},
        ]
        report = build_red_report(rows, ["A", "B"], {"A": 300.0, "B": 300.0})
        self.assertAlmostEqual(report["blues"]["B"]["normalized_vs_baseline"],
                               200.0 / 300.0)

    def test_red_task(self):
        blue = CandidateSpec("final-tuned", "Final-tuned")
        red = CandidateSpec("triple-majority", "Triple-majority")
        rows = red_task({
            "blue_spec": blue.to_dict(), "red_spec": red.to_dict(),
            "seeds": [1, 2], "rounds": 100,
            "blue_seed": 21, "red_seed": 22,
        })
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["blue"], "Final-tuned")
        self.assertEqual(rows[0]["red"], "Triple-majority")
        self.assertGreater(rows[0]["blue_score"], 0)

    def test_config_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "league.json"
            path.write_text(json.dumps({
                "name": "smoke",
                "seed_base": 1,
                "workers": 2,
                "candidates": [
                    {"name": "Final-tuned", "kind": "final-tuned",
                     "policy_seed": 123},
                    {"name": "Triple-majority", "kind": "triple-majority"},
                ],
                "suites": {
                    "heldout_v2": {
                        "kind": "opponents", "suite": "heldout_v2",
                        "seeds": 1, "seed_base": 2,
                    },
                },
                "red_policies": [],
            }), encoding="utf-8")
            cfg = load_league_config(path)
            self.assertEqual(cfg.name, "smoke")
            self.assertEqual(len(cfg.candidates), 2)
            self.assertEqual(cfg.candidates[0].policy_seed, 123)
            self.assertIsNone(cfg.candidates[1].policy_seed)
            self.assertEqual(cfg.suites["heldout_v2"].seeds, (2,))
            tasks = _suite_tasks(cfg, cfg.suites["heldout_v2"],
                                 cfg.candidates)
            count = len(build_suite("heldout_v2"))
            self.assertEqual(tasks[0]["policy_seed"], 123)
            self.assertEqual(tasks[count]["policy_seed"], 20261024 + 1009)


class LeagueCliTests(unittest.TestCase):
    def test_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "league.json"
            config.write_text(json.dumps({
                "name": "e0-smoke",
                "seed_base": 1,
                "workers": 2,
                "candidates": [
                    {"name": "Final-tuned", "kind": "final-tuned"},
                    {"name": "Triple-majority", "kind": "triple-majority"},
                ],
                "suites": {
                    "heldout_v2": {
                        "kind": "opponents", "suite": "heldout_v2",
                        "seeds": 1, "seed_base": 2,
                    },
                },
                "cross_play": {"seeds": 1, "seed_base": 3},
                "red_policies": [],
            }), encoding="utf-8")
            out = root / "report.json"
            md = root / "README.md"
            code = league_main([
                "--config", str(config), "--suites", "heldout_v2",
                "--skip-red", "--workers", "2",
                "--out", str(out), "--md-out", str(md),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "ipd-league-report-v1")
            self.assertIn("candidates", payload)
            self.assertIn("cross_play", payload)
            self.assertIn("heldout_v2", payload["suites"])
            self.assertEqual(payload["red"], {})
            self.assertTrue(md.is_file())
            self.assertIn("E0", md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
