"""Evaluation harness: matches, fields, CLIs and output schemas."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    from .helpers import ROOT
except ImportError:  # direct execution
    from helpers import ROOT  # noqa: F401

from ipd.core.simulator import AllC, C, D, TFT
from ipd.evaluation import common
from ipd.evaluation.benchmark import main as benchmark_main
from ipd.evaluation.blind import main as blind_main
from ipd.evaluation.matchup import main as matchup_main
from ipd.evaluation.matchup import type_task
from ipd.evaluation.split import main as split_main
from ipd.evaluation.split_config import load_split_config
from ipd.evaluation.suite import main as suite_main
from ipd.policies.registry import candidate_specs


class MatchTests(unittest.TestCase):
    def test_run_match(self):
        policy = candidate_specs(["final-tuned"])[0].build()
        result = common.run_match(policy, TFT())
        self.assertEqual(len(result.mine), 100)
        self.assertEqual(len(result.theirs), 100)
        self.assertEqual(result.opponent_name, "TFT")
        self.assertGreater(result.score, 0)

    def test_evaluate_field(self):
        policy = candidate_specs(["final-tuned"])[0].build()
        result = common.evaluate_field(policy, [AllC(), TFT()])
        self.assertEqual(result.n, 2)
        self.assertEqual(result.total, sum(result.scores))
        self.assertEqual(result.names, ["AllC", "TFT"])

    def test_evaluate_field_details(self):
        policy = candidate_specs(["final-tuned"])[0].build()
        result, by_opponent = common.evaluate_field_details(
            policy, [AllC(), AllC(), TFT()])
        self.assertEqual(result.n, 3)
        self.assertEqual(sorted(by_opponent), ["AllC", "TFT"])
        self.assertEqual(len(by_opponent["AllC"]), 2)
        self.assertEqual(len(by_opponent["TFT"]), 1)
        self.assertEqual(sum(len(v) for v in by_opponent.values()), 3)

    def test_eval_task_details(self):
        spec = candidate_specs(["final-tuned"])[0]
        task = {"spec": spec.to_dict(), "name": spec.name,
                "dist": "original", "seeds": [1], "n": 8}
        name, dist, values, details = common.eval_task_details(task)
        self.assertEqual((name, dist), ("Final-tuned", "original"))
        self.assertEqual(len(values), 1)
        self.assertTrue(details)
        self.assertEqual(sum(len(v) for v in details.values()), 8)

    def test_eval_task(self):
        spec = candidate_specs(["final-tuned"])[0]
        task = {"spec": spec.to_dict(), "name": spec.name,
                "dist": "original", "seeds": [1, 2], "n": 8}
        name, dist, values = common.eval_task(task)
        self.assertEqual(name, "Final-tuned")
        self.assertEqual(dist, "original")
        self.assertEqual(len(values), 2)
        self.assertTrue(all(v > 0 for v in values))

    def test_matchup_worker(self):
        spec = candidate_specs(["final-tuned"])[0]
        rows = type_task({"type": "TFT", "reps": 1, "seed": 1,
                          "candidates": [spec.to_dict()]})
        self.assertEqual(len(rows), 1)
        cand, typ, family, our, opp, diff, n = rows[0]
        self.assertEqual((cand, typ, family, n),
                         ("Final-tuned", "TFT", "nice", 1))

    def test_matchup_protocol_selection(self):
        from ipd.evaluation.matchup import _selected_specs
        from ipd.evaluation.protocol import build_catalog

        specs = _selected_specs("protocol")
        self.assertEqual(len(specs), len(build_catalog()))
        self.assertEqual(len({spec.name for spec in specs}), len(specs))

        spec = candidate_specs(["final-tuned"])[0]
        rows = type_task({
            "type": "anchor:AllC",
            "opponent_set": "protocol",
            "reps": 1,
            "rounds": 5,
            "seed": 1,
            "candidates": [spec.to_dict()],
        })
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:3], ("Final-tuned", "anchor:AllC",
                                       "anchor"))

    def test_matchup_suite_weighted_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "matchup.csv"
            code = matchup_main([
                "--candidates", "final-tuned",
                "--opponent-set", "core",
                "--reps", "1", "--workers", "4",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(
                out.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "ipd-matchup-v2")
            self.assertEqual(payload["opponent_set"], "core")
            self.assertEqual(len(payload["suite_scores"]["Final-tuned"]), 9)
            suite_csv = out.with_name(
                out.stem + "_suite_weighted.csv")
            self.assertTrue(suite_csv.is_file())
            self.assertIn("original", suite_csv.read_text(encoding="utf-8"))

    def test_matchup_spec_config_and_family_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "name": "matchup-smoke",
                "candidates": [
                    {"name": "Final-tuned", "kind": "final-tuned"},
                    {"name": "Resp3-tuned", "kind": "resp3-tuned"},
                ],
            }), encoding="utf-8")
            out = root / "matchup.csv"
            code = matchup_main([
                "--spec-config", str(config_path),
                "--opponent-set", "core",
                "--reps", "1", "--workers", "4",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(
                out.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["candidates"]), 2)
            self.assertIn("nice", payload["family_means"]["Final-tuned"])
            self.assertIn("family_means",
                          payload["candidate_summary"]["Resp3-tuned"])
            family_csv = out.with_name(out.stem + "_families.csv")
            self.assertTrue(family_csv.is_file())
            self.assertIn("Final-tuned",
                          family_csv.read_text(encoding="utf-8"))

    def test_pivot_matchup_tables(self):
        from ipd.tools.pivot_matchup import pivot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "long.csv"
            source.write_text(
                "candidate,type,family,our,opponent,diff,reps\n"
                "A,TFT,nice,604.0,594.0,10.0,1\n"
                "A,Grim,punitive,600.0,590.0,10.0,1\n"
                "B,TFT,nice,595.0,595.0,0.0,1\n"
                "B,Grim,punitive,590.0,590.0,0.0,1\n",
                encoding="utf-8")
            wide = root / "wide.csv"
            scores = root / "scores.csv"
            pivot(source, root / "missing.json", wide, scores)
            rows = list(csv.reader(wide.open(encoding="utf-8")))
            self.assertEqual(rows[0], ["candidate", "TFT", "Grim"])
            self.assertEqual(rows[1], ["A", "604|594", "600|590"])
            self.assertEqual(rows[2], ["B", "595|595", "590|590"])
            long_rows = list(csv.reader(scores.open(encoding="utf-8")))
            self.assertEqual(long_rows[0], ["candidate", "type", "family",
                                            "score"])
            self.assertEqual(long_rows[1], ["A", "TFT", "nice", "604|594"])


class CliIntegrationTests(unittest.TestCase):
    def test_benchmark_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench.json"
            code = benchmark_main([
                "--candidates", "final-tuned",
                "--fields", "1", "--n", "8", "--workers", "1",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "ipd-benchmark-v1")
            self.assertIn("Final-tuned", payload["summary"])
            self.assertEqual(set(payload["summary"]["Final-tuned"]["dists"]),
                             set(payload["dists"]))

    def test_suite_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite.json"
            code = suite_main([
                "--candidates", "final-tuned", "--suites", "original",
                "--fields", "1", "--n", "8", "--workers", "1",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "ipd-suite-v1")
            self.assertIn("Final-tuned", payload["summary"])

    def test_blind_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "blind.json"
            code = blind_main([
                "--candidates", "final-tuned",
                "--fields", "1", "--n", "8", "--workers", "1",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "ipd-blind-v1")
            self.assertIn("Final-tuned", payload["results"])

    def test_split_config_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "name": "smoke",
                "defaults": {
                    "benchmark": {"fields": 1, "n": 8, "seed": 1,
                                  "workers": 1},
                    "blind": {"fields": 1, "n": 8, "seed": 1,
                              "workers": 1},
                },
                "candidates": [
                    {"name": "Final-tuned", "kind": "final-tuned"},
                    {"name": "Resp3-tuned", "kind": "resp3-tuned",
                     "benchmark": {"fields": 2}},
                ],
            }), encoding="utf-8")
            config = load_split_config(config_path)
            self.assertEqual(len(config.candidates), 2)
            self.assertEqual(config.candidates[0].benchmark.fields, 1)
            self.assertEqual(config.candidates[1].benchmark.fields, 2)

            out_dir = root / "out"
            code = split_main([
                "--config", str(config_path),
                "--out-dir", str(out_dir),
                "--only", "Final-tuned",
                "--workers", "1", "--blind-workers", "1",
                "--force",
            ])
            self.assertEqual(code, 0)
            self.assertTrue(
                (out_dir / "by_candidate" / "Final-tuned"
                 / "benchmark.json").is_file())
            self.assertTrue(
                (out_dir / "by_candidate" / "Final-tuned"
                 / "blind.json").is_file())
            self.assertTrue(
                (out_dir / "by_candidate" / "Final-tuned"
                 / "opponents.json").is_file())
            self.assertTrue(any((out_dir / "by_opponent").glob("*.json")))
            summary = json.loads(
                (out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["schema"], "ipd-eval-split-v1")


if __name__ == "__main__":
    unittest.main()
