"""Evaluation harness: matches, fields, CLIs and output schemas."""
from __future__ import annotations

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
from ipd.evaluation.matchup import type_task
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


if __name__ == "__main__":
    unittest.main()
