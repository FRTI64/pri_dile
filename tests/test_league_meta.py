"""P3 league meta-game weights, manifests and Gate D."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ipd.training.league import (
    gate_d, load_opponent_manifest, main as meta_main,
    meta_weights_from_report, red_payoff_matrix, write_opponent_manifest,
)


def _report() -> dict:
    pairs = {}
    diffs = {
        "Blue|Red-A": 400.0,
        "Blue|Red-B": 410.0,
        "Blue|Red-C": 300.0,
        "Other|Red-A": 350.0,
        "Other|Red-B": 300.0,
        "Other|Red-C": 320.0,
    }
    for key, diff in diffs.items():
        blue, red = key.split("|")
        pairs[key] = {
            "blue": blue, "red": red,
            "blue_mean": 300.0, "red_mean": 300.0 + diff,
            "diff_mean": diff,
        }
    return {
        "candidates": {
            "Baseline": {
                "suites": {"official": {"mean": 554.0}},
                "overall": {"families": {
                    "official:original": 556.0,
                    "official:late": 550.0,
                }},
            },
            "Candidate": {
                "suites": {"official": {"mean": 553.0}},
                "overall": {"families": {
                    "official:original": 555.0,
                    "official:late": 548.0,
                }},
            },
        },
        "red": {"pairs": pairs},
    }


class PayoffMatrixTests(unittest.TestCase):
    def test_extracts_red_by_blue_matrix(self):
        matrix = red_payoff_matrix(
            _report()["red"]["pairs"], ["Red-A", "Red-B"], ["Blue", "Other"])
        self.assertEqual(matrix, [[400.0, 350.0], [410.0, 300.0]])

    def test_missing_pair_raises(self):
        with self.assertRaises(KeyError):
            red_payoff_matrix(_report()["red"]["pairs"], ["Missing"],
                              ["Blue"])


class MetaWeightTests(unittest.TestCase):
    def test_weights_sum_to_one_and_prefer_strong_red(self):
        result = meta_weights_from_report(
            _report(), ["Red-A", "Red-B", "Red-C"],
            blues=["Blue", "Other"], iterations=10000, seed=3,
            uniform_mix=0.0)
        weights = result["weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        self.assertGreater(weights["Red-A"], weights["Red-C"])
        self.assertLessEqual(result["worst_case"], 410.0 + 1e-9)

    def test_uniform_mix_regularises(self):
        result = meta_weights_from_report(
            _report(), ["Red-A", "Red-B", "Red-C"],
            blues=["Blue", "Other"], iterations=10000, seed=3,
            uniform_mix=0.5)
        self.assertTrue(all(value >= 0.5 / 3 - 1e-9
                            for value in result["weights"].values()))

    def test_deterministic(self):
        a = meta_weights_from_report(_report(), ["Red-A", "Red-B"],
                                     blues=["Blue"], seed=9)
        b = meta_weights_from_report(_report(), ["Red-A", "Red-B"],
                                     blues=["Blue"], seed=9)
        self.assertEqual(a["weights"], b["weights"])


class ManifestTests(unittest.TestCase):
    def test_roundtrip_and_weights(self):
        entries = [
            {"kind": "rl", "name": "Red-A",
             "params": {"weights": "weights/red_pool/red_rl_v1.json"}},
            {"kind": "final-tuned", "name": "Final-tuned"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.json"
            payload = write_opponent_manifest(
                path, entries, {"Red-A": 0.75, "Final-tuned": 0.25},
                name="demo")
            self.assertEqual(payload["schema"], "ipd-opponent-pool-v1")
            loaded = load_opponent_manifest(path)
            self.assertEqual([x["name"] for x in loaded],
                             ["Red-A", "Final-tuned"])
            self.assertAlmostEqual(loaded[0]["weight"], 0.75)
            self.assertAlmostEqual(loaded[1]["weight"], 0.25)

    def test_missing_weight_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyError):
                write_opponent_manifest(
                    Path(tmp) / "pool.json",
                    [{"kind": "rl", "name": "Red-A"}], {"Other": 1.0})


class GateDTests(unittest.TestCase):
    def test_pass_when_exploitability_drops(self):
        report = _report()
        # Candidate is harder to exploit on both held-out reds.
        for red in ("Red-A", "Red-B"):
            report["red"]["pairs"][f"Candidate|{red}"] = {
                "blue": "Candidate", "red": red,
                "blue_mean": 340.0, "red_mean": 640.0,
                "diff_mean": 300.0,
            }
            report["red"]["pairs"][f"Baseline|{red}"] = {
                "blue": "Baseline", "red": red,
                "blue_mean": 300.0, "red_mean": 700.0,
                "diff_mean": 400.0,
            }
        result = gate_d(report, "Baseline", "Candidate", ["Red-A", "Red-B"])
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["reduction"], 0.25)

    def test_fail_when_official_drops_too_much(self):
        report = _report()
        for red in ("Red-A",):
            report["red"]["pairs"][f"Candidate|{red}"] = {
                "blue": "Candidate", "red": red,
                "blue_mean": 340.0, "red_mean": 640.0,
                "diff_mean": 300.0,
            }
            report["red"]["pairs"][f"Baseline|{red}"] = {
                "blue": "Baseline", "red": red,
                "blue_mean": 300.0, "red_mean": 700.0,
                "diff_mean": 400.0,
            }
        report["candidates"]["Candidate"]["suites"]["official"]["mean"] = 551.0
        result = gate_d(report, "Baseline", "Candidate", ["Red-A"])
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["official"])

    def test_fail_on_family_collapse(self):
        report = _report()
        for red in ("Red-A",):
            report["red"]["pairs"][f"Candidate|{red}"] = {
                "blue": "Candidate", "red": red,
                "blue_mean": 340.0, "red_mean": 640.0,
                "diff_mean": 300.0,
            }
            report["red"]["pairs"][f"Baseline|{red}"] = {
                "blue": "Baseline", "red": red,
                "blue_mean": 300.0, "red_mean": 700.0,
                "diff_mean": 400.0,
            }
        report["candidates"]["Candidate"]["overall"]["families"][
            "official:late"] = 520.0
        result = gate_d(report, "Baseline", "Candidate", ["Red-A"])
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["families"])


class MetaCliTests(unittest.TestCase):
    def test_builds_weighted_manifest_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "league.json"
            config.write_text(json.dumps({
                "name": "meta-smoke",
                "candidates": [{"name": "Final-tuned",
                                "kind": "final-tuned"}],
                "red_policies": [
                    {"name": "Red-A", "kind": "final-tuned"},
                    {"name": "Red-B", "kind": "final-tuned"},
                ],
                "suites": {"heldout_v2": {
                    "kind": "opponents", "suite": "heldout_v2",
                    "seeds": 1, "seed_base": 1}},
            }), encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps({
                "candidates": {"Final-tuned": {
                    "suites": {"official": {"mean": 550.0}},
                    "overall": {"families": {"official:original": 550.0}},
                }},
                "red": {"pairs": {
                    "Final-tuned|Red-A": {
                        "blue": "Final-tuned", "red": "Red-A",
                        "blue_mean": 300.0, "red_mean": 700.0,
                        "diff_mean": 400.0,
                    },
                    "Final-tuned|Red-B": {
                        "blue": "Final-tuned", "red": "Red-B",
                        "blue_mean": 300.0, "red_mean": 650.0,
                        "diff_mean": 350.0,
                    },
                }},
            }), encoding="utf-8")
            out = root / "pool.json"
            summary = root / "summary.json"
            code = meta_main([
                "--config", str(config), "--report", str(report),
                "--train-reds", "Red-A", "--heldout-reds", "Red-B",
                "--out", str(out), "--summary-out", str(summary),
                "--iterations", "500",
            ])
            self.assertEqual(code, 0)
            loaded = load_opponent_manifest(out)
            self.assertEqual(loaded[0]["name"], "Red-A")
            self.assertAlmostEqual(loaded[0]["weight"], 1.0)
            self.assertTrue(summary.is_file())
            with self.assertRaises(SystemExit):
                meta_main([
                    "--config", str(config), "--report", str(report),
                    "--train-reds", "Red-A,Red-B",
                    "--heldout-reds", "Red-B", "--out", str(out),
                ])


if __name__ == "__main__":
    unittest.main()
