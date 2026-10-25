"""Unified training CLI, common helpers and importability."""
from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from .helpers import ROOT
except ImportError:  # direct execution
    from helpers import ROOT

from ipd.training.cli import TASKS, main as training_main
from ipd.training.common import TrainResult, load_json, save_json


class TrainingInterfaceTests(unittest.TestCase):
    def test_all_training_modules_expose_main(self):
        for task, (module_name, _) in TASKS.items():
            module = importlib.import_module(module_name)
            self.assertTrue(callable(getattr(module, "main", None)), task)
            params = inspect.signature(module.main).parameters
            self.assertIn("argv", params, task)

    def test_training_list(self):
        self.assertEqual(training_main(["--list"]), 0)
        self.assertEqual(training_main(["--help"]), 0)

    def test_unknown_task(self):
        self.assertEqual(training_main(["does-not-exist"]), 2)

    def test_adversarial_supports_rl_base(self):
        from ipd.training.adversarial import make_base

        deploy = make_base(
            "rl", str(ROOT / "weights" / "rl_policy_weights_bc.json"),
            seed=1, mode="sample")
        self.assertTrue(callable(getattr(deploy, "act", None)))
        checkpoint = make_base(
            "rl", str(ROOT / "weights" / "rl_policy_checkpoint_bc.json"),
            seed=1, mode="sample")
        self.assertTrue(callable(getattr(checkpoint, "act", None)))

    def test_common_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "data.json"
            save_json(path, {"x": 1})
            self.assertEqual(load_json(path), {"x": 1})
            result = TrainResult(task="demo", out=str(path), metrics={"a": 1})
            manifest = result.manifest()
            self.assertEqual(manifest["schema"], "ipd-training-result-v1")
            self.assertEqual(manifest["task"], "demo")
            result.write_manifest(Path(tmp) / "manifest.json")
            self.assertEqual(json.loads((Path(tmp) / "manifest.json").read_text(
                encoding="utf-8"))["task"], "demo")

    def test_direct_script_execution(self):
        cases = [
            ("ipd/training/neural.py", "--help"),
            ("ipd/training/search/multiobj.py", "--help"),
            ("ipd/training/__main__.py", "--list"),
        ]
        for script, flag in cases:
            proc = subprocess.run(
                [sys.executable, str(ROOT / script), flag],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")

    def test_standard_flags_are_visible(self):
        for task in ("neural", "gru", "rl", "adversarial",
                     "multiobj", "bayes", "joint", "wide"):
            proc = subprocess.run(
                [sys.executable, "-m", "ipd", "train", task, "--help"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for flag in ("--seed", "--out"):
                self.assertIn(flag, proc.stdout, f"{task} missing {flag}")
        for task in ("neural", "gru", "rl"):
            proc = subprocess.run(
                [sys.executable, "-m", "ipd", "train", task, "--help"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertIn("--device", proc.stdout)


if __name__ == "__main__":
    unittest.main()
