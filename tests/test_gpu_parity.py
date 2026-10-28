"""End-to-end parity tests for the Plan 2 GPU path.

The GPU path is compared against the reference CPU implementation with learned
policy opponents in argmax mode.  Sample mode is intentionally not batched, so
the fallback test checks that ``--mode sample`` produces the exact same result
file as the pre-Plan-2 CPU path.
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from ipd.evaluation.common import evaluate_field_details, run_match
from ipd.evaluation.protocol import (
    build_protocol_field, restore_field_rng, snapshot_field_rng,
)
from ipd.evaluation.protocol_cli import main as protocol_main
from ipd.policies.registry import candidate_specs

if torch is not None:
    from ipd.evaluation.gpu.blue import supports_gpu_policy
    from ipd.evaluation.gpu.simulator import evaluate_field_gpu

_LEARNED = {"NeuralPolicy", "RLPolicy", "GRUPolicy", "SelfPlayGRUPolicy"}
_CUDA = bool(torch is not None and torch.cuda.is_available())


def _argmax_field(tier: str, seed: int, n: int):
    field = build_protocol_field(tier, seed, n)
    for opponent in field:
        if type(opponent).__name__ in _LEARNED:
            opponent.mode = "argmax"
    return field


def _spec(name: str):
    return [spec for spec in candidate_specs(name, mode="argmax")][0]


@unittest.skipUnless(_CUDA, "CUDA is not available")
class SimulatorParityTests(unittest.TestCase):
    def test_learned_candidates_match_cpu_actions(self) -> None:
        field = _argmax_field("test_v7", 20261027, 32)
        snapshot = snapshot_field_rng(field)
        for name in ("rl-repair", "gru", "selfplay", "adv-mlp",
                     "psro-scratch"):
            with self.subTest(candidate=name):
                spec = _spec(name)
                restore_field_rng(field, snapshot)
                policy = spec.build()
                stats, _ = evaluate_field_details(policy, field, rounds=100)

                restore_field_rng(field, snapshot)
                gpu_policy = spec.build()
                simulation = evaluate_field_gpu(
                    gpu_policy, field, rounds=100, device="cuda",
                    dtype=torch.float64, keep_histories=True)
                self.assertEqual(stats.scores, simulation.scores)

                restore_field_rng(field, snapshot)
                cpu_policy = spec.build()
                for index, opponent in enumerate(field):
                    result = run_match(cpu_policy, opponent, rounds=100)
                    self.assertEqual(result.mine, simulation.mine[index],
                                     f"{name} row {index} blue")
                    self.assertEqual(result.theirs, simulation.theirs[index],
                                     f"{name} row {index} opponent")

    def test_batch_size_does_not_change_row_zero(self) -> None:
        from ipd.core.opponents.basic import AllC

        spec = _spec("selfplay")
        single = evaluate_field_gpu(
            spec.build(), [AllC()], rounds=100, device="cuda",
            dtype=torch.float64, keep_histories=True)
        many = evaluate_field_gpu(
            spec.build(), [AllC() for _ in range(32)], rounds=100,
            device="cuda", dtype=torch.float64, keep_histories=True)
        self.assertEqual(single.mine[0], many.mine[0])
        self.assertEqual(single.scores[0], many.scores[0])

    def test_multifield_batch_matches_single_field(self) -> None:
        # Use only stateless argmax policy opponents so the comparison tests
        # the field-batching layer itself, not stochastic rule RNG streams.
        spec = _spec("selfplay")
        fields = []
        for index in range(2):
            field = _argmax_field("test_v7", 20261027 + index * 7919, 24)
            fields.append([opponent for opponent in field
                           if type(opponent).__name__ in _LEARNED])
        from ipd.evaluation.gpu.simulator import evaluate_fields_gpu

        batched = evaluate_fields_gpu(
            spec.build(), fields, rounds=100, device="cuda",
            dtype=torch.float64, max_batch=32768)
        for index, field in enumerate(fields):
            single = evaluate_field_gpu(
                spec.build(), field, rounds=100, device="cuda",
                dtype=torch.float64)
            self.assertEqual(batched[index].scores, single.scores)


class CandidateRoutingTests(unittest.TestCase):
    def test_support_detection(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed")
        self.assertTrue(supports_gpu_policy(_spec("rl-repair").build()))
        self.assertTrue(supports_gpu_policy(_spec("selfplay").build()))
        self.assertTrue(supports_gpu_policy(_spec("adv-gru").build()))
        self.assertTrue(supports_gpu_policy(_spec("psro-scratch").build()))
        self.assertFalse(supports_gpu_policy(_spec("final-tuned").build()))
        self.assertFalse(supports_gpu_policy(_spec("psro").build()))


@unittest.skipUnless(_CUDA, "CUDA is not available")
class ProtocolCliGpuTests(unittest.TestCase):
    @staticmethod
    def _run(out: Path, *extra: str, fields: int = 2, n: int = 24) -> dict:
        code = protocol_main([
            "eval", "--candidates", "selfplay,gru,final-tuned",
            "--tiers", "test_v7", "--fields", str(fields), "--n", str(n),
            "--workers", "1", "--candidate-chunk", "2",
            "--field-reuse", "on", "--seed", "321",
            "--no-consume", "--out", str(out), *extra,
        ])
        if code != 0:
            raise AssertionError(f"protocol eval returned {code}")
        return json.loads(out.read_text(encoding="utf-8"))

    def test_cpu_and_gpu_result_values_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpu = self._run(
                Path(tmp) / "cpu.json", "--mode", "argmax",
                "--opponent-mode", "argmax", "--gpu", "off")
            gpu = self._run(
                Path(tmp) / "gpu.json", "--mode", "argmax",
                "--opponent-mode", "argmax", "--gpu", "on",
                "--gpu-dtype", "float64", "--gpu-field-batch", "2")
        self.assertFalse(cpu["gpu"]["enabled"])
        self.assertTrue(gpu["gpu"]["enabled"])
        for name in cpu["results"]:
            self.assertEqual(cpu["results"][name]["test_v7"]["values"],
                             gpu["results"][name]["test_v7"]["values"],
                             name)

    def test_sample_mode_stays_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._run(Path(tmp) / "plain.json", "--gpu", "off",
                              fields=1, n=16)
            auto = self._run(Path(tmp) / "auto.json", "--gpu", "on",
                             fields=1, n=16)
        self.assertFalse(auto["gpu"]["enabled"])
        for name in plain["results"]:
            self.assertEqual(plain["results"][name]["test_v7"]["values"],
                             auto["results"][name]["test_v7"]["values"],
                             name)


if __name__ == "__main__":
    unittest.main()
