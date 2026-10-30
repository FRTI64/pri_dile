"""Tests for Protocol v5 field-level scheduling and RNG reuse."""
from __future__ import annotations

import json
import random
import tempfile
import time
import unittest
from pathlib import Path

from ipd.core.primitives import Strategy
from ipd.evaluation.common import evaluate_field_details
from ipd.evaluation.protocol import (
    TEST_TIER, build_protocol_field, field_reuse_safe, restore_field_rng,
    snapshot_field_rng,
)
from ipd.evaluation.protocol_cli import _eval_field_task, _eval_task
from ipd.evaluation.protocol_cli import main as protocol_main
from ipd.evaluation.scheduler import (
    MemorySampler, ProfileAccumulator, chunk_specs, worker_budget,
)
from ipd.policies.registry import candidate_specs


def _spec(name: str):
    return {spec.name: spec for spec in candidate_specs("all")}[name]


class WorkerBudgetTests(unittest.TestCase):
    def test_auto_respects_cpu_ram_and_cap(self) -> None:
        budget = worker_budget("auto", cpu_count=24, available_gb=31.0)
        self.assertEqual(budget["resolved"], 20)
        self.assertEqual(budget["cpu_limit"], 22)
        self.assertGreaterEqual(budget["ram_limit"], 20)

    def test_auto_uses_ram_budget(self) -> None:
        budget = worker_budget("auto", cpu_count=24, available_gb=6.5)
        self.assertEqual(budget["resolved"], 4)
        budget = worker_budget("auto", cpu_count=24, available_gb=1.0)
        self.assertEqual(budget["resolved"], 1)

    def test_auto_uses_cpu_budget(self) -> None:
        budget = worker_budget("auto", cpu_count=6, available_gb=64.0)
        self.assertEqual(budget["resolved"], 4)

    def test_explicit_workers_are_not_capped(self) -> None:
        budget = worker_budget("32", cpu_count=24, available_gb=31.0)
        self.assertEqual(budget["resolved"], 32)
        with self.assertRaises(ValueError):
            worker_budget("zero")
        with self.assertRaises(ValueError):
            worker_budget(0)

    def test_chunks_are_balanced_and_lossless(self) -> None:
        chunks = chunk_specs(list(range(23)), 6)
        self.assertEqual(len(chunks), 4)
        self.assertEqual([len(chunk) for chunk in chunks], [6, 6, 6, 5])
        self.assertEqual(sorted(x for chunk in chunks for x in chunk),
                         list(range(23)))
        self.assertEqual(chunk_specs([], 6), [])


class MemorySamplerTests(unittest.TestCase):
    def test_sampler_records_process_tree(self) -> None:
        sampler = MemorySampler(interval=0.05)
        sampler.start()
        time.sleep(0.15)
        payload = sampler.stop()
        self.assertGreaterEqual(payload["samples"], 1)
        self.assertGreaterEqual(payload["processes_peak"], 1)
        self.assertGreater(payload["peak_pss_mb"], 0.0)
        self.assertGreaterEqual(payload["peak_rss_mb"],
                                payload["peak_pss_mb"])


class FieldRngTests(unittest.TestCase):
    def test_field_snapshot_restores_opponent_streams(self) -> None:
        field = build_protocol_field("dev_param", 777, 64)
        snapshot = snapshot_field_rng(field)
        self.assertTrue(field_reuse_safe(field))
        self.assertGreater(len(snapshot), 0)

        first = _spec("Final-tuned").build()
        baseline, _ = evaluate_field_details(first, field, rounds=100)

        restore_field_rng(field, snapshot)
        second = _spec("Resp3-tuned").build()
        reused, _ = evaluate_field_details(second, field, rounds=100)

        fresh_field = build_protocol_field("dev_param", 777, 64)
        fresh = _spec("Resp3-tuned").build()
        rebuilt, _ = evaluate_field_details(fresh, fresh_field, rounds=100)
        self.assertEqual(reused.scores, rebuilt.scores)
        self.assertEqual(reused.mean, rebuilt.mean)
        self.assertNotEqual(baseline.scores, reused.scores)

    def test_candidate_order_does_not_leak_rng(self) -> None:
        specs = [_spec("Final-tuned"), _spec("Resp3-tuned")]

        def run(order):
            field = build_protocol_field("dev_param", 999, 32)
            snapshot = snapshot_field_rng(field)
            values = {}
            for index, spec in enumerate(order):
                if index:
                    restore_field_rng(field, snapshot)
                policy = spec.build()
                stats, _ = evaluate_field_details(policy, field)
                values[spec.name] = stats.scores
            return values

        forward = run(specs)
        reverse = run(list(reversed(specs)))
        for spec in specs:
            self.assertEqual(forward[spec.name], reverse[spec.name],
                             spec.name)

    def test_snapshot_covers_learned_policy_opponents(self) -> None:
        field = build_protocol_field(TEST_TIER, 777, 512)
        snapshot = snapshot_field_rng(field)
        learned = [bot for bot in field
                   if type(bot).__name__ in
                   ("NeuralPolicy", "GRUPolicy", "SelfPlayGRUPolicy")]
        self.assertTrue(learned)
        states = {id(bot): bot.rng.getstate() for bot in learned}
        for bot in learned:
            bot.rng.random()
        self.assertNotEqual(learned[0].rng.getstate(),
                            states[id(learned[0])])
        restore_field_rng(field, snapshot)
        for bot in learned:
            self.assertEqual(bot.rng.getstate(), states[id(bot)])

    def test_unsafe_field_types_disable_reuse(self) -> None:
        class Darwin(Strategy):
            pass

        self.assertFalse(field_reuse_safe([Darwin(random.Random(1))]))


class FieldSchedulingTests(unittest.TestCase):
    def test_chunked_field_task_matches_rebuild_per_candidate(self) -> None:
        specs = [_spec("Final-tuned"), _spec("Triple-majority")]
        task = {
            "specs": [spec.to_dict() for spec in specs],
            "tier": "dev_param", "seed": 321, "n": 32,
            "rounds": 100, "field_reuse": "on",
        }
        chunked = _eval_field_task(task)
        self.assertTrue(chunked["field_reuse"])
        by_name = {row["candidate"]: row for row in chunked["results"]}
        for spec in specs:
            old = _eval_task({
                "spec": spec.to_dict(), "tier": "dev_param",
                "seed": 321, "n": 32, "rounds": 100,
            })
            self.assertEqual(by_name[spec.name]["scores"], old["scores"])
            self.assertEqual(by_name[spec.name]["family_sums"],
                             old["family_sums"])

    def test_test_tier_policy_opponents_match_rebuild(self) -> None:
        specs = [_spec("Final-tuned"), _spec("Harvest-tuned"),
                 _spec("GapStart"), _spec("Neural")]
        task = {
            "specs": [spec.to_dict() for spec in specs],
            "tier": TEST_TIER, "seed": 20261027, "n": 32,
            "rounds": 100, "field_reuse": "on",
        }
        chunked = _eval_field_task(task)
        by_name = {row["candidate"]: row for row in chunked["results"]}
        for spec in specs:
            old = _eval_task({
                "spec": spec.to_dict(), "tier": TEST_TIER,
                "seed": 20261027, "n": 32, "rounds": 100,
            })
            self.assertEqual(by_name[spec.name]["scores"], old["scores"],
                             spec.name)

    def test_field_reuse_off_rebuilds_and_matches(self) -> None:
        spec = _spec("Final-tuned")
        task = {
            "specs": [spec.to_dict(), spec.to_dict()],
            "tier": "dev_param", "seed": 654, "n": 32,
            "rounds": 100, "field_reuse": "off",
        }
        result = _eval_field_task(task)
        self.assertFalse(result["field_reuse"])
        self.assertEqual(result["results"][0]["scores"],
                         result["results"][1]["scores"])


class ProtocolCliSchedulerTests(unittest.TestCase):
    @staticmethod
    def _run(out: Path, workers: int, *extra: str) -> dict:
        code = protocol_main([
            "eval", "--candidates", "final-tuned,resp3-tuned,triple-majority",
            "--tiers", "dev_param", "--fields", "2", "--n", "256",
            "--workers", str(workers), "--candidate-chunk", "2",
            "--field-reuse", "on", "--seed", "123",
            "--out", str(out), *extra,
        ])
        if code != 0:
            raise AssertionError(f"protocol eval returned {code}")
        return json.loads(out.read_text(encoding="utf-8"))

    def test_worker_count_and_candidate_order_are_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            single = self._run(Path(tmp) / "single.json", 1)
            parallel = self._run(Path(tmp) / "parallel.json", 4)
        self.assertEqual(single["workers"], 1)
        self.assertEqual(parallel["workers"], 4)
        for name, tiers in single["results"].items():
            self.assertEqual(tiers["dev_param"]["values"],
                             parallel["results"][name]["dev_param"]["values"],
                             name)
            self.assertGreater(tiers["dev_param"]["mean"], 0.0)

    def test_profile_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eval.json"
            payload = self._run(out, 1, "--profile")
            profile_path = Path(payload["profile_path"])
            self.assertTrue(profile_path.exists())
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(profile["schema"], "ipd-protocol-profile-v1")
            self.assertEqual(profile["workers"], 1)
            self.assertEqual(payload["field_reuse"], "on")
            self.assertEqual(payload["candidate_chunk"], 2)
            self.assertGreater(payload["peak_rss_mb"], 0.0)
            self.assertIn("Final-tuned", profile["by_candidate"])
            self.assertTrue(profile["by_family"])
            for row in profile["by_candidate"].values():
                self.assertIn("eval_seconds", row)
                self.assertIn("candidate_build_seconds", row)
            self.assertEqual(profile["by_tier"]["dev_param"]["fields"], 2)

    def test_profile_accumulator_reports_family_cost(self) -> None:
        profile = ProfileAccumulator(
            workers=2, workers_requested="2", candidate_chunk=2,
            field_reuse="on")
        profile.add({
            "tier": "dev_param", "seed": 1, "field_reuse": True,
            "field_build_seconds": 2.0, "peak_rss_mb": 100.0,
            "results": [{
                "candidate": "A", "family_counts": {"nice": 2},
                "timings": {
                    "eval_seconds": 4.0,
                    "candidate_build_seconds": 0.5,
                    "family_seconds": {"nice": 3.0},
                },
            }],
        })
        payload = profile.payload(wall_seconds=8.0, peak_rss=101.0)
        self.assertEqual(payload["field_build_seconds"], 2.0)
        self.assertEqual(payload["by_candidate"]["A"]["eval_seconds"], 4.0)
        self.assertEqual(payload["by_family"]["nice"]["matches"], 2)
        self.assertEqual(payload["by_family"]["nice"]["seconds_per_match"],
                         1.5)


if __name__ == "__main__":
    unittest.main()
