"""Tests for Protocol v3 split catalog, audit and usage ledger."""
from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from ipd.evaluation.distributions import (
    TRAIN_DISTS, build_dist_field, build_train_dist_field,
)
from ipd.evaluation.protocol_cli import main as protocol_main
from ipd.evaluation.protocol import (
    AXELROD_FAMILIES, COMPLEX_FAMILIES, DEV_ONLY_FAMILIES, TEST_ONLY_FAMILIES, TEST_TIER,
    TRAIN_PROFILES, audit_protocol, build_catalog, build_protocol_field,
    build_protocol_red_field, build_training_field, catalog_sha256,
    guard_test_usage, load_test_policy, load_usage, record_usage,
    test_consumption_allowed, verify_manifest, write_manifest,
)


class ProtocolCatalogTests(unittest.TestCase):
    def test_catalog_is_deterministic(self) -> None:
        first = build_catalog()
        second = build_catalog()
        self.assertEqual(catalog_sha256(first), catalog_sha256(second))
        self.assertGreater(len(first), 100)

    def test_catalog_keys_are_unique(self) -> None:
        keys = [entry.key for entry in build_catalog()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_audit_has_no_errors(self) -> None:
        report = audit_protocol()
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["errors"], [])

    def test_no_concrete_config_overlap(self) -> None:
        seen: dict[tuple[str, str], str] = {}
        for entry in build_catalog():
            if entry.anchor:
                continue
            signature = entry.signature()
            if signature in seen:
                self.assertEqual(
                    seen[signature], entry.tier,
                    f"{entry.key} is shared across {seen[signature]} and "
                    f"{entry.tier}")
            else:
                seen[signature] = entry.tier

    def test_test_only_families_are_held_out(self) -> None:
        catalog = build_catalog()
        train = {e.family for e in catalog if e.tier == "train" and not e.anchor}
        dev = {e.family for e in catalog
               if e.tier in ("dev", "dev_family") and not e.anchor}
        for family in TEST_ONLY_FAMILIES:
            self.assertNotIn(family, train)
            self.assertNotIn(family, dev)

    def test_dev_family_is_held_out(self) -> None:
        catalog = build_catalog()
        train = {e.family for e in catalog if e.tier == "train" and not e.anchor}
        for family in DEV_ONLY_FAMILIES:
            self.assertNotIn(family, train)

    def test_every_tier_has_entries(self) -> None:
        counts = Counter(entry.tier for entry in build_catalog())
        for tier in ("train", "train_complex", "train_policy", "train_axelrod",
                     "dev", "dev_family", "dev_complex", "dev_axelrod",
                     "test", "test_complex", "test_policy", "test_axelrod",
                     "test_only"):
            self.assertGreaterEqual(counts[tier], 3, tier)

    def test_complex_families_are_split(self) -> None:
        catalog = build_catalog()
        for family in COMPLEX_FAMILIES:
            tiers = {entry.tier for entry in catalog
                     if entry.family == family}
            self.assertIn("train_complex", tiers, family)
            self.assertIn("dev_complex", tiers, family)
            self.assertIn("test_complex", tiers, family)

    def test_axelrod_families_are_split(self) -> None:
        catalog = build_catalog()
        for family in AXELROD_FAMILIES:
            tiers = {entry.tier for entry in catalog
                     if entry.family == family}
            self.assertIn("train_axelrod", tiers, family)
            self.assertIn("dev_axelrod", tiers, family)
            self.assertIn("test_axelrod", tiers, family)


class ProtocolFieldTests(unittest.TestCase):
    def test_field_size_and_determinism(self) -> None:
        first = build_protocol_field("dev_param", 12345, 64)
        second = build_protocol_field("dev_param", 12345, 64)
        self.assertEqual(len(first), 64)
        self.assertEqual([b.name for b in first], [b.name for b in second])

    def test_test_v7_contains_all_test_only_families(self) -> None:
        field = build_protocol_field(TEST_TIER, 777, 8000)
        names = Counter(bot.name.split(":", 1)[0] for bot in field)
        for family in TEST_ONLY_FAMILIES:
            self.assertGreater(names[family], 0, family)

    def test_test_v7_mixture_weights(self) -> None:
        field = build_protocol_field(TEST_TIER, 20261027, 8000)
        names = Counter(bot.name.split(":", 1)[0] for bot in field)
        total = sum(names.values())
        anchor_share = names["anchor"] / total
        test_only_share = sum(names[f] for f in TEST_ONLY_FAMILIES) / total
        complex_share = sum(names[f] for f in COMPLEX_FAMILIES) / total
        policy_share = names["policy"] / total
        axelrod_share = sum(names[f] for f in AXELROD_FAMILIES) / total
        self.assertAlmostEqual(anchor_share, 0.10, delta=0.03)
        self.assertAlmostEqual(test_only_share, 0.20, delta=0.04)
        self.assertAlmostEqual(complex_share, 0.15, delta=0.04)
        self.assertAlmostEqual(policy_share, 0.10, delta=0.04)
        self.assertAlmostEqual(axelrod_share, 0.15, delta=0.04)

    def test_dev_family_contains_only_heldout_families(self) -> None:
        field = build_protocol_field("dev_family", 20261027, 1000)
        families = {bot.name.split(":", 1)[0] for bot in field}
        self.assertTrue(families <= set(DEV_ONLY_FAMILIES) | {"anchor"})

    def test_dev_complex_contains_complex_families(self) -> None:
        field = build_protocol_field("dev_complex", 20261027, 1000)
        families = {bot.name.split(":", 1)[0] for bot in field}
        self.assertTrue(families <= set(COMPLEX_FAMILIES) | {"anchor"})

    def test_train_profiles_are_training_distributions(self) -> None:
        for profile in TRAIN_PROFILES:
            self.assertIn(profile, TRAIN_DISTS)
            field = build_train_dist_field(profile, 12345, 50)
            self.assertEqual(len(field), 50)
            field2 = build_training_field(profile, 12345, 50)
            self.assertEqual([b.name for b in field], [b.name for b in field2])

    def test_protocol_validation_distributions(self) -> None:
        for dist in ("dev_param", "dev_family", "dev_complex",
                     "dev_axelrod", "dev_ood"):
            field = build_train_dist_field(dist, 12345, 50)
            self.assertEqual(len(field), 50)
            self.assertEqual(len(build_dist_field(dist, 12345, 50)), 50)

    def test_red_pools_are_separate(self) -> None:
        dev = build_protocol_red_field("dev_red", 123, 2)
        test = build_protocol_red_field("test_red", 123, 2)
        self.assertEqual(len(dev), 2)
        self.assertEqual(len(test), 2)
        self.assertNotEqual(dev[0].name.split(":", 1)[0],
                            test[0].name.split(":", 1)[0])


class ProtocolCliTests(unittest.TestCase):
    def test_eval_cli_writes_dev_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eval.json"
            code = protocol_main([
                "eval", "--candidates", "final-tuned",
                "--tiers", "dev_param",
                "--fields", "1", "--n", "8", "--workers", "1",
                "--seed", "123", "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "ipd-protocol-eval-v1")
            self.assertIn("Final-tuned", payload["results"])
            self.assertEqual(payload["fields"], 1)


class ProtocolNoConsumeCliTests(unittest.TestCase):
    def test_test_tier_no_consume_is_provisional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.json"
            code = protocol_main([
                "eval", "--candidates", "final-tuned",
                "--tiers", TEST_TIER, "--fields", "1", "--n", "8",
                "--workers", "1", "--seed", "123", "--no-consume",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(payload["usage_consumed"])
            self.assertFalse(payload["selection_eligible"])
            self.assertTrue(payload["no_consume"])


class ProtocolTestPolicyTests(unittest.TestCase):
    def test_missing_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_policy.json"
            policy = load_test_policy(path)
            self.assertFalse(policy["consume_test"])
            self.assertFalse(test_consumption_allowed(path))
            self.assertIn("locked", policy["reason"])

    def test_only_policy_can_unlock_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_policy.json"
            path.write_text(json.dumps({"consume_test": False}),
                            encoding="utf-8")
            self.assertFalse(test_consumption_allowed(path))
            path.write_text(json.dumps({"consume_test": True}),
                            encoding="utf-8")
            self.assertTrue(test_consumption_allowed(path))
            path.write_text(json.dumps({"consume_test": "true"}),
                            encoding="utf-8")
            self.assertFalse(test_consumption_allowed(path))
            path.write_text("{not json", encoding="utf-8")
            self.assertFalse(test_consumption_allowed(path))

    def test_repository_policy_locks_consumption(self) -> None:
        self.assertFalse(load_test_policy()["consume_test"])
        self.assertFalse(test_consumption_allowed())


class ProtocolConsumptionLockCliTests(unittest.TestCase):
    @patch("ipd.evaluation.protocol_cli.test_consumption_allowed",
           return_value=False)
    def test_test_tier_still_requires_explicit_flag(self, _allowed) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.json"
            with self.assertRaises(SystemExit):
                protocol_main([
                    "eval", "--candidates", "final-tuned",
                    "--tiers", TEST_TIER, "--fields", "1", "--n", "8",
                    "--workers", "1", "--seed", "123", "--out", str(out),
                ])
            self.assertFalse(out.exists())

    @patch("ipd.evaluation.protocol_cli.test_consumption_allowed",
           return_value=False)
    def test_acknowledge_once_cannot_unlock_locked_test(
            self, _allowed) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.json"
            with self.assertRaises(SystemExit):
                protocol_main([
                    "eval", "--candidates", "final-tuned",
                    "--tiers", TEST_TIER, "--fields", "1", "--n", "8",
                    "--workers", "1", "--seed", "123", "--acknowledge-once",
                    "--out", str(out),
                ])
            self.assertFalse(out.exists())

    @patch("ipd.evaluation.protocol_cli.test_consumption_allowed",
           return_value=False)
    @patch("ipd.evaluation.protocol_cli.record_usage")
    @patch("ipd.evaluation.protocol_cli.guard_test_usage")
    def test_test_tier_is_provisional_with_no_consume(
            self, guard, record, _allowed) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.json"
            code = protocol_main([
                "eval", "--candidates", "final-tuned",
                "--tiers", TEST_TIER, "--fields", "1", "--n", "8",
                "--workers", "1", "--seed", "123", "--no-consume",
                "--acknowledge-once", "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(payload["usage_consumed"])
            self.assertFalse(payload["selection_eligible"])
            self.assertTrue(payload["no_consume"])
            self.assertTrue(payload["test_consumption_locked"])
            guard.assert_not_called()
            record.assert_not_called()

    @patch("ipd.evaluation.protocol_cli.test_consumption_allowed",
           return_value=True)
    @patch("ipd.evaluation.protocol_cli.record_usage")
    @patch("ipd.evaluation.protocol_cli.guard_test_usage")
    def test_unlocked_policy_can_consume(
            self, guard, record, _allowed) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.json"
            code = protocol_main([
                "eval", "--candidates", "final-tuned",
                "--tiers", TEST_TIER, "--fields", "1", "--n", "8",
                "--workers", "1", "--seed", "123", "--acknowledge-once",
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["usage_consumed"])
            self.assertTrue(payload["selection_eligible"])
            self.assertFalse(payload["no_consume"])
            self.assertFalse(payload["test_consumption_locked"])
            guard.assert_called_once()
            record.assert_called_once()

    @patch("ipd.evaluation.protocol_cli.test_consumption_allowed",
           return_value=False)
    @patch("ipd.evaluation.protocol_cli.record_usage")
    @patch("ipd.evaluation.protocol_cli.guard_test_usage")
    def test_red_test_tier_is_provisional_with_no_consume(
            self, guard, record, _allowed) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "red.json"
            code = protocol_main([
                "red", "--candidates", "final-tuned", "--tier", "test_red",
                "--seeds", "1", "--workers", "1", "--seed", "123",
                "--no-consume", "--acknowledge-once", "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(payload["usage_consumed"])
            self.assertFalse(payload["selection_eligible"])
            self.assertTrue(payload["no_consume"])
            self.assertTrue(payload["test_consumption_locked"])
            guard.assert_not_called()
            record.assert_not_called()


class ProtocolRedCliTests(unittest.TestCase):
    def test_red_cli_writes_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "red.json"
            code = protocol_main([
                "red", "--candidates", "final-tuned",
                "--tier", "dev_red", "--seeds", "1", "--workers", "1",
                "--seed", "123", "--out", str(out),
            ])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "ipd-protocol-red-eval-v1")
            self.assertIn("Final-tuned", payload["results"])


class ProtocolManifestTests(unittest.TestCase):
    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_manifest(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "ipd-protocol-v4")
            self.assertEqual(data["catalog_sha256"], catalog_sha256())
            self.assertTrue(verify_manifest(path)["ok"])
            data["catalog_sha256"] = "broken"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertFalse(verify_manifest(path)["ok"])


class ProtocolUsageTests(unittest.TestCase):
    def test_usage_guard_and_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            guard_test_usage("candidate-a", path=path, tier=TEST_TIER)
            record_usage(TEST_TIER, "candidate-a", "result.json", path=path,
                         allow_locked=True)
            self.assertEqual(len(load_usage(path)["entries"]), 1)
            with self.assertRaises(RuntimeError):
                guard_test_usage("candidate-a", path=path, tier=TEST_TIER)
            guard_test_usage("candidate-a", path=path, force=True,
                             tier=TEST_TIER)
            record_usage(TEST_TIER, "candidate-a", "result-2.json",
                         path=path, force=True, allow_locked=True)
            entries = load_usage(path)["entries"]
            self.assertEqual(len(entries), 2)
            self.assertTrue(entries[-1]["contaminated"])

    def test_different_candidate_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            record_usage(TEST_TIER, "candidate-a", "result.json", path=path,
                         allow_locked=True)
            guard_test_usage("candidate-b", path=path, tier=TEST_TIER)

    def test_record_usage_refuses_when_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            with patch("ipd.evaluation.protocol.test_consumption_allowed",
                       return_value=False):
                with self.assertRaises(RuntimeError):
                    record_usage(TEST_TIER, "candidate-a", "result.json",
                                 path=path)
            self.assertFalse(path.exists())

    def test_record_usage_allowed_when_unlocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.json"
            with patch("ipd.evaluation.protocol.test_consumption_allowed",
                       return_value=True):
                record_usage(TEST_TIER, "candidate-a", "result.json",
                             path=path)
            self.assertTrue(path.exists())
            self.assertEqual(len(load_usage(path)["entries"]), 1)


if __name__ == "__main__":
    unittest.main()
