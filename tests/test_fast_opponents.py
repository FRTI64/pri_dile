"""Plan 3 fast-rule parity, fallback, and integration tests."""
from __future__ import annotations

import random
import unittest

import numpy as np

from ipd.core.opponents.basic import TFT
from ipd.core.opponents.memory import RandomFSM
from ipd.evaluation.fast.bank import RuleBank
from ipd.evaluation.fast.native import available, build_error
from ipd.evaluation.fast.spec import (
    TYPE_IDS, FastRuleSpec, attach_fast_spec, encode_spec, get_fast_spec,
)
from ipd.evaluation.protocol import build_catalog, build_protocol_field
from ipd.training.selfplay.env import VectorEnv
from ipd.training.selfplay.opponents import make_training_bot

NATIVE = available()


def _run_python(bot, blue: list[int]) -> list[int]:
    mine: list[int] = []
    theirs: list[int] = []
    out: list[int] = []
    for t, reply in enumerate(blue):
        action = bot.act(t, mine, theirs)
        mine.append(action)
        theirs.append(int(reply))
        out.append(action)
    return out


@unittest.skipUnless(NATIVE, f"native rules unavailable: {build_error()}")
class SpecEncodingTests(unittest.TestCase):
    def test_type_count_matches_native(self) -> None:
        from ipd.evaluation.fast.native import load

        self.assertEqual(load().rb_type_count(), len(TYPE_IDS))

    def test_all_rule_configs_encode(self) -> None:
        for entry in build_catalog():
            if entry.family in ("axelrod", "axelrod_extra", "axelrod_mix",
                                "policy"):
                continue
            bot = entry.build(random.Random(7))
            spec = get_fast_spec(bot)
            if spec is None:
                continue
            encoded = encode_spec(spec)
            self.assertEqual(len(encoded["ip"]), 16)
            self.assertEqual(len(encoded["dp"]), 16)

    def test_unsupported_objects_have_no_spec(self) -> None:
        fsm = RandomFSM(random.Random(1), k=3)
        self.assertIsNone(get_fast_spec(fsm))
        bot = TFT(random.Random(1))
        self.assertIsNone(get_fast_spec(bot))


@unittest.skipUnless(NATIVE, f"native rules unavailable: {build_error()}")
class ParityTests(unittest.TestCase):
    def _check(self, bots, blue_matrix) -> None:
        bank = RuleBank(bots, rounds=len(blue_matrix[0]))
        self.assertTrue(bank.enabled)
        n = len(bots)
        native = [
            [0] * len(blue_matrix[0]) for _ in range(n)
        ]
        for t in range(len(blue_matrix[0])):
            actions = bank.act(t, blue_matrix[:, t])
            for i, value in enumerate(actions):
                native[i][t] = int(value)
            bank.observe(t, blue_matrix[:, t], actions)
        for i, bot in enumerate(bots):
            expected = _run_python(bot, list(blue_matrix[i]))
            self.assertEqual(native[i], expected,
                             msg=f"row {i}: {getattr(bot, 'name', bot)}")
        bank.close()

    def test_every_catalog_config_matches_python(self) -> None:
        rng = np.random.default_rng(20261028)
        blue = rng.integers(0, 2, size=(1, 100), dtype=np.int8)
        checked = 0
        for entry in build_catalog():
            if entry.family in ("axelrod", "axelrod_extra", "axelrod_mix",
                                "policy"):
                continue
            bot = entry.build(random.Random(12345))
            if get_fast_spec(bot) is None:
                continue
            bank = RuleBank([bot], rounds=100)
            native = []
            for t in range(100):
                action = int(bank.act(t, blue[:, t])[0])
                native.append(action)
                bank.observe(t, blue[:, t], [action])
            expected = _run_python(bot, list(blue[0]))
            self.assertEqual(native, expected, msg=entry.key)
            bank.close()
            checked += 1
        self.assertGreater(checked, 700)

    def test_stochastic_rows_match_exactly(self) -> None:
        # The native bank captures each Python RNG state and mirrors CPython's
        # MT19937 draw order, so stochastic rows are bit-identical, not merely
        # distributionally similar.
        entries = [e for e in build_catalog()
                   if e.type_name in ("NoisyTFT", "Joss", "GenerousTFT",
                                      "ExtortZD", "MemoryOne", "Reactive")
                   and e.family not in ("axelrod", "axelrod_extra",
                                        "axelrod_mix", "policy")]
        bots = [e.build(random.Random(1000 + i))
                for i, e in enumerate(entries[:24])]
        blue = np.random.default_rng(7).integers(
            0, 2, size=(len(bots), 100), dtype=np.int8)
        self._check(bots, blue)

    def test_voting_portfolio_is_reproducible(self) -> None:
        from ipd.evaluation.protocol import VotingPortfolio

        members = ("tft", "harvest", "adaptive")
        blue = [random.Random(5).randrange(2) for _ in range(100)]

        def run():
            bot = VotingPortfolio(members, "C", random.Random(7))
            mine: list[int] = []
            theirs: list[int] = []
            out: list[int] = []
            for t, reply in enumerate(blue):
                action = bot.act(t, mine, theirs)
                mine.append(action)
                theirs.append(reply)
                out.append(action)
            return out

        self.assertEqual(run(), run())

    def test_mixed_batch_matches_python(self) -> None:
        seeds = [11, 12, 13, 14, 15, 16]
        bots = [
            TFT(random.Random(seeds[0])),
            make_training_bot(random.Random(seeds[1]), "noisy"),
            make_training_bot(random.Random(seeds[2]), "extort"),
            make_training_bot(random.Random(seeds[3]), "special"),
            make_training_bot(random.Random(seeds[4]), "random"),
            make_training_bot(random.Random(seeds[5]), "good"),
        ]
        blue = np.random.default_rng(9).integers(0, 2, size=(len(bots), 100),
                                                 dtype=np.int8)
        self._check(bots, blue)


@unittest.skipUnless(NATIVE, f"native rules unavailable: {build_error()}")
class FallbackTests(unittest.TestCase):
    def test_irregular_objects_stay_on_python(self) -> None:
        fsm = RandomFSM(random.Random(3), k=4)
        fsm.name = "fsm"
        bot = TFT(random.Random(4))
        bot.name = "tft"
        bank = RuleBank([fsm, bot], rounds=100)
        self.assertTrue(bank.enabled)
        self.assertEqual(bank.fast_count, 1)
        self.assertEqual(bank.fallback_count, 1)
        blue = [random.Random(5).randrange(2) for _ in range(100)]
        native = []
        for t, reply in enumerate(blue):
            actions = bank.act(t, [reply, reply])
            native.append(actions[1])
            bank.observe(t, [reply, reply], actions)
        self.assertEqual(native[0], 0)
        self.assertEqual(native[:10], _run_python(bot, blue)[:10])
        bank.close()

    def test_reset_restores_initial_rng(self) -> None:
        bot = TFT(random.Random(8))
        bank = RuleBank([bot], rounds=100)
        blue = [0, 1] * 50
        first = []
        for t, reply in enumerate(blue):
            action = int(bank.act(t, [reply])[0])
            first.append(action)
            bank.observe(t, [reply], [action])
        bank.reset()
        second = []
        for t, reply in enumerate(blue):
            action = int(bank.act(t, [reply])[0])
            second.append(action)
            bank.observe(t, [reply], [action])
        self.assertEqual(first, second)
        bank.close()


@unittest.skipUnless(NATIVE, f"native rules unavailable: {build_error()}")
class TrainingIntegrationTests(unittest.TestCase):
    def test_vector_env_fast_matches_python(self) -> None:
        rng = random.Random(42)
        families = ["good", "noisy", "extort", "random", "special"] * 26
        seeds = [rng.randrange(1 << 30) for _ in families]
        # B=130 triggers the fast path (min_envs=128).
        families = families[:130]
        seeds = seeds[:130]
        actions = np.random.default_rng(3).integers(
            0, 2, size=(len(families), 100), dtype=np.int8)

        def factories():
            return [lambda s=s, f=f: make_training_bot(random.Random(s), f)
                    for s, f in zip(seeds, families)]

        fast = VectorEnv(factories(), families, fast_rules=True,
                         fast_rules_min_envs=128)
        self.assertTrue(fast.fast_enabled)
        slow = VectorEnv(factories(), families, fast_rules=False)
        for t in range(100):
            fast.step([int(x) for x in actions[:, t]])
            slow.step([int(x) for x in actions[:, t]])
        self.assertEqual(fast.scores, slow.scores)
        np.testing.assert_allclose(fast.observations(), slow.observations())

    def test_small_batch_keeps_python_path(self) -> None:
        env = VectorEnv([lambda: TFT(random.Random(1))],
                        fast_rules=True, fast_rules_min_envs=128)
        self.assertFalse(env.fast_enabled)


class GpuAdapterTests(unittest.TestCase):
    def test_auto_declines_mixed_field_and_accepts_rule_field(self) -> None:
        from ipd.evaluation.gpu.policy_opponents import BatchOpponentField
        import torch

        rule_field = build_protocol_field("dev_complex", 5, 32)
        adapter = BatchOpponentField(
            rule_field, list(range(len(rule_field))),
            device=torch.device("cpu"), dtype=torch.float32,
            rounds=100, fast_rules="auto")
        if NATIVE:
            self.assertIsNotNone(adapter.rules)
        # A synthetic mixed field (one fast rule, one irregular rule) is below
        # the 90% auto threshold.
        mixed = [TFT(random.Random(1)), RandomFSM(random.Random(2), k=3)]
        adapter2 = BatchOpponentField(
            mixed, [0, 1], device=torch.device("cpu"),
            dtype=torch.float32, rounds=100, fast_rules="auto")
        self.assertIsNone(adapter2.rules)


if __name__ == "__main__":
    unittest.main()
