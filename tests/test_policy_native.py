"""Bit-exact parity for the optional native CPU policy inference."""
from __future__ import annotations

import random
import unittest

from ipd.policies import native
from ipd.policies.gru import GRUPolicy
from ipd.policies.neural import NeuralPolicy
from ipd.policies.registry import candidate_specs
from ipd.policies.rl import RLPolicy
from ipd.policies.selfplay_gru import SelfPlayGRUPolicy

NATIVE = native.available()


@unittest.skipUnless(NATIVE, f"native policy ops unavailable: {native.build_error()}")
class NativePolicyParityTests(unittest.TestCase):
    def test_mlp_logits_match_python(self):
        rng = random.Random(7)
        for cls, path in (
            (NeuralPolicy, "weights/retrain_20261025/neural_policy_weights.json"),
            (RLPolicy, "weights/rl_policy_checkpoint.json"),
        ):
            fast = cls(path, mode="argmax")
            slow = cls(path, mode="argmax")
            self.assertIsNotNone(fast._native)
            slow._native = None
            for _ in range(200):
                features = [rng.uniform(-5.0, 5.0)
                            for _ in range(fast.n_features)]
                self.assertEqual(fast._logit(features),
                                 slow._logit(features))

    def test_gru_step_and_hidden_match_python(self):
        rng = random.Random(8)
        fast = GRUPolicy("weights/gru_policy_weights.json", mode="sample")
        slow = GRUPolicy("weights/gru_policy_weights.json", mode="sample")
        self.assertIsNotNone(fast._native)
        slow._native = None
        fast.reset()
        slow.reset()
        mine: list[int] = []
        theirs: list[int] = []
        for t in range(100):
            self.assertEqual(fast._step(mine, theirs),
                             slow._step(mine, theirs))
            self.assertEqual(fast.hidden_state, slow.hidden_state)
            mine.append(0)
            theirs.append(rng.randrange(2))

    def test_selfplay_forward_matches_python(self):
        rng = random.Random(9)
        fast = SelfPlayGRUPolicy("weights/selfplay_gru_policy.json",
                                 mode="sample")
        slow = SelfPlayGRUPolicy("weights/selfplay_gru_policy.json",
                                 mode="sample")
        self.assertIsNotNone(fast._native)
        slow._native = None
        fast.reset()
        slow.reset()
        for _ in range(100):
            features = [rng.uniform(-1.0, 1.0)
                        for _ in range(fast.n_features)]
            self.assertEqual(fast._forward(features),
                             slow._forward(features))
            self.assertEqual(fast.h, slow.h)

    def test_sample_actions_and_rng_match_python(self):
        rng = random.Random(10)
        for name in ("adv-mlp", "adv-gru", "adv-rl"):
            spec = candidate_specs([name], mode="sample")[0]
            fast = spec.build()
            slow = spec.build()
            for policy in (fast, slow):
                policy.reset()
            for obj in (slow, getattr(slow, "base", None)):
                if obj is not None and hasattr(obj, "_native"):
                    obj._native = None
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(100):
                a = fast.act(t, mine, theirs)
                b = slow.act(t, mine, theirs)
                self.assertEqual(a, b, f"{name} t={t}")
                mine.append(a)
                theirs.append(rng.randrange(2))
            self.assertEqual(fast.base.rng.getstate(),
                             slow.base.rng.getstate())


if __name__ == "__main__":
    unittest.main()
