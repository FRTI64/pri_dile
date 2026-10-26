"""Contract and smoke tests for the asymmetric self-play trainer."""
from __future__ import annotations

import contextlib
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ipd.core.opponents.basic import TFT
from ipd.policies.selfplay_gru import SelfPlayGRUPolicy
from ipd.policies.selfplay_gru_obs import features as std_features
from ipd.policies.selfplay_gru_obs import features_v2
from ipd.training.selfplay.config import smoke_config
from ipd.training.selfplay.env import IPDEnv, VectorEnv
from ipd.training.selfplay.models import export_weights, make_model
from ipd.training.selfplay.obs import N_FEATURES, features
from ipd.training.selfplay.opponents import (
    build_calibration_suite, build_validation_suite, sample_family_labels,
    sample_mixture, phase_weights,
)
from ipd.training.selfplay.red_pool import RedConstraint
from ipd.training.selfplay.rollout import Rollout


class ObservationContractTests(unittest.TestCase):
    def test_shape_and_padding(self):
        for t in (0, 1, 4, 99):
            f = features(t, [0] * t, [1] * t)
            self.assertEqual(len(f), N_FEATURES)
            self.assertTrue(all(math.isfinite(x) for x in f))
            self.assertAlmostEqual(f[0], t / 100)
            self.assertAlmostEqual(f[1], (100 - t) / 100)
        f0 = features(0, [], [])
        self.assertEqual(f0[2:6], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(f0[6:14], [-1.0] * 8)
        f1 = features(1, [0], [1])
        self.assertEqual(f1[2:6], [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(f1[6:8], [0.0, 1.0])
        self.assertEqual(f1[8:14], [-1.0] * 6)

    def test_training_and_inference_contract_match(self):
        rng = random.Random(4)
        mine: list[int] = []
        theirs: list[int] = []
        for t in range(100):
            self.assertEqual(features(t, mine, theirs),
                             std_features(t, mine, theirs))
            mine.append(rng.randrange(2))
            theirs.append(rng.randrange(2))

    def test_response_gap_separates_reciprocal_and_autonomous(self):
        t = 10
        mine = [0, 1] * (t // 2)
        # TFT responds to our previous move.
        tft = [0] + mine[:-1]
        gap_tft = features_v2(t, mine, tft)[17]
        # AllD/random-like opponent ignores our previous move.
        alld = [1] * t
        gap_alld = features_v2(t, mine, alld)[17]
        self.assertGreater(gap_tft, 0.9)
        self.assertAlmostEqual(gap_alld, 0.0)

    def test_endgame_flag(self):
        self.assertEqual(features(89, [0] * 89, [0] * 89)[20], 0.0)
        self.assertEqual(features(90, [0] * 90, [0] * 90)[20], 1.0)


class EnvironmentTests(unittest.TestCase):
    def test_vector_matches_single(self):
        seed = 12345
        actions = [(i * 7) % 2 for i in range(20)]

        single = IPDEnv()
        single.reset(lambda: TFT(random.Random(seed)))
        single_rows = []
        for a in actions:
            single_rows.append(single.observe())
            single.step(a)

        vec = VectorEnv([lambda: TFT(random.Random(seed))])
        vec_rows = []
        for a in actions:
            vec_rows.append(vec.observations()[0].tolist())
            vec.step([a])
        np.testing.assert_allclose(np.asarray(single_rows),
                                   np.asarray(vec_rows), atol=1e-6)
        self.assertEqual(single.own_score, vec.scores[0])

    def test_determinism(self):
        def run():
            env = VectorEnv([lambda: TFT(random.Random(9))])
            if hasattr(env.envs[0], "t"):
                pass
            rows = []
            for a in [0, 1] * 5:
                rows.append(env.observations().copy())
                env.step([a])
            return np.stack(rows), env.scores
        a, sa = run()
        b, sb = run()
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(sa, sb)

    def test_simultaneous_decisions(self):
        seen = []

        class Checker(TFT):
            def act(self, t, mine, theirs):
                seen.append((t, len(mine), len(theirs)))
                return super().act(t, mine, theirs)

        env = IPDEnv()
        env.reset(lambda: Checker(random.Random(0)))
        for t in range(5):
            env.step(t % 2)
        self.assertEqual(seen, [(t, t, t) for t in range(5)])


class ModelTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_step_matches_sequence(self):
        torch.manual_seed(0)
        model = make_model(None, device="cpu").eval()
        x = torch.randn(3, 11, N_FEATURES)
        logits, values, h = model.forward_sequence(x)
        h2 = model.init_hidden(3)
        outs = []
        vals = []
        for t in range(11):
            a, b, h2 = model.step(x[:, t], h2)
            outs.append(a)
            vals.append(b)
        self.assertLess(float((torch.stack(outs, 1) - logits).detach().abs().max()), 1e-5)
        self.assertLess(float((torch.stack(vals, 1) - values).detach().abs().max()), 1e-5)
        self.assertLess(float((h - h2).detach().abs().max()), 1e-5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_export_inference_matches_torch(self):
        torch.manual_seed(1)
        model = make_model(None, device="cpu").eval()
        policy = SelfPlayGRUPolicy(export_weights(model), mode="argmax")
        h = model.init_hidden(1)
        rng = random.Random(2)
        mine: list[int] = []
        theirs: list[int] = []
        for t in range(10):
            logit, value = policy.logit_value(t, mine, theirs)
            with torch.no_grad():
                x = torch.tensor(features(t, mine, theirs)).unsqueeze(0)
                ref_logit, ref_value, h = model.step(x, h)
            self.assertLess(abs(logit - float(ref_logit[0])), 1e-5)
            self.assertLess(abs(value - float(ref_value[0])), 1e-5)
            mine.append(rng.randrange(2))
            theirs.append(rng.randrange(2))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_gradient_isolation(self):
        from ipd.training.selfplay.models import clone_state
        from ipd.training.selfplay.ppo import PPOAgent

        cfg = smoke_config()
        cfg.ppo.epochs = 1
        blue = make_model(cfg, device="cpu")
        red = make_model(cfg, device="cpu")
        red_before = clone_state(red)
        agent = PPOAgent(blue, cfg.ppo, lr=1e-3, device="cpu")
        b, t, f = 2, 4, N_FEATURES
        roll = Rollout(
            obs=np.random.randn(b, t, f).astype("float32"),
            actions=np.random.randint(0, 2, (b, t)).astype("float32"),
            logprobs=np.full((b, t), -0.7, dtype="float32"),
            values=np.zeros((b, t), dtype="float32"),
            rewards=np.random.rand(b, t).astype("float32"),
            hidden=np.zeros((b, t, 2, 64), dtype="float32"),
        )
        roll.compute_gae()
        agent.update(roll)
        red_after = clone_state(red)
        for key in red_before:
            self.assertTrue(torch.equal(red_before[key], red_after[key]), key)


class RolloutTests(unittest.TestCase):
    def test_gae_reference(self):
        rewards = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        values = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        roll = Rollout(
            obs=np.zeros((1, 3, N_FEATURES), dtype=np.float32),
            actions=np.zeros((1, 3), dtype=np.float32),
            logprobs=np.zeros((1, 3), dtype=np.float32),
            values=values,
            rewards=rewards,
            hidden=np.zeros((1, 3, 2, 64), dtype=np.float32),
        ).compute_gae(gamma=1.0, lam=1.0)
        # delta = [0.5, 2.0, 3.0] -> advantage = [5.5, 5.0, 3.0]
        np.testing.assert_allclose(roll.advantages[0], [5.5, 4.5, 2.5])
        np.testing.assert_allclose(roll.returns[0], [6.0, 5.0, 3.0])


class RolloutConcatTests(unittest.TestCase):
    def test_concat_different_batch_sizes(self):
        from ipd.training.selfplay.rollout import concat_rollouts

        def make(b):
            return Rollout(
                obs=np.zeros((b, 3, N_FEATURES), dtype=np.float32),
                actions=np.zeros((b, 3), dtype=np.float32),
                logprobs=np.zeros((b, 3), dtype=np.float32),
                values=np.zeros((b, 3), dtype=np.float32),
                rewards=np.zeros((b, 3), dtype=np.float32),
                hidden=np.zeros((b, 3, 2, 4), dtype=np.float32),
            )
        out = concat_rollouts([make(2), make(3)])
        self.assertEqual(out.obs.shape, (5, 3, N_FEATURES))
        self.assertEqual(out.hidden.shape, (5, 3, 2, 4))


class PopulationTests(unittest.TestCase):
    def test_family_mixture_mean(self):
        rng = random.Random(0)
        prior = phase_weights("2")
        total = {k: 0.0 for k in prior}
        draws = 500
        for _ in range(draws):
            mix = sample_mixture(rng, prior)
            for k, v in mix.items():
                total[k] += v
        for k, p in prior.items():
            self.assertAlmostEqual(total[k] / draws, p, delta=0.06, msg=k)

    def test_labels_cover_all_families(self):
        labels = sample_family_labels(random.Random(1), "3", 5000)
        self.assertEqual(set(labels), set(phase_weights("3")))
        self.assertTrue(all(x in phase_weights("3") for x in labels))

    def test_validation_suite_is_held_out(self):
        names = {x.name for x in build_validation_suite()}
        self.assertIn("TFT", names)
        self.assertIn("Bully", names)
        self.assertIn("NoisyTFT.02", names)


class RedConstraintTests(unittest.TestCase):
    def test_no_penalty_before_window(self):
        rc = RedConstraint(alpha=0.85, lam=5.0, window_range=(15, 15))
        rewards, info = rc.episode_rewards([1.0] * 10, [1.0] * 10,
                                           random.Random(0))
        self.assertEqual(info["penalty"], 0.0)
        self.assertTrue(all(math.isfinite(x) for x in rewards))

    def test_penalty_formula(self):
        rc = RedConstraint(alpha=0.9, lam=10.0, window_range=(3, 3))
        red = [1.0] * 6
        target = [2.0] * 6
        rewards, info = rc.episode_rewards(red, target, random.Random(0))
        # red average 1, target 2 -> penalty 10*(0.9*2-1)^2 = 6.4
        self.assertAlmostEqual(rewards[-1], 1.0 - 2.0 - 6.4)
        self.assertAlmostEqual(info["penalty"], 6.4 * 4 / 6)

    def test_absolute_field_target(self):
        rc = RedConstraint(alpha=0.9, lam=10.0)
        rewards, info = rc.episode_rewards([1.0, 2.0], None,
                                           random.Random(0))
        self.assertEqual(rewards, [1.0, 2.0])
        self.assertEqual(info["violation"], 0.0)


class BCTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_bc_collect_and_train(self):
        from ipd.paths import weights_path
        from ipd.training.selfplay.bc import collect_episodes, train_bc

        cfg = smoke_config()
        cfg.rounds = 5
        cfg.envs = 2
        rows = collect_episodes(
            cfg, weights_path("rl_policy_checkpoint.json"), 4, seed=7)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["obs"].shape, (5, N_FEATURES))
        model, metrics = train_bc(cfg, rows, epochs=1, batch_size=2,
                                  lr=1e-3, device="cpu")
        self.assertEqual(len(metrics["history"]), 1)
        self.assertGreaterEqual(metrics["history"][0]["accuracy"], 0.0)
        self.assertTrue(np.isfinite(metrics["history"][0]["loss"]))


class SmokeRunTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_smoke_and_resume(self):
        from ipd.training.selfplay import train as train_mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            common = [
                "--smoke", "--device", "cpu", "--no-final-test",
                "--envs", "4", "--results-dir", str(tmp / "run"),
                "--out", str(tmp / "policy.json"),
                "--checkpoint-out", str(tmp / "checkpoint.pt"),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                code = train_mod.main(common + ["--episodes", "8"])
            self.assertEqual(code, 0)
            self.assertTrue((tmp / "policy.json").is_file())
            self.assertTrue((tmp / "checkpoint.pt").is_file())
            policy = json.loads((tmp / "policy.json").read_text())
            self.assertEqual(policy["format"], "ipd-selfplay-gru-v1")
            with contextlib.redirect_stdout(io.StringIO()):
                code = train_mod.main(common + [
                    "--episodes", "16", "--resume", str(tmp / "checkpoint.pt")])
            self.assertEqual(code, 0)
            hist = json.loads((tmp / "run" / "history.json").read_text())
            self.assertGreaterEqual(len(hist["history"]), 4)


if __name__ == "__main__":
    unittest.main()
