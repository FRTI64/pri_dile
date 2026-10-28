"""Plan 4 contract tests: GPU environment and GPU rollout storage."""
from __future__ import annotations

import random
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ipd.core.opponents.basic import AllC, Grim, TFT
from ipd.policies.rl import RLPolicy
from ipd.policies.selfplay_gru_obs import features as reference_features
from ipd.training.selfplay import N_FEATURES
from ipd.training.selfplay.config import TrainConfig
from ipd.training.selfplay.env import VectorEnv
from ipd.training.selfplay.gpu_env import GPUEnv
from ipd.training.selfplay.models import (
    TorchOpponent, make_model, snapshot,
)
from ipd.training.selfplay.opponents import make_training_bot
from ipd.training.selfplay.rollout import Rollout, TensorRollout


def _rule_factories() -> list:
    return [
        lambda: TFT(random.Random(1)),
        lambda: Grim(random.Random(2)),
        lambda: AllC(random.Random(3)),
    ]


class GPUEnvParityTests(unittest.TestCase):
    def test_rules_match_reference(self):
        actions = [(i * 7) % 2 for i in range(30)]
        cpu = VectorEnv(_rule_factories(), ["good"] * 3)
        gpu = GPUEnv(_rule_factories(), ["good"] * 3, device="cpu",
                     fast_rules=True, mode="argmax")
        for t, action in enumerate(actions):
            np.testing.assert_allclose(cpu.observations(), gpu.observations(),
                                       atol=1e-6, err_msg=str(t))
            own_c, other_c, done_c = cpu.step([action] * 3)
            own_g, other_g, done_g = gpu.step([action] * 3)
            np.testing.assert_array_equal(own_c, own_g)
            np.testing.assert_array_equal(other_c, other_g)
            np.testing.assert_array_equal(done_c, done_g)
        self.assertEqual(cpu.scores, gpu.scores)
        self.assertTrue(gpu.fast_enabled)

    def test_learned_policy_argmax_matches_reference(self):
        from ipd.paths import weights_path

        with open(weights_path("rl_policy_checkpoint.json"),
                  encoding="utf-8") as fh:
            import json

            weights = json.load(fh)

        def factories():
            return [lambda: TFT(random.Random(4)),
                    lambda: RLPolicy(weights, mode="argmax",
                                     seed=11, name="RL-final")]

        cpu = VectorEnv(factories(), ["good", "top"])
        gpu = GPUEnv(factories(), ["good", "top"], device="cpu",
                     mode="argmax")
        for t in range(100):
            np.testing.assert_allclose(cpu.observations(), gpu.observations(),
                                       atol=1e-6, err_msg=str(t))
            own_c, other_c, _ = cpu.step([t % 2, t % 2])
            own_g, other_g, _ = gpu.step([t % 2, t % 2])
            np.testing.assert_array_equal(own_c, own_g)
            np.testing.assert_array_equal(other_c, other_g)
        self.assertEqual(cpu.scores, gpu.scores)

    def test_torch_snapshot_argmax_matches_reference(self):
        cfg = TrainConfig()
        state = snapshot(make_model(cfg, device="cpu"))

        def factories():
            return [
                lambda: TorchOpponent(state, config=cfg, device="cpu",
                                      mode="argmax", seed=1),
                lambda: TorchOpponent(state, config=cfg, device="cpu",
                                      mode="argmax", seed=1),
            ]

        cpu = VectorEnv(factories(), ["red", "red"])
        gpu = GPUEnv(factories(), ["red", "red"], device="cpu",
                     mode="argmax")
        for t in range(100):
            np.testing.assert_allclose(cpu.observations(), gpu.observations(),
                                       atol=1e-6, err_msg=str(t))
            own_c, other_c, _ = cpu.step([t % 2, (t + 1) % 2])
            own_g, other_g, _ = gpu.step([t % 2, (t + 1) % 2])
            np.testing.assert_array_equal(own_c, own_g)
            np.testing.assert_array_equal(other_c, other_g)
        self.assertEqual(cpu.scores, gpu.scores)

    def test_mixed_training_population_matches_reference(self):
        rng = random.Random(9)
        # Rules and random rules only: learned opponents sample from their own
        # RNG streams, so exact parity is intentionally out of scope here.
        pool = ("good", "noisy", "extort", "random", "special")
        families = [rng.choice(pool) for _ in range(32)]
        seeds = [rng.randrange(1 << 30) for _ in range(len(families))]

        def factories():
            return [
                (lambda s=seed, f=family: (
                    lambda: make_training_bot(random.Random(s), f)))()
                for seed, family in zip(seeds, families)
            ]

        cpu = VectorEnv(factories(), families)
        gpu = GPUEnv(factories(), families, device="cpu", mode="argmax")
        for t in range(100):
            np.testing.assert_allclose(cpu.observations(), gpu.observations(),
                                       atol=1e-6, err_msg=str(t))
            own_c, other_c, _ = cpu.step([t % 2] * len(families))
            own_g, other_g, _ = gpu.step([t % 2] * len(families))
            np.testing.assert_array_equal(own_c, own_g)
            np.testing.assert_array_equal(other_c, other_g)
        self.assertEqual(cpu.scores, gpu.scores)

    def test_auto_reset_and_scores(self):
        env = GPUEnv(_rule_factories(), ["good"] * 3, device="cpu",
                     mode="argmax", auto_reset=True)
        total = 0
        for t in range(100):
            _own, _other, _done = env.step_tensor(
                torch.full((3,), t % 2, dtype=torch.int8))
            total = t + 1
        self.assertTrue(env.done)
        self.assertEqual(env.t, 100)
        # The completed scores survive until the next call starts a new
        # episode; auto_reset happens at the beginning of that call.
        completed = list(env.scores)
        env.step_tensor(torch.zeros(3, dtype=torch.int8))
        self.assertEqual(env.t, 1)
        self.assertEqual(env.scores, [6, 6, 6])  # fresh CC round
        self.assertNotEqual(completed, env.scores)

    def test_observations_are_incremental_21_features(self):
        env = GPUEnv([lambda: AllC(random.Random(0))],
                     ["good"], device="cpu", mode="argmax")
        mine: list[int] = []
        theirs: list[int] = []
        for t in range(100):
            expected = np.asarray(
                [reference_features(t, mine, theirs)], dtype=np.float32)
            np.testing.assert_allclose(env.observations(), expected,
                                       atol=1e-6, err_msg=str(t))
            env.step([t % 2])
            mine.append(t % 2)
            theirs.append(0)  # AllC always cooperates.


class TensorRolloutTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_gae_matches_numpy(self):
        rng = np.random.default_rng(3)
        rewards = rng.random((4, 100)).astype(np.float32)
        values = rng.random((4, 100)).astype(np.float32)
        cpu = Rollout(
            obs=np.zeros((4, 100, N_FEATURES), dtype=np.float32),
            actions=np.zeros((4, 100), dtype=np.float32),
            logprobs=np.zeros((4, 100), dtype=np.float32),
            values=values,
            rewards=rewards,
            hidden=np.zeros((4, 100, 2, 64), dtype=np.float32),
        ).compute_gae()
        gpu = TensorRollout(
            obs=torch.zeros(4, 100, N_FEATURES),
            actions=torch.zeros(4, 100),
            logprobs=torch.zeros(4, 100),
            values=torch.as_tensor(values),
            rewards=torch.as_tensor(rewards),
            hidden=torch.zeros(4, 100, 2, 64),
        ).compute_gae()
        np.testing.assert_allclose(cpu.advantages,
                                   gpu.advantages.numpy(), atol=1e-5)
        np.testing.assert_allclose(cpu.returns, gpu.returns.numpy(),
                                   atol=1e-5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_update_accepts_tensor_rollout(self):
        from ipd.training.selfplay.ppo import PPOAgent

        cfg = TrainConfig()
        cfg.ppo.epochs = 1
        cfg.ppo.batch_episodes = 2
        model = make_model(cfg, device="cpu")
        agent = PPOAgent(model, cfg.ppo, lr=1e-3, device="cpu")
        b, t = 4, 8
        rollout = TensorRollout(
            obs=torch.randn(b, t, N_FEATURES),
            actions=torch.randint(0, 2, (b, t)).float(),
            logprobs=torch.full((b, t), -0.7),
            values=torch.zeros(b, t),
            rewards=torch.rand(b, t),
            hidden=torch.zeros(b, t, 2, 64),
        ).compute_gae()
        stats = agent.update(rollout)
        self.assertEqual(stats.n_steps, b * t)
        self.assertTrue(np.isfinite(stats.policy_loss))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_concat_tensor_rollouts(self):
        from ipd.training.selfplay.rollout import concat_rollouts

        def make(b: int) -> TensorRollout:
            return TensorRollout(
                obs=torch.zeros(b, 3, N_FEATURES),
                actions=torch.zeros(b, 3),
                logprobs=torch.zeros(b, 3),
                values=torch.zeros(b, 3),
                rewards=torch.zeros(b, 3),
                hidden=torch.zeros(b, 3, 2, 4),
                opponent_rewards=torch.zeros(b, 3),
                families=["good"] * b,
            )

        out = concat_rollouts([make(2), make(3)])
        self.assertIsInstance(out, TensorRollout)
        self.assertEqual(tuple(out.obs.shape), (5, 3, N_FEATURES))
        self.assertEqual(tuple(out.opponent_rewards.shape), (5, 3))
        self.assertEqual(len(out.families), 5)


class GPUSmokeTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_trainer_gpu_env_smoke(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path

        from ipd.training.selfplay import train as train_mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = train_mod.main([
                    "--smoke", "--device", "cuda", "--gpu-env", "on",
                    "--no-final-test", "--envs", "8", "--episodes", "16",
                    "--results-dir", str(tmp / "run"),
                    "--out", str(tmp / "policy.json"),
                    "--checkpoint-out", str(tmp / "checkpoint.pt"),
                ])
            self.assertEqual(code, 0)
            history = json.loads(
                (tmp / "run" / "history.json").read_text(encoding="utf-8"))
            self.assertTrue(all("timing" in row
                                for row in history["history"]))

    @unittest.skipIf(torch is None, "torch not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_red_rollout_with_gpu_env(self):
        from ipd.training.selfplay.config import smoke_config
        from ipd.training.selfplay.ppo import PPOAgent
        from ipd.training.selfplay.rollout import concat_rollouts
        from ipd.training.selfplay.train import collect_red

        cfg = smoke_config()
        cfg.envs = 8
        cfg.device = "cuda"
        model = make_model(cfg, device="cuda")
        state = snapshot(model)
        rollout, stats = collect_red(
            cfg, model, cfg.phases[0], state, random.Random(3), "cuda",
            8, gpu_env=True, iteration=1)
        self.assertIsInstance(rollout, TensorRollout)
        self.assertEqual(rollout.opponent_rewards.device.type, "cuda")
        combined = concat_rollouts([rollout, rollout])
        combined.compute_gae()
        agent = PPOAgent(model, cfg.ppo, lr=1e-3, device="cuda")
        out = agent.update(combined)
        self.assertGreater(out.n_steps, 0)
        self.assertTrue(np.isfinite(out.policy_loss))
        self.assertGreaterEqual(len(stats["target_kinds"]), 1)

    @unittest.skipIf(torch is None, "torch not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu_env(self):
        actions = [(i * 5) % 2 for i in range(100)]
        cpu = GPUEnv(_rule_factories(), ["good"] * 3, device="cpu",
                     mode="argmax")
        gpu = GPUEnv(_rule_factories(), ["good"] * 3, device="cuda",
                     mode="argmax")
        for t, action in enumerate(actions):
            np.testing.assert_allclose(cpu.observations(), gpu.observations(),
                                       atol=1e-6, err_msg=str(t))
            own_c, other_c, _ = cpu.step([action] * 3)
            own_g, other_g, _ = gpu.step([action] * 3)
            np.testing.assert_array_equal(own_c, own_g)
            np.testing.assert_array_equal(other_c, other_g)
        self.assertEqual(cpu.scores, gpu.scores)


if __name__ == "__main__":
    unittest.main()
