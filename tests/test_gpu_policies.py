"""Parity tests for batched learned-policy inference."""
from __future__ import annotations

import math
import random
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from ipd.paths import weights_path
from ipd.policies.gru import GRUPolicy
from ipd.policies.neural import NeuralPolicy, extract_features
from ipd.policies.rl import RLPolicy
from ipd.policies.selfplay_gru import SelfPlayGRUPolicy
from ipd.policies.selfplay_gru_obs import features as features21

if torch is not None:
    from ipd.evaluation.gpu.features import FeatureState21, FeatureState85
    from ipd.evaluation.gpu.policies import (
        BatchGRU, BatchMLP, BatchSelfPlayGRU,
    )


def _histories(batch: int = 4, rounds: int = 100, seed: int = 3):
    rng = random.Random(seed)
    return [([rng.randrange(2) for _ in range(rounds)],
             [rng.randrange(2) for _ in range(rounds)])
            for _ in range(batch)]


def _neural_logit(policy: NeuralPolicy, mine, theirs) -> float:
    x = extract_features(mine, theirs)
    for layer in policy.layers[:-1]:
        x = [math.tanh(v) for v in policy._affine(x, layer)]
    return policy._affine(x, policy.layers[-1])[0]


@unittest.skipIf(torch is None, "torch is not installed")
class MLPPolicyTests(unittest.TestCase):
    def _check(self, path: str, cls) -> None:
        rows = _histories()
        batch = len(rows)
        mine = torch.tensor([m for m, _ in rows], dtype=torch.int8)
        theirs = torch.tensor([t for _, t in rows], dtype=torch.int8)
        cpu = cls(path, mode="argmax")
        gpu = BatchMLP(path, dtype=torch.float64)
        state = FeatureState85(batch, dtype=torch.float64)
        for t in range(101):
            features = state.features()
            logits = gpu.logits(features)
            for i, (m, x) in enumerate(rows):
                if cls is NeuralPolicy:
                    want = _neural_logit(cpu, m[:t], x[:t])
                else:
                    want = cpu.logit_value(extract_features(m[:t], x[:t]))[0]
                self.assertLess(abs(float(logits[i]) - want), 1e-10)
                no_defect = torch.tensor([1 not in x[:t]])
                got = int(gpu.act_from_logits(logits[i:i + 1], t,
                                              no_defect)[0])
                self.assertEqual(got, cpu.act(t, m[:t], x[:t]))
            if t < 100:
                state.append(mine[:, t], theirs[:, t])

    def test_neural_policy(self) -> None:
        self._check(weights_path("neural_policy_weights.json"), NeuralPolicy)

    def test_rl_policy_checkpoint(self) -> None:
        self._check(weights_path("rl_policy_checkpoint.json"), RLPolicy)

    def test_rl_deployable_export(self) -> None:
        self._check(weights_path("rl_policy_weights.json"), NeuralPolicy)


@unittest.skipIf(torch is None, "torch is not installed")
class RecurrentPolicyTests(unittest.TestCase):
    def test_gru_actions_and_logits(self) -> None:
        rows = _histories(seed=5)
        batch = len(rows)
        mine = torch.tensor([m for m, _ in rows], dtype=torch.int8)
        theirs = torch.tensor([t for _, t in rows], dtype=torch.int8)
        cpu = GRUPolicy(weights_path("gru_policy_weights.json"), mode="argmax")
        gpu = BatchGRU(weights_path("gru_policy_weights.json"),
                       dtype=torch.float64)
        gpu.reset(batch)
        state = FeatureState85(batch, dtype=torch.float64)
        refs = [[0.0] * batch for _ in range(101)]
        for i, (m, x) in enumerate(rows):
            cpu.reset()
            for t in range(101):
                refs[t][i] = cpu._step(m[:t], x[:t])
        for t in range(101):
            logits = gpu.step(state.features())
            for i, (m, x) in enumerate(rows):
                self.assertLess(abs(float(logits[i]) - refs[t][i]), 1e-10)
            if t < 100:
                state.append(mine[:, t], theirs[:, t])
                # CPU _step was called with the pre-action history; the
                # reference table already has every prefix.

    def test_selfplay_actions_and_logits(self) -> None:
        rows = _histories(seed=9)
        batch = len(rows)
        mine = torch.tensor([m for m, _ in rows], dtype=torch.int8)
        theirs = torch.tensor([t for _, t in rows], dtype=torch.int8)
        path = weights_path("selfplay_gru_policy.json")
        cpu = SelfPlayGRUPolicy(path, mode="argmax")
        gpu = BatchSelfPlayGRU(path, dtype=torch.float64)
        gpu.reset(batch)
        state = FeatureState21(batch, dtype=torch.float64)
        refs = [[0.0] * batch for _ in range(101)]
        for i, (m, x) in enumerate(rows):
            cpu.reset()
            for t in range(101):
                refs[t][i] = cpu._forward(features21(t, m[:t], x[:t]))[0]
        for t in range(101):
            logits = gpu.step(state.features())
            for i, (m, x) in enumerate(rows):
                self.assertLess(abs(float(logits[i]) - refs[t][i]), 1e-10)
            if t < 100:
                state.append(mine[:, t], theirs[:, t])


if __name__ == "__main__":
    unittest.main()
