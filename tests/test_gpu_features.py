"""Parity tests for the batched 85/21-dimensional feature extractors."""
from __future__ import annotations

import random
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from ipd.policies.neural import extract_features
from ipd.policies.selfplay_gru_obs import features as features21_v1
from ipd.policies.selfplay_gru_obs import features_v2 as features21_v2

if torch is not None:
    from ipd.evaluation.gpu.features import (
        FeatureState21, FeatureState85, features21, features85,
    )


def _histories(batch: int, rounds: int = 100, seed: int = 7):
    rng = random.Random(seed)
    rows = []
    for _ in range(batch):
        mine = [rng.randrange(2) for _ in range(rounds)]
        theirs = [rng.randrange(2) for _ in range(rounds)]
        rows.append((mine, theirs))
    return rows


@unittest.skipIf(torch is None, "torch is not installed")
class Feature85Tests(unittest.TestCase):
    def test_functional_and_incremental_match_reference(self) -> None:
        rows = _histories(8)
        batch = len(rows)
        rounds = 100
        mine = torch.tensor([m for m, _ in rows], dtype=torch.int8)
        theirs = torch.tensor([t for _, t in rows], dtype=torch.int8)
        state = FeatureState85(batch, dtype=torch.float64)
        for t in range(rounds + 1):
            reference = torch.tensor(
                [extract_features(m[:t], x[:t]) for m, x in rows],
                dtype=torch.float64)
            functional = features85(mine, theirs, t, dtype=torch.float64)
            incremental = state.features()
            self.assertEqual(tuple(functional.shape), (batch, 85))
            self.assertLess(float((functional - reference).abs().max()),
                            1e-12, f"functional mismatch at t={t}")
            self.assertLess(float((incremental - reference).abs().max()),
                            1e-12, f"incremental mismatch at t={t}")
            if t < rounds:
                state.append(mine[:, t], theirs[:, t])

    def test_float32_error_bound(self) -> None:
        rows = _histories(4)
        mine = torch.tensor([m for m, _ in rows], dtype=torch.int8)
        theirs = torch.tensor([t for _, t in rows], dtype=torch.int8)
        state = FeatureState85(4, dtype=torch.float32)
        for t in range(101):
            reference = torch.tensor(
                [extract_features(m[:t], x[:t]) for m, x in rows],
                dtype=torch.float32)
            got = state.features()
            self.assertLess(float((got - reference).abs().max()), 1e-6)
            if t < 100:
                state.append(mine[:, t], theirs[:, t])

    def test_constant_and_alternating_histories(self) -> None:
        histories = [
            ([0] * 100, [0] * 100),
            ([1] * 100, [1] * 100),
            ([i % 2 for i in range(100)], [i % 2 for i in range(100)]),
            ([0] * 100, [1] * 100),
        ]
        mine = torch.tensor([m for m, _ in histories], dtype=torch.int8)
        theirs = torch.tensor([t for _, t in histories], dtype=torch.int8)
        for t in (0, 1, 5, 10, 50, 99, 100):
            reference = torch.tensor(
                [extract_features(m[:t], x[:t]) for m, x in histories],
                dtype=torch.float64)
            got = features85(mine, theirs, t, dtype=torch.float64)
            self.assertLess(float((got - reference).abs().max()), 1e-12)


@unittest.skipIf(torch is None, "torch is not installed")
class Feature21Tests(unittest.TestCase):
    def test_v1_and_v2_functional_and_incremental(self) -> None:
        rows = _histories(5, seed=11)
        batch = len(rows)
        mine = torch.tensor([m for m, _ in rows], dtype=torch.int8)
        theirs = torch.tensor([t for _, t in rows], dtype=torch.int8)
        for version, reference_fn in (
            ("selfplay-21-v1", features21_v1),
            ("selfplay-21-v2", features21_v2),
        ):
            state = FeatureState21(batch, dtype=torch.float64,
                                   version=version)
            for t in range(101):
                reference = torch.tensor(
                    [reference_fn(t, m[:t], x[:t]) for m, x in rows],
                    dtype=torch.float64)
                functional = features21(mine, theirs, t, version=version,
                                        dtype=torch.float64)
                incremental = state.features()
                self.assertLess(float((functional - reference).abs().max()),
                                1e-12)
                self.assertLess(float((incremental - reference).abs().max()),
                                1e-12)
                if t < 100:
                    state.append(mine[:, t], theirs[:, t])

    def test_missing_micro_history_is_minus_one(self) -> None:
        mine = torch.zeros(1, 100, dtype=torch.int8)
        theirs = torch.zeros(1, 100, dtype=torch.int8)
        got = features21(mine, theirs, 2, dtype=torch.float64)
        self.assertEqual(float(got[0, 6]), 0.0)
        self.assertEqual(float(got[0, 7]), 0.0)
        self.assertEqual(float(got[0, 8]), 0.0)
        self.assertEqual(float(got[0, 9]), 0.0)
        self.assertEqual(float(got[0, 10]), -1.0)
        self.assertEqual(float(got[0, 11]), -1.0)


if __name__ == "__main__":
    unittest.main()
