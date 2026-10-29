"""Parity tests for the 97-dimensional v2 features."""
from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ipd.paths import cpp_path
from ipd.policies.neural import (
    FEATURE_COUNT_V2, extract_features, extract_features_v2,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

if torch is not None:
    from ipd.evaluation.gpu.features import (
        FeatureState97, features97,
    )


def _histories(batch: int, rounds: int = 100, seed: int = 17):
    rng = random.Random(seed)
    return [
        ([rng.randrange(2) for _ in range(rounds)],
         [rng.randrange(2) for _ in range(rounds)])
        for _ in range(batch)
    ]


class FeaturesV2PythonTests(unittest.TestCase):
    def test_v2_is_v1_prefix_plus_12_features(self) -> None:
        for mine, theirs in _histories(5, rounds=30):
            v1 = extract_features(mine, theirs)
            v2 = extract_features_v2(mine, theirs)
            self.assertEqual(len(v2), FEATURE_COUNT_V2)
            self.assertEqual(v2[:len(v1)], v1)
            self.assertTrue(all(0.0 <= x <= 1.0 for x in v2[len(v1):]))

    def test_periodic_and_response_features(self) -> None:
        # Opponent alternates; self always cooperates.
        theirs = [i % 2 for i in range(20)]
        mine = [0] * 20
        f = extract_features_v2(mine, theirs)[85:]
        self.assertAlmostEqual(f[0], 1.0)  # opponent period 2
        self.assertAlmostEqual(f[1], 0.0)  # opponent period 3
        self.assertAlmostEqual(f[4], 1.0)  # self period 2
        self.assertAlmostEqual(f[5], 1.0)  # self period 3
        self.assertAlmostEqual(f[8], 0.5)  # opponent mirrors half the time
        self.assertAlmostEqual(f[9], 0.5)  # self mirrors half the time


@unittest.skipIf(torch is None, "torch is not installed")
class FeaturesV2GpuTests(unittest.TestCase):
    def test_functional_and_incremental_match_python(self) -> None:
        rows = _histories(6)
        batch = len(rows)
        mine = torch.tensor([m for m, _ in rows], dtype=torch.int8)
        theirs = torch.tensor([t for _, t in rows], dtype=torch.int8)
        state = FeatureState97(batch, dtype=torch.float64)
        for t in range(101):
            reference = torch.tensor(
                [extract_features_v2(m[:t], x[:t]) for m, x in rows],
                dtype=torch.float64)
            functional = features97(mine, theirs, t, dtype=torch.float64)
            incremental = state.features()
            self.assertEqual(tuple(functional.shape), (batch, 97))
            self.assertLess(float((functional - reference).abs().max()),
                            1e-12, f"functional mismatch at t={t}")
            self.assertLess(float((incremental - reference).abs().max()),
                            1e-12, f"incremental mismatch at t={t}")
            if t < 100:
                state.append(mine[:, t], theirs[:, t])


@unittest.skipUnless(shutil.which("g++"), "g++ is not installed")
class FeaturesV2CppTests(unittest.TestCase):
    def test_cpp_v2_matches_python(self) -> None:
        mine = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
        theirs = [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
        source = """
#include <cstdio>
#include "ipd_features.hpp"
int main() {
    std::vector<int> mine = {0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,1};
    std::vector<int> theirs = {0,0,1,0,1,0,1,1,0,1,0,0,1,1,0,1};
    auto f = ipd::extract_features_v2(mine, theirs);
    for (size_t i = 85; i < f.size(); ++i) std::printf("%.17g\\n", f[i]);
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "check.cpp"
            exe = Path(tmp) / "check"
            src.write_text(source, encoding="utf-8")
            subprocess.run(
                ["g++", "-std=c++17", "-O2", f"-I{cpp_path('')}",
                 str(src), "-o", str(exe)],
                check=True, capture_output=True, text=True)
            got = [float(x) for x in subprocess.check_output(
                [str(exe)], text=True).split()]
        expected = extract_features_v2(mine, theirs)[85:]
        self.assertEqual(len(got), len(expected))
        for a, b in zip(got, expected):
            self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
