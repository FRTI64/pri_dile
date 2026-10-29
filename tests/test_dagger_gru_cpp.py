"""C++ action parity for the RL-DAgger-GRU policy."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ipd.paths import cpp_path
from ipd.policies.registry import make_policy

GPP = shutil.which("g++")


@unittest.skipUnless(GPP, "g++ is not installed")
class DaggerGRUCppTests(unittest.TestCase):
    def test_action_sequences_match_python(self) -> None:
        source = """
#include <cstdio>
#include <vector>
#include "policies/rl_dagger_gru_policy.hpp"
int main() {
    ipd::RLDAggerGRUParams params; params.sample = false;
    for (int mode = 0; mode < 3; ++mode) {
        ipd::RLDAggerGRUPolicy p(20261024u, params);
        std::vector<int> mine, theirs;
        for (int t = 0; t < 100; ++t) {
            int y = 0;
            if (mode == 0) y = (t < 2) ? 1 : 0;
            if (mode == 2) y = (t >= 6 && t % 3 == 0) ? 1 : 0;
            int a = p.act(t, mine, theirs);
            mine.push_back(a); theirs.push_back(y);
            std::printf("%d", a);
        }
        std::printf("\\n");
    }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "dagger_gru.cpp"
            exe = Path(tmp) / "dagger_gru"
            src.write_text(source, encoding="utf-8")
            subprocess.run(
                [GPP, "-std=c++17", "-O2", f"-I{cpp_path('')}",
                 str(src), "-o", str(exe)],
                check=True, capture_output=True, text=True)
            got = subprocess.check_output([str(exe)], text=True).splitlines()

        expected = []
        for mode in (0, 1, 2):
            policy = make_policy("rl-dagger-gru", mode="argmax")
            policy.reset()
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(100):
                y = 0
                if mode == 0:
                    y = 1 if t < 2 else 0
                if mode == 2:
                    y = 1 if (t >= 6 and t % 3 == 0) else 0
                mine.append(policy.act(t, mine, theirs))
                theirs.append(y)
            expected.append("".join(map(str, mine)))
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
