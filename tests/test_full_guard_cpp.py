"""C++ action parity for the RL-FullGuard meta policy."""
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
class FullGuardCppTests(unittest.TestCase):
    def test_action_sequences_match_python(self) -> None:
        source = """
#include <cstdio>
#include <vector>
#include "policies/rl_full_guard_policy.hpp"
int main() {
    ipd::RLFullGuardParams params; params.sample = false;
    for (int mode = 0; mode < 4; ++mode) {
        ipd::RLFullGuardPolicy p(20261024u, params);
        std::vector<int> mine, theirs;
        for (int t = 0; t < 100; ++t) {
            int y = 0;
            if (mode == 0) y = (t >= 5) ? 1 : 0;          // probe, always D
            else if (mode == 1) y = (t >= 10) ? 1 : 0;    // late, always D
            else if (mode == 2) y = (t == 20 || t == 21) ? 1 : 0;
            int a = p.act(t, mine, theirs);
            mine.push_back(a); theirs.push_back(y);
            std::printf("%d", a);
        }
        std::printf(" state=%s\\n", p.state());
    }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "full_guard.cpp"
            exe = Path(tmp) / "full_guard"
            src.write_text(source, encoding="utf-8")
            subprocess.run(
                [GPP, "-std=c++17", "-O2", f"-I{cpp_path('')}",
                 str(src), "-o", str(exe)],
                check=True, capture_output=True, text=True)
            got = subprocess.check_output([str(exe)], text=True).splitlines()
        expected = []
        for mode in range(4):
            policy = make_policy("rl-full-guard-c", mode="argmax")
            policy.reset()
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(100):
                if mode == 0:
                    y = 1 if t >= 5 else 0
                elif mode == 1:
                    y = 1 if t >= 10 else 0
                elif mode == 2:
                    y = 1 if t in (20, 21) else 0
                else:
                    y = 0
                mine.append(policy.act(t, mine, theirs))
                theirs.append(y)
            expected.append(
                "".join(map(str, mine)) + f" state={policy.state}")
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
