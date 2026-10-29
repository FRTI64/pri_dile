"""C++ action parity for the RL-Switch meta policy."""
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
class SwitchCppTests(unittest.TestCase):
    def test_action_sequences_match_python(self) -> None:
        source = """
#include <cstdio>
#include <vector>
#include "policies/rl_switch_policy.hpp"
int main() {
    ipd::RLSwitchParams params; params.sample = false;
    for (int mode = 0; mode < 2; ++mode) {
        ipd::RLSwitchPolicy p(20261024u, params);
        std::vector<int> mine, theirs;
        for (int t = 0; t < 100; ++t) {
            int y = (mode == 0) ? ((t < 2) ? 1 : 0) : 0;
            int a = p.act(t, mine, theirs);
            mine.push_back(a); theirs.push_back(y);
            std::printf("%d", a);
        }
        std::printf(" switch=%d\\n", (int)p.switched());
    }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "switch.cpp"
            exe = Path(tmp) / "switch"
            src.write_text(source, encoding="utf-8")
            subprocess.run(
                [GPP, "-std=c++17", "-O2", f"-I{cpp_path('')}",
                 str(src), "-o", str(exe)],
                check=True, capture_output=True, text=True)
            got = subprocess.check_output([str(exe)], text=True).splitlines()
        expected = []
        for mode in (0, 1):
            policy = make_policy("rl-switch", mode="argmax")
            policy.reset()
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(100):
                y = ((1 if t < 2 else 0) if mode == 0 else 0)
                mine.append(policy.act(t, mine, theirs))
                theirs.append(y)
            expected.append(
                "".join(map(str, mine)) + f" switch={int(policy.switched)}")
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
