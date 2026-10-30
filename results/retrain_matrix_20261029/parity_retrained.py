#!/usr/bin/env python3
"""Python/native action-parity check for the retrained weights.

This mirrors tests/test_policy_native.py but points at the new files.  It is a
diagnostic for deployment readiness; no C++ header is overwritten.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.policies.gru import GRUPolicy  # noqa: E402
from ipd.policies.meta import RLSwitchPolicy  # noqa: E402
from ipd.policies.neural import NeuralPolicy  # noqa: E402
from ipd.policies.rl import RLPolicy  # noqa: E402

EXP = ROOT / "results/retrain_matrix_20261029"
WEXP = ROOT / "weights/retrain_matrix_20261029"
REPAIR = ROOT / "weights/rl_policy_weights_repair.json"


def mlp(path: Path, n: int = 200) -> float:
    fmt = json.loads(path.read_text()).get("format")
    cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
    fast = cls(str(path), mode="argmax")
    slow = cls(str(path), mode="argmax")
    if fast._native is None:
        raise RuntimeError("native MLP ops unavailable")
    slow._native = None
    rng = random.Random(7)
    worst = 0.0
    for _ in range(n):
        x = [rng.uniform(-5.0, 5.0) for _ in range(fast.n_features)]
        worst = max(worst, abs(fast._logit(x) - slow._logit(x)))
    return worst


def gru(path: Path, n: int = 100) -> float:
    fast = GRUPolicy(str(path), mode="argmax")
    slow = GRUPolicy(str(path), mode="argmax")
    if fast._native is None:
        raise RuntimeError("native GRU ops unavailable")
    slow._native = None
    fast.reset()
    slow.reset()
    rng = random.Random(8)
    worst = 0.0
    mine: list[int] = []
    theirs: list[int] = []
    for _ in range(n):
        a = fast._step(mine, theirs)
        b = slow._step(mine, theirs)
        worst = max(worst, abs(a - b))
        their = rng.randrange(2)
        mine.append(0 if a < 0 else 1)
        theirs.append(their)
    return worst


def switch(base: Path, n: int = 100) -> float:
    fast = RLSwitchPolicy(str(base), str(REPAIR), mode="argmax")
    slow = RLSwitchPolicy(str(base), str(REPAIR), mode="argmax")
    for obj in (slow.base, slow.alt):
        if hasattr(obj, "_native"):
            obj._native = None
    fast.reset()
    slow.reset()
    rng = random.Random(9)
    worst = 0.0
    mine: list[int] = []
    theirs: list[int] = []
    for _ in range(n):
        a = fast.act(0, mine, theirs)
        b = slow.act(0, mine, theirs)
        worst = max(worst, abs(a - b))
        their = rng.randrange(2)
        mine.append(a)
        theirs.append(their)
    return worst


def main() -> None:
    rows = {}
    for name in ("rl_policy_weights_pattern_rm.json",
                 "rl_policy_weights_wide_rm.json",
                 "rl_policy_weights_wide160_rm.json",
                 "rl_policy_weights_wide160_rm_s2.json",
                 "rl_policy_weights_wide160_fix4.json"):
        path = WEXP / name
        diff = mlp(path)
        rows[name] = {"kind": "mlp", "max_abs_diff": diff,
                      "pass": diff < 1e-9}
    for name in ("rl_dagger_gru_weights.json",
                 "rl_dagger_gru_weights_seed20261044.json",
                 "rl_dagger_gru_weights_fix4.json",
                 "rl_dagger_gru_weights_fix4_seed20261049.json"):
        path = WEXP / name
        diff = gru(path)
        rows[name] = {"kind": "gru", "max_abs_diff": diff,
                      "pass": diff < 1e-9}
    base = EXP / "wide160_ckpts/iter_001.json"
    diff = switch(base)
    rows["rl_switch_rm"] = {"kind": "switch", "max_abs_diff": diff,
                            "pass": diff < 1e-9}
    out = EXP / "parity_retrained.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    for name, row in rows.items():
        print(f"{name:46} {row['kind']:6} max_abs_diff={row['max_abs_diff']:.3g} "
              f"{'PASS' if row['pass'] else 'FAIL'}")
    print("written", out)


if __name__ == "__main__":
    main()
