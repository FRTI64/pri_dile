#!/usr/bin/env python3
"""Small dev-only threshold scan for the retrained RL-Switch rule.

The registered ``rl-switch`` default (window=6, both sides >=2 D, opponent's
last move C) is kept unchanged.  This script only records whether an
alternative threshold would have helped on dev; it never writes a candidate
weight or the test ledger.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.core.primitives import C, D  # noqa: E402
from ipd.evaluation.common import evaluate_field_details  # noqa: E402
from ipd.evaluation.protocol import build_protocol_field  # noqa: E402
from ipd.policies.neural import NeuralPolicy  # noqa: E402
from ipd.policies.rl import RLPolicy  # noqa: E402

BASE = str(ROOT / "results/retrain_matrix_20261029/wide160_ckpts/iter_001.json")
ALT = str(ROOT / "weights/rl_policy_weights_repair.json")


def _load(path: str, seed: int):
    with open(path, encoding="utf-8") as fh:
        fmt = json.load(fh).get("format")
    cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
    return cls(path, seed=seed, mode="sample")


class ThresholdSwitch:
    def __init__(self, base: str, alt: str, window: int,
                 min_mine: int, min_opp: int, seed: int = 20261027):
        self.base = _load(base, seed)
        self.alt = _load(alt, seed + 1)
        self.window = window
        self.min_mine = min_mine
        self.min_opp = min_opp
        self.switched = False
        self.name = (f"switch-w{window}-m{min_mine}-o{min_opp}")

    def reset(self) -> None:
        self.base.reset()
        self.alt.reset()
        self.switched = False

    def set_rounds(self, rounds: int) -> None:
        for policy in (self.base, self.alt):
            fn = getattr(policy, "set_rounds", None)
            if callable(fn):
                fn(rounds)

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if not self.switched and t >= self.window:
            first_opp = theirs[:self.window]
            first_mine = mine[:self.window]
            if (first_opp.count(D) >= self.min_opp
                    and first_mine.count(D) >= self.min_mine
                    and theirs[-1] == C):
                self.switched = True
        return (self.alt if self.switched else self.base).act(t, mine, theirs)


def main() -> None:
    tiers = ("dev_param", "dev_complex", "dev_axelrod")
    fields = 5
    n = 1024
    seeds = [20261027 + i * 7919 for i in range(fields)]
    configs = [(6, 2, 2), (6, 2, 3), (6, 3, 3), (8, 2, 2),
               (8, 3, 3), (10, 2, 2)]
    out = Path(__file__).with_name("switch_thresholds.json")
    rows = {}
    started = time.time()
    for window, min_mine, min_opp in configs:
        label = f"w{window}-m{min_mine}-o{min_opp}"
        rows[label] = {}
        for tier in tiers:
            means = []
            for seed in seeds:
                field = build_protocol_field(tier, seed, n)
                policy = ThresholdSwitch(BASE, ALT, window, min_mine, min_opp)
                means.append(evaluate_field_details(policy, field)[0].mean)
            rows[label][tier] = {
                "mean": statistics.fmean(means),
                "values": means,
            }
        print(label, {t: round(rows[label][t]["mean"], 2) for t in tiers},
              flush=True)
    payload = {
        "note": "dev-only threshold scan; default candidate remains w6-m2-o2",
        "base": BASE, "alt": ALT, "tiers": tiers, "fields": fields,
        "n": n, "seeds": seeds, "seconds": round(time.time() - started, 2),
        "results": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
