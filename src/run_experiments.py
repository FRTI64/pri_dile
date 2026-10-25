#!/usr/bin/env python3
"""Reproducible comparison of the final strategy on synthetic 300-player fields.

The field contains the usual textbook strategies plus noisy, forgiving,
probing, late-defecting and adaptive variants.  Each field has 299 opponents;
the final strategy plays one 100-round match against each of them.

Run:
    python3 src/run_experiments.py --fields 10 --seed 2026
"""
from __future__ import annotations

import argparse
import random

from ipd_strategy import Machine
from ipd_field import (make_field, score_against, Responsive3, ProbeBot,
                       GenerousTFT, TFT2LastD)
from ipd_sim import AllC, AllD, TFT, TFTLastD


class FinalAdapter(Machine):
    name = "Final"

    def reset(self) -> None:
        super().reset()

    def act(self, t, mine, theirs):
        return self.move(t + 1, mine, theirs)


CANDIDATES = {
    "Final": lambda r: FinalAdapter(seed=20261024),
    "Responsive3": lambda r: Responsive3(rng=r),
    "TFT": lambda r: TFT(r),
    "TFT-lastD": lambda r: TFTLastD(r),
    "GTFT(0.2)-lastD": lambda r: GenerousTFT(0.2, True, r),
    "TFT2-lastD": lambda r: TFT2LastD(r),
    "AllC": lambda r: AllC(r),
    "AllD": lambda r: AllD(r),
    "ProbeBot": lambda r: ProbeBot(rng=r),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--fields", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    master = random.Random(args.seed)
    fields = [make_field(args.n, random.Random(master.random()))
              for _ in range(args.fields)]

    results = {}
    details = {}
    for name, factory in CANDIDATES.items():
        totals = []
        by_type = {}
        for field in fields:
            total, by = score_against(factory, field)
            totals.append(total)
            for k, v in by.items():
                by_type.setdefault(k, []).append(v)
        results[name] = totals
        details[name] = by_type

    print(f"{'strategy':18} {'mean total':>12} {'per opponent':>14} "
          f"{'min':>8} {'max':>8}")
    for name, vals in sorted(results.items(),
                             key=lambda kv: sum(kv[1]) / len(kv[1]),
                             reverse=True):
        mean = sum(vals) / len(vals)
        print(f"{name:18} {mean:12.1f} {mean/args.n:14.2f} "
              f"{min(vals):8d} {max(vals):8d}")

    print("\nFinal strategy: mean score by opponent type")
    for typ in sorted(details["Final"]):
        vals = details["Final"][typ]
        if len(vals) >= args.fields // 2:
            print(f"  {typ:20} {sum(vals)/len(vals):7.1f}")


if __name__ == "__main__":
    main()
