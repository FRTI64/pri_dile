#!/usr/bin/env python3
"""Screen the cost of an unconditional one-shot early D probe.

This is the control for the alternative "early-game active probe" branch.  It
does not train or register a policy; it only measures the immediate cost of a
single forced defection at several early days on heldout_v2.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics

from ipd.core.primitives import D
from ipd.evaluation.opponent_suite import build_suite
from ipd.paths import weights_path
from ipd.policies.neural import NeuralPolicy


def _play(probe_day: int | None, item, seed: int) -> int:
    policy = NeuralPolicy(
        weights_path("rl_policy_weights_repair.json"),
        seed=20261024, mode="sample")
    opponent = item.factory(random.Random(seed))
    mine: list[int] = []
    theirs: list[int] = []
    score = 0
    for t in range(100):
        action = int(policy.act(t, mine, theirs))
        if probe_day is not None and t == probe_day:
            action = D
        reply = int(opponent.act(t, theirs, mine))
        score += (6 if reply == 0 else 0) if action == 0 else (
            10 if reply == 0 else 1)
        mine.append(action)
        theirs.append(reply)
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="heldout_v2")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--probe-days", default="1,3,5,10,15,20,30,40")
    ap.add_argument("--out", default=(
        "results/p2_threeway_moe_20261026/probe_cost.json"))
    args = ap.parse_args()
    suite = build_suite(args.suite)
    probe_days = [int(x) for x in args.probe_days.split(",") if x.strip()]

    def by_family(probe_day: int | None) -> dict[str, float]:
        rows: dict[str, list[int]] = {}
        for item in suite:
            rows.setdefault(item.family, []).append(
                _play(probe_day, item, args.seed))
        return {name: statistics.fmean(values)
                for name, values in rows.items()}

    base = by_family(None)
    out: dict[str, object] = {
        "suite": args.suite,
        "seed": args.seed,
        "base": base,
        "base_mean": statistics.fmean(base.values()),
        "probes": {},
    }
    for day in probe_days:
        families = by_family(day)
        out["probes"][str(day)] = {
            "families": families,
            "mean": statistics.fmean(families.values()),
            "delta": statistics.fmean(families.values())
            - out["base_mean"],
            "family_delta": {name: families[name] - base[name]
                             for name in base},
        }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
