#!/usr/bin/env python3
"""One-shot Bayesian/EV probe screen.

At selected days we let the C1 belief planner choose an action (the planner
uses its particle posterior and a short rollout); on all other days RL-Repair
acts.  This is a lower bound on a full information-value probe.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics

from ipd.evaluation.opponent_suite import build_suite
from ipd.paths import weights_path
from ipd.policies.belief_planner import BeliefPlanner
from ipd.policies.neural import NeuralPolicy


def _run(items, seeds, probe_days, *, horizon, top_k, confidence, margin):
    rows: dict[str, list[int]] = {}
    for item in items:
        for seed in seeds:
            base = NeuralPolicy(
                weights_path("rl_policy_weights_repair.json"),
                seed=20261024, mode="sample")
            planner_base = NeuralPolicy(
                weights_path("rl_policy_weights_repair.json"),
                seed=20261024, mode="sample")
            planner = BeliefPlanner(
                planner_base, horizon=horizon, top_k=top_k,
                max_hypotheses=48, fallback_confidence=confidence,
                margin=margin, seed=20261024)
            opponent = item.factory(random.Random(seed))
            mine: list[int] = []
            theirs: list[int] = []
            score = 0
            for t in range(100):
                base_action = int(base.act(t, mine, theirs))
                planned = int(planner.act(t, mine, theirs))
                action = planned if t in probe_days else base_action
                reply = int(opponent.act(t, theirs, mine))
                score += ((6 if reply == 0 else 0) if action == 0
                          else (10 if reply == 0 else 1))
                mine.append(action)
                theirs.append(reply)
            rows.setdefault(item.family, []).append(score)
    return {name: statistics.fmean(values)
            for name, values in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="heldout_v2")
    ap.add_argument("--out", default=(
        "results/p2_repeated_calib_20261026/bayes_probe_screen.json"))
    args = ap.parse_args()
    items = build_suite(args.suite)
    base = _run(items, [1000], (), horizon=4, top_k=6,
                confidence=0.9, margin=10)
    base_mean = statistics.fmean(base.values())
    cases = {
        "h6_day10": dict(probe_days=(10,), horizon=6, top_k=8,
                         confidence=0.75, margin=0.0),
        "h6_day20": dict(probe_days=(20,), horizon=6, top_k=8,
                         confidence=0.75, margin=0.0),
        "h6_day30": dict(probe_days=(30,), horizon=6, top_k=8,
                         confidence=0.75, margin=0.0),
        "h6_day30_seeds3": dict(probe_days=(30,), horizon=6, top_k=8,
                                confidence=0.75, margin=0.0),
        "h4_day30": dict(probe_days=(30,), horizon=4, top_k=6,
                         confidence=0.9, margin=10.0),
    }
    out: dict[str, object] = {
        "suite": args.suite,
        "base": base,
        "base_mean": base_mean,
        "cases": {},
    }
    for name, config in cases.items():
        seeds = [1000, 1001, 1002] if name.endswith("seeds3") else [1000]
        result = _run(items, seeds, **config)
        out["cases"][name] = {
            "seeds": seeds,
            "config": {key: list(value) if isinstance(value, tuple)
                       else value for key, value in config.items()},
            "families": result,
            "mean": statistics.fmean(result.values()),
            "delta": statistics.fmean(result.values()) - base_mean,
            "family_delta": {key: result[key] - base[key] for key in base},
        }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
