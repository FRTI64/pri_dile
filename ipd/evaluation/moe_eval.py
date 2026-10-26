"""Offline evaluation for the P2 C3 three-way gate + MoE.

The baseline and the MoE use independent same-seed RL-Repair instances.  The
MoE routes ``safe`` to an exploitation expert and ``risky`` / ``unknown`` to
RL-Repair.  A planner expert gets its own base instance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ipd.evaluation.opponent_suite import build_suite
from ipd.evaluation.planner_eval import compare, evaluate_policy
from ipd.paths import result_path, weights_path
from ipd.policies.belief_planner import BeliefPlanner
from ipd.policies.moe import DefectExpert, ThreeWayMoE
from ipd.policies.neural import NeuralPolicy
from ipd.training.opponent_model import (
    ThreeWayGate, load_opponent_model,
)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline evaluation of the three-way gate + MoE.")
    ap.add_argument("--suite", default="validation,heldout_v2",
                    help="comma-separated frozen opponent suites")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--weights", default=weights_path(
        "rl_policy_weights_repair.json"))
    ap.add_argument("--policy-seed", type=int, default=20261024)
    ap.add_argument("--gate-model", default=weights_path(
        "opponent_gate_threeway.json"))
    ap.add_argument("--safe-threshold", type=float, default=0.6)
    ap.add_argument("--risky-threshold", type=float, default=0.6)
    ap.add_argument("--min-day", type=int, default=5)
    ap.add_argument("--response-model", default="",
                    help="optional C2 responsiveness veto")
    ap.add_argument("--max-response-probability", type=float, default=0.5)
    ap.add_argument("--safe-expert", choices=("alld", "planner", "base"),
                    default="alld")
    ap.add_argument("--planner-horizon", type=int, default=4)
    ap.add_argument("--planner-top-k", type=int, default=6)
    ap.add_argument("--planner-max-hypotheses", type=int, default=48)
    ap.add_argument("--planner-fallback-confidence", type=float, default=0.9)
    ap.add_argument("--planner-match-prob", type=float, default=0.85)
    ap.add_argument("--planner-margin", type=float, default=10.0)
    ap.add_argument("--out", default=result_path("moe_eval.json"))
    args = ap.parse_args(argv)

    seed_values = [args.seed_base + i for i in range(max(1, args.seeds))]
    suites = [x.strip() for x in args.suite.split(",") if x.strip()]
    response_model = (load_opponent_model(args.response_model)
                      if args.response_model else None)
    gate = ThreeWayGate(
        load_opponent_model(args.gate_model),
        safe_threshold=args.safe_threshold,
        risky_threshold=args.risky_threshold,
        min_day=args.min_day,
        response_model=response_model,
        max_response_probability=(
            args.max_response_probability if response_model else None))

    baseline = NeuralPolicy(args.weights, seed=args.policy_seed, mode="sample")
    fallback = NeuralPolicy(args.weights, seed=args.policy_seed, mode="sample")
    if args.safe_expert == "alld":
        safe_expert = DefectExpert()
    elif args.safe_expert == "planner":
        planner_base = NeuralPolicy(
            args.weights, seed=args.policy_seed, mode="sample")
        safe_expert = BeliefPlanner(
            planner_base, horizon=args.planner_horizon,
            top_k=args.planner_top_k,
            max_hypotheses=args.planner_max_hypotheses,
            fallback_confidence=args.planner_fallback_confidence,
            match_prob=args.planner_match_prob, margin=args.planner_margin,
            seed=args.policy_seed)
    else:
        safe_expert = fallback
    moe = ThreeWayMoE(fallback, gate, safe_expert=safe_expert)

    report = {
        "schema": "ipd-moe-eval-v1",
        "weights": args.weights,
        "gate_model": args.gate_model,
        "safe_expert": args.safe_expert,
        "config": {
            "seeds": seed_values,
            "rounds": args.rounds,
            "safe_threshold": args.safe_threshold,
            "risky_threshold": args.risky_threshold,
            "min_day": args.min_day,
            "response_model": args.response_model or None,
            "max_response_probability": (args.max_response_probability
                                         if response_model else None),
            "planner_horizon": args.planner_horizon,
            "planner_top_k": args.planner_top_k,
            "planner_fallback_confidence": args.planner_fallback_confidence,
            "planner_margin": args.planner_margin,
        },
        "suites": {},
    }
    for suite in suites:
        items = build_suite(suite)
        print(f"suite {suite}: {len(items)} opponents "
              f"x {len(seed_values)} seeds", flush=True)
        base_summary = evaluate_policy(
            baseline, items, seed_values, args.rounds)
        moe_summary = evaluate_policy(
            moe, items, seed_values, args.rounds,
            collect_diagnostics=True)
        compare(moe_summary, base_summary)
        report["suites"][suite] = {
            "base": base_summary,
            "moe": moe_summary,
        }
        print(f"  base mean={base_summary['mean']:.2f} "
              f"cvar={base_summary['cvar10_family']:.2f} "
              f"min={base_summary['min_family']:.2f}")
        print(f"  moe  mean={moe_summary['mean']:.2f} "
              f"delta={moe_summary['mean_delta']:+.2f} "
              f"min_norm={moe_summary['min_norm']} "
              f"safe={moe_summary['diagnostics']['safe_rate']:.3f}")
        for family in sorted(base_summary["families"]):
            delta = moe_summary["family_delta"].get(family, 0.0)
            print(f"    {family:16} {base_summary['families'][family]:7.2f} "
                  f"-> {moe_summary['families'][family]:7.2f} "
                  f"({delta:+.2f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written {out}")
    return 0


__all__ = ["main"]
