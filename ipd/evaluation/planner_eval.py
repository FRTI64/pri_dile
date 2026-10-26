"""Offline evaluation for the P2 C1 belief planner.

Runs the wrapped base policy and the planner on the same frozen opponent suite
and seeds, then reports per-family means, the normalized worst family and the
planner's override/fallback diagnostics.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from ipd.core.simulator import play_match
from ipd.evaluation.opponent_suite import build_suite
from ipd.paths import result_path, weights_path
from ipd.policies.belief_planner import BeliefPlanner
from ipd.policies.neural import NeuralPolicy


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[float]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(float(row["score"]))
    families = {name: statistics.fmean(values)
                for name, values in by_family.items()}
    means = list(families.values())
    means_sorted = sorted(means)
    k = max(1, len(means_sorted) // 10) if means_sorted else 0
    return {
        "n": len(rows),
        "mean": statistics.fmean(means) if means else 0.0,
        "families": families,
        "min_family": min(means) if means else 0.0,
        "argmin_family": min(families, key=families.get) if families else None,
        "cvar10_family": (statistics.fmean(means_sorted[:k])
                          if means_sorted else 0.0),
        "min_norm": None,
    }


def evaluate_policy(policy, items, seeds: Sequence[int], rounds: int = 100,
                    collect_diagnostics: bool = False) -> dict[str, Any]:
    rows = []
    diagnostics = []
    started = time.time()
    for item in items:
        for seed in seeds:
            opponent = item.factory(random.Random(int(seed)))
            if hasattr(opponent, "name"):
                opponent.name = item.name
            result = play_match(policy, opponent, rounds=rounds)
            rows.append({
                "family": item.family,
                "opponent": item.name,
                "seed": int(seed),
                "score": int(result.score_a),
                "opponent_score": int(result.score_b),
            })
        if collect_diagnostics:
            diagnostics.append(policy.diagnostics())
    summary = _summary(rows)
    summary["seconds"] = round(time.time() - started, 2)
    summary["rows"] = rows
    if collect_diagnostics:
        summary["diagnostics"] = {
            "matches": len(diagnostics),
            "overrides": sum(d["overrides"] for d in diagnostics),
            "rounds": sum(d["rounds"] for d in diagnostics),
            "fallback_low_confidence": sum(
                d["fallback_low_confidence"] for d in diagnostics),
            "fallback_margin": sum(d["fallback_margin"] for d in diagnostics),
            "override_rate": (sum(d["overrides"] for d in diagnostics)
                              / max(1, sum(d["rounds"] for d in diagnostics))),
        }
    return summary


def compare(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    base_families = baseline["families"]
    ratios = {name: summary["families"][name] / value
              for name, value in base_families.items()
              if name in summary["families"] and abs(value) > 1e-9}
    summary["min_norm"] = min(ratios.values()) if ratios else None
    summary["family_delta"] = {
        name: summary["families"][name] - base_families[name]
        for name in base_families if name in summary["families"]
    }
    summary["mean_delta"] = summary["mean"] - baseline["mean"]
    summary["cvar_delta"] = (summary["cvar10_family"]
                             - baseline["cvar10_family"])
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline evaluation of the Bayesian belief planner.")
    ap.add_argument("--suite", default="validation,heldout_v2",
                    help="comma-separated frozen opponent suites")
    ap.add_argument("--seeds", type=int, default=1,
                    help="seeds per opponent (seed = 1000 + i)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--weights", default=weights_path(
        "rl_policy_weights_repair.json"))
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=48)
    ap.add_argument("--fallback-confidence", type=float, default=0.75)
    ap.add_argument("--match-prob", type=float, default=0.85)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--policy-seed", type=int, default=20261024)
    ap.add_argument("--out", default=result_path("planner_eval.json"))
    args = ap.parse_args(argv)

    seed_values = [args.seed_base + i for i in range(max(1, args.seeds))]
    suites = [x.strip() for x in args.suite.split(",") if x.strip()]
    base = NeuralPolicy(args.weights, seed=args.policy_seed, mode="sample")
    planner = BeliefPlanner(
        base, horizon=args.horizon, top_k=args.top_k,
        max_hypotheses=args.max_hypotheses,
        fallback_confidence=args.fallback_confidence,
        match_prob=args.match_prob, margin=args.margin,
        seed=args.policy_seed)
    report: dict[str, Any] = {
        "schema": "ipd-planner-eval-v1",
        "weights": args.weights,
        "suites": {},
        "config": {
            "seeds": seed_values,
            "rounds": args.rounds,
            "horizon": args.horizon,
            "top_k": args.top_k,
            "max_hypotheses": args.max_hypotheses,
            "fallback_confidence": args.fallback_confidence,
            "match_prob": args.match_prob,
            "margin": args.margin,
        },
    }
    for suite in suites:
        items = build_suite(suite)
        print(f"suite {suite}: {len(items)} opponents "
              f"x {len(seed_values)} seeds", flush=True)
        base_summary = evaluate_policy(base, items, seed_values, args.rounds)
        planner_summary = evaluate_policy(
            planner, items, seed_values, args.rounds,
            collect_diagnostics=True)
        compare(planner_summary, base_summary)
        report["suites"][suite] = {
            "base": base_summary,
            "planner": planner_summary,
        }
        print(f"  base    mean={base_summary['mean']:.2f} "
              f"cvar={base_summary['cvar10_family']:.2f} "
              f"min={base_summary['min_family']:.2f}")
        print(f"  planner mean={planner_summary['mean']:.2f} "
              f"delta={planner_summary['mean_delta']:+.2f} "
              f"min_norm={planner_summary['min_norm']} "
              f"override={planner_summary['diagnostics']['override_rate']:.3f}")
        for family in sorted(base_summary["families"]):
            delta = planner_summary["family_delta"].get(family, 0.0)
            print(f"    {family:16} {base_summary['families'][family]:7.2f} "
                  f"-> {planner_summary['families'][family]:7.2f} "
                  f"({delta:+.2f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written {out}")
    return 0


__all__ = ["evaluate_policy", "compare", "main"]
