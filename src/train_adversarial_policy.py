#!/usr/bin/env python3
"""Constrained adversarial tuning for the neural IPD policies.

This is a black-box adversarial training step, not another supervised network
fit.  It searches the parameters of the late-harvester / non-responsive
exploitation rules in ``adversarial_policy.py`` against the opponent
distribution and keeps only candidates that do **not** lower the mean score of
any of the four validation distributions.

Pipeline:

1. evaluate the frozen base policy on disjoint search fields;
2. sample exploitation configurations and keep the ones whose worst
   distribution delta is non-negative;
3. re-evaluate the best candidates on larger validation fields;
4. write the winning ``adversarial_policy_config.json``.

Example::

    python3 src/train_adversarial_policy.py --arch mlp --seed 20261026
    python3 src/train_adversarial_policy.py --arch gru --weights weights/gru_policy_weights.json
"""
from __future__ import annotations

from paths import config_path, result_path, weights_path

import argparse
import json
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from ipd_sim import payoff
from train_multiobj import build_dist_field

DISTS = ("original", "late", "grim30", "wide")


def make_base(arch: str, weights: str, seed: int, mode: str):
    if arch == "gru":
        from gru_policy import GRUPolicy
        return GRUPolicy(weights, seed=seed, mode=mode)
    from neural_policy import NeuralPolicy
    return NeuralPolicy(weights, seed=seed, mode=mode)


def make_policy(arch: str, weights: str, seed: int, mode: str,
                cfg: dict[str, Any] | None):
    base = make_base(arch, weights, seed, mode)
    if cfg is None:
        return base
    from adversarial_policy import AdversarialPolicy
    return AdversarialPolicy(base, **cfg)


def eval_task(task):
    name, arch, weights, seed, mode, cfg, dist, seeds, n = task
    vals = []
    for field_seed in seeds:
        field = build_dist_field(dist, field_seed, n)
        p = make_policy(arch, weights, seed, mode, cfg)
        total = 0
        for opp in field:
            p.reset()
            opp.reset()
            mine: list[int] = []
            theirs: list[int] = []
            s = 0
            for t in range(100):
                a = p.act(t, mine, theirs) if hasattr(p, "act") else p.move(t + 1, mine, theirs)
                b = opp.act(t, theirs, mine)
                mine.append(a)
                theirs.append(b)
                s += payoff(a, b)
            total += s
        vals.append(total / n)
    return name, dist, vals


def evaluate(name: str, cfg: dict[str, Any] | None, args,
             seeds: dict[str, list[int]], n: int):
    tasks = [
        (name, args.arch, args.weights, args.seed, args.mode, cfg, dist,
         seeds[dist], n)
        for dist in DISTS
    ]
    out = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for _, dist, vals in ex.map(eval_task, tasks, chunksize=1):
            out[dist] = statistics.fmean(vals)
    return out


def sample_configs(count: int, rng: random.Random) -> list[dict[str, Any]]:
    seeded = [
        {"gap": 0.10, "min_c": 8, "min_d": 3, "late_day": 50, "repair": False},
        {"gap": 0.10, "min_c": 8, "min_d": 3, "late_day": 60, "repair": False},
        {"gap": 0.05, "min_c": 8, "min_d": 3, "late_day": 50, "repair": False},
        {"gap": 0.15, "min_c": 8, "min_d": 3, "late_day": 50, "repair": True},
        {"gap": 0.10, "min_c": 8, "min_d": 3, "late_day": None, "repair": False},
        {"gap": None, "min_c": 8, "min_d": 3, "late_day": 50, "repair": False},
        {"gap": 0.10, "min_c": 8, "min_d": 3, "late_day": 50, "repair": True},
        {"gap": 0.0, "min_c": 8, "min_d": 3, "late_day": 50, "repair": False},
    ]
    out = seeded[:count]
    seen = {json.dumps(c, sort_keys=True) for c in out}
    while len(out) < count:
        gap = rng.choice([None, 0.0, 0.05, 0.10, 0.15, 0.20])
        late = rng.choice([None, 40, 45, 50, 55, 60, 65, 70])
        if gap is None and late is None:
            continue
        cfg = {
            "gap": gap,
            "min_c": rng.choice([6, 8, 10, 12]),
            "min_d": rng.choice([2, 3, 4]),
            "late_day": late,
            "repair": rng.choice([False, True]),
        }
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def deltas(scores: dict[str, float], base: dict[str, float]):
    return {d: scores[d] - base[d] for d in DISTS}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Constrained adversarial tuning of exploitation rules.")
    ap.add_argument("--arch", default="mlp", choices=("mlp", "gru"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--mode", default="sample", choices=("sample", "argmax"))
    ap.add_argument("--candidates", type=int, default=48)
    ap.add_argument("--finalists", type=int, default=6)
    ap.add_argument("--search-fields", type=int, default=4)
    ap.add_argument("--search-n", type=int, default=100)
    ap.add_argument("--val-fields", type=int, default=12)
    ap.add_argument("--val-n", type=int, default=299)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="allowed worst-distribution loss during search")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results", default=result_path("adversarial_search_results.json"))
    args = ap.parse_args()

    if args.weights is None:
        args.weights = (weights_path("gru_policy_weights.json") if args.arch == "gru"
                        else weights_path("neural_policy_weights.json"))
    if args.out is None:
        args.out = (config_path("adversarial_gru_policy_tuned_config.json")
                    if args.arch == "gru"
                    else config_path("adversarial_policy_tuned_config.json"))

    rng = random.Random(args.seed)
    search_seeds = {d: [rng.randrange(1 << 30)
                        for _ in range(args.search_fields)] for d in DISTS}
    val_seeds = {d: [rng.randrange(1 << 30)
                     for _ in range(args.val_fields)] for d in DISTS}

    print(f"base arch={args.arch} weights={args.weights}")
    base_search = evaluate("base", None, args, search_seeds, args.search_n)
    print("search base:", {d: round(v, 2) for d, v in base_search.items()})

    cfgs = sample_configs(args.candidates, rng)
    rows = []
    for i, cfg in enumerate(cfgs, 1):
        scores = evaluate(f"cand{i}", cfg, args, search_seeds, args.search_n)
        d = deltas(scores, base_search)
        rows.append({
            "cfg": cfg,
            "scores": scores,
            "avg": statistics.fmean(scores.values()),
            "worst_delta": min(d.values()),
            "delta": d,
        })
        print(f"{i:3d}/{len(cfgs)} worst={rows[-1]['worst_delta']:+7.2f} "
              f"avg={rows[-1]['avg']:7.2f} cfg={cfg}", flush=True)

    feasible = [r for r in rows if r["worst_delta"] >= -abs(args.tolerance)]
    if not feasible:
        print("no candidate met the no-loss constraint; using the best worst-delta")
        feasible = sorted(rows, key=lambda r: (r["worst_delta"], r["avg"]),
                          reverse=True)[:1]
    feasible.sort(key=lambda r: r["avg"], reverse=True)
    finalists = feasible[:args.finalists]

    print(f"\nvalidating {len(finalists)} finalists on "
          f"{args.val_fields} fields x {args.val_n} opponents")
    val_base = evaluate("base", None, args, val_seeds, args.val_n)
    print("val base:", {d: round(v, 2) for d, v in val_base.items()})
    val_rows = []
    for i, r in enumerate(finalists, 1):
        scores = evaluate(f"final{i}", r["cfg"], args, val_seeds, args.val_n)
        d = deltas(scores, val_base)
        row = {
            "cfg": r["cfg"],
            "scores": scores,
            "avg": statistics.fmean(scores.values()),
            "worst_delta": min(d.values()),
            "delta": d,
        }
        val_rows.append(row)
        print(f"final{i} worst={row['worst_delta']:+7.2f} "
              f"avg={row['avg']:7.2f} cfg={row['cfg']}", flush=True)

    passing = [r for r in val_rows if r["worst_delta"] >= 0.0]
    if not passing:
        print("warning: no finalist preserved every validation distribution; "
              "falling back to the best worst-delta")
        passing = sorted(val_rows, key=lambda r: (r["worst_delta"], r["avg"]),
                         reverse=True)[:1]
    best = max(passing, key=lambda r: r["avg"])
    print(f"\nselected: {best['cfg']} avg={best['avg']:.2f} "
          f"worst_delta={best['worst_delta']:+.2f}")

    config = {
        "format": "ipd-adversarial-config-v1",
        "base_arch": args.arch,
        "base_weights": args.weights,
        "gap": best["cfg"]["gap"],
        "min_c": best["cfg"]["min_c"],
        "min_d": best["cfg"]["min_d"],
        "late_day": best["cfg"]["late_day"],
        "repair": best["cfg"]["repair"],
        "meta": {
            "method": "constrained black-box adversarial tuning",
            "constraint": "validation mean may not decrease in any distribution",
            "seed": args.seed,
            "search_fields": args.search_fields,
            "search_n": args.search_n,
            "val_fields": args.val_fields,
            "val_n": args.val_n,
            "base_scores": val_base,
            "selected_scores": best["scores"],
            "selected_delta": best["delta"],
            "candidates": len(rows),
        },
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    with open(args.results, "w", encoding="utf-8") as fh:
        json.dump({
            "search_base": base_search,
            "search": sorted(rows, key=lambda r: r["avg"], reverse=True)[:30],
            "val_base": val_base,
            "validation": val_rows,
            "selected": best,
        }, fh, ensure_ascii=False, indent=2)
    print(f"written {args.out}")
    print(f"written {args.results}")


if __name__ == "__main__":
    main()
