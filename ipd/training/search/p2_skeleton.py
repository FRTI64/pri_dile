#!/usr/bin/env python3
"""P2e: evolutionary/parameter search on a compact skeleton.

Skeleton: cooperate until day `start`; afterwards use the two-state
gap rule learned by P2c:
    gap = (pD - pC + 1) / 2
    defect iff gap <= threshold
This removes free-form FSM complexity while keeping the useful
responsive/non-responsive feature.
"""
from __future__ import annotations

if __package__ in (None, ""):  # allow: python ipd/training/<file>.py
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break


from ipd.paths import result_path

import argparse
import json
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

from ipd.core.simulator import C, D, payoff
from ipd.policies.rules import GapStartPolicy
from ipd.training.search.common import build_field, N_PLAYERS
from ipd.training.search.p2_fsm import fields as train_specs, val_fields, q
from ipd.training.search.common import evaluate_parallel, FieldSet, mean_per_opp


def eval_one(args):
    start, threshold, kind, seed, n = args
    field, _ = build_field(kind, seed, n)
    p = GapStartPolicy(start, threshold)
    total = 0
    for opp in field:
        p.reset(); opp.reset()
        mine = []; theirs = []; s = 0
        for t in range(100):
            x = p.act(t, mine, theirs)
            y = opp.act(t, theirs, mine)
            mine.append(x); theirs.append(y)
            s += payoff(x, y)
        total += s
    return total / n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=result_path("p2_skeleton_results.json"))
    args = ap.parse_args(argv)

    tr_specs, n_tr = train_specs(args.seed, 100, 60)
    va_specs, n_va = val_fields(args.seed, 20, N_PLAYERS)
    starts = [None, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80]
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
    configs = [(s, th) for s in starts for th in thresholds]
    print("configs", len(configs), "train fields", len(tr_specs))

    # Training: evaluate all configs on 100+100 fields of 60 opponents.
    results = []
    for ci, (start, th) in enumerate(configs):
        tasks = [(start, th, kind, seed, n_tr) for kind, seed in tr_specs]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            vals = list(ex.map(eval_one, tasks))
        orig = [v for v, (k, _) in zip(vals, tr_specs) if k == "original"]
        late = [v for v, (k, _) in zip(vals, tr_specs) if k == "late"]
        mean = statistics.fmean(vals)
        fit = 0.5*mean + 0.25*min(statistics.fmean(orig), statistics.fmean(late)) + 0.25*q(vals, .05)
        results.append((fit, mean, statistics.fmean(orig), statistics.fmean(late),
                        q(vals, .05), start, th))
        if (ci+1) % 10 == 0:
            print("train", ci+1, "/", len(configs), flush=True)
    results.sort(key=lambda r: r[0], reverse=True)
    print("\nTop training:")
    for r in results[:12]:
        print(f"fit={r[0]:7.2f} mean={r[1]:7.2f} orig={r[2]:7.2f} "
              f"late={r[3]:7.2f} p05={r[4]:7.2f} start={r[5]} th={r[6]}")

    # Validation of top 10 + baselines.
    print("\nValidation:")
    rows = []
    top = results[:10]
    for _, _, _, _, _, start, th in top:
        tasks = [(start, th, kind, seed, n_va) for kind, seed in va_specs]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            vals = list(ex.map(eval_one, tasks))
        rows.append((statistics.fmean(vals), start, th, vals))
        print(f"start={start} th={th}: mean={statistics.fmean(vals):7.2f}")
    # baselines
    fs = FieldSet(va_specs)
    base = {}
    for name, spec in [("Final", {"kind": "final"}),
                       ("Harvest45", {"kind": "harvest", "HARVEST_START": 45}),
                       ("EarlyProbe", {"kind": "early"})]:
        vals = [x / n_va for x in evaluate_parallel(spec, fs, args.workers, n_va)]
        base[name] = statistics.fmean(vals)
    print("baselines", {k: round(v, 2) for k, v in base.items()})
    valid = [{"mean": r[0], "start": r[1], "threshold": r[2],
              "scores": r[3]} for r in rows]
    valid.sort(key=lambda r: r["mean"], reverse=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"train_top": [{"fit": r[0], "mean": r[1], "orig": r[2],
                                  "late": r[3], "p05": r[4], "start": r[5],
                                  "threshold": r[6]} for r in results[:30]],
                   "validation": valid, "baselines": base}, f,
                  ensure_ascii=False, indent=2)
    print("written", args.out)


if __name__ == "__main__":
    main()
