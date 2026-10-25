#!/usr/bin/env python3
"""Multi-objective fine tuning across four distributions.

Distributions: original, late-harvest, high-Grim, wide/smart.
Families: Final, Harvest, GapStart, Responsive3, BayesPlus thresholds.
Fitness: mean + lower quantile + worst-distribution mean.
"""
from __future__ import annotations

from ipd.paths import result_path
import argparse, json, random, statistics, time
from concurrent.futures import ProcessPoolExecutor

from ipd.core.simulator import C, D, payoff
from ipd.evaluation.distributions import build_dist_field, sample_wide
from ipd.policies.rules import (
    GapStartPolicy, ParamBayes, ParamHarvest, ParamPolicy, ParamResp,
)


def make_policy(kind, cfg):
    if kind == "final":
        return ParamPolicy(seed=20261024, **cfg)
    if kind == "harvest":
        return ParamHarvest(seed=20261024, **cfg)
    if kind == "gap":
        return GapStartPolicy(cfg["start"], cfg["threshold"])
    if kind == "resp":
        return ParamResp(seed=20261024, **cfg)
    if kind == "bayes":
        return ParamBayes(**cfg)
    raise ValueError(kind)

def eval_task(task):
    kind, cfg, seed_map, n = task
    out = {}
    for dist, seeds in seed_map.items():
        vals = []
        for seed in seeds:
            field = build_dist_field(dist, seed, n)
            p = make_policy(kind, cfg)
            total = 0
            for opp in field:
                p.reset(); opp.reset()
                mine = []; theirs = []; s = 0
                for t in range(100):
                    if hasattr(p, "act"):
                        x = p.act(t, mine, theirs)
                    else:
                        x = p.move(t + 1, mine, theirs)
                    y = opp.act(t, theirs, mine)
                    mine.append(x); theirs.append(y); s += payoff(x, y)
                total += s
            vals.append(total / n)
        out[dist] = vals
    return out

def fitness(out):
    allv = [x for vs in out.values() for x in vs]
    mean = statistics.fmean(allv)
    q05 = sorted(allv)[max(0, int(.05 * len(allv)))]
    worst_dist = min(statistics.fmean(vs) for vs in out.values())
    return 0.45 * mean + 0.25 * worst_dist + 0.20 * q05 + 0.10 * min(allv), mean, worst_dist

def configs():
    rng = random.Random(3030)
    out = []
    for _ in range(36):
        out.append(("final", {
            "P_UNPROVOKED": rng.uniform(.10, .60),
            "P_MUTUAL": rng.uniform(.15, .70),
            "P_RESPONSIVE": rng.uniform(.80, 1.0),
            "MIN_C": rng.randint(3, 7),
            "MIN_D": rng.randint(2, 5),
            "RESP_GAP": rng.uniform(.25, .55),
            "DEFECT_RATE": rng.uniform(.10, .26),
            "PEACE_TESTS": rng.randint(1, 3),
            "PEACE_COOLDOWN": rng.randint(4, 14),
        }))
    for _ in range(36):
        out.append(("harvest", {
            "P_UNPROVOKED": rng.uniform(.10, .55),
            "P_MUTUAL": rng.uniform(.15, .65),
            "P_RESPONSIVE": rng.uniform(.85, 1.0),
            "MIN_C": rng.randint(3, 6),
            "MIN_D": rng.randint(2, 4),
            "RESP_GAP": rng.uniform(.28, .52),
            "DEFECT_RATE": rng.uniform(.12, .24),
            "PEACE_TESTS": rng.randint(1, 3),
            "PEACE_COOLDOWN": rng.randint(4, 14),
            "HARVEST_START": rng.randint(15, 75),
        }))
    for _ in range(30):
        out.append(("gap", {
            "start": rng.randint(1, 60),
            "threshold": rng.uniform(.35, .65),
        }))
    for _ in range(30):
        out.append(("resp", {
            "g_unprov": rng.uniform(.25, .80),
            "g_mutual": rng.uniform(.10, .65),
            "g_resp": rng.uniform(.80, 1.0),
            "min_c": rng.randint(3, 7),
            "min_d": rng.randint(2, 5),
            "resp": rng.uniform(.25, .55),
            "rate": rng.uniform(.10, .25),
            "max_tests": rng.randint(1, 3),
            "cooldown": rng.randint(4, 14),
        }))
    for _ in range(20):
        out.append(("bayes", {
            "NONRESP_THRESHOLD": rng.uniform(.25, .60),
            "PUNITIVE_THRESHOLD": rng.uniform(.40, .80),
        }))
    return out

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9090)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--train-per-dist", type=int, default=16)
    ap.add_argument("--val-per-dist", type=int, default=16)
    ap.add_argument("--out", default=result_path("train_multiobj_results.json"))
    args = ap.parse_args(argv)
    cfgs = configs()
    rng = random.Random(args.seed)
    train_map = {d: [rng.randrange(1 << 30) for _ in range(args.train_per_dist)]
                 for d in ("original", "late", "grim30", "wide")}
    val_map = {d: [rng.randrange(1 << 30) for _ in range(args.val_per_dist)]
               for d in ("original", "late", "grim30", "wide")}
    tasks = [(kind, cfg, train_map, args.n) for kind, cfg in cfgs]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        outs = list(ex.map(eval_task, tasks, chunksize=1))
    print("train eval", round(time.time()-t0,1), flush=True)
    rows = []
    for (kind, cfg), out in zip(cfgs, outs):
        f, mean, worst = fitness(out)
        rows.append({"kind": kind, "cfg": cfg, "fit": f, "mean": mean,
                     "worst": worst, "out": out})
    print("\nTop train per family:")
    selected = []
    for kind in ("final", "harvest", "gap", "resp", "bayes"):
        rs = [r for r in rows if r["kind"] == kind]
        rs.sort(key=lambda r: r["fit"], reverse=True)
        for r in rs[:3]:
            print(f"{kind:8} fit={r['fit']:7.2f} mean={r['mean']:7.2f} "
                  f"worst={r['worst']:7.2f} cfg={r['cfg']}")
            selected.append(r)
    print("\nValidation:")
    val_tasks = [(r["kind"], r["cfg"], val_map, 299) for r in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        val_outs = list(ex.map(eval_task, val_tasks, chunksize=1))
    val_rows = []
    for r, out in zip(selected, val_outs):
        f, mean, worst = fitness(out)
        row = {"kind": r["kind"], "cfg": r["cfg"], "train_fit": r["fit"],
               "val_fit": f, "val_mean": mean, "val_worst": worst,
               "val_out": out}
        val_rows.append(row)
        print(f"{r['kind']:8} val_mean={mean:7.2f} val_worst={worst:7.2f} "
              f"val_fit={f:7.2f}")
    # baselines
    base = [("Final", "final", {}),
            ("Harvest45", "harvest", {"HARVEST_START": 45}),
            ("GapStart", "gap", {"start": 1, "threshold": .5}),
            ("Responsive3", "resp", {}),
            ("BayesPlus", "bayes", {})]
    print("\nBaselines:")
    base_rows = []
    for name, kind, cfg in base:
        out = eval_task((kind, cfg, val_map, 299))
        f, mean, worst = fitness(out)
        base_rows.append({"name": name, "val_fit": f, "val_mean": mean,
                          "val_worst": worst, "val_out": out})
        print(f"{name:12} val_mean={mean:7.2f} val_worst={worst:7.2f} val_fit={f:7.2f}")
    val_rows.sort(key=lambda r: r["val_fit"], reverse=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"train": sorted(rows, key=lambda r: r["fit"], reverse=True)[:50],
                   "validation": val_rows, "baselines": base_rows},
                  fh, ensure_ascii=False, indent=2)
    print("written", args.out)

if __name__ == "__main__":
    main()
