#!/usr/bin/env python3
"""Wide-bank training: more models, fine-grained parameters, random mixtures.

Adds:
- fine GTFT/Joss/NoisyTFT/Random parameter grids;
- all late-harvest onsets and delayed TFT variants;
- our own smart strategies as opponents;
- random field priors and random smart-opponent shares;
- continuous parameter search for Final / Harvest / GapStart;
- disjoint validation fields.
"""
from __future__ import annotations

from ipd.paths import result_path
import argparse, copy, json, math, random, statistics, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import ipd.core.opponents_extended as ext
from ipd.core.simulator import payoff
from ipd.evaluation.distributions import EXTRA, sample_wide
from ipd.policies.opponents import SMART, SmartAdapter
from ipd.policies.rules import GapStartPolicy, ParamHarvest, ParamPolicy

ALL_BASE = list(ext.TYPE_BANK) + EXTRA


def add_extra(name: str, weight: float, factory) -> None:
    """Legacy helper: append a fine-grained opponent to the wide bank."""
    EXTRA.append(ext.TypeSpec(name, "fine", weight, factory))

def eval_cfg(args):
    kind, cfg, seeds, n = args
    scores = []
    for seed in seeds:
        field, _, _ = sample_wide(seed, n, None)
        if kind == "final":
            p = ParamPolicy(seed=20261024, **cfg)
        elif kind == "harvest":
            p = ParamHarvest(seed=20261024, **cfg)
        elif kind == "gap":
            p = GapStartPolicy(cfg["start"], cfg["threshold"])
        else:
            raise ValueError(kind)
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
        scores.append(total / n)
    return scores

def make_configs():
    rng = random.Random(2026)
    configs = []
    # Final family
    for _ in range(100):
        configs.append(("final", {
            "P_UNPROVOKED": rng.uniform(.10, .60),
            "P_MUTUAL": rng.uniform(.15, .75),
            "P_RESPONSIVE": rng.uniform(.80, 1.0),
            "MIN_C": rng.randint(3, 7),
            "MIN_D": rng.randint(2, 5),
            "RESP_GAP": rng.uniform(.25, .55),
            "DEFECT_RATE": rng.uniform(.10, .26),
            "PEACE_TESTS": rng.randint(1, 3),
            "PEACE_COOLDOWN": rng.randint(4, 14),
        }))
    # Harvest family
    for _ in range(100):
        configs.append(("harvest", {
            "P_UNPROVOKED": rng.uniform(.10, .50),
            "P_MUTUAL": rng.uniform(.15, .65),
            "P_RESPONSIVE": rng.uniform(.85, 1.0),
            "MIN_C": rng.randint(3, 6),
            "MIN_D": rng.randint(2, 4),
            "RESP_GAP": rng.uniform(.28, .52),
            "DEFECT_RATE": rng.uniform(.12, .24),
            "PEACE_TESTS": rng.randint(1, 3),
            "PEACE_COOLDOWN": rng.randint(4, 14),
            "HARVEST_START": rng.randint(20, 80),
        }))
    # Gap family
    for _ in range(80):
        configs.append(("gap", {
            "start": rng.randint(1, 75),
            "threshold": rng.uniform(.35, .65),
        }))
    return configs

def fitness(scores):
    mean = statistics.fmean(scores)
    low = sorted(scores)[max(0, int(.05 * len(scores)))]
    mn = min(scores)
    return 0.55 * mean + 0.25 * low + 0.20 * mn

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--configs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--train-fields", type=int, default=100)
    ap.add_argument("--val-fields", type=int, default=100)
    ap.add_argument("--out", default=result_path("train_wide_results.json"))
    args = ap.parse_args(argv)
    configs = make_configs()
    if args.configs:
        configs = configs[:args.configs]
    rng = random.Random(args.seed)
    train_seeds = [rng.randrange(1 << 30) for _ in range(args.train_fields)]
    val_seeds = [rng.randrange(1 << 30) for _ in range(args.val_fields)]
    print("configs", len(configs), "train", len(train_seeds), "val", len(val_seeds))
    tasks = [(kind, cfg, train_seeds, args.n) for kind, cfg in configs]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        all_scores = list(ex.map(eval_cfg, tasks, chunksize=1))
    results = []
    for (kind, cfg), scores in zip(configs, all_scores):
        fit = fitness(scores)
        results.append({"kind": kind, "cfg": cfg, "fit": fit,
                        "mean": statistics.fmean(scores),
                        "min": min(scores), "scores": scores})
    print("train done elapsed", round(time.time()-t0,1), flush=True)
    # top by family
    print("\nTop train by family:")
    selected = []
    for kind in ("final", "harvest", "gap"):
        rs = [r for r in results if r["kind"] == kind]
        rs.sort(key=lambda r: r["fit"], reverse=True)
        for r in rs[:4]:
            print(f"{kind:8} fit={r['fit']:7.2f} mean={r['mean']:7.2f} "
                  f"min={r['min']:7.2f} cfg={r['cfg']}")
            selected.append(r)
    print("\nValidation:")
    val_tasks = [(r["kind"], r["cfg"], val_seeds, 299) for r in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        val_scores = list(ex.map(eval_cfg, val_tasks, chunksize=1))
    val_rows = []
    for r, scores in zip(selected, val_scores):
        vfit = fitness(scores)
        row = {"kind": r["kind"], "cfg": r["cfg"], "train_fit": r["fit"],
               "val_fit": vfit, "val_mean": statistics.fmean(scores),
               "val_min": min(scores)}
        val_rows.append(row)
        print(f"{r['kind']:8} val_mean={row['val_mean']:7.2f} "
              f"val_min={row['val_min']:7.2f} val_fit={vfit:7.2f}")
    val_rows.sort(key=lambda r: r["val_fit"], reverse=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"train": sorted(results, key=lambda r: r["fit"], reverse=True)[:50],
                   "validation": val_rows}, f, ensure_ascii=False, indent=2)
    print("written", args.out)

if __name__ == "__main__":
    main()
