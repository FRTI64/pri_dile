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

from paths import result_path
import argparse, copy, json, math, random, statistics, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import ipd_extended as ext
from ipd_sim import C, D, payoff
from ipd_strategy import Machine
from ipd_harvest import HarvestMachine
from ipd_early import EarlyProbeMachine
from ipd_bayes_plus import BayesPlusMachine
from ipd_field import Responsive3, GenerousTFT, NoisyTFT, Joss, RandomP
from p2_skeleton_search import GapStartPolicy
from search_common import ParamPolicy, ParamHarvest, ORIGINAL_SCENARIO_WEIGHTS
from run_smart_opponents import SMART, SmartAdapter

# Fine extra opponents on top of the 85-type bank.
EXTRA = []
def add_extra(name, weight, factory):
    EXTRA.append(ext.TypeSpec(name, "fine", weight, factory))

for g in (0.01, 0.03, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70):
    add_extra(f"GTFT.fine{g}", 1.0,
              lambda r, g=g: GenerousTFT(g, True, r))
for p in (0.01, 0.03, 0.08, 0.12, 0.18, 0.25, 0.35):
    add_extra(f"Joss.fine{p}", 1.0, lambda r, p=p: Joss(p, r))
    add_extra(f"Noisy.fine{p}", 1.0, lambda r, p=p: NoisyTFT(p, r))
for p in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60):
    add_extra(f"Random.fine{p}", 1.0, lambda r, p=p: RandomP(p, r))
for k in (1, 2, 3, 6, 8, 12, 18, 22, 28, 35, 45, 55, 60, 65, 70, 75, 80):
    from ipd_field import LateDefector
    add_extra(f"LateD.fine{k}", 1.0, lambda r, k=k: LateDefector(k, r))
for d in (3, 4, 5):
    from ipd_field import DelayedTFT
    add_extra(f"TFTdelay.fine{d}", 1.0, lambda r, d=d: DelayedTFT(d, r))

ALL_BASE = list(ext.TYPE_BANK) + EXTRA

def sample_wide(seed, n=60, smart_share=None):
    rng = random.Random(seed)
    scen = rng.choices(list(ORIGINAL_SCENARIO_WEIGHTS),
                       weights=list(ORIGINAL_SCENARIO_WEIGHTS.values()), k=1)[0]
    if rng.random() < 0.25:
        scen = "late_harvest"
    base_specs, base_w = ext.effective_weights(scen)
    # include fine extras at modest weight
    specs = list(base_specs) + list(EXTRA) + list(SMART)
    weights = list(base_w) + [s.weight for s in EXTRA] + [0.0] * len(SMART)
    if smart_share is None:
        smart_share = rng.choice([0.0, 0.03, 0.08, 0.13, 0.20, 0.30])
    base_total = sum(weights)
    smart_total = sum(s.weight for s in SMART)
    factor = 0.0 if smart_share <= 0 else (
        smart_share / (1.0 - smart_share) * base_total / smart_total)
    weights[-len(SMART):] = [factor * s.weight for s in SMART]
    pop = []
    for _ in range(n):
        spec = rng.choices(specs, weights=weights, k=1)[0]
        obj = spec.factory(rng)
        obj.name = spec.name
        pop.append(obj)
    return pop, scen, smart_share

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--train-fields", type=int, default=100)
    ap.add_argument("--val-fields", type=int, default=100)
    ap.add_argument("--out", default=result_path("train_wide_results.json"))
    args = ap.parse_args()
    configs = make_configs()
    if args.configs:
        configs = configs[:args.configs]
    rng = random.Random(4242)
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
