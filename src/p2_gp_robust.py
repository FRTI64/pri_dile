#!/usr/bin/env python3
"""P2b: robust multi-objective Genetic Programming.

Fitness penalises tail risk and the worse of the two field families:
    mean + min(original, late) + lower-quantile - complexity.
"""
from __future__ import annotations

from paths import result_path

import argparse
import copy
import json
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from ipd_sim import C, D, payoff
from search_common import build_field, N_PLAYERS
from p2_gp_search import (
    FEATURES, F, TreePolicy, rand_tree, mutate, crossover, nodes, seeds, leaf,
)

def val_fields(seed=2026, n=N_PLAYERS, n_each=8):
    rng = random.Random(seed + 555)
    specs = ([("original", rng.randrange(1 << 30)) for _ in range(n_each)] +
             [("late", rng.randrange(1 << 30)) for _ in range(n_each)])
    return specs, n


def train_fields(seed=2026, n=100, n_each=10):
    rng = random.Random(seed + 333)
    specs = ([("original", rng.randrange(1 << 30)) for _ in range(n_each)] +
             [("late", rng.randrange(1 << 30)) for _ in range(n_each)])
    return specs, n


def eval_tree_field_scores(tree, specs, n):
    scores = []
    policy = TreePolicy(tree)
    for kind, seed in specs:
        field, _ = build_field(kind, seed, n)
        total = 0
        for opp in field:
            mine = []; theirs = []; s = 0
            for t in range(100):
                x = policy.act(t, mine, theirs)
                y = opp.act(t, theirs, mine)
                mine.append(x); theirs.append(y)
                s += payoff(x, y)
            total += s
        scores.append(total / n)
    return scores


def _eval_chunk(args):
    trees, specs, n = args
    return [eval_tree_field_scores(t, specs, n) for t in trees]


def q(vals, p):
    xs = sorted(vals)
    pos = (len(xs) - 1) * p
    lo = int(pos); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (pos - lo)) + xs[hi] * (pos - lo)


def fitness_from_scores(scores, tree):
    orig = [x for x, (k, _) in zip(scores, SPECS) if k == "original"]
    late = [x for x, (k, _) in zip(scores, SPECS) if k == "late"]
    mean = statistics.fmean(scores)
    worst_group = min(statistics.fmean(orig), statistics.fmean(late))
    lower = q(scores, 0.20)
    return 0.50 * mean + 0.25 * worst_group + 0.25 * lower - 0.08 * nodes(tree), mean, worst_group, lower


def main():
    global SPECS
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=80)
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=result_path("p2_robust_results.json"))
    args = ap.parse_args()

    SPECS, N_TRAIN = train_fields(args.seed, 100, 10)
    specs_val, n_val = val_fields(args.seed, N_PLAYERS, 8)
    rng = random.Random(args.seed)
    domain = seeds()
    # A hand-designed robust seed: cooperate by default; after the first
    # opponent D, use the conditional response gap to forgive or punish.
    domain.append(("n", F["ever_opp_d"], 0.5, leaf(C),
                   ("n", F["gap"], 0.55, leaf(D), leaf(C))))
    # A more forgiving variant.
    domain.append(("n", F["ever_opp_d"], 0.5, leaf(C),
                   ("n", F["gap"], 0.40, leaf(D), leaf(C))))
    pop = domain + [rand_tree(rng, 4, 5)
                    for _ in range(args.population - len(domain))]
    pop = pop[:args.population]
    cache = {}
    t0 = time.time()

    def eval_population(pop):
        todo = [t for t in pop if repr(t) not in cache]
        if todo:
            chunks = [todo[i::args.workers] for i in range(args.workers)]
            tasks = [(ch, SPECS, N_TRAIN) for ch in chunks if ch]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(_eval_chunk, tasks))
            flat = [s for sub in res for s in sub]
            for tree, scores in zip(todo, flat):
                cache[repr(tree)] = scores
        return [cache[repr(t)] for t in pop]

    for gen in range(args.generations):
        all_scores = eval_population(pop)
        fits = []
        meta = []
        for t, sc in zip(pop, all_scores):
            f, mean, wg, low = fitness_from_scores(sc, t)
            fits.append(f); meta.append((mean, wg, low))
        order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        elite = [pop[i] for i in order[:8]]
        print(f"gen {gen:2d} best={fits[order[0]]:7.2f} "
              f"mean={meta[order[0]][0]:7.2f} worstgrp={meta[order[0]][1]:7.2f} "
              f"p20={meta[order[0]][2]:7.2f} nodes={nodes(pop[order[0]])} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
        if gen == args.generations - 1:
            break
        new = elite[:]
        while len(new) < args.population:
            p1 = pop[rng.choice(order[:max(10, len(order)//2)])]
            if rng.random() < 0.55:
                p2 = pop[rng.choice(order[:max(10, len(order)//2)])]
                child = crossover(p1, p2, rng)
            else:
                child = copy.deepcopy(p1)
            if rng.random() < 0.80:
                child = mutate(child, rng)
            if nodes(child) > 50:
                child = rand_tree(rng, 3, 4)
            new.append(child)
        pop = new

    all_scores = eval_population(pop)
    fits = []
    for t, sc in zip(pop, all_scores):
        f, mean, wg, low = fitness_from_scores(sc, t)
        fits.append((f, mean, wg, low, nodes(t), t))
    fits.sort(reverse=True, key=lambda x: x[0])
    best = [x[-1] for x in fits[:10]]

    print("\nValidation on 8+8 full 299-player fields:")
    rows = []
    for t in best:
        sc = eval_tree_field_scores(t, specs_val, n_val)
        mean = statistics.fmean(sc)
        orig = statistics.fmean([x for x,(k,_) in zip(sc,specs_val) if k=="original"])
        late = statistics.fmean([x for x,(k,_) in zip(sc,specs_val) if k=="late"])
        low = min(sc)
        rows.append((mean, orig, late, low, nodes(t), t))
    rows.sort(reverse=True, key=lambda x: x[0])
    for mean, orig, late, low, nd, _ in rows:
        print(f"  mean={mean:7.2f} orig={orig:7.2f} late={late:7.2f} "
              f"min={low:7.2f} nodes={nd}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES,
                   "best": [{"mean": m, "orig": o, "late": l, "min": mn,
                             "nodes": nd, "tree": t}
                            for m,o,l,mn,nd,t in rows]},
                  f, ensure_ascii=False, indent=2)
    print("written", args.out)


if __name__ == "__main__":
    main()
