#!/usr/bin/env python3
"""P2: Genetic Programming over small decision trees.

Features are compact history statistics (not hidden state).  A tree chooses
C/D each day.  Fitness is the mean score over original and late-heavy training
fields, with a small complexity penalty.  Top trees are re-evaluated on
disjoint validation fields against Final and Harvest45.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from ipd_sim import C, D, payoff
from search_common import build_field, N_PLAYERS

# Feature names/order.  All numeric features are normalised to [0,1].
FEATURES = [
    "day", "remaining", "my_last", "opp_last", "ever_opp_d",
    "ever_my_d", "opp_d_rate", "recent5_d", "recent10_d",
    "pC", "pD", "gap", "recent_pC", "recent_pD", "recent_gap",
    "opp_c_run", "opp_d_run", "my_c_run", "mutual_dd",
    "alternating8", "first_d_day", "time_since_d",
]
F = {name: i for i, name in enumerate(FEATURES)}


def feature_vector(day, mine, theirs):
    n = len(theirs)
    day0 = day - 1
    my_last = mine[-1] if mine else C
    opp_last = theirs[-1] if theirs else C
    ever_opp_d = int(D in theirs)
    ever_my_d = int(D in mine)
    d_rate = sum(theirs) / n if n else 0.0
    recent5 = theirs[-5:]
    recent10 = theirs[-10:]
    r5 = sum(recent5) / len(recent5) if recent5 else 0.0
    r10 = sum(recent10) / len(recent10) if recent10 else 0.0

    nC = nD = dC = dD = 0
    for i in range(1, n):
        if mine[i - 1] == C:
            nC += 1; dC += theirs[i] == D
        else:
            nD += 1; dD += theirs[i] == D
    pC = dC / nC if nC else 0.5
    pD = dD / nD if nD else 0.5
    gap = (pD - pC + 1.0) / 2.0

    # Recent conditional stats over the last 12 transitions.
    rc = rd = rdc = rdd = 0
    start = max(1, n - 12)
    for i in range(start, n):
        if mine[i - 1] == C:
            rc += 1; rdc += theirs[i] == D
        else:
            rd += 1; rdd += theirs[i] == D
    rpC = rdc / rc if rc else 0.5
    rpD = rdd / rd if rd else 0.5
    rgap = (rpD - rpC + 1.0) / 2.0

    oc = od = 0
    for x in reversed(theirs):
        if x == C:
            if od: break
            oc += 1
        else:
            if oc: break
            od += 1
    mc = 0
    for x in reversed(mine):
        if x == C: mc += 1
        else: break
    mutual_dd = float(n >= 2 and mine[-1] == D and theirs[-1] == D)
    alt = float(n >= 8 and all(theirs[i] != theirs[i - 1]
                               for i in range(n - 7, n)))
    first_d = 100
    for i, x in enumerate(theirs):
        if x == D:
            first_d = i + 1; break
    since = (day0 - first_d) if first_d < 100 and day0 >= first_d else 0

    return [
        day0 / 100.0, (100 - day) / 100.0, float(my_last), float(opp_last),
        float(ever_opp_d), float(ever_my_d), d_rate, r5, r10,
        pC, pD, gap, rpC, rpD, rgap,
        min(oc, 100) / 100.0, min(od, 100) / 100.0, min(mc, 100) / 100.0,
        mutual_dd, alt, first_d / 100.0, since / 100.0,
    ]


class TreePolicy:
    def __init__(self, tree):
        self.tree = tree

    def act(self, t, mine, theirs):
        if t >= 99:
            return D
        x = feature_vector(t + 1, mine, theirs)
        node = self.tree
        while node[0] == "n":
            _, feat, thr, left, right = node
            node = left if x[feat] <= thr else right
        return node[1]


def leaf(action):
    return ("l", action)


def rand_tree(rng, depth=3, max_depth=4):
    if depth <= 0 or rng.random() < 0.28:
        return leaf(C if rng.random() < 0.6 else D)
    feat = rng.randrange(len(FEATURES))
    thr = rng.random()
    return ("n", feat, thr,
            rand_tree(rng, depth - 1, max_depth),
            rand_tree(rng, depth - 1, max_depth))


def nodes(tree):
    if tree[0] == "l":
        return 1
    return 1 + nodes(tree[3]) + nodes(tree[4])


def subtrees(tree):
    out = [tree]
    if tree[0] == "n":
        out += subtrees(tree[3]) + subtrees(tree[4])
    return out


def replace_random_subtree(tree, replacement, rng, target_depth=0):
    # Replace one random subtree (including the root) with `replacement`.
    candidates = []
    def walk(t, parent):
        candidates.append((t, parent))
        if t[0] == "n":
            walk(t[3], ("n", t[1], t[2], t[3], t[4], 3))
            walk(t[4], ("n", t[1], t[2], t[3], t[4], 4))
    walk(tree, None)
    _, parent = rng.choice(candidates)
    if parent is None:
        return replacement
    _, f, th, l, r, side = parent
    if side == 3:
        return ("n", f, th, replacement, r)
    return ("n", f, th, l, replacement)


def mutate(tree, rng, max_nodes=40):
    if rng.random() < 0.45:
        # leaf flip / subtree replacement
        return replace_random_subtree(tree, rand_tree(rng, 2), rng)
    # change a random internal node
    internal = []
    def walk(t):
        if t[0] == "n":
            internal.append(t); walk(t[3]); walk(t[4])
    walk(tree)
    if not internal:
        return rand_tree(rng, 2)
    old = rng.choice(internal)
    new = ("n", rng.randrange(len(FEATURES)), rng.random(),
           old[3], old[4])
    return replace_random_subtree(tree, new, rng)


def crossover(a, b, rng):
    sub = copy.deepcopy(rng.choice(subtrees(b)))
    return replace_random_subtree(a, sub, rng)


def evaluate_tree_fields(tree, specs, n):
    total = 0
    policy = TreePolicy(tree)
    for kind, seed in specs:
        field, _ = build_field(kind, seed, n)
        for opp in field:
            mine = []; theirs = []; s = 0
            for t in range(100):
                x = policy.act(t, mine, theirs)
                y = opp.act(t, theirs, mine)
                mine.append(x); theirs.append(y)
                s += payoff(x, y)
            total += s
    return total


def _eval_chunk(args):
    trees, specs, n = args
    return [evaluate_tree_fields(t, specs, n) for t in trees]


def train_fields(seed=2026, n=80):
    rng = random.Random(seed)
    specs = ([("original", rng.randrange(1 << 30)) for _ in range(6)] +
             [("late", rng.randrange(1 << 30)) for _ in range(6)])
    return specs, n


def val_fields(seed=2026, n=N_PLAYERS):
    rng = random.Random(seed + 777)
    specs = ([("original", rng.randrange(1 << 30)) for _ in range(8)] +
             [("late", rng.randrange(1 << 30)) for _ in range(8)])
    return specs, n


def seeds():
    out = [leaf(C), leaf(D)]
    # TFT-like and a few scheduled harvest policies.
    out.append(("n", F["opp_last"], 0.5, leaf(D), leaf(C)))
    for day in (40, 50, 60, 70, 80, 90):
        out.append(("n", F["day"], day / 100.0, leaf(D), leaf(C)))
    # Forgive a first D if the opponent has not defected yet.
    out.append(("n", F["ever_opp_d"], 0.5, leaf(D), leaf(C)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=60)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="p2_results.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    specs_tr, n_tr = train_fields(args.seed, 80)
    specs_val, n_val = val_fields(args.seed, N_PLAYERS)
    pop = seeds() + [rand_tree(rng, 4, 5) for _ in range(args.population - len(seeds()))]
    pop = pop[:args.population]
    cache = {}
    t0 = time.time()

    def eval_population(pop):
        todo = [t for t in pop if repr(t) not in cache]
        if todo:
            chunks = [todo[i::args.workers] for i in range(args.workers)]
            tasks = [(ch, specs_tr, n_tr) for ch in chunks if ch]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(_eval_chunk, tasks))
            flat = [x for sub in res for x in sub]
            for tree, total in zip(todo, flat):
                cache[repr(tree)] = total
        return [cache[repr(t)] for t in pop]

    for gen in range(args.generations):
        scores = eval_population(pop)
        fitness = [s / (len(specs_tr) * n_tr) - 0.12 * nodes(t)
                   for s, t in zip(scores, pop)]
        order = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)
        elite = [pop[i] for i in order[:6]]
        avg = statistics.fmean(fitness)
        best = fitness[order[0]]
        print(f"gen {gen:2d} best={best:7.2f} avg={avg:7.2f} "
              f"nodes={nodes(pop[order[0]])} elapsed={time.time()-t0:.1f}s",
              flush=True)
        if gen == args.generations - 1:
            break
        new = elite[:]
        while len(new) < args.population:
            p1 = pop[rng.choice(order[:max(8, len(order) // 2)])]
            if rng.random() < 0.55:
                p2 = pop[rng.choice(order[:max(8, len(order) // 2)])]
                child = crossover(p1, p2, rng)
            else:
                child = copy.deepcopy(p1)
            if rng.random() < 0.75:
                child = mutate(child, rng)
            if nodes(child) > 45:
                child = rand_tree(rng, 3, 4)
            new.append(child)
        pop = new

    scores = eval_population(pop)
    fitness = [s / (len(specs_tr) * n_tr) - 0.12 * nodes(t)
               for s, t in zip(scores, pop)]
    order = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)
    best_trees = [pop[i] for i in order[:10]]

    print("\nValidating top trees on full 299-player fields...")
    chunks = [best_trees[i::args.workers] for i in range(args.workers)]
    tasks = [(ch, specs_val, n_val) for ch in chunks if ch]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_eval_chunk, tasks))
    val_scores = [x for sub in res for x in sub]
    # Need map back in order because of striding; recompute simple serial for
    # the top trees to avoid ordering ambiguity.
    val_rows = []
    for t in best_trees:
        s = evaluate_tree_fields(t, specs_val, n_val)
        val_rows.append((s / (len(specs_val) * n_val), nodes(t), t))
    val_rows.sort(reverse=True, key=lambda x: x[0])
    for s, nd, _ in val_rows[:10]:
        print(f"  val/opp={s:7.2f} nodes={nd}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "features": FEATURES,
            "best_val": [{"score": s, "nodes": nd, "tree": t}
                         for s, nd, t in val_rows[:10]],
        }, f, ensure_ascii=False, indent=2)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
