#!/usr/bin/env python3
"""P2c: complexity-limited stateful GP (small finite-state machines).

123 implementation:
1. stateful FSM genome instead of a stateless decision tree;
2. 100 original + 100 late-heavy training fields;
3. multi-objective fitness: mean, worst-family mean, 5% quantile and
   rank against a baseline reference set.

Complexity is capped at 6 states; each state has one action and one binary
feature test controlling the next state.
"""
from __future__ import annotations

from paths import result_path

import argparse
import copy
import json
import math
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from ipd_sim import C, D, payoff
from search_common import (
    build_field, N_PLAYERS, make_policy, evaluate_parallel, FieldSet,
)
from p2_gp_search import FEATURES, feature_vector

MAX_STATES = 6


class FSMPolicy:
    def __init__(self, genome):
        self.g = genome
        self.state = 0

    def reset(self):
        self.state = 0

    def act(self, t, mine, theirs):
        if t >= 99:
            return D
        g = self.g
        s = self.state
        a = g["act"][s]
        x = feature_vector(t + 1, mine, theirs)
        self.state = g["left"][s] if x[g["feat"][s]] <= g["thr"][s] else g["right"][s]
        return a


def new_fsm(rng, n_states=None):
    # Initial genomes stay moderate; mutation can grow them up to MAX_STATES.
    n = n_states if n_states is not None else rng.randint(2, min(MAX_STATES, 8))
    n = max(1, min(MAX_STATES, n))
    return {
        "act": [C if rng.random() < 0.62 else D for _ in range(n)],
        "feat": [rng.randrange(len(FEATURES)) for _ in range(n)],
        "thr": [rng.random() for _ in range(n)],
        "left": [rng.randrange(n) for _ in range(n)],
        "right": [rng.randrange(n) for _ in range(n)],
    }


def fsm_seeds():
    out = []
    # constant C / D
    out.append({"act": [C], "feat": [0], "thr": [0.5], "left": [0], "right": [0]})
    out.append({"act": [D], "feat": [0], "thr": [0.5], "left": [0], "right": [0]})
    # TFT: state 0 C, state 1 D; feature opp_last
    f = FEATURES.index("opp_last")
    out.append({"act": [C, D], "feat": [f, f], "thr": [0.5, 0.5],
                "left": [0, 0], "right": [1, 1]})
    # simple conditional reciprocity using gap
    fg = FEATURES.index("gap")
    out.append({"act": [C, D], "feat": [fg, fg], "thr": [0.55, 0.55],
                "left": [1, 1], "right": [0, 0]})
    # scheduled harvest at several days
    fd = FEATURES.index("day")
    for day in (40, 50, 60, 70, 80, 90):
        out.append({"act": [C, D], "feat": [fd, fd], "thr": [day/100.0, day/100.0],
                    "left": [0, 1], "right": [1, 1]})
    return out


def norm_fsm(g):
    n = max(1, min(MAX_STATES, len(g["act"])))
    for k in ("act", "feat", "thr", "left", "right"):
        g[k] = list(g[k])[:n]
    for k in ("left", "right"):
        g[k] = [max(0, min(n - 1, int(x))) for x in g[k]]
    for i in range(n):
        g["feat"][i] = max(0, min(len(FEATURES)-1, int(g["feat"][i])))
        g["thr"][i] = max(0.0, min(1.0, float(g["thr"][i])))
        g["act"][i] = C if g["act"][i] == C else D
    return g


def mutate(g, rng):
    g = copy.deepcopy(g)
    n = len(g["act"])
    r = rng.random()
    if r < 0.10 and n < MAX_STATES:
        # add a state
        g["act"].append(C if rng.random() < .6 else D)
        g["feat"].append(rng.randrange(len(FEATURES)))
        g["thr"].append(rng.random())
        g["left"].append(rng.randrange(n + 1))
        g["right"].append(rng.randrange(n + 1))
        n += 1
    elif r < 0.18 and n > 1:
        # delete a state, remap indices
        k = rng.randrange(n)
        for key in ("act", "feat", "thr", "left", "right"):
            g[key].pop(k)
        for key in ("left", "right"):
            g[key] = [x - 1 if x > k else x for x in g[key]]
        n -= 1
    elif r < 0.45:
        i = rng.randrange(n); g["act"][i] = C if g["act"][i] == D else D
    elif r < 0.70:
        i = rng.randrange(n); g["feat"][i] = rng.randrange(len(FEATURES))
        g["thr"][i] = rng.random()
    else:
        i = rng.randrange(n); side = rng.choice(("left", "right"))
        g[side][i] = rng.randrange(n)
    return norm_fsm(g)


def crossover(a, b, rng):
    a = copy.deepcopy(a); b = copy.deepcopy(b)
    n = min(len(a["act"]), len(b["act"]))
    # uniform state crossover, keeping parent A's index space
    for i in range(n):
        if rng.random() < 0.5:
            a["act"][i] = b["act"][i]
            a["feat"][i] = b["feat"][i]
            a["thr"][i] = b["thr"][i]
            a["left"][i] = min(len(a["act"])-1, b["left"][i])
            a["right"][i] = min(len(a["act"])-1, b["right"][i])
    return norm_fsm(a)


def eval_fsm_field(g, kind, seed, n):
    field, _ = build_field(kind, seed, n)
    pol = FSMPolicy(g)
    total = 0
    for opp in field:
        pol.reset(); opp.reset()
        mine = []; theirs = []; s = 0
        for t in range(100):
            x = pol.act(t, mine, theirs)
            y = opp.act(t, theirs, mine)
            mine.append(x); theirs.append(y)
            s += payoff(x, y)
        total += s
    return total / n


def _eval_chunk(args):
    gs, specs, n = args
    return [[eval_fsm_field(g, kind, seed, n) for kind, seed in specs]
            for g in gs]


def q(vals, p):
    xs = sorted(vals); pos = (len(xs)-1)*p
    lo = int(pos); hi = min(lo+1, len(xs)-1)
    return xs[lo]*(1-(pos-lo)) + xs[hi]*(pos-lo)


def reference_scores(specs, n, workers):
    refs = {
        "Final": {"kind": "final"},
        "Harvest45": {"kind": "harvest", "HARVEST_START": 45},
        "EarlyProbe": {"kind": "early"},
        "GTFT2": {"kind": "param", "P_UNPROVOKED": .2, "P_MUTUAL": .4,
                  "P_RESPONSIVE": .9, "HARVEST_START": None},
        "TFT": {"kind": "param", "P_UNPROVOKED": .0, "P_MUTUAL": .0,
                "P_RESPONSIVE": .0, "MIN_C": 99, "MIN_D": 99,
                "RESP_GAP": 2.0, "DEFECT_RATE": 2.0, "HARVEST_START": None},
    }
    # The above param specs are not exact TFT/GTFT; use only the meaningful
    # reference entries for rank.  Their absolute values are not critical.
    out = {}
    fs = FieldSet(specs)
    for name, spec in refs.items():
        out[name] = [x / n for x in evaluate_parallel(spec, fs, workers, n)]
    return out


def fields(seed=2026, n_each=100, n=60):
    rng = random.Random(seed + 4242)
    specs = ([("original", rng.randrange(1 << 30)) for _ in range(n_each)] +
             [("late", rng.randrange(1 << 30)) for _ in range(n_each)])
    return specs, n


def val_fields(seed=2026, n_each=20, n=N_PLAYERS):
    rng = random.Random(seed + 9090)
    specs = ([("original", rng.randrange(1 << 30)) for _ in range(n_each)] +
             [("late", rng.randrange(1 << 30)) for _ in range(n_each)])
    return specs, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=60)
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=result_path("p2_fsm_results.json"))
    ap.add_argument("--max-states", type=int, default=6)
    args = ap.parse_args()

    global MAX_STATES
    MAX_STATES = max(1, args.max_states)
    specs, n_train = fields(args.seed, 100, 60)
    v_specs, v_n = val_fields(args.seed, 20, N_PLAYERS)
    print("precomputing reference scores...", flush=True)
    refs = reference_scores(specs, n_train, args.workers)
    ref_means = {k: statistics.fmean(v) for k, v in refs.items()}
    print("refs", {k: round(v, 1) for k, v in ref_means.items()}, flush=True)

    rng = random.Random(args.seed)
    seeds = fsm_seeds()
    pop = seeds + [new_fsm(rng) for _ in range(args.population - len(seeds))]
    pop = pop[:args.population]
    cache = {}
    t0 = time.time()

    def evaluate_population(pop):
        todo = [g for g in pop if repr(g) not in cache]
        if todo:
            chunks = [todo[i::args.workers] for i in range(args.workers)]
            tasks = [(ch, specs, n_train) for ch in chunks if ch]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(_eval_chunk, tasks))
            flat = [x for sub in res for x in sub]
            for g, sc in zip(todo, flat):
                cache[repr(g)] = sc
        return [cache[repr(g)] for g in pop]

    def fit_one(scores, g):
        orig = [x for x, (k, _) in zip(scores, specs) if k == "original"]
        late = [x for x, (k, _) in zip(scores, specs) if k == "late"]
        mean = statistics.fmean(scores)
        worst = min(statistics.fmean(orig), statistics.fmean(late))
        low = q(scores, 0.05)
        # rank against reference strategies on each field
        ranks = []
        for i, s in enumerate(scores):
            rr = 1 + sum(1 for k in refs if refs[k][i] > s)
            ranks.append(rr)
        mean_rank = statistics.fmean(ranks)
        rank_score = 100.0 * (len(refs) + 1 - mean_rank) / len(refs)
        f = (0.42*mean + 0.22*worst + 0.18*low + 0.18*rank_score
             - 0.45*len(g["act"]))
        return f, mean, worst, low, mean_rank

    for gen in range(args.generations):
        all_scores = evaluate_population(pop)
        fits = [fit_one(sc, g) for g, sc in zip(pop, all_scores)]
        order = sorted(range(len(pop)), key=lambda i: fits[i][0], reverse=True)
        elite = [pop[i] for i in order[:8]]
        f0 = fits[order[0]]
        print(f"gen {gen:2d} fit={f0[0]:7.2f} mean={f0[1]:7.2f} "
              f"worst={f0[2]:7.2f} p05={f0[3]:7.2f} rank={f0[4]:5.2f} "
              f"states={len(pop[order[0]]['act'])} elapsed={time.time()-t0:.1f}s",
              flush=True)
        if gen == args.generations - 1:
            break
        new = elite[:]
        while len(new) < args.population:
            p1 = pop[rng.choice(order[:max(10, len(order)//2)])]
            if rng.random() < 0.5:
                p2 = pop[rng.choice(order[:max(10, len(order)//2)])]
                child = crossover(p1, p2, rng)
            else:
                child = copy.deepcopy(p1)
            if rng.random() < 0.8:
                child = mutate(child, rng)
            new.append(norm_fsm(child))
        pop = new

    all_scores = evaluate_population(pop)
    fits = [fit_one(sc, g) for g, sc in zip(pop, all_scores)]
    order = sorted(range(len(pop)), key=lambda i: fits[i][0], reverse=True)
    best = [pop[i] for i in order[:12]]

    print("\nValidation on 20+20 full 299-player fields...", flush=True)
    rows = []
    for g in best:
        sc = [eval_fsm_field(g, kind, seed, v_n) for kind, seed in v_specs]
        orig = statistics.fmean([x for x, (k, _) in zip(sc, v_specs) if k == "original"])
        late = statistics.fmean([x for x, (k, _) in zip(sc, v_specs) if k == "late"])
        mean = statistics.fmean(sc)
        low = min(sc)
        # rank among the same reference bank on validation fields
        # (computed only for the aggregate score here)
        rows.append({"mean": mean, "orig": orig, "late": late, "min": low,
                     "states": len(g["act"]), "g": g})
    rows.sort(key=lambda r: 0.5*r["mean"] + 0.25*min(r["orig"], r["late"]) + 0.25*r["min"],
              reverse=True)
    for r in rows[:10]:
        print(f"  mean={r['mean']:7.2f} orig={r['orig']:7.2f} "
              f"late={r['late']:7.2f} min={r['min']:7.2f} states={r['states']}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "best": rows}, f,
                  ensure_ascii=False, indent=2)
    print("written", args.out)


if __name__ == "__main__":
    main()
