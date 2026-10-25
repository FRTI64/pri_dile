#!/usr/bin/env python3
"""Compare strategies on the original mixed prior (no late-harvest-heavy prior).

The 85-type opponent bank is used, but the 'late_harvest' scenario is removed
from the scenario mixture.  Late-defector types still occur at their normal
low weights through the 'defector' and 'naive' priors.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import statistics

import ipd_extended as ext
from ipd_strategy import Machine
from ipd_ema import EMAMachine
from ipd_ema_forgive import EMAForgiveMachine
from ipd_bayes import BayesMachine
from ipd_bayes_plus import BayesPlusMachine
from ipd_harvest import HarvestMachine
from ipd_early import EarlyProbeMachine
from ipd_field import Responsive3, ProbeBot, GenerousTFT, TFT2LastD
from ipd_sim import AllC, AllD, TFT, TFTLastD, play_match
from run_large_experiments import evaluate, q, summarize

class Final(Machine):
    name = "Final"
    def reset(self): super().reset()
    def act(self, t, mine, theirs): return self.move(t + 1, mine, theirs)

CANDIDATES = {
    "Final": lambda r: Final(seed=20261024),
    "EMAMachine": lambda r: EMAMachine(seed=20261024),
    "EMAForgive": lambda r: EMAForgiveMachine(seed=20261024),
    "Bayes": lambda r: BayesMachine(seed=20261024),
    "BayesPlus": lambda r: BayesPlusMachine(seed=20261024),
    "Harvest45": lambda r: HarvestMachine(seed=20261024),
    "EarlyProbe": lambda r: EarlyProbeMachine(seed=20261024),
    "ProbeBot": lambda r: ProbeBot(rng=r),
    "Responsive3": lambda r: Responsive3(rng=r),
    "GTFT.2-lastD": lambda r: GenerousTFT(0.2, True, r),
    "TFT2-lastD": lambda r: TFT2LastD(r),
    "TFT-lastD": lambda r: TFTLastD(r),
    "TFT": lambda r: TFT(r),
    "AllC": lambda r: AllC(r),
    "AllD": lambda r: AllD(r),
}

# The scenario weights used before adding the late-harvest-heavy scenario.
ORIGINAL_SCENARIO_WEIGHTS = {
    "balanced": 22, "cooperative": 8, "defector": 8, "naive": 8,
    "sophisticated": 22, "noisy": 12, "punitive": 10, "wild": 10,
}

def original_population(n, rng):
    scenario = rng.choices(list(ORIGINAL_SCENARIO_WEIGHTS),
                           weights=list(ORIGINAL_SCENARIO_WEIGHTS.values()),
                           k=1)[0]
    specs, probs = ext.sample_mixture(rng, scenario)
    pop = []
    for _ in range(n):
        spec = rng.choices(specs, weights=probs, k=1)[0]
        obj = spec.factory(rng)
        obj.name = spec.name
        pop.append(obj)
    return pop, scenario

def full_rank(field, cls):
    pop = [cls(seed=20261024)] + list(field)
    n = len(pop); scores = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            r = play_match(pop[i], pop[j])
            scores[i] += r.score_a
            scores[j] += r.score_b
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return order.index(0) + 1, scores[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=300)
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--rank-fields", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    master = random.Random(args.seed)
    fields = []
    scen_count = {}
    for _ in range(args.fields):
        pop, scen = original_population(args.n, random.Random(master.random()))
        fields.append(pop)
        scen_count[scen] = scen_count.get(scen, 0) + 1
    print("scenario counts", dict(sorted(scen_count.items())))

    totals = {name: [] for name in CANDIDATES}
    for fi, field in enumerate(fields):
        for name, factory in CANDIDATES.items():
            total, _, _ = evaluate(factory, copy.deepcopy(field))
            totals[name].append(total)
        if (fi + 1) % 50 == 0:
            print(f"field {fi + 1}/{args.fields} done", flush=True)

    print(f"\n{'strategy':16} {'mean/opp':>9} {'median':>9} {'p05':>8} "
          f"{'p95':>8} {'worst':>8}")
    for name, vals in sorted(totals.items(),
                             key=lambda kv: statistics.fmean(kv[1]),
                             reverse=True):
        s = summarize(vals)
        print(f"{name:16} {s['mean']/args.n:9.2f} {s['median']/args.n:9.2f} "
              f"{s['p05']/args.n:8.2f} {s['p95']/args.n:8.2f} "
              f"{s['min']/args.n:8.2f}")

    if args.rank_fields:
        print(f"\nFull rank on {args.rank_fields} fields")
        for name in ("Final", "BayesPlus", "Harvest45", "EarlyProbe"):
            cls = None
            # Build a small factory class with .name for play_match.
            if name == "Final":
                cls = Final
            elif name == "BayesPlus":
                cls = BayesPlusMachine
            elif name == "Harvest45":
                cls = HarvestMachine
            elif name == "EarlyProbe":
                cls = EarlyProbeMachine
            ranks = []; scores = []
            for i in range(min(args.rank_fields, len(fields))):
                r, s = full_rank(copy.deepcopy(fields[i]), cls)
                ranks.append(r); scores.append(s)
            print(f"{name:12} score/opp={statistics.fmean(scores)/args.n:7.2f} "
                  f"meanrank={statistics.fmean(ranks):6.2f} "
                  f"rank1={sum(x == 1 for x in ranks)}/{len(ranks)} "
                  f"top3={sum(x <= 3 for x in ranks)}/{len(ranks)} "
                  f"top10={sum(x <= 10 for x in ranks)}/{len(ranks)}")

if __name__ == "__main__":
    main()
