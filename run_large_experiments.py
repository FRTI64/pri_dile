#!/usr/bin/env python3
"""Large-sample mixed-population experiments for the final IPD strategy.

- 85 opponent types (textbook, parameterised, random memory-one/reactive/FSM,
  zero-determinant, adaptive and endgame strategies).
- Population weights are drawn from a Dirichlet mixture around eight different
  scenario priors, so both the type proportions and the concentration vary.
- 300 participants per field (our machine + 299 opponents).
- Optional full round-robin ranking on a subset of fields.

Examples:
    python3 run_large_experiments.py --fields 300 --seed 2026
    python3 run_large_experiments.py --fields 300 --rank-fields 100
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import statistics
from collections import Counter, defaultdict

from ipd_strategy import Machine
from ipd_sim import C, D, payoff, play_match
from ipd_extended import (
    TYPE_BANK, TYPE_BY_NAME, SCENARIOS,
    sample_mixed_population,
)
from ipd_field import Responsive3, ProbeBot, GenerousTFT, TFT2LastD
from ipd_sim import AllC, AllD, TFT, TFTLastD
from ipd_harvest import HarvestMachine


class FinalAdapter(Machine):
    name = "Final"

    def reset(self) -> None:
        super().reset()

    def act(self, t, mine, theirs):
        return self.move(t + 1, mine, theirs)


CANDIDATES = {
    "Final": lambda r: FinalAdapter(seed=20261024),
    "Harvest45": lambda r: HarvestMachine(seed=20261024),
    "Responsive3": lambda r: Responsive3(rng=r),
    "GTFT.2-lastD": lambda r: GenerousTFT(0.2, True, r),
    "TFT2-lastD": lambda r: TFT2LastD(r),
    "TFT-lastD": lambda r: TFTLastD(r),
    "TFT": lambda r: TFT(r),
    "AllC": lambda r: AllC(r),
    "AllD": lambda r: AllD(r),
    "ProbeBot": lambda r: ProbeBot(rng=r),
}

TYPE_FAMILY = {spec.name: spec.family for spec in TYPE_BANK}


def evaluate(factory, field, rounds=100):
    """Return total, per-type sum, and per-type count for one field."""
    me = factory(random.Random(1234567))
    total = 0
    sums = Counter()
    counts = Counter()
    for opp in field:
        me.reset()
        opp.reset()
        mh: list[int] = []
        oh: list[int] = []
        s = 0
        for t in range(rounds):
            x = me.act(t, mh, oh)
            y = opp.act(t, oh, mh)
            mh.append(x)
            oh.append(y)
            s += payoff(x, y)
        total += s
        sums[opp.name] += s
        counts[opp.name] += 1
    return total, sums, counts


def q(vals, p):
    xs = sorted(vals)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize(vals):
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "p05": q(vals, 0.05),
        "p95": q(vals, 0.95),
        "min": min(vals),
        "max": max(vals),
    }


def full_rank(field, rounds=100):
    """Full round robin: our machine plus the 299 field opponents."""
    pop = [FinalAdapter(seed=20261024)] + list(field)
    n = len(pop)
    scores = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            result = play_match(pop[i], pop[j], rounds)
            scores[i] += result.score_a
            scores[j] += result.score_b
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    rank = order.index(0) + 1
    return rank, scores[0], scores[order[1]], pop[order[1]].name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=300)
    ap.add_argument("--n", type=int, default=299,
                    help="opponents per field; total participants = n+1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rank-fields", type=int, default=0)
    ap.add_argument("--csv", default="ipd_extended_results.csv")
    args = ap.parse_args()

    master = random.Random(args.seed)
    # A field is used identically for every candidate.
    fields = []
    for _ in range(args.fields):
        pop, scenario = sample_mixed_population(args.n, random.Random(master.random()))
        fields.append((pop, scenario))

    totals = {name: [] for name in CANDIDATES}
    type_sums = {name: Counter() for name in CANDIDATES}
    type_counts = {name: Counter() for name in CANDIDATES}
    by_scenario = {name: defaultdict(list) for name in CANDIDATES}
    scenario_counts = Counter()

    for fi, (field, scenario) in enumerate(fields):
        scenario_counts[scenario] += 1
        for name, factory in CANDIDATES.items():
            # Deep copies make the random opponents see exactly the same
            # initial state for every candidate.
            total, sums, counts = evaluate(factory, copy.deepcopy(field))
            totals[name].append(total)
            type_sums[name].update(sums)
            type_counts[name].update(counts)
            by_scenario[name][scenario].append(total)
        if (fi + 1) % 25 == 0:
            print(f"field {fi + 1}/{args.fields} done", flush=True)

    print(f"\nSample: {args.fields} fields x {args.n + 1} participants, "
          f"{len(TYPE_BANK)} types, seed={args.seed}")
    print("Scenario counts:", dict(sorted(scenario_counts.items())))
    print(f"\n{'strategy':16} {'mean/opp':>9} {'median':>9} {'p05':>8} "
          f"{'p95':>8} {'min':>8} {'max':>8} {'std':>8}")
    for name, vals in sorted(totals.items(),
                             key=lambda kv: statistics.fmean(kv[1]),
                             reverse=True):
        s = summarize(vals)
        print(f"{name:16} {s['mean']/args.n:9.2f} {s['median']/args.n:9.2f} "
              f"{s['p05']/args.n:8.2f} {s['p95']/args.n:8.2f} "
              f"{s['min']/args.n:8.2f} {s['max']/args.n:8.2f} "
              f"{s['std']/args.n:8.2f}")

    print("\nFinal strategy mean score by scenario (points/opponent):")
    for scen in SCENARIOS:
        vals = by_scenario["Final"].get(scen, [])
        if vals:
            print(f"  {scen:15} n={len(vals):3d} "
                  f"mean={statistics.fmean(vals)/args.n:7.2f} "
                  f"p05={q(vals, 0.05)/args.n:7.2f} "
                  f"p95={q(vals, 0.95)/args.n:7.2f}")

    # Detailed exact per-type averages for the final strategy.
    rows = []
    for typ, cnt in type_counts["Final"].items():
        mean = type_sums["Final"][typ] / cnt
        row = {
            "type": typ,
            "family": TYPE_FAMILY.get(typ, "other"),
            "matches": cnt,
            "final": mean,
        }
        for name in CANDIDATES:
            if name == "Final":
                continue
            c = type_counts[name].get(typ, 0)
            row[name] = (type_sums[name][typ] / c) if c else ""
        rows.append(row)
    rows.sort(key=lambda r: r["final"], reverse=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-type details written to {args.csv}")

    print("\nBest/worst opponent types for Final:")
    for r in rows[:8]:
        print(f"  {r['type']:24} {r['family']:12} {r['final']:7.1f}")
    print("  ...")
    for r in rows[-8:]:
        print(f"  {r['type']:24} {r['family']:12} {r['final']:7.1f}")

    if args.rank_fields:
        print(f"\nFull round-robin ranking on {args.rank_fields} fields...")
        ranks = []
        wins = 0
        top3 = 0
        top10 = 0
        rank_rows = []
        for i in range(min(args.rank_fields, len(fields))):
            field, scenario = fields[i]
            rank, score, second, second_name = full_rank(copy.deepcopy(field))
            ranks.append(rank)
            wins += rank == 1
            top3 += rank <= 3
            top10 += rank <= 10
            rank_rows.append((i + 1, scenario, rank, score, second_name))
            if (i + 1) % 10 == 0:
                print(f"  rank field {i + 1}/{args.rank_fields} done", flush=True)
        print(f"rank: mean={statistics.fmean(ranks):.2f} "
              f"median={statistics.median(ranks):.0f} "
              f"best={min(ranks)} worst={max(ranks)}")
        print(f"rank 1: {wins}/{len(ranks)} ({wins/len(ranks):.1%})   "
              f"top3: {top3/len(ranks):.1%}   top10: {top10/len(ranks):.1%}")
        with open("ipd_rank_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["field", "scenario", "rank", "score", "second_name"])
            w.writerows(rank_rows)


if __name__ == "__main__":
    main()
