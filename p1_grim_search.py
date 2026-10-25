#!/usr/bin/env python3
"""P1b: parameter search trained on a fixed high-Grim population."""
from __future__ import annotations
import argparse, json, random, statistics, time
from search_common import FieldSet, evaluate_parallel, mean_per_opp, N_PLAYERS
from p1_parameter_search import BASE, random_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grim", type=int, default=30)
    ap.add_argument("--configs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="p1_grim_results.json")
    args = ap.parse_args()
    train = FieldSet([(f"grim{args.grim}", 12000 + i) for i in range(30)])
    val = FieldSet([(f"grim{args.grim}", 22000 + i) for i in range(30)])
    rng = random.Random(2026)
    configs = [("Final", {"kind": "final"}),
               ("Harvest90", {"kind": "harvest", "HARVEST_START": 90}),
               ("Harvest80", {"kind": "harvest", "HARVEST_START": 80}),
               ("EarlyProbe", {"kind": "early"})]
    seen = {json.dumps(c, sort_keys=True) for _, c in configs}
    while len(configs) < args.configs + 4:
        cfg = random_config(rng)
        key = json.dumps(cfg, sort_keys=True)
        if key in seen: continue
        seen.add(key); configs.append((f"cfg{len(configs)-3}", cfg))
    res = []
    t0 = time.time()
    for i, (name, cfg) in enumerate(configs):
        scores = evaluate_parallel(cfg, train, args.workers, N_PLAYERS)
        m = mean_per_opp(scores)
        res.append((m, name, cfg))
        if (i+1) % 10 == 0:
            print(f"train {i+1}/{len(configs)} elapsed={time.time()-t0:.1f}s", flush=True)
    res.sort(reverse=True, key=lambda r: r[0])
    print("\nTop train:")
    for m, name, cfg in res[:12]:
        print(f"{name:10} train={m:7.2f} harvest={cfg.get('HARVEST_START')} "
              f"pu={cfg.get('P_UNPROVOKED')} pm={cfg.get('P_MUTUAL')}")
    print("\nValidation:")
    rows = []
    for m, name, cfg in res[:12]:
        scores = evaluate_parallel(cfg, val, args.workers, N_PLAYERS)
        vm = mean_per_opp(scores)
        rows.append((vm, m, name, cfg))
        print(f"{name:10} val={vm:7.2f} train={m:7.2f} "
              f"harvest={cfg.get('HARVEST_START')}")
    rows.sort(reverse=True, key=lambda r: r[0])
    # ensure baselines are validated too
    base_rows = []
    for name, cfg in [("Final", {"kind": "final"}),
                      ("Harvest90", {"kind": "harvest", "HARVEST_START": 90}),
                      ("EarlyProbe", {"kind": "early"})]:
        scores = evaluate_parallel(cfg, val, args.workers, N_PLAYERS)
        base_rows.append((mean_per_opp(scores), name))
    print("baselines", [(n, round(v, 2)) for v, n in base_rows])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"grim_weight": args.grim,
                   "train": [{"name": n, "score": m, "cfg": c}
                             for m, n, c in res],
                   "validation": [{"name": n, "score": vm, "train": m, "cfg": c}
                                  for vm, m, n, c in rows],
                   "baselines": [{"name": n, "score": v} for v, n in base_rows]},
                  f, ensure_ascii=False, indent=2)
    print("written", args.out)

if __name__ == "__main__":
    main()
