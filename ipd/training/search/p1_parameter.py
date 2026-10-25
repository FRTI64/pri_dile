#!/usr/bin/env python3
"""P1: parameter search + held-out validation.

Searches a compact set of thresholds for the Final/Harvest state machine on
two training distributions:
  - original mixed field (8 priors, late harvest not emphasized)
  - late-harvest-heavy field
Then re-evaluates the best candidates on disjoint validation fields.
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
import time
from pathlib import Path

from ipd.training.search.common import (
    make_fieldsets, evaluate_parallel, mean_per_opp, N_PLAYERS,
)

BASE = {
    "kind": "param",
    "P_UNPROVOKED": 0.25,
    "P_MUTUAL": 0.45,
    "P_RESPONSIVE": 0.95,
    "MIN_C": 5,
    "MIN_D": 3,
    "RESP_GAP": 0.38,
    "DEFECT_RATE": 0.18,
    "PEACE_TESTS": 2,
    "PEACE_COOLDOWN": 8,
    "HARVEST_START": None,
}

def random_config(rng: random.Random):
    cfg = dict(BASE)
    cfg["P_UNPROVOKED"] = rng.choice([0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    cfg["P_MUTUAL"] = rng.choice([0.20, 0.30, 0.35, 0.45, 0.55, 0.65])
    cfg["P_RESPONSIVE"] = rng.choice([0.85, 0.90, 0.95, 1.0])
    cfg["MIN_C"] = rng.choice([3, 4, 5, 6, 7])
    cfg["MIN_D"] = rng.choice([2, 3, 4])
    cfg["RESP_GAP"] = rng.choice([0.28, 0.34, 0.38, 0.44, 0.50])
    cfg["DEFECT_RATE"] = rng.choice([0.12, 0.15, 0.18, 0.22, 0.26])
    cfg["PEACE_TESTS"] = rng.choice([1, 2, 3])
    cfg["PEACE_COOLDOWN"] = rng.choice([4, 6, 8, 10, 14])
    hs = rng.choice([None, 40, 45, 50, 55, 60, 70, 80])
    if hs is None:
        cfg["kind"] = "param"
        cfg["HARVEST_START"] = None
    else:
        cfg["kind"] = "param_harvest"
        cfg["HARVEST_START"] = hs
    return cfg

def score_cfg(cfg, fields, workers):
    vals = evaluate_parallel(cfg, fields, workers)
    return mean_per_opp(vals), vals

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=result_path("p1_results.json"))
    args = ap.parse_args(argv)

    fs = make_fieldsets(seed=args.seed, n_orig=20, n_late=20)
    rng = random.Random(args.seed)
    configs = []
    seen = set()
    # Include hand-designed baselines.
    for name, cfg in [
        ("Final", {"kind": "final"}),
        ("Harvest45", {"kind": "harvest", "HARVEST_START": 45}),
        ("Harvest50", {"kind": "harvest", "HARVEST_START": 50}),
        ("Harvest60", {"kind": "harvest", "HARVEST_START": 60}),
    ]:
        configs.append((name, cfg)); seen.add(json.dumps(cfg, sort_keys=True))
    while len(configs) < args.configs + 4:
        cfg = random_config(rng)
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        configs.append((f"cfg{len(configs)-3}", cfg))

    results = []
    t0 = time.time()
    for i, (name, cfg) in enumerate(configs):
        so, _ = score_cfg(cfg, fs["train_original"], args.workers)
        sl, _ = score_cfg(cfg, fs["train_late"], args.workers)
        combined = 0.5 * so + 0.5 * sl
        results.append({"name": name, "cfg": cfg, "train_original": so,
                        "train_late": sl, "combined": combined})
        if (i + 1) % 10 == 0:
            print(f"train {i+1}/{len(configs)} elapsed={time.time()-t0:.1f}s",
                  flush=True)

    results.sort(key=lambda r: r["combined"], reverse=True)
    print("\nTop training candidates:")
    for r in results[:12]:
        print(f"{r['name']:10} orig={r['train_original']:7.2f} "
              f"late={r['train_late']:7.2f} comb={r['combined']:7.2f} "
              f"harvest={r['cfg'].get('HARVEST_START')}")

    # Validation on disjoint fields.
    print("\nValidation of top candidates + baselines:")
    val = []
    baselines = results[:12]
    # Ensure the three hand-designed baselines are included.
    names = {r["name"] for r in baselines}
    for r in results:
        if r["name"] in ("Final", "Harvest45", "Harvest50", "Harvest60") and r["name"] not in names:
            baselines.append(r); names.add(r["name"])
    for r in baselines:
        vo, _ = score_cfg(r["cfg"], fs["val_original"], args.workers)
        vl, _ = score_cfg(r["cfg"], fs["val_late"], args.workers)
        vc = 0.5 * vo + 0.5 * vl
        val.append({"name": r["name"], "cfg": r["cfg"],
                    "val_original": vo, "val_late": vl, "val_combined": vc,
                    "train_combined": r["combined"]})
        print(f"{r['name']:10} orig={vo:7.2f} late={vl:7.2f} "
              f"comb={vc:7.2f} train={r['combined']:7.2f} "
              f"harvest={r['cfg'].get('HARVEST_START')}")
    val.sort(key=lambda r: r["val_combined"], reverse=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"train": results, "validation": val}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nBest validation: {val[0]['name']} {val[0]['val_combined']:.2f}")
    print(f"written {args.out}")

if __name__ == "__main__":
    main()
