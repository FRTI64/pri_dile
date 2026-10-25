#!/usr/bin/env python3
"""Evaluate policy families under fixed elevated Grim proportions."""
from __future__ import annotations
import statistics
from ipd.training.search.common import FieldSet, evaluate_parallel, mean_per_opp, N_PLAYERS

def main(argv=None):
    specs = {
        "Final": {"kind": "final"},
        "Harvest40": {"kind": "harvest", "HARVEST_START": 40},
        "Harvest50": {"kind": "harvest", "HARVEST_START": 50},
        "Harvest60": {"kind": "harvest", "HARVEST_START": 60},
        "Harvest70": {"kind": "harvest", "HARVEST_START": 70},
        "Harvest80": {"kind": "harvest", "HARVEST_START": 80},
        "Harvest90": {"kind": "harvest", "HARVEST_START": 90},
        "EarlyProbe": {"kind": "early"},
    }
    print(f"{'Grim weight':>12} " + " ".join(f"{k:>11}" for k in specs))
    for g in (10, 20, 30, 40, 50, 60):
        fs = FieldSet([(f"grim{g}", 9000 + g * 100 + i) for i in range(50)])
        vals = {}
        for name, spec in specs.items():
            scores = evaluate_parallel(spec, fs, 24, N_PLAYERS)
            vals[name] = mean_per_opp(scores)
        print(f"{g:12d} " + " ".join(f"{vals[k]:11.1f}" for k in specs))

if __name__ == "__main__":
    main()
