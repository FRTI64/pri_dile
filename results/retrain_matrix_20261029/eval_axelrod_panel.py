#!/usr/bin/env python3
"""Targeted Axelrod panel: ZD, Meta and rewritten historical-threshold bots.

This is a diagnostic only.  It complements the Protocol v4 ``dev_axelrod``
family aggregate (axelrod / axelrod_extra / axelrod_mix) with direct matches
against the strategies whose internal behaviour changed most.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.core.opponents import axelrod_bank as ab  # noqa: E402
from ipd.evaluation.common import run_match  # noqa: E402
from ipd.policies.registry import CandidateSpec  # noqa: E402

PANEL = {
    "zd": ["ZDExtort2", "ZDExtort2v2", "ZDExtort3", "ZDExtort4",
           "ZDExtortion", "ZDGTFT2", "ZDGen2", "ZDMem2", "ZDMischief",
           "ZDSet2"],
    "meta": ["MetaHunter", "MetaHunterAggressive", "MetaMajority",
             "MetaMajorityMemoryOne", "MetaMixer", "MetaWinner",
             "MetaWinnerDeterministic", "MetaWinnerEnsemble",
             "MetaWinnerMemoryOne", "MetaWinnerStochastic", "NiceMetaWinner"],
    "historical": ["FirstByTidemanAndChieruzzi",
                   "SecondByTidemanAndChieruzzi", "FirstByGraaskamp",
                   "SecondByGraaskampKatzen", "SecondByRowsam",
                   "SecondByTranquilizer", "Stalker"],
}


def _task(spec_data: dict, strategy: str, seeds: list[int]) -> tuple:
    import torch
    torch.set_num_threads(1)
    ab.require_axelrod()
    spec = CandidateSpec.from_dict(spec_data)
    policy = spec.build()
    scores = []
    for seed in seeds:
        opponent = ab.axelrod_by_name()[strategy].factory(random.Random(seed))
        opponent.reset()
        result = run_match(policy, opponent, rounds=100)
        scores.append(result.score)
    return spec.name, strategy, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    specs = [CandidateSpec(kind=row["kind"], name=row["name"],
                           params={k: v for k, v in row.items()
                                   if k in ("weights", "mode")})
             for row in json.loads(Path(args.specs).read_text())["candidates"]]
    seeds = [20261027 + i * 7919 for i in range(args.seeds)]
    tasks = [(spec.to_dict(), strategy, seeds)
             for spec in specs
             for names in PANEL.values() for strategy in names]
    started = time.time()
    raw: dict[tuple[str, str], list[int]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_task, *task) for task in tasks]
        for fut in as_completed(futures):
            name, strategy, scores = fut.result()
            raw[(name, strategy)] = scores
    out = {"seed": 20261027, "seeds": seeds, "rounds": 100,
           "seconds": round(time.time() - started, 2),
           "note": "targeted diagnostic; not used for selection",
           "panel": PANEL, "candidates": {}}
    for spec in specs:
        rows = {}
        for category, names in PANEL.items():
            per = {}
            for strategy in names:
                vals = raw[(spec.name, strategy)]
                per[strategy] = {"mean": statistics.fmean(vals),
                                 "values": vals}
            rows[category] = {
                "mean": statistics.fmean(
                    v["mean"] for v in per.values()),
                "strategies": per,
            }
        out["candidates"][spec.name] = rows
    path = Path(args.out)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    for spec in specs:
        rows = out["candidates"][spec.name]
        print(f"{spec.name:18} " + "  ".join(
            f"{cat}={rows[cat]['mean']:.2f}" for cat in PANEL))
    print(f"written {path}")


if __name__ == "__main__":
    main()
