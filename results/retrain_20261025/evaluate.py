#!/usr/bin/env python3
"""Evaluate every retrained candidate in one benchmark + one blind run.

The repository CLIs accept a single ``--weights`` / ``--config`` override, so
they cannot score several freshly trained checkpoints side by side.  This
script builds one ``CandidateSpec`` per checkpoint and reuses the same worker
helpers as ``ipd benchmark`` / ``ipd blind``.  The blind opponent round robin is
therefore computed once per field and shared by every candidate.

Run from the repository root::

    .venv/bin/python results/retrain_20261025/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.evaluation.blind import field_task
from ipd.evaluation.common import eval_task, seed_map
from ipd.evaluation.distributions import DISTS
from ipd.policies.registry import CandidateSpec

WEIGHTS = ROOT / "weights" / "retrain_20261025"
CONFIGS = ROOT / "configs" / "retrain_20261025"
OUT = Path(__file__).resolve().parent


def specs() -> list[CandidateSpec]:
    rows = [
        ("Neural-retrain", "neural",
         {"weights": str(WEIGHTS / "neural_policy_weights.json"),
          "mode": "sample"}),
        ("GRU-retrain", "gru",
         {"weights": str(WEIGHTS / "gru_policy_weights.json"),
          "mode": "sample"}),
        ("Adv-MLP-retrain", "adv-mlp",
         {"weights": str(WEIGHTS / "neural_policy_weights.json"),
          "config": str(CONFIGS / "adversarial_policy_aggressive.json"),
          "mode": "sample"}),
        ("Adv-GRU-retrain", "adv-gru",
         {"weights": str(WEIGHTS / "gru_policy_weights.json"),
          "config": str(CONFIGS / "adversarial_gru_policy_aggressive.json"),
          "mode": "sample"}),
        ("RL-final-retrain", "rl",
         {"weights": str(WEIGHTS / "rl_policy_weights.json"),
          "mode": "sample"}),
        ("RL-BC-retrain", "rl",
         {"weights": str(WEIGHTS / "rl_policy_weights_bc.json"),
          "mode": "sample"}),
        ("RL-scratch-base-retrain", "rl",
         {"weights": str(WEIGHTS / "rl_policy_weights_scratch_base.json"),
          "mode": "sample"}),
        ("RL-stage1-retrain", "rl",
         {"weights": str(WEIGHTS / "rl_policy_weights_stage1.json"),
          "mode": "sample"}),
        ("Adv-RL-retrain", "adv-rl",
         {"weights": str(WEIGHTS / "rl_policy_weights_bc.json"),
          "config": str(CONFIGS / "adversarial_rl_policy_aggressive.json"),
          "mode": "sample"}),
        ("Adv-RL-conservative-retrain", "adv-rl",
         {"weights": str(WEIGHTS / "rl_policy_weights_bc.json"),
          "config": str(ROOT / "configs" /
                        "adversarial_rl_policy_config.json"),
          "mode": "sample"}),
        ("Adv-RL-final-retrain", "adv-rl",
         {"weights": str(WEIGHTS / "rl_policy_weights.json"),
          "config": str(CONFIGS / "adversarial_rl_policy_aggressive.json"),
          "mode": "sample"}),
    ]
    out: list[CandidateSpec] = []
    for name, kind, params in rows:
        files = [params.get("weights"), params.get("config")]
        missing = [p for p in files if p and not Path(p).is_file()]
        if missing:
            print(f"skip {name}: missing {missing}")
            continue
        out.append(CandidateSpec(kind=kind, name=name, params=params))
    return out


def run_benchmark(rows: list[CandidateSpec], fields: int, n: int,
                  workers: int, seed: int) -> dict:
    seeds = seed_map(seed, DISTS, fields)
    tasks = [
        {"spec": spec.to_dict(), "name": spec.name, "dist": dist,
         "seeds": seeds[dist], "n": n, "rounds": 100}
        for spec in rows for dist in DISTS
    ]
    started = time.time()
    results: dict[str, dict[str, list[float]]] = {spec.name: {} for spec in rows}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, dist, values in ex.map(eval_task, tasks, chunksize=1):
            results[name][dist] = values
            print(f"{name:26} {dist:9} mean={statistics.fmean(values):7.2f}",
                  flush=True)
    summary = {}
    for spec in rows:
        means = {d: statistics.fmean(results[spec.name][d]) for d in DISTS}
        summary[spec.name] = {
            "kind": spec.kind,
            "dists": {
                d: {
                    "mean": means[d],
                    "min": min(results[spec.name][d]),
                    "values": results[spec.name][d],
                }
                for d in DISTS
            },
            "avg": statistics.fmean(means.values()),
            "worst": min(means.values()),
        }
    payload = {
        "schema": "ipd-retrain-benchmark-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "seed": seed, "fields": fields, "n": n,
        "candidates": [spec.to_dict() for spec in rows],
        "summary": summary,
    }
    path = OUT / "benchmark_all.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\nwritten {path}")
    return summary


def run_blind(rows: list[CandidateSpec], fields: int, n: int,
              workers: int, seed: int) -> dict:
    seeds = [seed + i * 7919 for i in range(fields)]
    tasks = [
        {"seed": s, "n": n, "specs": [spec.to_dict() for spec in rows]}
        for s in seeds
    ]
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (field_seed, result) in enumerate(
                ex.map(field_task, tasks, chunksize=1), 1):
            results.append(result)
            print(f"field {i}/{fields} seed={field_seed} "
                  + ", ".join(f"{name}:{v['rank']}"
                              for name, v in result.items()), flush=True)
    summary = {}
    for spec in rows:
        scores = [row[spec.name]["score"] / n for row in results]
        ranks = [row[spec.name]["rank"] for row in results]
        summary[spec.name] = {
            "kind": spec.kind,
            "score": statistics.fmean(scores),
            "mean_rank": statistics.fmean(ranks),
            "rank1": sum(x == 1 for x in ranks),
            "top10": sum(x <= 10 for x in ranks),
            "worst": max(ranks),
            "scores": scores,
            "ranks": ranks,
        }
    payload = {
        "schema": "ipd-retrain-blind-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "seed": seed, "fields": fields, "n": n,
        "candidates": [spec.to_dict() for spec in rows],
        "results": summary,
    }
    path = OUT / "blind_all.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\nwritten {path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-fields", type=int, default=20)
    ap.add_argument("--blind-fields", type=int, default=10)
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--benchmark-workers", type=int, default=20)
    ap.add_argument("--blind-workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--blind-seed", type=int, default=20261025)
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--skip-blind", action="store_true")
    args = ap.parse_args()

    rows = specs()
    if not rows:
        raise SystemExit("no retrained candidates found")
    if not args.skip_benchmark:
        run_benchmark(rows, args.benchmark_fields, args.n,
                      args.benchmark_workers, args.seed)
    if not args.skip_blind:
        run_blind(rows, args.blind_fields, args.n,
                  args.blind_workers, args.blind_seed)


if __name__ == "__main__":
    main()
