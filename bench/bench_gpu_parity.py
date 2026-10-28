#!/usr/bin/env python3
"""Measure float32/float64 boundary flips for the Plan 2 GPU path.

The parity contract uses float64.  The production default is float32, so this
script records how many actions (if any) change when the same field is
replayed with both precisions.

Usage::

    python3 bench/bench_gpu_parity.py --n 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.evaluation.common import evaluate_field_details
from ipd.evaluation.gpu.simulator import evaluate_field_gpu
from ipd.evaluation.protocol import (
    build_protocol_field, restore_field_rng, snapshot_field_rng,
)
from ipd.paths import result_path
from ipd.policies.registry import candidate_specs

_LEARNED = {"NeuralPolicy", "RLPolicy", "GRUPolicy", "SelfPlayGRUPolicy"}


def _field(n: int, seed: int):
    field = build_protocol_field("test_v7", seed, n)
    for opponent in field:
        if type(opponent).__name__ in _LEARNED:
            opponent.mode = "argmax"
    return field


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--candidates",
                        default="rl-repair,gru,selfplay,adv-mlp")
    parser.add_argument("--out", default=result_path(
        "protocol_v4/plan2_gpu_parity.json"))
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 2
    field = _field(args.n, args.seed)
    snapshot = snapshot_field_rng(field)
    rows: list[dict[str, object]] = []
    for name in [x.strip() for x in args.candidates.split(",") if x.strip()]:
        spec = [s for s in candidate_specs(name, mode="argmax")][0]

        restore_field_rng(field, snapshot)
        cpu_started = time.perf_counter()
        cpu = evaluate_field_details(spec.build(), field, rounds=args.rounds)[0]
        cpu_seconds = time.perf_counter() - cpu_started

        results = {}
        for dtype_name, dtype in (("float64", torch.float64),
                                  ("float32", torch.float32)):
            restore_field_rng(field, snapshot)
            started = time.perf_counter()
            simulation = evaluate_field_gpu(
                spec.build(), field, rounds=args.rounds, device="cuda",
                dtype=dtype, keep_histories=True)
            results[dtype_name] = (simulation,
                                   time.perf_counter() - started)
        f64, f32 = results["float64"][0], results["float32"][0]
        action_diffs = sum(
            1 for i in range(len(field))
            for a, b in zip(f64.mine[i], f32.mine[i]) if a != b)
        opponent_diffs = sum(
            1 for i in range(len(field))
            for a, b in zip(f64.theirs[i], f32.theirs[i]) if a != b)
        score_diffs = sum(1 for a, b in zip(f64.scores, f32.scores)
                          if a != b)
        row = {
            "candidate": spec.name,
            "fields": 1,
            "n": args.n,
            "rounds": args.rounds,
            "cpu_seconds": round(cpu_seconds, 6),
            "gpu_float64_seconds": round(results["float64"][1], 6),
            "gpu_float32_seconds": round(results["float32"][1], 6),
            "float64_vs_cpu_score_diffs": sum(
                1 for a, b in zip(cpu.scores, f64.scores) if a != b),
            "float32_blue_action_diffs": action_diffs,
            "float32_opponent_action_diffs": opponent_diffs,
            "float32_score_diffs": score_diffs,
        }
        rows.append(row)
        print(f"{spec.name:12} f64={row['gpu_float64_seconds']:.2f}s "
              f"f32={row['gpu_float32_seconds']:.2f}s "
              f"blue_diffs={action_diffs} opp_diffs={opponent_diffs} "
              f"score_diffs={score_diffs}", flush=True)
    payload = {
        "schema": "ipd-plan2-gpu-parity-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
