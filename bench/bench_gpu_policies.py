#!/usr/bin/env python3
"""Plan 2 component benchmark: batched GPU vs sequential CPU inference.

The benchmark measures the policy component only (no rule opponents), for
``Neural``/``RL`` MLPs, the 85-feature GRU, and the SelfPlay-GRU.  CPU is run
at a smaller batch and linearly extrapolated to the GPU batch sizes; the script
records the extrapolation explicitly rather than hiding it.

Usage::

    python3 bench/bench_gpu_policies.py
    python3 bench/bench_gpu_policies.py --cpu-batch 256 --batches 2048,8192
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.evaluation.gpu.features import FeatureState21, FeatureState85
from ipd.evaluation.gpu.policies import (
    BatchGRU, BatchMLP, BatchSelfPlayGRU,
)
from ipd.paths import result_path, weights_path
from ipd.policies.gru import GRUPolicy
from ipd.policies.neural import NeuralPolicy
from ipd.policies.rl import RLPolicy
from ipd.policies.selfplay_gru import SelfPlayGRUPolicy


def _opponent_history(batch: int, rounds: int, seed: int = 1234
                      ) -> list[list[int]]:
    rng = random.Random(seed)
    # A mix of periodic and random actions exercises every feature branch.
    return [[rng.randrange(2) if i % 17 else (i // 17) % 2
             for i in range(rounds)] for _ in range(batch)]


def _cpu_component(factory: Callable[[], Any], batch: int, rounds: int
                   ) -> tuple[float, list[int]]:
    opponents = _opponent_history(batch, rounds)
    policies = [factory() for _ in range(batch)]
    mine = [[] for _ in range(batch)]
    theirs = [[] for _ in range(batch)]
    blue_all: list[list[int]] = [[] for _ in range(batch)]
    started = time.perf_counter()
    for t in range(rounds):
        for i in range(batch):
            action = policies[i].act(t, mine[i], theirs[i])
            mine[i].append(action)
            theirs[i].append(opponents[i][t])
            blue_all[i].append(action)
    return time.perf_counter() - started, blue_all


def _gpu_component(kind: str, path: str, batch: int, rounds: int,
                   dtype: torch.dtype) -> tuple[float, None, Any]:
    opponents = _opponent_history(batch, rounds)
    theirs = torch.tensor(opponents, dtype=torch.int8, device="cuda")
    mine = torch.zeros(batch, rounds, dtype=torch.int8, device="cuda")
    state_cls = FeatureState21 if kind == "selfplay" else FeatureState85
    state = state_cls(batch, device="cuda", dtype=dtype)
    if kind == "mlp":
        policy = BatchMLP(path, device="cuda", dtype=dtype)
    elif kind == "gru":
        policy = BatchGRU(path, device="cuda", dtype=dtype)
    else:
        policy = BatchSelfPlayGRU(path, device="cuda", dtype=dtype)
    policy.reset(batch)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for t in range(rounds):
        features = state.features()
        if kind in ("gru", "selfplay"):
            logits = policy.step(features)
        else:
            logits = policy.logits(features)
        action = policy.act_from_logits(
            logits, t, ~state.any_defect("theirs"))
        mine[:, t] = action
        state.append(action, theirs[:, t])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, None, policy


def _gpu_info() -> dict[str, Any]:
    try:
        raw = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,power.draw,power.limit,clocks.sm",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip()
        fields = [value.strip() for value in raw.split(",")]
        return {
            "name": fields[0],
            "memory_mb": float(fields[1]) if len(fields) > 1 else None,
            "power_w": float(fields[2]) if len(fields) > 2 else None,
            "power_limit_w": float(fields[3]) if len(fields) > 3 else None,
            "clocks_sm_mhz": float(fields[4]) if len(fields) > 4 else None,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"name": torch.cuda.get_device_name(0)}


POLICIES: list[tuple[str, str, str]] = [
    ("rl-repair", "mlp", weights_path("rl_policy_checkpoint_repair.json")),
    ("neural", "mlp", weights_path("neural_policy_weights.json")),
    ("gru", "gru", weights_path("gru_policy_weights.json")),
    ("selfplay", "selfplay", weights_path("selfplay_gru_policy.json")),
]


def _cpu_factory(kind: str, path: str) -> Callable[[], Any]:
    if kind == "mlp":
        if "checkpoint" in Path(path).name:
            return lambda: RLPolicy(path, mode="argmax")
        return lambda: NeuralPolicy(path, mode="argmax")
    if kind == "gru":
        return lambda: GRUPolicy(path, mode="argmax")
    return lambda: SelfPlayGRUPolicy(path, mode="argmax")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="2048,8192,32768,61440")
    parser.add_argument("--cpu-batch", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--dtype", choices=("float32", "float64"),
                        default="float32")
    parser.add_argument("--out", default=result_path(
        "protocol_v4/plan2_gpu_component.json"))
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is not available; component benchmark requires a GPU",
              file=sys.stderr)
        return 2
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for name, kind, path in POLICIES:
        cpu_seconds, _ = _cpu_component(
            _cpu_factory(kind, path), args.cpu_batch, args.rounds)
        cpu_per_item = cpu_seconds / args.cpu_batch
        for batch in batches:
            gpu_seconds, _, _ = _gpu_component(
                kind, path, batch, args.rounds, dtype)
            extrapolated = cpu_per_item * batch
            row = {
                "policy": name,
                "kind": kind,
                "batch": batch,
                "rounds": args.rounds,
                "dtype": args.dtype,
                "cpu_batch": args.cpu_batch,
                "cpu_seconds": round(cpu_seconds, 6),
                "cpu_seconds_per_item": round(cpu_per_item, 9),
                "cpu_extrapolated_seconds": round(extrapolated, 6),
                "gpu_seconds": round(gpu_seconds, 6),
                "speedup": round(extrapolated / gpu_seconds, 3),
            }
            rows.append(row)
            print(f"{name:10} B={batch:6d} cpu~{extrapolated:8.3f}s "
                  f"gpu={gpu_seconds:8.3f}s speedup={row['speedup']:.1f}x",
                  flush=True)
    payload = {
        "schema": "ipd-plan2-gpu-component-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": _gpu_info(),
        "rounds": args.rounds,
        "dtype": args.dtype,
        "cpu_batch": args.cpu_batch,
        "batches": batches,
        "rows": rows,
        "notes": [
            "CPU timings use sequential stdlib inference and linear "
            "extrapolation from cpu_batch to the GPU batch size.",
            "GPU timings include feature extraction and policy forward, but "
            "no rule opponents or protocol field construction.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
