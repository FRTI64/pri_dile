#!/usr/bin/env python3
"""Plan 3 benchmark: Python rules vs native C++ bank vs GPU spike.

Three scenarios are recorded:

1. ``spike`` -- 2048x100 matches for TFT and HarvestFactor, the two families
   used by Gate 3A.  Includes setup/transfer and a mixed heterogeneous batch.
2. ``eval`` -- one Protocol v4 field (default ``test_v7``, n=2048) with all
   non-Axelrod/non-policy rules on the fast bank.
3. ``training`` -- the opponent-step of a self-play-style rollout for several
   batch sizes.  Only the opponent step is timed; the learner is unchanged.

Usage::

    python3 bench/bench_fast_opponents.py
    python3 bench/bench_fast_opponents.py --quick
    python3 bench/bench_fast_opponents.py --scenarios spike,training
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.core.opponents.basic import TFT
from ipd.evaluation.fast.bank import RuleBank
from ipd.evaluation.fast.gpu_spike import SpikeSpec, run_spike
from ipd.evaluation.fast.native import NativeRuleBank, available, build_error
from ipd.evaluation.fast.spec import FastRuleSpec
from ipd.evaluation.protocol import build_protocol_field
from ipd.paths import result_path
from ipd.evaluation.hard2 import HarvestFactor
from ipd.training.selfplay.opponents import (
    make_training_bot, phase_weights, sample_family_labels,
)


def _blue_matrix(batch: int, rounds: int, seed: int = 20261028) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(batch, rounds), dtype=np.int8)


def _cpu_python(bots: Sequence[Any], blue: np.ndarray,
                rounds: int) -> tuple[float, list[list[int]]]:
    started = time.perf_counter()
    out: list[list[int]] = []
    for i, bot in enumerate(bots):
        mine: list[int] = []
        theirs: list[int] = []
        row: list[int] = []
        for t in range(rounds):
            action = bot.act(t, mine, theirs)
            mine.append(action)
            theirs.append(int(blue[i, t]))
            row.append(action)
        out.append(row)
    return time.perf_counter() - started, out


def _native_run(specs: Sequence[FastRuleSpec], bots: Sequence[Any],
                blue: np.ndarray, rounds: int) -> tuple[float, float, Any]:
    started = time.perf_counter()
    bank = NativeRuleBank(specs, bots, rounds=rounds)
    setup = time.perf_counter() - started
    out = np.zeros((len(specs), rounds), dtype=np.int8)
    started = time.perf_counter()
    for t in range(rounds):
        actions = bank.act(t, blue[:, t])
        out[:, t] = actions
        bank.observe(t, blue[:, t], actions)
    elapsed = time.perf_counter() - started
    bank.close()
    return setup, elapsed, out


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
        try:
            import torch

            return {"name": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu"}
        except Exception:
            return {"name": "unknown"}


def _spike(batch: int, rounds: int) -> dict[str, Any]:
    torch = __import__("torch")
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA unavailable"}
    rng = random.Random(20261028)
    seeds = [rng.randrange(1 << 30) for _ in range(batch)]
    specs: list[FastRuleSpec] = []
    spike_specs: list[SpikeSpec] = []
    for i in range(batch):
        if i % 2 == 0:
            spec = FastRuleSpec("TFT", {})
            spike = SpikeSpec("TFT")
        else:
            spec = FastRuleSpec("HarvestFactor", {
                "start": 25 + (i % 40), "repairable": bool(i % 3),
                "p_defect": 0.95, "last_d": True})
            spike = SpikeSpec("HarvestFactor", start=25 + (i % 40),
                              repairable=bool(i % 3), p_defect=0.95)
        specs.append(spec)
        spike_specs.append(spike)

    def make_bots() -> list[Any]:
        out: list[Any] = []
        for i, seed in enumerate(seeds):
            if i % 2 == 0:
                out.append(TFT(random.Random(seed)))
            else:
                out.append(HarvestFactor(start=25 + (i % 40),
                                         repairable=bool(i % 3),
                                         p_defect=0.95,
                                         rng=random.Random(seed)))
        return out

    blue = _blue_matrix(batch, rounds)
    cpu_seconds, cpu_out = _cpu_python(make_bots(), blue, rounds)
    native_setup, native_seconds, native_out = _native_run(
        specs, make_bots(), blue, rounds)
    parity = bool(np.array_equal(np.asarray(cpu_out, dtype=np.int8),
                                 native_out))

    gpu_blue = torch.tensor(blue, dtype=torch.int8, device="cuda")
    gpu_out, gpu_seconds = run_spike(
        spike_specs, gpu_blue, device="cuda", dtype=torch.float32,
        seed=20261028)

    # A second GPU call excludes the first-call context/warmup cost.
    _, gpu_seconds_warm = run_spike(
        spike_specs, gpu_blue, device="cuda", dtype=torch.float32,
        seed=20261028)
    return {
        "available": True,
        "batch": batch,
        "rounds": rounds,
        "mix": "50% TFT + 50% HarvestFactor",
        "cpu_python_seconds": round(cpu_seconds, 6),
        "native_setup_seconds": round(native_setup, 6),
        "native_seconds": round(native_seconds, 6),
        "native_parity": parity,
        "gpu_seconds_cold": round(gpu_seconds, 6),
        "gpu_seconds": round(gpu_seconds_warm, 6),
        "native_speedup_vs_python": round(cpu_seconds / native_seconds, 2)
        if native_seconds else None,
        "gpu_speedup_vs_python": round(cpu_seconds / gpu_seconds_warm, 2)
        if gpu_seconds_warm else None,
        "native_vs_gpu": round(gpu_seconds_warm / native_seconds, 2)
        if native_seconds and gpu_seconds_warm else None,
        "gate_3a": ("C++" if gpu_seconds_warm >= 2 * native_seconds
                    else "GPU"),
    }


def _eval_field(tier: str, n: int, rounds: int, seed: int) -> dict[str, Any]:
    field = build_protocol_field(tier, seed, n)
    blue = _blue_matrix(n, rounds, seed=seed + 1)
    started = time.perf_counter()
    cpu_out = []
    for i, bot in enumerate(field):
        mine: list[int] = []
        theirs: list[int] = []
        row: list[int] = []
        for t in range(rounds):
            action = bot.act(t, mine, theirs)
            mine.append(action)
            theirs.append(int(blue[i, t]))
            row.append(action)
        cpu_out.append(row)
    cpu_seconds = time.perf_counter() - started

    # Rebuild to reset all RNG streams before the bank captures them.
    field = build_protocol_field(tier, seed, n)
    bank = RuleBank(field, rounds=rounds)
    started = time.perf_counter()
    fast_out = np.zeros((n, rounds), dtype=np.int8)
    for t in range(rounds):
        actions = bank.act(t, blue[:, t])
        fast_out[:, t] = actions
        bank.observe(t, blue[:, t], actions)
    fast_seconds = time.perf_counter() - started
    # Parity is checked on fast rows only; fallback rows are the original
    # Python objects and can be affected by Axelrod's global state.
    parity = True
    for row_index in bank.fast_rows:
        if list(fast_out[row_index]) != cpu_out[row_index]:
            parity = False
            break
    result = {
        "tier": tier,
        "n": n,
        "rounds": rounds,
        "fast_rows": bank.fast_count,
        "fallback_rows": bank.fallback_count,
        "fast_fraction": round(bank.fast_fraction, 4),
        "cpu_python_seconds": round(cpu_seconds, 6),
        "native_seconds": round(fast_seconds, 6),
        "speedup": round(cpu_seconds / fast_seconds, 2) if fast_seconds else None,
        "parity_fast_rows": parity,
    }
    bank.close()
    return result


def _training(batch: int, rounds: int, seed: int,
              family_pool: Sequence[str] | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    if family_pool is None:
        families = sample_family_labels(rng, "2", batch)
    else:
        families = [rng.choice(list(family_pool)) for _ in range(batch)]
    bot_seeds = [rng.randrange(1 << 30) for _ in range(batch)]

    def make_bots() -> list[Any]:
        return [make_training_bot(random.Random(s), family)
                for s, family in zip(bot_seeds, families)]

    blue = _blue_matrix(batch, rounds, seed=seed + 1)
    cpu_seconds, cpu_out = _cpu_python(make_bots(), blue, rounds)
    # Rebuild the same objects because the CPU run consumed their RNG.
    bank = RuleBank(make_bots(), rounds=rounds)
    fast_out = np.zeros((batch, rounds), dtype=np.int8)
    started = time.perf_counter()
    for t in range(rounds):
        actions = bank.act(t, blue[:, t])
        fast_out[:, t] = actions
        bank.observe(t, blue[:, t], actions)
    fast_seconds = time.perf_counter() - started
    fast_rows = set(bank.fast_rows)
    parity = all(list(fast_out[i]) == cpu_out[i]
                 for i in range(batch) if i in fast_rows)
    result = {
        "batch": batch,
        "rounds": rounds,
        "fast_rows": bank.fast_count,
        "fallback_rows": bank.fallback_count,
        "fast_fraction": round(bank.fast_fraction, 4),
        "cpu_python_seconds": round(cpu_seconds, 6),
        "native_seconds": round(fast_seconds, 6),
        "speedup": round(cpu_seconds / fast_seconds, 2) if fast_seconds else None,
        "parity_fast_rows": bool(parity),
    }
    bank.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios",
                        default="spike,eval,eval_rule,training,training_rules")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--tier", default="test_v7")
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--batches", default="32,128,512,2048")
    parser.add_argument("--out", default=result_path(
        "protocol_v4/plan3_fast_opponents.json"))
    args = parser.parse_args(argv)
    scenarios = {x.strip() for x in args.scenarios.split(",") if x.strip()}
    if not available():
        print(f"native rules unavailable: {build_error()}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "schema": "ipd-plan3-fast-opponents-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cxx": os.environ.get("CXX", "g++"),
        "gpu": _gpu_info(),
        "quick": bool(args.quick),
        "rounds": args.rounds,
    }
    if "spike" in scenarios:
        batch = 256 if args.quick else 2048
        payload["spike"] = _spike(batch, args.rounds)
        print(f"spike: {payload['spike']}", flush=True)
    if "eval" in scenarios:
        n = 256 if args.quick else args.n
        payload["eval"] = _eval_field(args.tier, n, args.rounds, args.seed)
        print(f"eval: {payload['eval']}", flush=True)
    if "eval_rule" in scenarios:
        n = 256 if args.quick else args.n
        payload["eval_rule"] = _eval_field(
            "dev_complex", n, args.rounds, args.seed)
        print(f"eval_rule: {payload['eval_rule']}", flush=True)
    if "training" in scenarios:
        batches = [32, 128] if args.quick else [
            int(x) for x in args.batches.split(",") if x.strip()]
        payload["training"] = [
            _training(batch, args.rounds, args.seed + batch)
            for batch in batches
        ]
        for row in payload["training"]:
            print(f"training B={row['batch']}: {row}", flush=True)
    if "training_rules" in scenarios:
        batches = [32, 128] if args.quick else [
            int(x) for x in args.batches.split(",") if x.strip()]
        rules = ("good", "noisy", "extort", "random", "special")
        payload["training_rules"] = [
            _training(batch, args.rounds, args.seed + batch, rules)
            for batch in batches
        ]
        for row in payload["training_rules"]:
            print(f"training_rules B={row['batch']}: {row}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
