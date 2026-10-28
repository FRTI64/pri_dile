#!/usr/bin/env python3
"""Plan 4 benchmark: training profile, same-B parity, and large-B scaling.

Scenarios:

1. ``profile`` -- decompose one training rollout and update into opponent,
   feature, learner-forward, sampling, storage, GAE and minibatch time.
   This is the Gate 4.1 input: if the opponent/env share is below 40% at the
   production batch size, do not start a GPU VectorEnv rewrite for that size.
2. ``same_b`` -- ten fixed-seed iterations at ``envs=32`` on the CPU and GPU
   paths, with the pre-registered curve tolerances recorded in the result.
3. ``large_b`` -- CPU vs GPU rollout at B=512/2048/4096 (default) for the full
   training mix and for rules/self-play only.  The GPU path is warmed once
   before timing so CUDA context and cuDNN algorithm selection are excluded.

Usage::

    python3 bench/bench_training.py
    python3 bench/bench_training.py --quick
    python3 bench/bench_training.py --scenarios profile,large_b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.paths import result_path
from ipd.training.selfplay.config import TrainConfig
from ipd.training.selfplay.gpu_env import GPUEnv
from ipd.training.selfplay.models import make_model
from ipd.training.selfplay.opponents import make_training_bot
from ipd.training.selfplay.ppo import PPOAgent
from ipd.training.selfplay.rollout import Rollout, TensorRollout
from ipd.training.selfplay.train import _rollout_cpu, _rollout_gpu

FULL_POOL = ("good", "top", "noisy", "extort", "random", "special")
RULE_POOL = ("good", "noisy", "extort", "random", "special")


def _gpu_info() -> dict[str, Any]:
    try:
        raw = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,power.draw,power.limit,clocks.sm",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip()
        fields = [value.strip() for value in raw.split(",")]

        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        return {
            "name": fields[0],
            "memory_mb": number(fields[1]) if len(fields) > 1 else None,
            "power_w": number(fields[2]) if len(fields) > 2 else None,
            "power_limit_w": number(fields[3]) if len(fields) > 3 else None,
            "clocks_sm_mhz": number(fields[4]) if len(fields) > 4 else None,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        try:
            return {"name": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu"}
        except Exception:
            return {"name": "unknown"}


def _factories(pool: Sequence[str], batch: int, seed: int = 20261028):
    rng = random.Random(seed)
    families = [rng.choice(tuple(pool)) for _ in range(batch)]
    seeds = [rng.randrange(1 << 30) for _ in range(batch)]
    factories = [
        (lambda s=seed, f=family: (
            lambda: make_training_bot(random.Random(s), f)))()
        for seed, family in zip(seeds, families)
    ]
    return factories, families


def _cpu_rollout_profile(cfg: TrainConfig, model, env, device
                         ) -> tuple[Rollout, dict[str, float]]:
    """Instrumented copy of the reference rollout loop."""
    model.eval()
    h = model.init_hidden(env.n, device)
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    logp_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    times = {key: 0.0 for key in
             ("observe", "forward", "sample", "env_step", "store")}
    with torch.no_grad():
        for _t in range(cfg.rounds):
            started = time.perf_counter()
            obs = env.observations()
            times["observe"] += time.perf_counter() - started

            started = time.perf_counter()
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            logits, values, h_next = model.step(x, h)
            times["forward"] += time.perf_counter() - started

            started = time.perf_counter()
            dist = Bernoulli(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            times["sample"] += time.perf_counter() - started

            started = time.perf_counter()
            hidden_rows.append(h.detach().cpu().permute(1, 0, 2).numpy())
            own, _other, _done = env.step(actions.cpu().tolist())
            times["env_step"] += time.perf_counter() - started

            started = time.perf_counter()
            obs_rows.append(obs)
            action_rows.append(actions.cpu().numpy())
            logp_rows.append(logp.cpu().numpy())
            value_rows.append(values.cpu().numpy())
            reward_rows.append(own)
            times["store"] += time.perf_counter() - started
            h = h_next

    rollout = Rollout(
        obs=np.stack(obs_rows, axis=1),
        actions=np.stack(action_rows, axis=1),
        logprobs=np.stack(logp_rows, axis=1),
        values=np.stack(value_rows, axis=1),
        rewards=np.stack(reward_rows, axis=1) * cfg.ppo.reward_scale,
        hidden=np.stack(hidden_rows, axis=1),
        families=env.families,
    )
    return rollout, times


def _gpu_rollout_profile(cfg: TrainConfig, model, env, device
                         ) -> tuple[TensorRollout, dict[str, float]]:
    """GPU rollout with host-side component buckets.

    CUDA kernels are asynchronous, so the buckets are wall-clock attribution
    rather than per-kernel GPU time.  The final ``torch.cuda.synchronize`` is
    included in ``rollout`` and the caller records the total separately.
    """
    model.eval()
    h = model.init_hidden(env.n, device)
    obs_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    logp_rows: list[torch.Tensor] = []
    value_rows: list[torch.Tensor] = []
    reward_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    times = {key: 0.0 for key in
             ("observe", "forward", "sample", "env_step", "store")}
    with torch.no_grad():
        for _t in range(cfg.rounds):
            started = time.perf_counter()
            obs = env.observations_tensor()
            times["observe"] += time.perf_counter() - started

            started = time.perf_counter()
            logits, values, h_next = model.step(obs, h)
            times["forward"] += time.perf_counter() - started

            started = time.perf_counter()
            actions = torch.bernoulli(torch.sigmoid(logits)).to(torch.int8)
            logp = F.logsigmoid(torch.where(actions == 1, logits, -logits))
            times["sample"] += time.perf_counter() - started

            started = time.perf_counter()
            hidden_rows.append(h.permute(1, 0, 2))
            own, _other, _done = env.step_tensor(actions)
            times["env_step"] += time.perf_counter() - started

            started = time.perf_counter()
            obs_rows.append(obs)
            action_rows.append(actions)
            logp_rows.append(logp)
            value_rows.append(values)
            reward_rows.append(own)
            times["store"] += time.perf_counter() - started
            h = h_next
    torch.cuda.synchronize()

    rollout = TensorRollout(
        obs=torch.stack(obs_rows, dim=1),
        actions=torch.stack(action_rows, dim=1).to(torch.float32),
        logprobs=torch.stack(logp_rows, dim=1),
        values=torch.stack(value_rows, dim=1),
        rewards=torch.stack(reward_rows, dim=1) * cfg.ppo.reward_scale,
        hidden=torch.stack(hidden_rows, dim=1),
        families=env.families,
    )
    return rollout, times


def _make_cpu_env(cfg: TrainConfig, factories, families):
    from ipd.training.selfplay.env import VectorEnv

    return VectorEnv(factories, families, fast_rules=cfg.fast_rules,
                     fast_rules_min_envs=cfg.fast_rules_min_envs)


def _make_gpu_env(cfg: TrainConfig, factories, families, device, seed):
    return GPUEnv(factories, families, device=device, rounds=cfg.rounds,
                  fast_rules=cfg.fast_rules,
                  fast_rules_min_envs=cfg.fast_rules_min_envs,
                  mode="sample", seed=seed)


def _profile_batch(batch: int, update: bool = True) -> dict[str, Any]:
    cfg = TrainConfig()
    cfg.envs = batch
    cfg.ppo.epochs = 2
    cfg.ppo.batch_episodes = 8
    device = torch.device("cuda")
    model = make_model(cfg, value_bias=5.5, device=device).eval()

    # Warm CUDA kernels before the CPU measurement too.  The reference path
    # uses the same GPU learner/PPO update, and a cold first call otherwise
    # dwarfs the environment difference.
    warm_factories, warm_families = _factories(FULL_POOL, batch)
    warm_env = _make_cpu_env(cfg, warm_factories, warm_families)
    warm_roll, _ = _cpu_rollout_profile(cfg, model, warm_env, device)
    agent = PPOAgent(model, cfg.ppo, lr=3e-4, device=device)
    if update:
        warm_roll.compute_gae()
        agent.update(warm_roll, rng=np.random.default_rng(0))
    del warm_roll, warm_env

    factories, families = _factories(FULL_POOL, batch)
    started = time.perf_counter()
    cpu_env = _make_cpu_env(cfg, factories, families)
    cpu_build = time.perf_counter() - started
    started = time.perf_counter()
    cpu_roll, times = _cpu_rollout_profile(cfg, model, cpu_env, device)
    cpu_rollout = time.perf_counter() - started
    started = time.perf_counter()
    cpu_roll.compute_gae()
    cpu_gae = time.perf_counter() - started
    cpu_update = 0.0
    if update:
        started = time.perf_counter()
        agent.update(cpu_roll, rng=np.random.default_rng(0))
        cpu_update = time.perf_counter() - started

    factories, families = _factories(FULL_POOL, batch)
    started = time.perf_counter()
    gpu_env = _make_gpu_env(cfg, factories, families, device, seed=7)
    gpu_build = time.perf_counter() - started
    # Warm the CUDA/cuDNN kernels and the TensorRollout update path; do not
    # include the first call in the KPI.
    warm_gpu_roll, _ = _gpu_rollout_profile(cfg, model, gpu_env, device)
    if update:
        warm_gpu_roll.compute_gae()
        agent.update(warm_gpu_roll, rng=np.random.default_rng(0))
    del warm_gpu_roll
    gpu_env.reset()
    torch.cuda.synchronize()
    started = time.perf_counter()
    gpu_roll, gpu_times = _gpu_rollout_profile(cfg, model, gpu_env, device)
    gpu_rollout = time.perf_counter() - started
    started = time.perf_counter()
    gpu_roll.compute_gae()
    gpu_gae = time.perf_counter() - started
    gpu_update = 0.0
    if update:
        started = time.perf_counter()
        agent.update(gpu_roll, rng=np.random.default_rng(0))
        gpu_update = time.perf_counter() - started

    def pct(seconds: float, total: float) -> float:
        return round(100.0 * seconds / max(1e-9, total), 2)

    result = {
        "batch": batch,
        "samples": batch * cfg.rounds,
        "cpu": {
            "env_build": round(cpu_build, 4),
            "rollout": round(cpu_rollout, 4),
            "components": {k: round(v, 4) for k, v in times.items()},
            "component_share": {k: pct(v, cpu_rollout)
                                for k, v in times.items()},
            "opponent_share": pct(times["env_step"], cpu_rollout),
            "feature_share": pct(times["observe"], cpu_rollout),
            "gae": round(cpu_gae, 4),
            "update": round(cpu_update, 4),
            "samples_per_s_rollout": round(batch * cfg.rounds /
                                           max(1e-9, cpu_rollout), 1),
            "samples_per_s_total": round(
                batch * cfg.rounds / max(1e-9, cpu_rollout + cpu_gae
                                         + cpu_update), 1),
        },
        "gpu": {
            "env_build": round(gpu_build, 4),
            "rollout": round(gpu_rollout, 4),
            "components": {k: round(v, 4) for k, v in gpu_times.items()},
            "component_share": {k: pct(v, gpu_rollout)
                                for k, v in gpu_times.items()},
            "opponent_share": pct(gpu_times["env_step"], gpu_rollout),
            "feature_share": pct(gpu_times["observe"], gpu_rollout),
            "gae": round(gpu_gae, 4),
            "update": round(gpu_update, 4),
            "samples_per_s_rollout": round(batch * cfg.rounds /
                                           max(1e-9, gpu_rollout), 1),
            "samples_per_s_total": round(
                batch * cfg.rounds / max(1e-9, gpu_rollout + gpu_gae
                                         + gpu_update), 1),
            "vram_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 2**30, 3),
            "vram_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 2**30, 3),
        },
        "rollout_speedup": round(cpu_rollout / max(1e-9, gpu_rollout), 2),
        "update_speedup": round(cpu_update / max(1e-9, gpu_update), 2)
        if gpu_update else None,
    }
    del cpu_roll, gpu_roll, cpu_env, gpu_env
    torch.cuda.empty_cache()
    return result


def profile_scenario(quick: bool) -> dict[str, Any]:
    batches = [32, 128] if quick else [32, 128, 512, 2048]
    rows = [_profile_batch(batch, update=batch <= 2048)
            for batch in batches]
    production = next(row for row in rows if row["batch"] == 32)
    large = rows[-1]
    gate = {
        "gate_4_1": "env/opponent step is not the bottleneck at envs=32",
        "decision": (
            "Do not rewrite the B=32 CPU env in isolation.  The learner "
            "forward/sampling and PPO update dominate; use the GPU rollout "
            "storage path for the same-B gain and target GPUEnv at large B."
        ),
        "production_batch": production["batch"],
        "production_opponent_share": production["cpu"]["opponent_share"],
        "production_feature_share": production["cpu"]["feature_share"],
        "large_batch": large["batch"],
        "large_opponent_share": large["cpu"]["opponent_share"],
        "large_feature_share": large["cpu"]["feature_share"],
        "large_rollout_speedup": large["rollout_speedup"],
    }
    return {"rows": rows, "gate": gate}


def same_b_scenario(iterations: int = 10, envs: int = 32) -> dict[str, Any]:
    """Run the real trainer for a fixed number of iterations on both paths."""
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        for label, gpu_env in (("cpu", "off"), ("gpu", "on")):
            run_dir = tmp / label
            command = [
                sys.executable, "-m", "ipd", "train", "selfplay",
                "--device", "cuda", "--no-final-test",
                "--envs", str(envs), "--iterations", str(iterations),
                "--gpu-env", gpu_env,
                "--results-dir", str(run_dir),
                "--out", str(tmp / f"{label}.json"),
                "--checkpoint-out", str(tmp / f"{label}.pt"),
            ]
            started = time.perf_counter()
            proc = subprocess.run(command, cwd=str(ROOT), env=env,
                                  text=True, capture_output=True)
            wall = time.perf_counter() - started
            if proc.returncode != 0:
                raise RuntimeError(
                    f"training subprocess failed ({label}):\n"
                    f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            payload = json.loads(
                (run_dir / "history.json").read_text(encoding="utf-8"))
            history = payload["history"]
            results[label] = {
                "code": proc.returncode,
                "wall_s": round(wall, 3),
                "elapsed_s": round(
                    float(payload["meta"].get("elapsed_s", wall)), 3),
                "scores": [round(row["blue_rollout"]["score_mean"], 4)
                           for row in history],
                "rewards": [round(row["blue_rollout"]["reward_mean"], 6)
                            for row in history],
                "policy_loss": [round(row["blue"]["policy_loss"], 6)
                                for row in history],
                "collect_s": [round(row.get("timing", {}).get(
                    "blue_collect_s", 0.0), 4) for row in history],
                "update_s": [round(row.get("timing", {}).get(
                    "blue_update_s", 0.0), 4) for row in history],
            }
    cpu = results["cpu"]
    gpu = results["gpu"]
    score_diff = [abs(a - b) for a, b in zip(cpu["scores"], gpu["scores"])]
    reward_diff = [abs(a - b) for a, b in zip(cpu["rewards"], gpu["rewards"])]
    tolerances = {"max_score_diff": 25.0, "mean_score_diff": 10.0,
                  "max_reward_diff": 0.10, "mean_reward_diff": 0.05}
    passed = (max(score_diff) <= tolerances["max_score_diff"]
              and float(np.mean(score_diff)) <= tolerances["mean_score_diff"]
              and max(reward_diff) <= tolerances["max_reward_diff"]
              and float(np.mean(reward_diff)) <= tolerances["mean_reward_diff"])
    cpu_collect = float(np.median(cpu["collect_s"][1:]))
    gpu_collect = float(np.median(gpu["collect_s"][1:]))
    cpu_update = float(np.median(cpu["update_s"][1:]))
    gpu_update = float(np.median(gpu["update_s"][1:]))
    results["comparison"] = {
        "iterations": iterations,
        "envs": envs,
        "max_score_diff": round(max(score_diff), 4),
        "mean_score_diff": round(float(np.mean(score_diff)), 4),
        "max_reward_diff": round(max(reward_diff), 6),
        "mean_reward_diff": round(float(np.mean(reward_diff)), 6),
        "tolerances": tolerances,
        "passed": bool(passed),
        "median_collect_cpu_s": round(cpu_collect, 4),
        "median_collect_gpu_s": round(gpu_collect, 4),
        "rollout_speedup": round(cpu_collect / max(1e-9, gpu_collect), 2),
        "median_update_cpu_s": round(cpu_update, 4),
        "median_update_gpu_s": round(gpu_update, 4),
        "wall_speedup": round(cpu["wall_s"] / max(1e-9, gpu["wall_s"]), 2),
        "train_speedup": round(
            cpu["elapsed_s"] / max(1e-9, gpu["elapsed_s"]), 2),
    }
    results["meta"] = {
        "schema": "ipd-plan4-same-b-v1",
        "command": ("python -m ipd train selfplay --device cuda --envs "
                    f"{envs} --iterations {iterations} --gpu-env off|on"),
        "note": ("GPU sampling changes the RNG stream; this is a curve "
                 "equivalence check, not bitwise parity."),
    }
    return results


def _large_b_row(batch: int, pool: Sequence[str], label: str) -> dict[str, Any]:
    cfg = TrainConfig()
    cfg.envs = batch
    device = torch.device("cuda")
    model = make_model(cfg, value_bias=5.5, device=device).eval()
    factories, families = _factories(pool, batch)
    cpu_env = _make_cpu_env(cfg, factories, families)
    cpu_started = time.perf_counter()
    cpu_roll, _own, _other, _rate = _rollout_cpu(cfg, model, cpu_env, device)
    cpu_rollout = time.perf_counter() - cpu_started
    del cpu_roll, cpu_env

    factories, families = _factories(pool, batch)
    started = time.perf_counter()
    gpu_env = _make_gpu_env(cfg, factories, families, device, seed=11)
    gpu_build = time.perf_counter() - started
    # Warm-up call, then reset to the same initial opponent RNG state.
    _rollout_gpu(cfg, model, gpu_env, device)
    gpu_env.reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_started = time.perf_counter()
    gpu_roll, _own, _other, _rate = _rollout_gpu(cfg, model, gpu_env, device)
    gpu_rollout = time.perf_counter() - gpu_started
    vram = torch.cuda.max_memory_allocated() / 2**30
    del gpu_roll, gpu_env
    torch.cuda.empty_cache()
    return {
        "label": label,
        "batch": batch,
        "samples": batch * cfg.rounds,
        "cpu_rollout_s": round(cpu_rollout, 4),
        "gpu_build_s": round(gpu_build, 4),
        "gpu_rollout_s": round(gpu_rollout, 4),
        "speedup": round(cpu_rollout / max(1e-9, gpu_rollout), 2),
        "gpu_vram_allocated_gib": round(vram, 3),
        "cpu_samples_per_s": round(batch * cfg.rounds / cpu_rollout, 1),
        "gpu_samples_per_s": round(batch * cfg.rounds / gpu_rollout, 1),
    }


def large_b_scenario(quick: bool) -> dict[str, Any]:
    batches = [512, 2048] if quick else [512, 2048, 4096]
    rows = []
    for batch in batches:
        rows.append(_large_b_row(batch, FULL_POOL, "full_mix"))
        rows.append(_large_b_row(batch, RULE_POOL, "rules"))
    full = [row for row in rows if row["label"] == "full_mix"]
    rules = [row for row in rows if row["label"] == "rules"]
    best_full = max(full, key=lambda row: row["speedup"])
    best_rules = max(rules, key=lambda row: row["speedup"])
    return {
        "rows": rows,
        "gate": {
            "full_mix_best": best_full,
            "rules_best": best_rules,
            "supported_subset_10x": bool(best_rules["speedup"] >= 10.0),
            "full_mix_10x": bool(best_full["speedup"] >= 10.0),
            "large_b_10x_passed": bool(best_rules["speedup"] >= 10.0),
            "note": ("The Plan 4 target applies to the supported "
                     "rules/self-play set.  Final/Triple/Axelrod stay on the "
                     "CPU fallback by design, which caps the full training "
                     "mix; both numbers are reported."),
            "best_vram_gib": max(row["gpu_vram_allocated_gib"]
                                 for row in rows),
            "vram_under_20_gib": max(row["gpu_vram_allocated_gib"]
                                     for row in rows) < 20.0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="profile,same_b,large_b")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--out", default=result_path(
        "protocol_v4/plan4_training.json"))
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required for the Plan 4 benchmark", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    scenarios = {x.strip() for x in args.scenarios.split(",") if x.strip()}
    payload: dict[str, Any] = {
        "schema": "ipd-plan4-training-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": _gpu_info(),
        "torch": torch.__version__,
        "quick": bool(args.quick),
    }
    if "profile" in scenarios:
        started = time.perf_counter()
        payload["profile"] = profile_scenario(args.quick)
        payload["profile"]["wall_s"] = round(time.perf_counter() - started, 2)
        print(f"profile: gate={payload['profile']['gate']}", flush=True)
    if "same_b" in scenarios:
        payload["same_b"] = same_b_scenario(
            iterations=args.iterations, envs=args.envs)
        print(f"same_b: {payload['same_b']['comparison']}", flush=True)
    if "large_b" in scenarios:
        payload["large_b"] = large_b_scenario(args.quick)
        print(f"large_b: {payload['large_b']['gate']}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    if "profile" in payload:
        profile_out = out.parent / "plan4_profile.json"
        profile_out.write_text(json.dumps({
            "schema": "ipd-plan4-profile-v1",
            "created": payload.get("created"),
            "gpu": payload.get("gpu"),
            "torch": payload.get("torch"),
            "profile": payload["profile"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written {profile_out}")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
