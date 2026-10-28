#!/usr/bin/env python3
"""Three-phase asymmetric self-play PPO for 100-round IPD.

Usage::

    python3 -m ipd train selfplay --config configs/selfplay/full.json --device auto
    python3 -m ipd train selfplay --smoke

The implementation follows ``docs/research/selfplay-plan.md``.  Blue and red have
independent GRU actor-critics and optimizers; red rollouts use a constrained
difference reward, and blue periodically faces snapshots from the red league.
"""
from __future__ import annotations

if __package__ in (None, ""):  # allow: python ipd/training/selfplay/train.py
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from ipd.core.primitives import C, D
from ipd.training.neural import describe_device, resolve_device

from .config import TrainConfig, load_config, smoke_config
from .env import VectorEnv
from .gpu_env import GPUEnv
from .evaluate import (
    calibration_report, evaluate_model, fast_validation, final_test,
    heldout_red_tasks, passes_thresholds, save_json, selection_score,
    suite_tasks, summarize_rows, validation_report,
)
from .models import (
    GRUActorCritic, export_weights, load_state, make_model, snapshot,
    TorchOpponent,
)
from .opponents import (
    build_calibration_suite, make_training_bot, phase_weights,
    sample_family_labels,
)
from .ppo import PPOAgent, UpdateStats
from .red_pool import RedConstraint, RedLeague, calibration_scores
from .rollout import Rollout, TensorRollout, concat_rollouts


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _bot_factory(family: str, seed: int):
    def factory():
        return make_training_bot(random.Random(seed), family)
    return factory


def _red_factory(league: RedLeague, cfg: TrainConfig, device, seed: int,
                 rng: random.Random):
    snap = league.sample(rng)
    if snap is None:
        return _bot_factory("top", seed)
    return lambda: TorchOpponent(snap.state, config=cfg, device=device,
                                 mode="sample", seed=seed,
                                 name=f"red-{snap.iteration}")


def _make_env(cfg: TrainConfig, factories, families, device, *,
              use_gpu: bool, gpu_env: bool, iteration: int):
    """Build the CPU reference env or the Plan 4 GPU env."""
    if not gpu_env:
        return VectorEnv(factories, families, fast_rules=cfg.fast_rules,
                         fast_rules_min_envs=cfg.fast_rules_min_envs)
    if not use_gpu:
        # An explicit --device cpu request must keep the reference path.
        return VectorEnv(factories, families, fast_rules=cfg.fast_rules,
                         fast_rules_min_envs=cfg.fast_rules_min_envs)
    return GPUEnv(factories, families, device=device,
                  rounds=cfg.rounds,
                  fast_rules=cfg.fast_rules,
                  fast_rules_min_envs=cfg.fast_rules_min_envs,
                  mode="sample",
                  seed=cfg.seed + 1009 * iteration)


def _rollout_cpu(cfg: TrainConfig, model: GRUActorCritic, env: VectorEnv,
                 device) -> tuple[Rollout, np.ndarray, np.ndarray, float]:
    """Reference rollout collection (unchanged Plan 1-3 behaviour)."""
    model.eval()
    h = model.init_hidden(env.n, device)
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    logp_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    own_rows: list[np.ndarray] = []
    other_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    d_rate = 0.0
    with torch.no_grad():
        for _t in range(cfg.rounds):
            obs = env.observations()
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            logits, values, h_next = model.step(x, h)
            dist = Bernoulli(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            hidden_rows.append(h.detach().cpu().permute(1, 0, 2).numpy())
            own, other, _done = env.step(actions.cpu().tolist())
            obs_rows.append(obs)
            action_rows.append(actions.cpu().numpy())
            logp_rows.append(logp.cpu().numpy())
            value_rows.append(values.cpu().numpy())
            own_rows.append(own)
            other_rows.append(other)
            d_rate += float(actions.float().mean())
            h = h_next

    rollout = Rollout(
        obs=np.stack(obs_rows, axis=1),
        actions=np.stack(action_rows, axis=1),
        logprobs=np.stack(logp_rows, axis=1),
        values=np.stack(value_rows, axis=1),
        rewards=np.stack(own_rows, axis=1) * cfg.ppo.reward_scale,
        hidden=np.stack(hidden_rows, axis=1),
        opponent_rewards=np.stack(other_rows, axis=1),
        families=env.families,
    )
    return (rollout, np.stack(own_rows, axis=1),
            np.stack(other_rows, axis=1), d_rate / max(1, cfg.rounds))


def _rollout_gpu(cfg: TrainConfig, model: GRUActorCritic, env: GPUEnv,
                 device) -> tuple[TensorRollout, np.ndarray, np.ndarray, float]:
    """GPU-resident rollout; one host transfer at the end of the episode."""
    model.eval()
    h = model.init_hidden(env.n, device)
    obs_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    logp_rows: list[torch.Tensor] = []
    value_rows: list[torch.Tensor] = []
    own_rows: list[torch.Tensor] = []
    other_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    d_rate = torch.zeros((), device=device)
    with torch.no_grad():
        for _t in range(cfg.rounds):
            obs = env.observations_tensor()
            logits, values, h_next = model.step(obs, h)
            actions = torch.bernoulli(torch.sigmoid(logits)).to(torch.int8)
            logp = F.logsigmoid(torch.where(actions == 1, logits, -logits))
            hidden_rows.append(h.permute(1, 0, 2))
            own, other, _done = env.step_tensor(actions)
            obs_rows.append(obs)
            action_rows.append(actions)
            logp_rows.append(logp)
            value_rows.append(values)
            own_rows.append(own)
            other_rows.append(other)
            d_rate += actions.float().mean()
            h = h_next

    if torch.device(device).type == "cuda":
        # Make the measured rollout honest: the next GAE/update would sync
        # here anyway, and benchmarks must not report enqueue time as
        # completed work.
        torch.cuda.synchronize()
    own = torch.stack(own_rows, dim=1)
    other = torch.stack(other_rows, dim=1)
    rollout = TensorRollout(
        obs=torch.stack(obs_rows, dim=1),
        actions=torch.stack(action_rows, dim=1).to(torch.float32),
        logprobs=torch.stack(logp_rows, dim=1),
        values=torch.stack(value_rows, dim=1),
        rewards=own * cfg.ppo.reward_scale,
        hidden=torch.stack(hidden_rows, dim=1),
        opponent_rewards=other,
        families=env.families,
    )
    return (rollout, own.detach().cpu().numpy(),
            other.detach().cpu().numpy(), float(d_rate) / max(1, cfg.rounds))


def _rollout(cfg: TrainConfig, model: GRUActorCritic, env, device
             ) -> tuple[Rollout | TensorRollout, np.ndarray, np.ndarray, float]:
    if isinstance(env, GPUEnv):
        return _rollout_gpu(cfg, model, env, device)
    return _rollout_cpu(cfg, model, env, device)


def collect_blue(cfg: TrainConfig, model: GRUActorCritic, league: RedLeague,
                 phase, rng: random.Random, device, iteration: int,
                 gpu_env: bool = False
                 ) -> tuple[Rollout | TensorRollout, dict[str, Any]]:
    n_red_learner = 0
    if phase.red_fraction[1] > 0:
        n_red_learner = int(round(cfg.envs * rng.uniform(*phase.red_fraction)))
        n_red_learner = max(0, min(cfg.envs - 1, n_red_learner))
    n_blue = cfg.envs - n_red_learner

    # A fraction of the blue-controlled environments faces a red snapshot,
    # so blue does not overfit to the red learner's current parameters.
    n_red_opp = 0
    if phase.red_fraction[1] > 0 and len(league):
        n_red_opp = int(round(n_blue * rng.uniform(*phase.red_fraction)))
        n_red_opp = max(0, min(n_blue, n_red_opp))
    n_bot = n_blue - n_red_opp

    families = sample_family_labels(rng, phase.name, n_bot)
    factories = [_bot_factory(family, rng.randrange(1 << 30))
                 for family in families]
    for _ in range(n_red_opp):
        families.append("red")
        factories.append(_red_factory(league, cfg, device,
                                      rng.randrange(1 << 30), rng))

    env = _make_env(cfg, factories, families, device,
                    use_gpu=torch.device(device).type == "cuda",
                    gpu_env=gpu_env, iteration=iteration)
    rollout, _own, _other, d_rate = _rollout(cfg, model, env, device)
    rollout.opponent_rewards = None
    stats = {
        "n_red": n_red_opp,
        "n_red_learner": n_red_learner,
        "n_blue": n_blue,
        "d_rate": d_rate,
        "scores": env.scores,
        "family_counts": {f: families.count(f) for f in set(families)},
        "gpu_env": isinstance(env, GPUEnv),
    }
    if isinstance(env, GPUEnv):
        env.close()
    return rollout, stats


def _red_target_factories(cfg: TrainConfig, phase, blue_state, rng, device,
                          n_red: int):
    """Build (factory, family, target_kind, seed) rows for one red rollout."""
    n = max(1, min(cfg.envs, int(n_red)))
    # Phase 2: 50/30/20, Phase 3: 30/40/30.
    if phase.name == "2":
        base_p, blue_p = 0.50, 0.30
    else:
        base_p, blue_p = 0.30, 0.40
    rows = []
    for _ in range(n):
        seed = rng.randrange(1 << 30)
        u = rng.random()
        if u < base_p:
            kind = rng.choices(("final", "triple", "rl"),
                               weights=(5, 3, 2), k=1)[0]
            family = f"base-{kind}"
            rows.append((_strong_bot_factory(kind, seed), family,
                         "base", seed))
        elif u < base_p + blue_p:
            rows.append((lambda s=blue_state, sd=seed: TorchOpponent(
                s, config=cfg, device=device, mode="sample", seed=sd,
                name="blue-snapshot"), "blue", "blue", seed))
        else:
            fam = rng.choices(("special", "noisy", "extort", "good",
                               "random"), weights=(3, 2, 2, 2, 2), k=1)[0]
            rows.append((_bot_factory(fam, seed), f"field-{fam}",
                         "field", seed))
    return rows


def _strong_bot_factory(kind: str, seed: int):
    from .opponents import _final_bot, _rl_bot, _triple_bot

    def factory():
        rng = random.Random(seed)
        if kind == "final":
            return _final_bot(rng)
        if kind == "triple":
            return _triple_bot(rng)
        return _rl_bot(rng)
    return factory


def collect_red(cfg: TrainConfig, model: GRUActorCritic, phase, blue_state,
                rng: random.Random, device, n_red: int,
                gpu_env: bool = False, iteration: int = 0
                ) -> tuple[Rollout | TensorRollout, dict[str, Any]]:
    rows = _red_target_factories(cfg, phase, blue_state, rng, device, n_red)
    families = [row[1] for row in rows]
    targets = [row[2] for row in rows]
    env = _make_env(cfg, [row[0] for row in rows], families, device,
                    use_gpu=torch.device(device).type == "cuda",
                    gpu_env=gpu_env, iteration=iteration + 1)
    rollout, own, other, d_rate = _rollout(cfg, model, env, device)
    constraint = RedConstraint(phase.alpha, phase.lam)
    reward_rows = np.zeros_like(own)
    violations = []
    penalties = []
    for j, kind in enumerate(targets):
        if kind == "field":
            rewards, info = constraint.episode_rewards(
                own[j].tolist(), None, rng)
        else:
            rewards, info = constraint.episode_rewards(
                own[j].tolist(), other[j].tolist(), rng)
        reward_rows[j] = rewards
        violations.append(info.get("violation", 0.0))
        penalties.append(info.get("penalty", 0.0))

    if isinstance(rollout, TensorRollout):
        rollout.rewards = torch.as_tensor(
            reward_rows * cfg.ppo.reward_scale, dtype=torch.float32,
            device=rollout.device)
        rollout.opponent_rewards = torch.as_tensor(
            other, dtype=torch.float32, device=rollout.device)
    else:
        rollout.rewards = reward_rows * cfg.ppo.reward_scale
        rollout.opponent_rewards = other
    stats = {
        "n_red": env.n,
        "d_rate": d_rate,
        "scores": env.scores,
        "target_scores": [int(x) for x in other.sum(axis=1).tolist()],
        "red_scores": [int(x) for x in own.sum(axis=1).tolist()],
        "violation_rate": (sum(1 for x in violations if x > 0)
                           / max(1, len(violations))),
        "mean_violation": statistics.fmean(violations) if violations else 0.0,
        "mean_penalty": statistics.fmean(penalties) if penalties else 0.0,
        "target_kinds": {k: targets.count(k) for k in set(targets)},
        "gpu_env": isinstance(env, GPUEnv),
    }
    if isinstance(env, GPUEnv):
        env.close()
    return rollout, stats


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _rng_state(rng: random.Random, np_rng: np.random.Generator) -> dict:
    return {"python": rng.getstate(), "numpy": np_rng.bit_generator.state,
            "torch": torch.get_rng_state()}


def _restore_rng(state: dict, rng: random.Random,
                 np_rng: np.random.Generator) -> None:
    if not state:
        return
    if "python" in state:
        rng.setstate(state["python"])
    if "numpy" in state:
        np_rng.bit_generator.state = state["numpy"]
    if "torch" in state:
        torch.set_rng_state(state["torch"])


def save_checkpoint(path: str, *, cfg: TrainConfig, blue: PPOAgent,
                    red: PPOAgent, league: RedLeague, iteration: int,
                    episodes_seen: int, history: list, best: dict,
                    best_any: dict, top: list[dict], rng: random.Random,
                    np_rng: np.random.Generator, anchor: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ipd-selfplay-checkpoint-v1",
        "config": _config_json(cfg),
        "blue": blue.state_dict(),
        "red": red.state_dict(),
        "league": [
            {"iteration": s.iteration, "state": s.state, "score": s.score,
             "family_scores": s.family_scores}
            for s in league.snapshots
        ],
        "iteration": iteration,
        "episodes_seen": episodes_seen,
        "history": history,
        "best": best,
        "best_any": best_any,
        "top": top,
        "rng": _rng_state(rng, np_rng),
        "anchor": anchor,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, blue: PPOAgent, red: PPOAgent,
                    league: RedLeague) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    blue.load_state_dict(payload["blue"])
    red.load_state_dict(payload["red"])
    league.snapshots.clear()
    for row in payload.get("league", []):
        from .red_pool import RedSnapshot
        league.snapshots.append(RedSnapshot(
            row["iteration"], row["state"], row.get("score", 0.0),
            row.get("family_scores", {})))
    return payload


def _config_json(cfg: TrainConfig) -> dict:
    def conv(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: conv(getattr(obj, k)) for k in obj.__dataclass_fields__}
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, list):
            return [conv(x) for x in obj]
        if isinstance(obj, dict):
            return {k: conv(v) for k, v in obj.items()}
        return obj
    return conv(cfg)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _reset_critic(model: GRUActorCritic) -> None:
    """Blue's critic predicts absolute blue returns; red needs a fresh one."""
    for module in model.critic:
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
    with torch.no_grad():
        model.critic[-1].bias.zero_()


def _red_gate(model, anchor: dict, cfg: TrainConfig, device,
             seed: int) -> dict:
    """Quick per-family calibration before a red snapshot enters the league."""
    if not anchor:
        return {"worst_ratio": 1.0, "ratios": {}}
    suite = build_calibration_suite()
    rows = evaluate_model(model, suite_tasks(suite, [seed]),
                          device=device, mode="sample", batch_size=32)
    summary = summarize_rows(rows)
    ratios = {family: summary["families"].get(family, 0.0)
              / max(1e-9, base) for family, base in anchor.items()}
    return {"worst_ratio": min(ratios.values()) if ratios else 0.0,
            "ratios": ratios}


def _phase_start(cfg: TrainConfig, phase) -> int:
    idx = cfg.phases.index(phase)
    return 0 if idx == 0 else cfg.phases[idx - 1].until_episodes


def _lr_for_phase(cfg: TrainConfig, phase, episodes_seen: int) -> float:
    if phase.name != "3":
        return cfg.ppo.blue_lr
    start = _phase_start(cfg, phase)
    span = max(1, phase.until_episodes - start)
    p = min(1.0, max(0.0, (episodes_seen - start) / span))
    return cfg.ppo.final_lr + (cfg.ppo.blue_lr - cfg.ppo.final_lr) * 0.5 * (
        1.0 + math.cos(math.pi * p))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Asymmetric self-play PPO for 100-round IPD.")
    ap.add_argument("--config", default=None,
                    help="JSON config (configs/selfplay/full.json)")
    ap.add_argument("--smoke", action="store_true",
                    help="short 200-episode end-to-end smoke run")
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cuda", "npu", "mps", "cpu"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None,
                    help="override total episodes")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override total episodes = iterations * envs")
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--fast-rules", dest="fast_rules", action="store_true",
                    default=None,
                    help="use the native Plan 3 rule bank for opponent steps")
    ap.add_argument("--no-fast-rules", dest="fast_rules",
                    action="store_false",
                    help="keep all rule opponents on the Python path")
    ap.add_argument("--fast-rules-min-envs", type=int, default=None,
                    help="minimum batch size for the native rule bank "
                         "(default: 128)")
    ap.add_argument("--gpu-env", choices=("auto", "on", "off"), default=None,
                    help="Plan 4 GPU-resident rollout environment "
                         "(auto: on when CUDA and envs >= min batch)")
    ap.add_argument("--gpu-env-min-batch", type=int, default=None,
                    help="minimum envs for --gpu-env auto (default: 128)")
    ap.add_argument("--out", default=None,
                    help="deployable JSON policy output")
    ap.add_argument("--checkpoint-out", default=None,
                    help="resumable checkpoint output")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--resume", default=None,
                    help="resume from an ipd-selfplay-checkpoint-v1 file")
    ap.add_argument("--init-checkpoint", default=None,
                    help="initialise blue from a PyTorch state dict or a "
                         "self-play checkpoint (optional BC warm start)")
    ap.add_argument("--final-test", action="store_true",
                    help="run the one-shot held-out final test after training")
    ap.add_argument("--no-final-test", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--freeze-blue", action="store_true",
                    help="train only red against the frozen blue policy")
    ap.add_argument("--red-out", default=None,
                    help="export the trained red policy as self-play JSON")
    ap.add_argument("--eval-only", action="store_true",
                    help="load a checkpoint and only run validation/final test")
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = smoke_config() if args.smoke else load_config(args.config)
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.envs is not None:
        overrides["envs"] = args.envs
    if args.fast_rules is not None:
        overrides["fast_rules"] = bool(args.fast_rules)
    if args.fast_rules_min_envs is not None:
        overrides["fast_rules_min_envs"] = int(args.fast_rules_min_envs)
    if args.gpu_env is not None:
        overrides["gpu_env"] = args.gpu_env
    if args.gpu_env_min_batch is not None:
        overrides["gpu_env_min_batch"] = int(args.gpu_env_min_batch)
    if args.iterations is not None:
        overrides["total_episodes"] = args.iterations * (args.envs or cfg.envs)
    elif args.episodes is not None:
        overrides["total_episodes"] = args.episodes
    if args.out:
        overrides["out"] = args.out
    if args.checkpoint_out:
        overrides["checkpoint_out"] = args.checkpoint_out
    if args.results_dir:
        overrides["results_dir"] = args.results_dir
    if overrides:
        cfg = load_config(None, overrides, base=cfg)
    cfg.device = args.device
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyTorch is required for self-play training.  Install with:\n"
            "  python3 -m venv .venv && . .venv/bin/activate\n"
            "  pip install -r requirements-train.txt")

    device = resolve_device(cfg.device)
    use_gpu_env = (device.type == "cuda"
                   and (cfg.gpu_env == "on"
                        or (cfg.gpu_env == "auto"
                            and cfg.envs >= cfg.gpu_env_min_batch)))
    print(f"device: {describe_device(device)}", flush=True)
    print(f"gpu_env={'on' if use_gpu_env else 'off'} "
          f"(requested {cfg.gpu_env}, min_batch {cfg.gpu_env_min_batch})",
          flush=True)
    print(f"episodes={cfg.total_episodes} envs={cfg.envs} "
          f"iterations={cfg.iterations} phases="
          + ",".join(f"{p.name}:{p.until_episodes}" for p in cfg.phases),
          flush=True)
    print(f"results_dir={cfg.results_dir}", flush=True)

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(1)

    blue_model = make_model(cfg, value_bias=5.5, device=device)
    red_model = make_model(cfg, value_bias=0.0, device=device)
    if args.init_checkpoint and not args.resume:
        init = torch.load(args.init_checkpoint, map_location="cpu",
                          weights_only=False)
        if isinstance(init, dict) and "blue" in init:
            init = init["blue"]["model"]
        elif isinstance(init, dict) and "model" in init:
            init = init["model"]
        load_state(blue_model, init)
        print(f"blue initialised from {args.init_checkpoint}", flush=True)
    ref_model = None
    if args.init_checkpoint and cfg.ppo.bc_kl_coef > 0:
        ref_model = make_model(cfg, value_bias=5.5, device=device)
        load_state(ref_model, snapshot(blue_model))
    blue_agent = PPOAgent(blue_model, cfg.ppo, lr=cfg.ppo.blue_lr,
                          device=device, ref_model=ref_model)
    red_agent = PPOAgent(red_model, cfg.ppo, lr=cfg.ppo.red_lr,
                         device=device)
    league = RedLeague(cfg.red_league_size)

    rng = random.Random(cfg.seed + 700001)
    np_rng = np.random.default_rng(cfg.seed + 700002)
    episodes_seen = 0
    iteration = 0
    history: list[dict] = []
    best: dict[str, Any] = {"selection": -1e18, "iteration": -1,
                            "state": None, "summary": None}
    best_any: dict[str, Any] = {"selection": -1e18, "iteration": -1,
                                "state": None, "summary": None}
    top: list[dict] = []
    failures = 0
    lr_scale = 1.0
    red_initialized = False
    anchor: dict[str, Any] = {}
    red_parts: list[Rollout] = []
    log_path = Path(cfg.results_dir) / "train_log.jsonl"

    if args.resume:
        payload = load_checkpoint(args.resume, blue_agent, red_agent, league)
        episodes_seen = int(payload.get("episodes_seen", 0))
        iteration = int(payload.get("iteration", 0))
        history = list(payload.get("history", []))
        best = dict(payload.get("best", best))
        best_any = dict(payload.get("best_any", best_any))
        top = list(payload.get("top", []))
        anchor = dict(payload.get("anchor", {}))
        _restore_rng(payload.get("rng", {}), rng, np_rng)
        red_initialized = bool(league)
        print(f"resumed {args.resume} at iteration {iteration} "
              f"episodes {episodes_seen}", flush=True)

    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume")
        state = best.get("state")
        if state is not None:
            load_state(blue_model, state)
        report = validation_report(
            blue_model, device=device,
            seeds=[cfg.seed + i * 7919
                   for i in range(cfg.evaluation.full_seeds)])
        save_json(str(Path(cfg.results_dir) / "eval_only.json"), report)
        print(f"eval-only mean={report['mean']:.2f} "
              f"min_family={report['min_family']:.2f}", flush=True)
        if args.final_test:
            test = final_test(
                blue_model, device=device,
                seeds=[cfg.seed + 100000 + i * 7919
                       for i in range(cfg.evaluation.final_seeds)],
                heldout_red_weights=cfg.evaluation.heldout_red_weights)
            save_json(str(Path(cfg.results_dir) / "final_test.json"), test)
        return 0

    validation_seeds = [cfg.seed + i * 7919
                        for i in range(cfg.evaluation.full_seeds)]
    fast_every = max(1, cfg.evaluation.fast_every_episodes)
    full_every = max(1, cfg.evaluation.full_every_episodes)
    next_fast = ((episodes_seen // fast_every) + 1) * fast_every
    next_full = ((episodes_seen // full_every) + 1) * full_every
    start_time = time.time()

    with log_path.open("a", encoding="utf-8") as log_fh:
        while episodes_seen < cfg.total_episodes:
            iteration += 1
            phase = cfg.phase_for_episode(episodes_seen)
            phase_start = _phase_start(cfg, phase)

            # Initialise red from the current blue when phase 2 begins.
            if phase.red_fraction[1] > 0 and not red_initialized:
                load_state(red_model, snapshot(blue_model))
                _reset_critic(red_model)
                league.add(red_model, iteration, score=0.0)
                red_initialized = True
                if not anchor:
                    base_suite = build_calibration_suite()
                    from .opponents import _final_bot, _rl_bot, _triple_bot
                    import random as _random
                    bots = [
                        _final_bot(_random.Random(cfg.seed + 1)),
                        _triple_bot(_random.Random(cfg.seed + 2)),
                        _rl_bot(_random.Random(cfg.seed + 3)),
                    ]
                    anchor_raw = calibration_scores(
                        base_suite, [cfg.seed + i * 31 for i in range(3)],
                        bots)
                    anchor = anchor_raw["families"]
                    save_json(str(Path(cfg.results_dir) /
                                   "red_base_anchor.json"), anchor_raw)
                print(f"red initialised from blue at iter {iteration}; "
                      f"anchor families={len(anchor)}", flush=True)

            lr = _lr_for_phase(cfg, phase, episodes_seen) * lr_scale
            blue_agent.set_lr(lr)
            red_agent.set_lr(min(lr, cfg.ppo.red_lr) * lr_scale)
            schedule = getattr(cfg.ppo, "entropy_by_phase", None)
            if schedule:
                cfg.ppo.entropy_coef = float(
                    schedule.get(phase.name, cfg.ppo.entropy_coef))

            # Blue rollout and update.
            iteration_started = time.perf_counter()
            blue_rollout, blue_stats = collect_blue(
                cfg, blue_model, league, phase, rng, device, iteration,
                gpu_env=use_gpu_env)
            blue_collect_s = time.perf_counter() - iteration_started
            blue_rollout.compute_gae(cfg.ppo.gamma, cfg.ppo.lam)
            blue_update_started = time.perf_counter()
            if args.freeze_blue:
                blue_update = UpdateStats(n_steps=blue_rollout.n_steps)
            else:
                blue_update = blue_agent.update(blue_rollout, rng=np_rng)
            blue_update_s = time.perf_counter() - blue_update_started

            red_stats: dict[str, Any] = {}
            red_update = None
            red_gate_info: dict[str, Any] | None = None
            red_collect_s = 0.0
            red_update_s = 0.0
            if phase.red_fraction[1] > 0 and red_initialized:
                blue_state = snapshot(blue_model)
                red_started = time.perf_counter()
                red_rollout, red_stats = collect_red(
                    cfg, red_model, phase, blue_state, rng, device,
                    blue_stats["n_red_learner"], gpu_env=use_gpu_env,
                    iteration=iteration)
                red_collect_s = time.perf_counter() - red_started
                red_parts.append(red_rollout)
                if iteration % max(1, cfg.red_update_every) == 0:
                    combined = concat_rollouts(red_parts)
                    combined.compute_gae(cfg.ppo.gamma, cfg.ppo.lam)
                    red_update_started = time.perf_counter()
                    red_update = red_agent.update(combined, rng=np_rng)
                    red_update_s = time.perf_counter() - red_update_started
                    red_parts = []
                    if iteration % max(1, cfg.red_snapshot_every) == 0:
                        red_gate_info = _red_gate(
                            red_model, anchor, cfg, device,
                            cfg.seed + 500000 + iteration)
                        if (red_gate_info["worst_ratio"]
                                >= phase.alpha):
                            league.add(red_model, iteration, score=0.0,
                                       family_scores=red_gate_info["ratios"])
                        else:
                            print(f"  red snapshot {iteration} quarantined: "
                                  f"worst_ratio="
                                  f"{red_gate_info['worst_ratio']:.2f} "
                                  f"< {phase.alpha:.2f}", flush=True)

            episodes_seen += cfg.envs
            row: dict[str, Any] = {
                "iteration": iteration,
                "episodes_seen": episodes_seen,
                "phase": phase.name,
                "lr": lr,
                "timing": {
                    "blue_collect_s": round(blue_collect_s, 4),
                    "blue_update_s": round(blue_update_s, 4),
                    "red_collect_s": round(red_collect_s, 4),
                    "red_update_s": round(red_update_s, 4),
                },
                "blue": blue_update.to_dict(),
                "blue_rollout": {
                    "reward_mean": float(
                        blue_rollout.rewards.mean() / cfg.ppo.reward_scale),
                    "d_rate": blue_stats["d_rate"],
                    "n_red": blue_stats["n_red"],
                    "family_counts": blue_stats["family_counts"],
                    "score_mean": statistics.fmean(blue_stats["scores"]),
                    "score_min": min(blue_stats["scores"]),
                    "score_max": max(blue_stats["scores"]),
                },
            }
            if red_update is not None:
                row["red"] = red_update.to_dict()
            if red_gate_info is not None:
                row["red_gate"] = red_gate_info
            if red_stats:
                row["red_rollout"] = {
                    "d_rate": red_stats["d_rate"],
                    "score_mean": statistics.fmean(red_stats["red_scores"]),
                    "violation_rate": red_stats["violation_rate"],
                    "mean_violation": red_stats["mean_violation"],
                    "mean_penalty": red_stats["mean_penalty"],
                    "target_kinds": red_stats["target_kinds"],
                }

            do_fast = (episodes_seen >= next_fast
                       and episodes_seen >= cfg.evaluation.warmup_episodes)
            do_full = (episodes_seen >= next_full
                       and episodes_seen >= cfg.evaluation.warmup_episodes)
            if do_fast:
                next_fast += cfg.evaluation.fast_every_episodes
                fast = fast_validation(
                    blue_model, device=device,
                    opponents=cfg.evaluation.fast_opponents,
                    seeds=[cfg.seed + i * 104729
                           for i in range(cfg.evaluation.fast_seeds)])
                row["fast_val"] = {k: fast[k] for k in
                                   ("mean", "min_family", "worst", "sigma")}
            if do_full:
                next_full += cfg.evaluation.full_every_episodes
                full = validation_report(
                    blue_model, device=device, seeds=validation_seeds)
                # Held-out red: a strong RL checkpoint never used in training.
                red_rows = evaluate_model(
                    blue_model,
                    heldout_red_tasks(
                        [cfg.seed + 200000 + i * 7919
                         for i in range(min(20, len(validation_seeds) * 4))],
                        cfg.evaluation.heldout_red_weights),
                    device=device, mode="sample", batch_size=32)
                red_summary = summarize_rows(red_rows)
                full["heldout_red"] = {
                    "mean": red_summary["mean"],
                    "worst": red_summary["worst"],
                }
                ok, checks = passes_thresholds(full, cfg)
                full["passes_thresholds"] = ok
                full["threshold_checks"] = checks
                if red_summary["mean"] < cfg.evaluation.heldout_red_min:
                    ok = False
                    full["threshold_checks"]["heldout_red"] = False
                full["selection"] = selection_score(full)
                row["full_val"] = full
                save_json(str(Path(cfg.results_dir) /
                              f"eval_full_{iteration:04d}.json"), full)
                if full["selection"] > best_any["selection"]:
                    best_any = {
                        "selection": full["selection"],
                        "iteration": iteration,
                        "state": snapshot(blue_model),
                        "summary": {k: v for k, v in full.items()
                                    if k != "rows"},
                    }

                if phase.red_fraction[1] > 0 and anchor and league:
                    cal = calibration_report(
                        red_model, anchor, device=device,
                        seeds=[cfg.seed + i * 31 for i in range(2)])
                    full["red_calibration"] = {
                        "worst_ratio": cal["worst_ratio"],
                        "ratios": cal["ratios"],
                        "families": cal["families"],
                    }
                    row["red_calibration"] = full["red_calibration"]
                    if cal["worst_ratio"] < phase.alpha:
                        ok = False

                if ok and full["selection"] > best["selection"]:
                    best = {
                        "selection": full["selection"],
                        "iteration": iteration,
                        "state": snapshot(blue_model),
                        "summary": {k: v for k, v in full.items()
                                    if k not in ("rows",)},
                    }
                    save_json(str(Path(cfg.results_dir) / "best_val.json"),
                              best["summary"])
                    torch.save(snapshot(blue_model),
                               str(Path(cfg.results_dir) / "best_blue.pt"))
                    top.append({
                        "selection": full["selection"],
                        "iteration": iteration,
                        "state": snapshot(blue_model),
                        "summary": best["summary"],
                    })
                    top.sort(key=lambda x: x["selection"], reverse=True)
                    top = top[:3]
                    print(f"  new best iter {iteration} "
                          f"mean={full['mean']:.2f} "
                          f"min_family={full['min_family']:.2f} "
                          f"sel={full['selection']:.2f}", flush=True)

                # Absolute red lines filter checkpoints.  A rollback is only
                # triggered by a large regression from the best observed
                # validation score, so the normal early curriculum (where TFT
                # is still far from 594) is not punished as a catastrophe.
                regression = (best_any.get("state") is not None
                              and full["selection"]
                              < best_any["selection"] - 30.0)
                if not ok and regression:
                    failures += 1
                    rollback = best if best.get("state") is not None else best_any
                    if (failures >= cfg.evaluation.failures_before_action
                            and not args.no_early_stop
                            and not args.freeze_blue
                            and episodes_seen >= cfg.evaluation.warmup_episodes
                            and rollback.get("state") is not None):
                        print(f"  early-stop action at iter {iteration}: "
                              f"rollback to iter {rollback['iteration']}, "
                              f"lr_scale {lr_scale:.3f}->{lr_scale * 0.5:.3f}",
                              flush=True)
                        load_state(blue_model, rollback["state"])
                        lr_scale *= 0.5
                        failures = 0
                else:
                    failures = 0

            if iteration % max(1, args.log_every) == 0:
                log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_fh.flush()
                print(f"iter {iteration:4d} ep {episodes_seen:6d} "
                      f"ph={phase.name} r={row['blue_rollout']['score_mean']:6.2f} "
                      f"D={row['blue_rollout']['d_rate']:.2f} "
                      f"pi={blue_update.policy_loss:+.4f} "
                      f"v={blue_update.value_loss:.3f} "
                      f"H={blue_update.entropy:.3f} "
                      f"kl={blue_update.approx_kl:.4f}"
                      + (f" red={row['red_rollout']['score_mean']:.2f}"
                         f"/viol={row['red_rollout']['violation_rate']:.2f}"
                         if red_stats else "")
                      + (f" val={row['fast_val']['mean']:.1f}"
                         if "fast_val" in row else ""),
                      flush=True)

            history.append(row)
            if iteration % 10 == 0 or episodes_seen >= cfg.total_episodes:
                save_checkpoint(
                    cfg.checkpoint_out, cfg=cfg, blue=blue_agent,
                    red=red_agent, league=league, iteration=iteration,
                    episodes_seen=episodes_seen, history=history, best=best,
                    best_any=best_any, top=top, rng=rng, np_rng=np_rng,
                    anchor=anchor)
                save_json(str(Path(cfg.results_dir) / "history.json"), {
                    "meta": {"config": _config_json(cfg),
                             "best_iteration": best["iteration"],
                             "best_summary": best["summary"],
                             "elapsed_s": time.time() - start_time},
                    "history": history,
                })

    # Save the final current checkpoint as well.
    save_checkpoint(
        cfg.checkpoint_out, cfg=cfg, blue=blue_agent, red=red_agent,
        league=league, iteration=iteration, episodes_seen=episodes_seen,
        history=history, best=best, best_any=best_any, top=top,
        rng=rng, np_rng=np_rng, anchor=anchor)
    chosen = best if best.get("state") is not None else best_any
    if chosen.get("state") is not None:
        load_state(blue_model, chosen["state"])
    # Persist the top three eligible checkpoints for later diagnosis.
    ckpt_dir = Path(cfg.results_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for rank, item in enumerate(top[:3], 1):
        torch.save(item["state"], str(ckpt_dir / f"best_{rank}.pt"))
    if top:
        save_json(str(ckpt_dir / "top.json"), [
            {"rank": rank, "iteration": item["iteration"],
             "selection": item["selection"],
             "summary": {k: v for k, v in item["summary"].items()
                         if k != "rows"}}
            for rank, item in enumerate(top[:3], 1)
        ])
    weights = export_weights(blue_model, meta={
        "algorithm": "asymmetric-selfplay-ppo",
        "iterations": iteration,
        "episodes": episodes_seen,
        "best_iteration": chosen["iteration"],
        "best_summary": chosen["summary"],
        "config": _config_json(cfg),
    })
    save_json(cfg.out, weights)
    if args.red_out:
        red_weights = export_weights(red_model, meta={
            "algorithm": "selfplay-red",
            "iterations": iteration,
            "episodes": episodes_seen,
            "freeze_blue": bool(args.freeze_blue),
            "blue_init": args.init_checkpoint,
        })
        save_json(args.red_out, red_weights)
        print(f"written {args.red_out}", flush=True)
    save_json(str(Path(cfg.results_dir) / "history.json"), {
        "meta": {"config": _config_json(cfg),
                 "best_iteration": chosen["iteration"],
                 "best_summary": chosen["summary"],
                 "elapsed_s": time.time() - start_time},
        "history": history,
    })
    print(f"selected iteration {chosen['iteration']} "
          f"selection={chosen['selection']:.3f} "
          f"eligible={best.get('state') is not None}", flush=True)
    print(f"written {cfg.out}", flush=True)
    print(f"written {cfg.checkpoint_out}", flush=True)

    if args.final_test:
        test = final_test(
            blue_model, device=device,
            seeds=[cfg.seed + 300000 + i * 7919
                   for i in range(cfg.evaluation.final_seeds)],
            heldout_red_weights=cfg.evaluation.heldout_red_weights)
        save_json(str(Path(cfg.results_dir) / "final_test.json"), test)
        print(f"final test mean={test['summary']['mean']:.2f} "
              f"min_family={test['summary']['min_family']:.2f} "
              f"heldout_red={test['heldout_red'].get('mean')}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
