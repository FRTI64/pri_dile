#!/usr/bin/env python3
"""Train the IPD policy with full reinforcement learning (PPO).

Unlike ``ipd/training/neural.py`` (behaviour cloning + DAgger), this script
optimises the actual tournament objective: the sum of the 100 daily payoffs.
The agent observes the same 85 history features as the other neural policies,
samples cooperate/defect from a Bernoulli policy, and is trained with PPO
against the four evaluation distributions (original, late-harvest, high-Grim,
wide/smart).

Design
------
* Shared tanh trunk with a policy head and a value head.
* Rollouts run in parallel standard-library workers, so the GPU is only used
  for optimisation and workers do not import PyTorch.
* GAE (gamma=1, lambda=0.95) is computed per 100-day episode inside workers.
* Hard safety rules (day 1 cooperate, no proactive probe, day 100 defect) are
  used by default; ``--allow-probe`` removes the no-probe restriction and lets
  RL learn whether early probing is worth the Grim risk.
* The best validation checkpoint is exported both as an actor-critic
  checkpoint and as the deployable ``ipd-neural-policy-v1`` MLP format.

Setup::

    python3 -m venv .venv
    . .venv/bin/activate
    pip install torch

Train::

    python3 -m ipd train rl --device auto --iterations 150

Evaluate::

    python3 -m ipd benchmark --candidates core,rl --fields 20 --workers 20
    python3 -m ipd blind --candidates core,rl --fields 10 --workers 12
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


from ipd.paths import result_path, weights_path

import argparse
import json
import math
import os
import random
import statistics
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

from ipd.core.simulator import C, D, payoff
from ipd.evaluation.distributions import (
    DISTS, PROTOCOL_VAL_DISTS, TRAIN_DISTS, build_train_dist_field,
)
from ipd.policies.neural import FEATURE_VERSION, extract_features
from ipd.training.neural import describe_device, resolve_device
from ipd.training.risk import (
    competitive_reward, cvar_episode_weights, dirichlet_weights,
    dro_update_weights,
)

N_FEATURES = len(extract_features([], []))
REWARD_SCALE = 0.01  # payoff / 100; keeps value targets near 5.5

# ---------------------------------------------------------------------------
# Rollout workers (standard library only)
# ---------------------------------------------------------------------------

def _gae(values: list[float], rewards: list[float],
         gamma: float, lam: float) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(values)
    returns = [0.0] * len(values)
    adv = 0.0
    for i in range(len(values) - 1, -1, -1):
        next_value = 0.0 if i == len(values) - 1 else values[i + 1]
        delta = rewards[i] + gamma * next_value - values[i]
        adv = delta + gamma * lam * adv
        advantages[i] = adv
        returns[i] = adv + values[i]
    return advantages, returns


def rollout_task(task: dict[str, Any]):
    """Play one field and return packed PPO transitions or validation scores."""
    from ipd.policies.rl import RLPolicy

    dist = task["dist"]
    seed = task["seed"]
    n = int(task["n"])
    checkpoint = task["checkpoint"]
    safe = bool(task.get("safe", True))
    collect = bool(task.get("collect", True))
    mode = task.get("mode", "sample")
    policy_seed = int(task.get("policy_seed", seed))
    gamma = float(task.get("gamma", 1.0))
    lam = float(task.get("lam", 0.95))
    center_rewards = bool(task.get("center_rewards", False))
    center_advantages = bool(task.get("center_advantages", True))
    opponent_penalty = float(task.get("opponent_penalty", 0.0))

    opponent_pool = task.get("opponent_pool")
    opponent_weights = task.get("opponent_weights")
    if opponent_pool or opponent_weights:
        from ipd.evaluation.policy_pool import build_policy_field

        field = build_policy_field(opponent_pool or opponent_weights,
                                   seed=seed, n=n, mode="sample")
    else:
        field = build_train_dist_field(dist, seed, n)
    policy = RLPolicy(checkpoint, seed=policy_seed, mode=mode, safe=safe)

    scores: list[float] = []
    if not collect:
        for opp in field:
            policy.reset()
            opp.reset()
            mine: list[int] = []
            theirs: list[int] = []
            score = 0
            for t in range(100):
                action = policy.act(t, mine, theirs)
                reply = opp.act(t, theirs, mine)
                mine.append(action)
                theirs.append(reply)
                score += payoff(action, reply)
            scores.append(score)
        return {"scores": scores, "n_steps": 0, "dist": dist}

    features = array("f")
    actions = bytearray()
    logprobs = array("f")
    values = array("f")
    rewards = array("f")
    masks = bytearray()
    advantages = array("f")
    returns = array("f")

    for opp in field:
        policy.reset()
        opp.reset()
        mine: list[int] = []
        theirs: list[int] = []
        ep_features: list[list[float]] = []
        ep_actions: list[int] = []
        ep_logprobs: list[float] = []
        ep_values: list[float] = []
        ep_rewards: list[float] = []
        ep_masks: list[int] = []
        score = 0

        for t in range(100):
            f = extract_features(mine, theirs)
            logit, value = policy.logit_value(f)
            action, logprob, forced = policy.select(t, theirs, logit)
            reply = opp.act(t, theirs, mine)
            own = payoff(action, reply)
            reward = competitive_reward(
                own, payoff(reply, action), opponent_penalty)

            ep_features.append(f)
            ep_actions.append(action)
            ep_logprobs.append(logprob)
            ep_values.append(value)
            ep_rewards.append(reward * REWARD_SCALE)
            ep_masks.append(0 if forced else 1)

            mine.append(action)
            theirs.append(reply)
            score += own

        if center_rewards:
            # Optional: remove the per-episode mean reward before GAE.  The
            # critic then learns relative value.  Kept off by default because
            # centring advantages (below) is less invasive.
            baseline = statistics.fmean(ep_rewards) if ep_rewards else 0.0
            ep_rewards = [r - baseline for r in ep_rewards]
        adv, ret = _gae(ep_values, ep_rewards, gamma, lam)
        if center_advantages:
            # Subtract a constant per-episode baseline from the advantages.
            # This removes most of the variance from opponent identity
            # (AllC vs Grim) without changing the optimal policy.  Value
            # targets remain absolute, so the critic stays useful for GAE.
            baseline = statistics.fmean(adv) if adv else 0.0
            adv = [a - baseline for a in adv]
        for i in range(100):
            features.extend(ep_features[i])
            actions.append(ep_actions[i])
            logprobs.append(ep_logprobs[i])
            values.append(ep_values[i])
            rewards.append(ep_rewards[i])
            masks.append(ep_masks[i])
            advantages.append(adv[i])
            returns.append(ret[i])
        scores.append(score)

    return {
        "features": features.tobytes(),
        "actions": bytes(actions),
        "logprobs": logprobs.tobytes(),
        "values": values.tobytes(),
        "rewards": rewards.tobytes(),
        "masks": bytes(masks),
        "advantages": advantages.tobytes(),
        "returns": returns.tobytes(),
        "scores": scores,
        "n_steps": len(actions),
        "dist": dist,
    }


# ---------------------------------------------------------------------------
# Torch model and PPO update
# ---------------------------------------------------------------------------

def make_model(hidden: list[int], value_bias: float = 0.0):
    import torch.nn as nn

    def make_trunk():
        layers: list[nn.Module] = []
        prev = N_FEATURES
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        return nn.Sequential(*layers), prev

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            # Separate trunks keep the large value-regression gradient from
            # distorting the policy representation.  The deployable export only
            # contains the policy trunk.
            self.policy_trunk, prev = make_trunk()
            self.value_trunk, _ = make_trunk()
            self.policy = nn.Linear(prev, 1)
            self.value = nn.Linear(prev, 1)
            nn.init.zeros_(self.value.weight)
            nn.init.constant_(self.value.bias, float(value_bias))

        def forward(self, x):
            hp = self.policy_trunk(x)
            hv = self.value_trunk(x)
            return self.policy(hp).squeeze(-1), self.value(hv).squeeze(-1)

    return ActorCritic()


def _torch_from_payload(payload: dict[str, Any], device):
    import torch

    features = torch.frombuffer(bytearray(payload["features"]),
                                dtype=torch.float32).reshape(-1, N_FEATURES)
    actions = torch.frombuffer(bytearray(payload["actions"]),
                               dtype=torch.uint8).float()
    logprobs = torch.frombuffer(bytearray(payload["logprobs"]),
                                dtype=torch.float32)
    values = torch.frombuffer(bytearray(payload["values"]),
                              dtype=torch.float32)
    rewards = torch.frombuffer(bytearray(payload["rewards"]),
                               dtype=torch.float32)
    masks = torch.frombuffer(bytearray(payload["masks"]),
                             dtype=torch.uint8).bool()
    advantages = torch.frombuffer(bytearray(payload["advantages"]),
                                  dtype=torch.float32)
    returns = torch.frombuffer(bytearray(payload["returns"]),
                               dtype=torch.float32)
    return {
        "x": features.to(device),
        "actions": actions.to(device),
        "logprobs": logprobs.to(device),
        "values": values.to(device),
        "rewards": rewards.to(device),
        "masks": masks.to(device),
        "advantages": advantages.to(device),
        "returns": returns.to(device),
    }


def load_initial_weights(model, path: str, load_value: bool = True) -> tuple[list[float], list[float], float]:
    """Load BC or actor-critic weights.  Returns feature mean/std/temperature."""
    import torch

    with open(path, encoding="utf-8") as fh:
        weights = json.load(fh)
    mean = [float(x) for x in weights["feature_mean"]]
    std = [float(x) for x in weights["feature_std"]]
    temperature = float(weights.get("temperature", 1.0))

    fmt = weights.get("format")
    value_trunk = None
    if fmt == "ipd-rl-policy-v1":
        trunk = weights["trunk"]
        policy_head = weights["policy_head"]
        value_head = weights.get("value_head")
        value_trunk = weights.get("value_trunk")
    elif fmt == "ipd-neural-policy-v1":
        layers = weights["layers"]
        if len(layers) < 2:
            raise ValueError("neural weights need at least a trunk and a head")
        trunk = layers[:-1]
        policy_head = layers[-1]
        value_head = None
    else:
        raise ValueError(f"unsupported initial weights format: {fmt}")

    def load_trunk(module_list, layers):
        linears = [m for m in module_list if isinstance(m, torch.nn.Linear)]
        if len(linears) != len(layers):
            raise ValueError(
                f"trunk depth mismatch: weights={len(layers)} model={len(linears)}")
        for module, layer in zip(linears, layers):
            w = torch.tensor(layer["w"], dtype=torch.float32)
            b = torch.tensor(layer["b"], dtype=torch.float32)
            if tuple(module.weight.shape) != tuple(w.shape):
                raise ValueError("trunk layer shape mismatch")
            module.weight.data.copy_(w.to(module.weight.device))
            module.bias.data.copy_(b.to(module.bias.device))

    load_trunk(model.policy_trunk, trunk)
    if value_trunk is not None:
        load_trunk(model.value_trunk, value_trunk)

    pw = torch.tensor(policy_head["w"], dtype=torch.float32)
    pb = torch.tensor(policy_head["b"], dtype=torch.float32)
    model.policy.weight.data.copy_(pw.to(model.policy.weight.device))
    model.policy.bias.data.copy_(pb.to(model.policy.bias.device))

    if load_value and value_head is not None:
        vw = torch.tensor(value_head["w"], dtype=torch.float32)
        vb = torch.tensor(value_head["b"], dtype=torch.float32)
        model.value.weight.data.copy_(vw.to(model.value.weight.device))
        model.value.bias.data.copy_(vb.to(model.value.bias.device))

    return mean, std, temperature


def export_checkpoint(model, mean: list[float], std: list[float],
                      temperature: float, safe: bool,
                      meta: dict[str, Any]) -> dict[str, Any]:
    import torch

    def linear(module):
        return {
            "w": module.weight.detach().cpu().tolist(),
            "b": module.bias.detach().cpu().tolist(),
        }

    trunk = [linear(m) for m in model.policy_trunk
             if isinstance(m, torch.nn.Linear)]
    value_trunk = [linear(m) for m in model.value_trunk
                   if isinstance(m, torch.nn.Linear)]
    return {
        "format": "ipd-rl-policy-v1",
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "activation": "tanh",
        "trunk": trunk,
        "value_trunk": value_trunk,
        "policy_head": linear(model.policy),
        "value_head": linear(model.value),
        "feature_mean": [float(x) for x in mean],
        "feature_std": [float(x) for x in std],
        "temperature": float(temperature),
        "safe": bool(safe),
        "meta": meta,
    }


def train_ppo(model, payload: dict[str, Any], mean: list[float],
              std: list[float], device, args, iteration: int,
              ref_model=None) -> dict[str, float]:
    import torch
    import torch.nn.functional as F
    from torch.distributions import Bernoulli

    data = _torch_from_payload(payload, device)
    x = data["x"]
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).clamp_min(1e-3)
    x = (x - mean_t) / std_t

    actions = data["actions"]
    old_logprobs = data["logprobs"]
    values_old = data["values"]
    returns = data["returns"]
    masks = data["masks"]
    advantages = data["advantages"]

    # Normalise advantages over transitions the policy actually controls.
    if masks.any():
        a = advantages[masks]
        advantages = (advantages - a.mean()) / a.std().clamp_min(1e-6)
    else:
        advantages = advantages - advantages.mean()

    n = x.shape[0]
    episode_scores = [float(v) for v in payload.get("scores", [])]
    cvar_return = None
    risk_weight_max = 1.0
    risk_weight_min = 1.0
    if (getattr(args, "risk_objective", "none") == "cvar"
            and float(getattr(args, "risk_coef", 0.0)) > 0.0
            and episode_scores):
        q = float(getattr(args, "risk_quantile", 0.2))
        episode_weights = cvar_episode_weights(
            episode_scores, quantile=q, coef=float(args.risk_coef))
        if len(episode_scores) * 100 == n:
            step_weights = torch.repeat_interleave(
                torch.tensor(episode_weights, dtype=torch.float32,
                             device=device), 100)
            advantages = advantages * step_weights
            risk_weight_max = max(episode_weights)
            risk_weight_min = min(episode_weights)
        k = max(1, int(math.ceil(min(1.0, max(0.0, q))
                                 * len(episode_scores))))
        cvar_return = statistics.fmean(sorted(episode_scores)[:k])
    batch_size = min(args.batch_size, n)
    epochs = args.epochs
    policy_losses = []
    value_losses = []
    entropies = []
    kl_values = []
    clip_fracs = []
    bc_kls = []

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_kl = 0.0
        epoch_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            logits, values = model(x[idx])
            dist = Bernoulli(logits=logits)
            new_logprobs = dist.log_prob(actions[idx])
            log_ratio = new_logprobs - old_logprobs[idx]
            ratio = log_ratio.exp()
            mask = masks[idx]

            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1.0 - args.clip,
                                1.0 + args.clip) * advantages[idx]
            policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask.sum().clamp_min(1)
            value_loss = 0.5 * F.mse_loss(values, returns[idx], reduction="mean")
            entropy = dist.entropy()
            entropy_bonus = (entropy * mask).sum() / mask.sum().clamp_min(1)

            bc_kl = torch.tensor(0.0, device=device)
            if ref_model is not None and args.bc_kl_coef > 0:
                with torch.no_grad():
                    ref_logits, _ = ref_model(x[idx])
                ref_dist = Bernoulli(logits=ref_logits)
                bc_kl = torch.distributions.kl_divergence(dist, ref_dist)
                bc_kl = (bc_kl * mask).sum() / mask.sum().clamp_min(1)

            loss = (policy_loss + args.vf_coef * value_loss
                    - args.ent_coef * entropy_bonus
                    + args.bc_kl_coef * bc_kl)
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            args.optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio)
                approx_kl = (approx_kl * mask).sum() / mask.sum().clamp_min(1)
                clip_frac = (((ratio < 1.0 - args.clip)
                              | (ratio > 1.0 + args.clip)).float() * mask).sum() \
                    / mask.sum().clamp_min(1)
            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy_bonus.detach()))
            kl_values.append(float(approx_kl))
            clip_fracs.append(float(clip_frac))
            bc_kls.append(float(bc_kl.detach()))
            epoch_kl += float(approx_kl)
            epoch_batches += 1

        if epoch_batches and epoch_kl / epoch_batches > args.target_kl:
            break

    with torch.no_grad():
        _, values_pred = model(x)
        var_ret = float(torch.var(returns))
        explained = 1.0 - float(torch.var(returns - values_pred)) / max(1e-9, var_ret)

    return {
        "policy_loss": statistics.fmean(policy_losses) if policy_losses else 0.0,
        "value_loss": statistics.fmean(value_losses) if value_losses else 0.0,
        "entropy": statistics.fmean(entropies) if entropies else 0.0,
        "approx_kl": statistics.fmean(kl_values) if kl_values else 0.0,
        "clip_frac": statistics.fmean(clip_fracs) if clip_fracs else 0.0,
        "bc_kl": statistics.fmean(bc_kls) if bc_kls else 0.0,
        "explained_var": explained,
        "n_steps": int(n),
        "cvar_return": cvar_return,
        "risk_weight_max": risk_weight_max,
        "risk_weight_min": risk_weight_min,
    }


# ---------------------------------------------------------------------------
# Rollout aggregation and validation
# ---------------------------------------------------------------------------

def _concat_payloads(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("features", "actions", "logprobs", "values", "rewards",
            "masks", "advantages", "returns")
    by_dist: dict[str, list[float]] = {}
    for r in results:
        by_dist.setdefault(r["dist"], []).extend(r["scores"])
    return {
        **{key: b"".join(r[key] for r in results) for key in keys},
        "scores": [s for r in results for s in r["scores"]],
        "scores_by_dist": by_dist,
        "n_steps": sum(r["n_steps"] for r in results),
    }


def collect_rollouts(pool: ProcessPoolExecutor, checkpoint: dict[str, Any],
                     dists: tuple[str, ...], fields_per_dist: int, n: int,
                     seed: int, safe: bool, workers: int,
                     gamma: float, lam: float,
                     center_rewards: bool = False,
                     center_advantages: bool = True,
                     dist_weights: dict[str, float] | None = None,
                     dirichlet_concentration: float = 0.0,
                     opponent_weights: Sequence[str] | None = None,
                     opponent_penalty: float = 0.0,
                     opponent_fraction: float = 1.0,
                     opponent_pool: Sequence[dict[str, Any]] | None = None
                     ) -> dict[str, Any]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    def add_task(dist: str, red: bool) -> None:
        task = {
            "dist": dist,
            "seed": rng.randrange(1 << 30),
            "n": n,
            "checkpoint": checkpoint,
            "safe": safe,
            "collect": True,
            "mode": "sample",
            "policy_seed": rng.randrange(1 << 30),
            "gamma": gamma,
            "lam": lam,
            "center_rewards": center_rewards,
            "center_advantages": center_advantages,
            "opponent_penalty": opponent_penalty,
        }
        if red:
            if opponent_pool:
                task["opponent_pool"] = [dict(entry)
                                         for entry in opponent_pool]
            else:
                task["opponent_weights"] = list(opponent_weights or ())
        tasks.append(task)

    base = {dist: (1.0 if dist_weights is None
                   else float(dist_weights.get(dist, 1.0)))
            for dist in dists}
    base_total = sum(base.values()) or float(len(dists))
    frac = min(1.0, max(0.0, float(opponent_fraction)))
    total_fields = max(1, int(round(fields_per_dist * base_total)))
    has_pool = bool(opponent_pool or opponent_weights)
    n_red = (int(round(total_fields * frac)) if has_pool else 0)
    n_official = total_fields - n_red
    counts = {dist: 0 for dist in dists}
    if n_official > 0:
        if dirichlet_concentration > 0.0:
            sampled = dirichlet_weights(base, rng, dirichlet_concentration)
            mixture = {dist: sampled[dist] * base_total for dist in dists}
        else:
            mixture = base
        raw = {dist: n_official * mixture.get(dist, 1.0) / base_total
               for dist in dists}
        counts = {dist: int(raw[dist]) for dist in dists}
        remaining = n_official - sum(counts.values())
        for dist in sorted(dists, key=lambda d: raw[d] - counts[d],
                           reverse=True)[:remaining]:
            counts[dist] += 1
        for dist in dists:
            for _ in range(max(0, counts[dist])):
                add_task(dist, red=False)
    for _ in range(n_red):
        add_task("blue_pool", red=True)
    results = list(pool.map(rollout_task, tasks, chunksize=1))
    payload = _concat_payloads(results)
    payload["mixture"] = {
        **{dist: (counts.get(dist, 0) / total_fields)
           for dist in dists},
        "blue_pool": n_red / total_fields,
    }
    return payload


def validate(pool: ProcessPoolExecutor, checkpoint: dict[str, Any],
             dists: tuple[str, ...], fields_per_dist: int, n: int,
             seeds: dict[str, list[int]], safe: bool, mode: str,
             policy_seed: int = 20261024) -> dict[str, Any]:
    tasks = []
    for dist in dists:
        for seed in seeds[dist]:
            tasks.append({
                "dist": dist,
                "seed": seed,
                "n": n,
                "checkpoint": checkpoint,
                "safe": safe,
                "collect": False,
                "mode": mode,
                "policy_seed": policy_seed,
            })
    results = list(pool.map(rollout_task, tasks, chunksize=1))
    by_dist: dict[str, list[float]] = {d: [] for d in dists}
    i = 0
    for dist in dists:
        for _ in seeds[dist]:
            by_dist[dist].append(statistics.fmean(results[i]["scores"]))
            i += 1
    means = {d: statistics.fmean(v) for d, v in by_dist.items()}
    avg = statistics.fmean(means.values())
    worst = min(means.values())
    return {"by_dist": means, "avg": avg, "worst": worst,
            "selection": 0.75 * avg + 0.25 * worst}


def selection_score(val: dict[str, Any], metric: str) -> float:
    avg = float(val["avg"])
    worst = float(val["worst"])
    wide = float(val["by_dist"].get("wide", avg))
    if metric == "avg":
        return avg
    if metric == "wide":
        return 0.5 * avg + 0.5 * wide
    if metric == "blind":
        return 0.4 * avg + 0.4 * wide + 0.2 * worst
    return 0.75 * avg + 0.25 * worst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Train an IPD policy with full PPO reinforcement learning.")
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cuda", "npu", "mps", "cpu"))
    ap.add_argument("--dists", default=",".join(DISTS),
                    help="four official validation distributions")
    ap.add_argument("--train-dists", default=None,
                    help="training distributions (default: same as --dists); "
                         "add 'hard' for tail-risk opponent mining")
    ap.add_argument("--fields-per-dist", type=int, default=6,
                    help="training fields per iteration per distribution")
    ap.add_argument("--dist-weights", default=None,
                    help="optional comma-separated weights, e.g. 1,1,1,2 "
                         "matching --train-dists")
    ap.add_argument("--n", type=int, default=80,
                    help="opponents per training field")
    ap.add_argument("--val-fields-per-dist", type=int, default=3)
    ap.add_argument("--val-n", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--select-metric", default="balanced",
                    choices=("balanced", "avg", "wide", "blind"),
                    help="checkpoint selection score")
    ap.add_argument("--save-all-checkpoints", default=None,
                    help="directory for every validation checkpoint")
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--bc-kl-coef", type=float, default=0.0,
                    help="KL penalty towards the initial (BC) policy")
    ap.add_argument("--risk-objective", default="none",
                    choices=("none", "cvar", "dro"),
                    help="tail-risk surrogate: cvar weights worst episodes; "
                         "dro reweights worst distributions")
    ap.add_argument("--risk-coef", type=float, default=0.0,
                    help="extra advantage weight for worst episodes")
    ap.add_argument("--risk-quantile", type=float, default=0.2,
                    help="fraction of worst episodes for CVaR weighting")
    ap.add_argument("--dro-eta", type=float, default=0.5,
                    help="exponentiated-gradient step for DRO")
    ap.add_argument("--dro-quantile", type=float, default=0.5,
                    help="fraction of worst distributions for DRO blend")
    ap.add_argument("--dro-blend", type=float, default=0.5,
                    help="blend between EG and worst-set target")
    ap.add_argument("--dro-floor", type=float, default=0.1,
                    help="minimum mixture weight per distribution")
    ap.add_argument("--dirichlet-concentration", type=float, default=0.0,
                    help="sample per-iteration dist mixture; 0 disables")
    ap.add_argument("--freeze-policy-trunk", action="store_true",
                    help="only train the policy head (and the value network)")
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--center-advantages", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="subtract the per-episode mean advantage after GAE")
    ap.add_argument("--center-rewards", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="subtract the per-episode mean reward before GAE")
    ap.add_argument("--value-bias", type=float, default=None,
                    help="initial value-head bias (default: 5.5 for raw "
                         "scores, 0 when --center-rewards)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--init-weights", default=weights_path("neural_policy_weights.json"),
                    help="BC or actor-critic weights used for initialisation")
    ap.add_argument("--from-scratch", action="store_true",
                    help="do not load initial policy weights (normalisation is kept)")
    ap.add_argument("--allow-probe", action="store_true",
                    help="remove the day-1/no-proactive-probe safety rules")
    ap.add_argument("--opponent-weights", default="",
                    help="comma-separated frozen blue policy JSONs; train a "
                         "red best response against this pool")
    ap.add_argument("--opponent-pool", default="",
                    help="weighted opponent manifest (JSON) with path/spec "
                         "entries; supports Adv-RL/rules and meta-Nash mixes")
    ap.add_argument("--opponent-penalty", type=float, default=0.0,
                    help="subtract this fraction of opponent payoff from the "
                         "PPO reward (1.0 = score-difference objective)")
    ap.add_argument("--opponent-fraction", type=float, default=1.0,
                    help="fraction of training fields drawn from "
                         "--opponent-weights; 1.0 = red-only, 0.5 = mixed")
    ap.add_argument("--no-validation", action="store_true",
                    help="skip official validation/selection and export the "
                         "last iteration (used for red best responses)")
    ap.add_argument("--out", default=weights_path("rl_policy_weights.json"))
    ap.add_argument("--checkpoint-out", default=weights_path("rl_policy_checkpoint.json"))
    ap.add_argument("--history-out", default=result_path("rl_training_history.json"))
    ap.add_argument("--no-value-load", action="store_true",
                    help="keep the fresh value head when resuming")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "PyTorch is required for RL training.  Create a virtual environment:\n"
            "  python3 -m venv .venv && . .venv/bin/activate\n"
            "  pip install -r requirements-train.txt"
        )

    val_dists = tuple(x.strip() for x in args.dists.split(",") if x.strip())
    unknown = [d for d in val_dists if d not in DISTS + PROTOCOL_VAL_DISTS]
    if unknown:
        raise SystemExit(f"unknown validation distributions: {unknown}")

    opponent_weights = tuple(
        x.strip() for x in args.opponent_weights.split(",") if x.strip())
    opponent_pool: list[dict[str, Any]] = []
    if args.opponent_pool and opponent_weights:
        raise SystemExit("use either --opponent-pool or --opponent-weights")
    if args.opponent_pool:
        from ipd.training.league import load_opponent_manifest

        opponent_pool = load_opponent_manifest(args.opponent_pool)
    train_dists = (tuple(x.strip() for x in args.train_dists.split(",")
                         if x.strip())
                   if args.train_dists else val_dists)
    has_pool = bool(opponent_pool or opponent_weights)
    if has_pool:
        unknown_train = []
    else:
        unknown_train = [d for d in train_dists if d not in TRAIN_DISTS]
    if unknown_train:
        raise SystemExit(f"unknown training distributions: {unknown_train}")

    dist_weights = None
    if args.dist_weights:
        vals = [float(x) for x in args.dist_weights.split(",") if x.strip()]
        if len(vals) != len(train_dists):
            raise SystemExit("--dist-weights must match --train-dists in length")
        dist_weights = dict(zip(train_dists, vals))

    device = resolve_device(args.device)
    print(f"device: {describe_device(device)}", flush=True)
    risk_desc = (f"cvar:{args.risk_coef:g}@{args.risk_quantile:g}"
                 if args.risk_objective == "cvar"
                 else f"dro:eta={args.dro_eta:g},q={args.dro_quantile:g},"
                      f"blend={args.dro_blend:g},floor={args.dro_floor:g}"
                 if args.risk_objective == "dro" else "none")
    print(f"train_dists={','.join(train_dists)} "
          f"val_dists={','.join(val_dists)} "
          f"fields/iter/dist={args.fields_per_dist} "
          f"n={args.n} iterations={args.iterations} safe={not args.allow_probe} "
          f"risk={risk_desc} "
          f"dirichlet={args.dirichlet_concentration:g} "
          f"blue_pool={len(opponent_pool) or len(opponent_weights)}",
          flush=True)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    value_bias = (0.0 if args.center_rewards else 5.5)
    if args.value_bias is not None:
        value_bias = float(args.value_bias)
    model = make_model(args.hidden, value_bias=value_bias).to(device)
    mean = [0.0] * N_FEATURES
    std = [1.0] * N_FEATURES
    temperature = 1.0
    if not args.from_scratch:
        mean, std, temperature = load_initial_weights(
            model, args.init_weights, load_value=not args.no_value_load)
        print(f"initialised from {args.init_weights}", flush=True)
    else:
        # Keep the feature scaler from the BC checkpoint; it is only a fixed
        # normalisation, not part of the learned policy.
        if os.path.exists(args.init_weights):
            with open(args.init_weights, encoding="utf-8") as fh:
                norm = json.load(fh)
            mean = [float(x) for x in norm["feature_mean"]]
            std = [float(x) for x in norm["feature_std"]]
        print("initialised from scratch (fixed BC feature scaler)", flush=True)

    # Force temperature 1.0: the PPO objective trains the logits directly.
    temperature = 1.0
    safe = not args.allow_probe

    if args.freeze_policy_trunk:
        if args.from_scratch:
            raise SystemExit("--freeze-policy-trunk requires an initial policy")
        for param in model.policy_trunk.parameters():
            param.requires_grad_(False)
        print("policy trunk frozen; training the policy head only", flush=True)

    ref_model = None
    if args.bc_kl_coef > 0 and not args.from_scratch:
        ref_model = make_model(args.hidden).to(device)
        load_initial_weights(ref_model, args.init_weights, load_value=False)
        for param in ref_model.parameters():
            param.requires_grad_(False)
        ref_model.eval()
        print(f"using reference-policy KL penalty "
              f"(coef={args.bc_kl_coef})", flush=True)

    args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        args.optimizer, T_max=max(1, args.iterations))

    val_rng = random.Random(args.seed + 900000)
    val_seeds = {d: [val_rng.randrange(1 << 30)
                     for _ in range(args.val_fields_per_dist)] for d in val_dists}
    dro_scale = (sum(dist_weights.values()) if dist_weights
                 else float(len(train_dists)))
    dro_weights = (dict(dist_weights) if dist_weights
                   else {d: 1.0 for d in train_dists})

    meta_base = {
        "algorithm": "ppo",
        "seed": args.seed,
        "dists": list(val_dists),
        "train_dists": list(train_dists),
        "opponent_weights": list(opponent_weights),
        "opponent_pool": list(opponent_pool),
        "opponent_pool_path": args.opponent_pool or None,
        "opponent_penalty": args.opponent_penalty,
        "opponent_fraction": args.opponent_fraction,
        "fields_per_dist": args.fields_per_dist,
        "dist_weights": dist_weights,
        "n": args.n,
        "val_fields_per_dist": args.val_fields_per_dist,
        "val_n": args.val_n,
        "hidden": args.hidden,
        "lr": args.lr,
        "clip": args.clip,
        "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef,
        "bc_kl_coef": args.bc_kl_coef,
        "risk_objective": args.risk_objective,
        "risk_coef": args.risk_coef,
        "risk_quantile": args.risk_quantile,
        "dro_eta": args.dro_eta,
        "dro_quantile": args.dro_quantile,
        "dro_blend": args.dro_blend,
        "dro_floor": args.dro_floor,
        "dirichlet_concentration": args.dirichlet_concentration,
        "freeze_policy_trunk": args.freeze_policy_trunk,
        "select_metric": args.select_metric,
        "save_all_checkpoints": args.save_all_checkpoints,
        "gamma": args.gamma,
        "lam": args.lam,
        "center_rewards": args.center_rewards,
        "center_advantages": args.center_advantages,
        "value_bias": value_bias,
        "safe": safe,
        "init_weights": None if args.from_scratch else args.init_weights,
        "feature_version": FEATURE_VERSION,
    }

    history: list[dict[str, Any]] = []
    best = {"selection": -1e9, "iteration": -1, "checkpoint": None, "val": None}

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        # Baseline validation of the initial policy.
        ckpt = export_checkpoint(model, mean, std, temperature, safe, meta_base)
        initial_val = validate(pool, ckpt, val_dists, args.val_fields_per_dist,
                               args.val_n, val_seeds, safe, "sample")
        print(f"initial val avg={initial_val['avg']:.2f} "
              f"worst={initial_val['worst']:.2f} "
              + " ".join(f"{d}={v:.2f}" for d, v in initial_val["by_dist"].items()),
              flush=True)
        history.append({"iteration": 0, "val": initial_val, "phase": "init"})
        initial_score = selection_score(initial_val, args.select_metric)
        best = {"selection": initial_score, "iteration": 0,
                "checkpoint": ckpt, "val": initial_val}

        t0 = time.time()
        for iteration in range(1, args.iterations + 1):
            model.eval()
            ckpt = export_checkpoint(model, mean, std, temperature, safe, meta_base)
            rollout_start = time.time()
            rollout_weights = (dro_weights
                               if args.risk_objective == "dro"
                               else dist_weights)
            payload = collect_rollouts(
                pool, ckpt, train_dists, args.fields_per_dist, args.n,
                args.seed + iteration * 100003, safe, args.workers,
                args.gamma, args.lam, args.center_rewards,
                args.center_advantages, rollout_weights,
                dirichlet_concentration=args.dirichlet_concentration,
                opponent_weights=opponent_weights,
                opponent_penalty=args.opponent_penalty,
                opponent_fraction=args.opponent_fraction,
                opponent_pool=opponent_pool)
            rollout_time = time.time() - rollout_start

            train_start = time.time()
            metrics = train_ppo(model, payload, mean, std, device, args,
                                iteration, ref_model)
            scheduler.step()
            train_time = time.time() - train_start

            scores = payload["scores"]
            reward_by_dist = {
                d: statistics.fmean(v)
                for d, v in payload.get("scores_by_dist", {}).items()
            }
            row: dict[str, Any] = {
                "iteration": iteration,
                "reward_mean": statistics.fmean(scores),
                "reward_by_dist": reward_by_dist,
                "reward_min": min(scores),
                "reward_max": max(scores),
                "rollout_s": rollout_time,
                "train_s": train_time,
                "lr": args.optimizer.param_groups[0]["lr"],
                "mixture": payload.get("mixture"),
                **metrics,
            }
            if args.risk_objective == "dro":
                normalised = dro_update_weights(
                    dro_weights, reward_by_dist, eta=args.dro_eta,
                    quantile=args.dro_quantile, blend=args.dro_blend,
                    floor=args.dro_floor)
                dro_weights = {name: value * dro_scale
                               for name, value in normalised.items()}
                row["dro_next"] = dict(normalised)
            if args.no_validation:
                ckpt = export_checkpoint(model, mean, std, temperature, safe,
                                         meta_base)
                best = {"selection": 0.0, "iteration": iteration,
                        "checkpoint": ckpt, "val": None}
            do_val = (not args.no_validation
                      and (iteration % args.val_every == 0
                           or iteration == args.iterations))
            if do_val:
                model.eval()
                ckpt = export_checkpoint(model, mean, std, temperature, safe,
                                         meta_base)
                val = validate(pool, ckpt, val_dists, args.val_fields_per_dist,
                               args.val_n, val_seeds, safe, "sample")
                row["val"] = val
                if args.save_all_checkpoints:
                    os.makedirs(args.save_all_checkpoints, exist_ok=True)
                    save_json(os.path.join(args.save_all_checkpoints,
                                           f"iter_{iteration:03d}.json"), ckpt)
                score = selection_score(val, args.select_metric)
                if score > best["selection"]:
                    best = {"selection": score,
                            "iteration": iteration,
                            "checkpoint": ckpt, "val": val}
                    print(f"  new best at iter {iteration}: "
                          f"avg={val['avg']:.2f} worst={val['worst']:.2f}",
                          flush=True)

            history.append(row)
            print(f"iter {iteration:3d} reward={row['reward_mean']:7.2f} "
                  f"pi={metrics['policy_loss']:+.4f} v={metrics['value_loss']:.4f} "
                  f"H={metrics['entropy']:.4f} kl={metrics['approx_kl']:.4f} "
                  f"bckl={metrics['bc_kl']:.4f} "
                  f"clip={metrics['clip_frac']:.3f} ev={metrics['explained_var']:+.3f} "
                  f"({rollout_time:.1f}s+{train_time:.1f}s)", flush=True)
            if do_val:
                print(f"      train " + " ".join(
                    f"{d}={v:.2f}" for d, v in reward_by_dist.items())
                    + f" | val avg={row['val']['avg']:.2f} "
                    f"worst={row['val']['worst']:.2f} "
                    + " ".join(f"{d}={v:.2f}"
                               for d, v in row["val"]["by_dist"].items()),
                    flush=True)
            if iteration % 10 == 0 or iteration == args.iterations:
                save_json(args.history_out,
                          {"meta": {**meta_base, "best_iteration": best["iteration"],
                                    "best_val": best["val"]},
                           "history": history})
                save_json(args.checkpoint_out, best["checkpoint"])
                from ipd.policies.rl import RLPolicy
                save_json(args.out,
                          RLPolicy(best["checkpoint"]).export_policy_weights())
                val_avg = (best["val"]["avg"]
                           if best.get("val") else None)
                print(f"  saved checkpoint (best iter {best['iteration']}, "
                      f"val avg={val_avg if val_avg is not None else 'none'})",
                      flush=True)

    # Final export of the best validation policy.
    from ipd.policies.rl import RLPolicy
    best_policy = RLPolicy(best["checkpoint"])
    policy_weights = best_policy.export_policy_weights()
    policy_weights["meta"] = {
        **policy_weights.get("meta", {}),
        **meta_base,
        "best_iteration": best["iteration"],
        "best_val": best["val"],
        "samples_seen": sum(r.get("n_steps", 0) for r in history),
        "device": describe_device(device),
        "torch": torch.__version__,
    }
    save_json(args.checkpoint_out, best["checkpoint"])
    save_json(args.out, policy_weights)
    save_json(args.history_out,
              {"meta": {**meta_base, "best_iteration": best["iteration"],
                        "best_val": best["val"]},
               "history": history})
    print(f"best iteration {best['iteration']} val={best['val']}", flush=True)
    print(f"written {args.checkpoint_out}", flush=True)
    print(f"written {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
    print(f"written {args.history_out}", flush=True)

    # Quick sanity check.
    p = RLPolicy(best["checkpoint"], mode="argmax")
    mine: list[int] = []
    theirs: list[int] = []
    for t in range(100):
        a = p.act(t, mine, theirs)
        b = C if t < 90 else D
        mine.append(a)
        theirs.append(b)
    print(f"vs AllC-until-90: C={mine.count(C)} D={mine.count(D)} "
          f"score={sum(payoff(a, b) for a, b in zip(mine, theirs))}")


if __name__ == "__main__":
    main()
