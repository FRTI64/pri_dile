"""Recurrent PPO-Clip update (full-episode sequence minibatches)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from .models import GRUActorCritic, clone_state, load_state
from .rollout import Rollout, iter_batches, normalize_advantages


@dataclass
class UpdateStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    bc_kl: float = 0.0
    explained_var: float = 0.0
    n_steps: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_frac": self.clip_frac,
            "bc_kl": self.bc_kl,
            "explained_var": self.explained_var,
            "n_steps": self.n_steps,
        }


class PPOAgent:
    def __init__(self, model: GRUActorCritic, config,
                 lr: float, device: str | torch.device = "cpu",
                 ref_model: GRUActorCritic | None = None):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, eps=1e-5)
        self.ref_model = (ref_model.to(device).eval()
                          if ref_model is not None else None)
        if self.ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def update(self, rollout: Rollout, *,
               rng: np.random.Generator | None = None) -> UpdateStats:
        cfg = self.config
        rng = rng or np.random.default_rng(0)
        data = rollout.tensors(self.device)
        obs = data["obs"]
        actions = data["actions"]
        old_logprobs = data["logprobs"]
        returns = data["returns"]
        advantages = data["advantages"]
        if getattr(cfg, "center_advantages", True):
            # Remove opponent-identity variance without changing the optimal
            # policy: subtract the per-episode mean advantage.
            advantages = advantages - advantages.mean(dim=1, keepdim=True)
        advantages = normalize_advantages(advantages)

        n = obs.shape[0]
        epochs = int(cfg.epochs)
        batch = max(1, int(cfg.batch_episodes))
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fracs: list[float] = []
        bc_kls: list[float] = []

        self.model.train()
        stop = False
        for _ in range(epochs):
            epoch_kl: list[float] = []
            for idx in iter_batches(n, batch, rng):
                x = obs[idx]
                logits, values, _ = self.model.forward_sequence(x)
                dist = Bernoulli(logits=logits)
                new_logprobs = dist.log_prob(actions[idx])
                log_ratio = new_logprobs - old_logprobs[idx]
                ratio = log_ratio.exp()
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip,
                                    1.0 + cfg.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()
                bc_kl = torch.tensor(0.0, device=logits.device)
                if (self.ref_model is not None
                        and getattr(cfg, "bc_kl_coef", 0.0) > 0):
                    with torch.no_grad():
                        ref_logits, _, _ = self.ref_model.forward_sequence(x)
                    ref_dist = Bernoulli(logits=ref_logits)
                    bc_kl = torch.distributions.kl_divergence(
                        dist, ref_dist).mean()
                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy
                        + getattr(cfg, "bc_kl_coef", 0.0) * bc_kl)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio < 1.0 - cfg.clip)
                                 | (ratio > 1.0 + cfg.clip)).float().mean()
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.detach()))
                kls.append(float(approx_kl))
                clip_fracs.append(float(clip_frac))
                bc_kls.append(float(bc_kl.detach()))
                epoch_kl.append(float(approx_kl))
            if epoch_kl and statistics.fmean(epoch_kl) > cfg.target_kl:
                stop = True
            if stop:
                break

        with torch.no_grad():
            _, values_pred, _ = self.model.forward_sequence(obs)
            var_ret = float(torch.var(returns))
            explained = 1.0 - float(torch.var(returns - values_pred)) / max(
                1e-9, var_ret)

        return UpdateStats(
            policy_loss=statistics.fmean(policy_losses) if policy_losses else 0.0,
            value_loss=statistics.fmean(value_losses) if value_losses else 0.0,
            entropy=statistics.fmean(entropies) if entropies else 0.0,
            approx_kl=statistics.fmean(kls) if kls else 0.0,
            clip_frac=statistics.fmean(clip_fracs) if clip_fracs else 0.0,
            bc_kl=statistics.fmean(bc_kls) if bc_kls else 0.0,
            explained_var=explained,
            n_steps=int(obs.shape[0] * obs.shape[1]),
        )

    def state_dict(self) -> dict:
        return {
            "model": clone_state(self.model),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        load_state(self.model, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


__all__ = ["PPOAgent", "UpdateStats"]
