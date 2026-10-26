"""Recurrent rollout storage, GAE, and sequence minibatches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch


@dataclass
class Rollout:
    obs: np.ndarray          # [B, T, F]
    actions: np.ndarray      # [B, T]
    logprobs: np.ndarray     # [B, T]
    values: np.ndarray       # [B, T]
    rewards: np.ndarray      # [B, T] (already scaled if requested)
    hidden: np.ndarray       # [B, T, layers, hidden]
    opponent_rewards: np.ndarray | None = None
    families: Sequence[str] | None = None
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    @property
    def n_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.obs.shape[0] * self.obs.shape[1])

    def compute_gae(self, gamma: float = 1.0, lam: float = 0.95) -> "Rollout":
        rewards = self.rewards.astype(np.float64)
        values = self.values.astype(np.float64)
        b, t = rewards.shape
        advantages = np.zeros_like(rewards)
        last = np.zeros(b, dtype=np.float64)
        for i in range(t - 1, -1, -1):
            next_value = np.zeros(b) if i == t - 1 else values[:, i + 1]
            delta = rewards[:, i] + gamma * next_value - values[:, i]
            last = delta + gamma * lam * last
            advantages[:, i] = last
        self.advantages = advantages.astype(np.float32)
        self.returns = (advantages + values).astype(np.float32)
        return self

    def tensors(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        if self.advantages is None or self.returns is None:
            raise ValueError("call compute_gae() before tensors()")
        return {
            "obs": torch.as_tensor(self.obs, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions, dtype=torch.float32,
                                       device=device),
            "logprobs": torch.as_tensor(self.logprobs, dtype=torch.float32,
                                        device=device),
            "rewards": torch.as_tensor(self.rewards, dtype=torch.float32,
                                       device=device),
            "values": torch.as_tensor(self.values, dtype=torch.float32,
                                      device=device),
            "advantages": torch.as_tensor(self.advantages, dtype=torch.float32,
                                          device=device),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32,
                                       device=device),
        }

    def take(self, index: np.ndarray | Sequence[int]) -> "Rollout":
        return Rollout(
            obs=self.obs[index], actions=self.actions[index],
            logprobs=self.logprobs[index], values=self.values[index],
            rewards=self.rewards[index], hidden=self.hidden[index],
            opponent_rewards=(None if self.opponent_rewards is None
                              else self.opponent_rewards[index]),
            families=(None if self.families is None
                      else [self.families[i] for i in index]),
            advantages=(None if self.advantages is None
                        else self.advantages[index]),
            returns=(None if self.returns is None else self.returns[index]),
        )


def concat_rollouts(parts: Sequence[Rollout]) -> Rollout:
    parts = [p for p in parts if p.n_episodes]
    if not parts:
        raise ValueError("no rollouts to concatenate")
    return Rollout(
        obs=np.concatenate([p.obs for p in parts], axis=0),
        actions=np.concatenate([p.actions for p in parts], axis=0),
        logprobs=np.concatenate([p.logprobs for p in parts], axis=0),
        values=np.concatenate([p.values for p in parts], axis=0),
        rewards=np.concatenate([p.rewards for p in parts], axis=0),
        hidden=np.concatenate([p.hidden for p in parts], axis=0),
        opponent_rewards=(
            np.concatenate([p.opponent_rewards for p in parts], axis=0)
            if all(p.opponent_rewards is not None for p in parts) else None),
        families=([f for p in parts for f in (p.families or [])]
                  if all(p.families is not None for p in parts) else None),
    )


def iter_batches(n: int, batch_size: int, rng: np.random.Generator,
                 shuffle: bool = True) -> Iterator[np.ndarray]:
    index = np.arange(n)
    if shuffle:
        rng.shuffle(index)
    for start in range(0, n, max(1, batch_size)):
        yield index[start:start + max(1, batch_size)]


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)


__all__ = ["Rollout", "concat_rollouts", "iter_batches",
           "normalize_advantages"]
