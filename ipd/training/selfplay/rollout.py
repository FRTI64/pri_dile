"""Recurrent rollout storage, GAE, and sequence minibatches.

``Rollout`` keeps the original NumPy representation used by the CPU trainer.
``TensorRollout`` is the Plan 4 GPU-resident variant: observations, actions,
rewards and GAE stay on the training device, and minibatches are indexed with
device tensors instead of round-tripping through NumPy.
"""
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

    def batches(self, batch_size: int, rng: np.random.Generator,
                shuffle: bool = True) -> Iterator[np.ndarray]:
        """Yield episode indices; ``TensorRollout`` overrides with tensors."""
        return iter_batches(self.n_episodes, batch_size, rng, shuffle)

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


@dataclass
class TensorRollout:
    """GPU-resident rollout with the same public API as :class:`Rollout`.

    The storage layout matches ``Rollout`` so a checkpoint or diagnostic can be
    compared field by field.  ``obs``/``actions``/... are torch tensors on one
    device; ``hidden`` is ``[B, T, layers, hidden]`` exactly like NumPy.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    hidden: torch.Tensor
    opponent_rewards: torch.Tensor | None = None
    families: Sequence[str] | None = None
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.obs.device

    @property
    def n_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.obs.shape[0] * self.obs.shape[1])

    def batches(self, batch_size: int, rng: np.random.Generator,
                shuffle: bool = True) -> Iterator[torch.Tensor]:
        for index in iter_batches(self.n_episodes, batch_size, rng, shuffle):
            yield torch.as_tensor(index, dtype=torch.long, device=self.device)

    def compute_gae(self, gamma: float = 1.0, lam: float = 0.95
                    ) -> "TensorRollout":
        rewards = self.rewards.to(torch.float64)
        values = self.values.to(torch.float64)
        b, t = rewards.shape
        advantages = torch.zeros_like(rewards)
        last = torch.zeros(b, dtype=torch.float64, device=self.device)
        for i in range(t - 1, -1, -1):
            next_value = (torch.zeros(b, dtype=torch.float64,
                                      device=self.device)
                          if i == t - 1 else values[:, i + 1])
            delta = rewards[:, i] + gamma * next_value - values[:, i]
            last = delta + gamma * lam * last
            advantages[:, i] = last
        self.advantages = advantages.to(torch.float32)
        self.returns = (advantages + values).to(torch.float32)
        return self

    def tensors(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        if self.advantages is None or self.returns is None:
            raise ValueError("call compute_gae() before tensors()")
        target = torch.device(device)
        out = {
            "obs": self.obs,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "values": self.values,
            "advantages": self.advantages,
            "returns": self.returns,
        }
        return {key: value.to(target) if value.device != target else value
                for key, value in out.items()}

    def take(self, index: Sequence[int] | np.ndarray | torch.Tensor
             ) -> "TensorRollout":
        if not isinstance(index, torch.Tensor):
            index = torch.as_tensor(index, dtype=torch.long,
                                    device=self.device)
        else:
            index = index.to(self.device)
        return TensorRollout(
            obs=self.obs[index], actions=self.actions[index],
            logprobs=self.logprobs[index], values=self.values[index],
            rewards=self.rewards[index], hidden=self.hidden[index],
            opponent_rewards=(None if self.opponent_rewards is None
                              else self.opponent_rewards[index]),
            families=(None if self.families is None
                      else [self.families[i] for i in index.tolist()]),
            advantages=(None if self.advantages is None
                        else self.advantages[index]),
            returns=(None if self.returns is None else self.returns[index]),
        )

    def to_numpy(self) -> dict[str, np.ndarray]:
        """One explicit host transfer for logging or NumPy-only consumers."""
        out = {"obs": self.obs, "actions": self.actions,
               "logprobs": self.logprobs, "values": self.values,
               "rewards": self.rewards, "hidden": self.hidden}
        if self.opponent_rewards is not None:
            out["opponent_rewards"] = self.opponent_rewards
        return {key: value.detach().cpu().numpy()
                for key, value in out.items()}


def concat_rollouts(parts: Sequence[Rollout | TensorRollout]
                    ) -> Rollout | TensorRollout:
    parts = [p for p in parts if p.n_episodes]
    if not parts:
        raise ValueError("no rollouts to concatenate")
    if isinstance(parts[0], TensorRollout):
        return TensorRollout(
            obs=torch.cat([p.obs for p in parts], dim=0),
            actions=torch.cat([p.actions for p in parts], dim=0),
            logprobs=torch.cat([p.logprobs for p in parts], dim=0),
            values=torch.cat([p.values for p in parts], dim=0),
            rewards=torch.cat([p.rewards for p in parts], dim=0),
            hidden=torch.cat([p.hidden for p in parts], dim=0),
            opponent_rewards=(
                torch.cat([p.opponent_rewards for p in parts], dim=0)
                if all(p.opponent_rewards is not None for p in parts) else None),
            families=([f for p in parts for f in (p.families or [])]
                      if all(p.families is not None for p in parts) else None),
        )
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


__all__ = ["Rollout", "TensorRollout", "concat_rollouts", "iter_batches",
           "normalize_advantages"]
