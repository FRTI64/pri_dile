"""Red threat population, league, and the competitive reward constraint."""
from __future__ import annotations

import copy
import random
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ipd.core.primitives import C, D

from .env import play_episode
from .models import TorchOpponent, snapshot


@dataclass
class RedSnapshot:
    iteration: int
    state: dict[str, torch.Tensor]
    score: float = 0.0
    family_scores: dict[str, float] = field(default_factory=dict)


class RedLeague:
    """Recent red checkpoints, plus the current in-training red policy."""

    def __init__(self, keep: int = 8):
        self.keep = max(1, int(keep))
        self.snapshots: deque[RedSnapshot] = deque(maxlen=self.keep)

    def add(self, model, iteration: int, score: float = 0.0,
            family_scores: dict[str, float] | None = None) -> RedSnapshot:
        snap = RedSnapshot(iteration, snapshot(model), float(score),
                           dict(family_scores or {}))
        self.snapshots.append(snap)
        return snap

    def __len__(self) -> int:
        return len(self.snapshots)

    def sample(self, rng: random.Random) -> RedSnapshot | None:
        if not self.snapshots:
            return None
        return rng.choice(list(self.snapshots))

    def latest(self) -> RedSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def states(self) -> list[dict[str, torch.Tensor]]:
        return [s.state for s in self.snapshots]


@dataclass
class RedConstraint:
    """Per-step difference reward with a rolling-window competitiveness penalty.

    ``R_t = red_t - target_t - lam * max(0, alpha * mean_W(target) - mean_W(red))^2``

    The penalty is zero before ``W`` rounds and for absolute-score (field)
    targets.  ``W`` is resampled per episode from ``window_range``.
    """

    alpha: float = 0.85
    lam: float = 5.0
    window_range: tuple[int, int] = (15, 25)
    enabled: bool = True

    def episode_rewards(self, red_scores: Sequence[float],
                        target_scores: Sequence[float] | None,
                        rng: random.Random) -> tuple[list[float], dict]:
        n = len(red_scores)
        if target_scores is None or not self.enabled:
            return [float(x) for x in red_scores], {
                "penalty": 0.0, "violation": 0.0, "window": 0}
        window = rng.randint(*self.window_range)
        out: list[float] = []
        penalties: list[float] = []
        red_hist: list[float] = []
        target_hist: list[float] = []
        for t, (red, target) in enumerate(zip(red_scores, target_scores)):
            red_hist.append(float(red))
            target_hist.append(float(target))
            penalty = 0.0
            if t + 1 >= window:
                red_avg = statistics.fmean(red_hist[-window:])
                target_avg = statistics.fmean(target_hist[-window:])
                penalty = self.lam * max(
                    0.0, self.alpha * target_avg - red_avg) ** 2
            penalties.append(penalty)
            out.append(float(red) - float(target) - penalty)
        red_total = sum(red_scores)
        target_total = sum(target_scores)
        violation = max(0.0, self.alpha * target_total - red_total)
        return out, {
            "penalty": statistics.fmean(penalties) if penalties else 0.0,
            "violation": violation,
            "window": window,
            "red_total": red_total,
            "target_total": target_total,
        }


def calibration_scores(suite, seeds: Sequence[int],
                       bots: Sequence) -> dict[str, dict[str, float]]:
    """Per-family reference scores of frozen strong bots.

    The reference is the *best* of the supplied bots on each family, which
    makes the red anchor conservative rather than easy to game.
    """
    out: dict[str, dict[str, float]] = {}
    for bot in bots:
        for item in suite:
            for seed in seeds:
                rng = random.Random(seed)
                opponent = item.factory(rng)
                score, _ = play_episode(bot, opponent)
                out.setdefault(item.family, {}).setdefault(bot.name, []).append(
                    float(score))  # type: ignore[arg-type]
    result: dict[str, float] = {}
    for family, by_bot in out.items():
        per_bot = [statistics.fmean(v) for v in by_bot.values()]
        result[family] = max(per_bot)
    return {"families": result, "raw": out}


def red_snapshot_factory(state: dict[str, torch.Tensor], config,
                         device: str | torch.device, seed: int,
                         name: str = "red-snapshot"):
    """Return an opponent factory usable by :class:`VectorEnv`."""
    def factory():
        return TorchOpponent(state, config=config, device=device,
                             mode="sample", seed=seed, name=name)
    return factory


__all__ = ["RedSnapshot", "RedLeague", "RedConstraint",
           "calibration_scores", "red_snapshot_factory"]
