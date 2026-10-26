"""Lock-step 100-round IPD environments.

The environment layer is deliberately small and dependency-free apart from
NumPy for stacking observations.  It never lets the two players observe each
other's current action: both actions in a round are computed from the same
pre-step histories, then appended together.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np

from ipd.core.primitives import payoff

from .obs import features, features_batch

ROUNDS = 100
OpponentFactory = Callable[[], object]


class IPDEnv:
    """One learner versus one opponent."""

    def __init__(self) -> None:
        self.mine: list[int] = []
        self.theirs: list[int] = []
        self.t = 0
        self.opponent = None
        self.family = "unknown"
        self.opponent_name = "unknown"

    def reset(self, factory: OpponentFactory, *,
              family: str = "unknown") -> None:
        self.mine = []
        self.theirs = []
        self.t = 0
        self.family = family
        self.opponent = factory()
        if hasattr(self.opponent, "reset"):
            self.opponent.reset()
        self.opponent_name = getattr(self.opponent, "name",
                                     type(self.opponent).__name__)

    def observe(self) -> list[float]:
        return features(self.t, self.mine, self.theirs)

    def step(self, action: int) -> tuple[float, float, bool]:
        """Apply one simultaneous round; return (own, opponent, done)."""
        if self.t >= ROUNDS:
            raise RuntimeError("episode already finished")
        # Both actors see exactly the histories before this round.
        reply = int(self.opponent.act(self.t, self.theirs, self.mine))
        own = payoff(action, reply)
        other = payoff(reply, action)
        self.mine.append(int(action))
        self.theirs.append(reply)
        self.t += 1
        return float(own), float(other), self.t >= ROUNDS

    @property
    def done(self) -> bool:
        return self.t >= ROUNDS

    @property
    def own_score(self) -> int:
        return int(sum(payoff(a, b) for a, b in zip(self.mine, self.theirs)))

    @property
    def opponent_score(self) -> int:
        return int(sum(payoff(b, a) for a, b in zip(self.mine, self.theirs)))


class VectorEnv:
    """A list of IPD envs stepped in lock-step.

    This is the 32-environment simulator from the plan.  Opponent logic is
    still Python (the bank is heterogeneous), but observations and the learner
    forward pass are batched.
    """

    def __init__(self, factories: Sequence[OpponentFactory],
                 families: Sequence[str] | None = None):
        self.envs = [IPDEnv() for _ in factories]
        self.factories = list(factories)
        self.families = list(families or ["unknown"] * len(factories))
        self.reset()

    def reset(self, factories: Sequence[OpponentFactory] | None = None,
              families: Sequence[str] | None = None) -> None:
        if factories is not None:
            self.factories = list(factories)
            if len(self.factories) != len(self.envs):
                raise ValueError("factory count does not match environment count")
        if families is not None:
            self.families = list(families)
        for env, factory, family in zip(self.envs, self.factories,
                                        self.families):
            env.reset(factory, family=family)

    @property
    def n(self) -> int:
        return len(self.envs)

    def observations(self) -> np.ndarray:
        return features_batch(self.envs)

    def step(self, actions: Sequence[int]) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]:
        if len(actions) != len(self.envs):
            raise ValueError("action count does not match environment count")
        own = np.zeros(len(self.envs), dtype=np.float32)
        other = np.zeros(len(self.envs), dtype=np.float32)
        done = np.zeros(len(self.envs), dtype=np.bool_)
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            a, b, d = env.step(int(action))
            own[i] = a
            other[i] = b
            done[i] = d
        return own, other, done

    @property
    def done(self) -> bool:
        return all(env.done for env in self.envs)

    @property
    def scores(self) -> list[int]:
        return [env.own_score for env in self.envs]


def make_factory(make_bot: Callable[[random.Random], object],
                 seed: int, *args, **kwargs) -> OpponentFactory:
    """Return a deterministic per-episode factory."""
    rng = random.Random(seed)

    def factory():
        return make_bot(rng, *args, **kwargs)

    return factory


def play_episode(learner, opponent, rounds: int = ROUNDS) -> tuple[int, int]:
    """Convenience for tests and calibration: learner acts first each round."""
    learner.reset()
    if hasattr(opponent, "reset"):
        opponent.reset()
    mine: list[int] = []
    theirs: list[int] = []
    own = other = 0
    for t in range(rounds):
        a = int(learner.act(t, mine, theirs))
        b = int(opponent.act(t, theirs, mine))
        mine.append(a)
        theirs.append(b)
        own += payoff(a, b)
        other += payoff(b, a)
    return own, other


__all__ = ["ROUNDS", "IPDEnv", "VectorEnv", "OpponentFactory",
           "make_factory", "play_episode"]
