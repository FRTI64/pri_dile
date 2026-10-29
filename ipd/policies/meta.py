"""Small meta-policy that switches from RL-Wide160 to RL-Repair on an
early mutual-defection signal.

The switch is intentionally narrow: it only fires after the first six rounds
when both sides have already defected at least twice and the opponent's last
move is cooperation.  In the full opponent scan this pattern is common for
zero-determinant extortion and noisy/Joss opponents, where RL-Repair is more
robust, while TFT/Grim/AllC remain on the safe RL-Wide160 path.
"""
from __future__ import annotations

import json

from ..paths import weights_path
from .neural import NeuralPolicy
from .rl import RLPolicy


def _load(path: str, seed: int, mode: str):
    with open(path, encoding="utf-8") as fh:
        fmt = json.load(fh).get("format")
    cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
    return cls(path, seed=seed, mode=mode)


class RLSwitchPolicy:
    """Base RL-Wide160 with a narrow early switch to RL-Repair."""

    def __init__(self, base_weights: str, alt_weights: str,
                 seed: int = 20261024, mode: str = "sample",
                 name: str = "RL-Switch"):
        self.base = _load(base_weights, seed, mode)
        self.alt = _load(alt_weights, seed + 1, mode)
        self.mode = mode
        self.name = name
        self.switched = False

    def reset(self) -> None:
        self.base.reset()
        self.alt.reset()
        self.switched = False

    def set_rounds(self, rounds: int) -> None:
        for policy in (self.base, self.alt):
            fn = getattr(policy, "set_rounds", None)
            if callable(fn):
                fn(rounds)

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if not self.switched and t >= 6:
            first_opp = theirs[:6]
            first_mine = mine[:6]
            if (first_opp.count(1) >= 2 and first_mine.count(1) >= 2
                    and theirs[-1] == 0):
                self.switched = True
        policy = self.alt if self.switched else self.base
        return policy.act(t, mine, theirs)

    def observe(self, mine: list[int], theirs: list[int]) -> None:
        """No internal observation state; histories are passed per action."""

    def __repr__(self) -> str:
        return self.name


__all__ = ["RLSwitchPolicy"]
