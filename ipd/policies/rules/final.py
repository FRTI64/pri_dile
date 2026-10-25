"""Final state-machine policies."""
from __future__ import annotations

from typing import Any

from ...core.machine import Machine


class FinalPolicy(Machine):
    name = "Final"

    def reset(self) -> None:
        super().reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        return self.move(t + 1, mine, theirs)


class ParamPolicy(Machine):
    """Final state machine with per-instance thresholds."""

    name = "Final-tuned"

    def __init__(self, seed: int = 20261024, **params: Any):
        for key, value in params.items():
            setattr(self, key, value)
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        return self.move(t + 1, mine, theirs)
