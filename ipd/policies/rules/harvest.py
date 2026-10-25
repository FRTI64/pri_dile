"""Harvest variants of the Final state machine."""
from __future__ import annotations

from typing import Any

from ...core.harvest import HarvestMachine


class HarvestPolicy(HarvestMachine):
    name = "Harvest"

    def __init__(self, seed: int = 20261024, **params: Any):
        for key, value in params.items():
            setattr(self, key, value)
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()


class ParamHarvest(HarvestMachine):
    name = "Harvest-tuned"

    def __init__(self, seed: int = 20261024, **params: Any):
        for key, value in params.items():
            setattr(self, key, value)
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        return self.move(t + 1, mine, theirs)
