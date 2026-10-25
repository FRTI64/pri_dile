"""Final + Responsive3 hybrid policy."""
from __future__ import annotations

import random
from typing import Any

from ...core.simulator import C, D
from .final import ParamPolicy
from .responsive3 import ParamResp


class Hybrid:
    """Final + Responsive3; agree on action, otherwise sample/defect."""

    name = "Final+Resp3"

    def __init__(self, final_cfg: dict[str, Any], resp_cfg: dict[str, Any],
                 p_coop: float = 1.0, seed: int = 20261025):
        self.f = ParamPolicy(seed=20261024, **final_cfg)
        self.r = ParamResp(seed=20261024, **resp_cfg)
        self.p = p_coop
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.f.reset()
        self.r.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        a = self.f.act(t, mine, theirs)
        b = self.r.act(t, mine, theirs)
        if a == b:
            return a
        return C if self.rng.random() < self.p else D
