"""Responsive3 submission policy with tunable parameters."""
from __future__ import annotations

import random
from typing import Any

from ...core.opponents import Responsive3


class ParamResp(Responsive3):
    """Responsive3 with per-instance parameters."""

    name = "Resp3-tuned"

    def __init__(self, seed: int = 20261024, **params: Any):
        super().__init__(rng=random.Random(seed))
        for key, value in params.items():
            setattr(self, key, value)

    def reset(self) -> None:
        super().reset()
