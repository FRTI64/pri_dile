"""Triple-majority / voting fusion policy."""
from __future__ import annotations

import random
from typing import Any

from ...core.simulator import C, D
from .bayes import ParamBayesPrior
from .final import ParamPolicy
from .responsive3 import ParamResp


class Triple:
    """Final, Responsive3 and BayesPlus fused by a voting rule."""

    name = "Triple-majority"

    def __init__(self, f_cfg: dict[str, Any], r_cfg: dict[str, Any],
                 b_cfg: dict[str, Any], mode: str = "or",
                 conf: float = 0.65, seed: int = 20261027):
        self.f = ParamPolicy(seed=20261024, **f_cfg)
        self.r = ParamResp(seed=20261024, **r_cfg)
        self.b = ParamBayesPrior(b_cfg)
        self.mode = mode
        self.conf = conf
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.f.reset()
        self.r.reset()
        self.b.reset()

    def groups(self, mine: list[int], theirs: list[int]):
        post = self.b._posterior(mine, theirs)
        resp = nonresp = punitive = 0.0
        for name, p in post.items():
            if name.startswith(("TFT", "sTFT", "GTFT", "Joss", "NoisyTFT",
                                "Pavlov")):
                resp += p
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                punitive += p
            else:
                nonresp += p
        return resp, nonresp, punitive, max(post.values())

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        a = self.f.act(t, mine, theirs)
        b = self.r.act(t, mine, theirs)
        c = self.b.act(t, mine, theirs)
        if self.mode == "or":
            return D if (a == D or b == D or c == D) else C
        if self.mode == "and":
            return C if (a == C or b == C or c == C) else D
        if self.mode == "majority":
            return C if (a + b + c) <= 1 else D
        resp, nonresp, punitive, mx = self.groups(mine, theirs)
        if self.mode == "bayes_conf":
            if mx >= self.conf:
                return c
            return D if (a == D or b == D) else C
        if self.mode == "bayes_punitive":
            if punitive >= 0.55:
                return c
            return D if (a == D or b == D) else C
        if self.mode == "bayes_nonresp":
            if punitive >= 0.55 or nonresp >= 0.40:
                return c
            return D if (a == D or b == D) else C
        raise ValueError(self.mode)
