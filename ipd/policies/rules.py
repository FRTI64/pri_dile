"""Rule-based submission policies.

These are the strategies that can be submitted without PyTorch.  They are
independent of training scripts and evaluation harnesses.
"""
from __future__ import annotations

import math
import random
from typing import Any

from ..core.bayes_plus import BayesPlusMachine
from ..core.early import EarlyProbeMachine
from ..core.features import FEATURES, feature_vector
from ..core.harvest import HarvestMachine
from ..core.machine import Machine
from ..core.opponents import Responsive3
from ..core.simulator import C, D


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


class ParamResp(Responsive3):
    """Responsive3 with per-instance parameters."""

    name = "Resp3-tuned"

    def __init__(self, seed: int = 20261024, **params: Any):
        super().__init__(rng=random.Random(seed))
        for key, value in params.items():
            setattr(self, key, value)

    def reset(self) -> None:
        super().reset()


class ParamBayes(BayesPlusMachine):
    name = "Bayes-param"

    def __init__(self, seed: int = 20261024, **params: Any):
        for key, value in params.items():
            setattr(self, key, value)
        self.reset()


class ParamBayesPrior(BayesPlusMachine):
    """BayesPlus with scaled model priors (tuned by ``train_bayes_priors``)."""

    name = "Bayes-tuned"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        super().reset()
        c = self.cfg
        for i, (name, prior, fn) in enumerate(self.models):
            if name.startswith("LateD"):
                scale = c["late_scale"]
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                scale = c["punitive_scale"]
            elif name.startswith(("Joss", "NoisyTFT", "Random")):
                scale = c["noise_scale"]
            elif name.startswith(("Tester", "Prober", "Remorseful")):
                scale = c["probe_scale"]
            else:
                scale = c["coop_scale"]
            self.models[i] = (name, prior * scale, fn)
        self._priors = [x[1] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0
        self.NONRESP_THRESHOLD = c["nonresp"]
        self.PUNITIVE_THRESHOLD = c["punitive_thr"]


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


class GapStartPolicy:
    """Cooperate until ``start``; then defect iff the gap is small."""

    name = "GapStart"

    def __init__(self, start: int | None = 1, threshold: float = 0.55):
        self.start = start
        self.threshold = threshold

    def reset(self) -> None:
        pass

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if t >= 99:
            return D
        if self.start is not None and t + 1 < self.start:
            return C
        gap = feature_vector(t + 1, mine, theirs)[FEATURES.index("gap")]
        return D if gap <= self.threshold else C


class EarlyProbePolicy(EarlyProbeMachine):
    name = "EarlyProbe"
