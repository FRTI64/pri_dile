"""Candidate: Triple-majority (also triple-or / triple-and variants).

Self-contained voting candidate.  It constructs Final, Responsive3 and
BayesPlus locally and embeds their tuned parameters; it does not import another
candidate file.
"""
from __future__ import annotations

import math
import random

from ...core.bayes_plus import BayesPlusMachine
from ...core.machine import Machine
from ...core.opponents.reciprocal import Responsive3
from ...core.simulator import C, D

KIND = "triple-majority"
DISPLAY = "Triple-majority"
SEED = 20261027

FINAL_TUNED = {
    "P_UNPROVOKED": 0.2395699546614404,
    "P_MUTUAL": 0.6399155368157192,
    "P_RESPONSIVE": 0.9544467968114831,
    "MIN_C": 5,
    "MIN_D": 2,
    "RESP_GAP": 0.3871088255996472,
    "DEFECT_RATE": 0.1170465108152342,
    "PEACE_TESTS": 1,
    "PEACE_COOLDOWN": 9,
}

RESP_TUNED = {
    "g_unprov": 0.31711381663242244,
    "g_mutual": 0.26173130084075913,
    "g_resp": 0.9804446329081643,
    "min_c": 5,
    "min_d": 2,
    "resp": 0.46788262082576104,
    "rate": 0.197064176521839,
    "max_tests": 3,
    "cooldown": 14,
}

BAYES_TUNED = {
    "coop_scale": 1.8369823500713045,
    "punitive_scale": 2.4705958305576154,
    "late_scale": 0.5216017158776368,
    "noise_scale": 0.5817914534720606,
    "probe_scale": 3.315461421931959,
    "nonresp": 0.27974873408266854,
    "punitive_thr": 0.7531347429866336,
}


class _TunedBayes(BayesPlusMachine):
    name = "Triple-Bayes"

    def reset(self) -> None:
        super().reset()
        for i, (name, prior, fn) in enumerate(self.models):
            if name.startswith("LateD"):
                scale = BAYES_TUNED["late_scale"]
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                scale = BAYES_TUNED["punitive_scale"]
            elif name.startswith(("Joss", "NoisyTFT", "Random")):
                scale = BAYES_TUNED["noise_scale"]
            elif name.startswith(("Tester", "Prober", "Remorseful")):
                scale = BAYES_TUNED["probe_scale"]
            else:
                scale = BAYES_TUNED["coop_scale"]
            self.models[i] = (name, prior * scale, fn)
        self._priors = [x[1] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0
        self.NONRESP_THRESHOLD = BAYES_TUNED["nonresp"]
        self.PUNITIVE_THRESHOLD = BAYES_TUNED["punitive_thr"]


class TripleMajorityPolicy:
    """Final, Responsive3 and BayesPlus fused by a voting rule."""

    name = DISPLAY

    def __init__(self, voting: str = "majority", seed: int = SEED):
        self.f = Machine(seed=20261024)
        for key, value in FINAL_TUNED.items():
            setattr(self.f, key, value)
        self.r = Responsive3(rng=random.Random(20261024), **RESP_TUNED)
        self.b = _TunedBayes()
        self.mode = voting
        self.conf = 0.65
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
        a = self.f.move(t + 1, mine, theirs)
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


def build(*, voting: str = "majority", name: str | None = None,
          **_: object) -> TripleMajorityPolicy:
    policy = TripleMajorityPolicy(voting=voting)
    policy.name = name or DISPLAY
    return policy
