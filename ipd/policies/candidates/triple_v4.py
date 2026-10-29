"""Candidate: Triple-v4.

A protocol-v4-tuned variant of Triple-majority.  Final, Responsive3 and
BayesPlus vote as before, but the Final component gains one narrow
noise-tolerant rule:

    when the opponent has repeatedly retaliated after our defection
    (pD high) while also defecting occasionally after our cooperation
    (pC moderate), treat it as a noisy reciprocator and cooperate after
    its defections instead of starting another punishment spiral.

The condition is deliberately narrow (pD >= 0.70, pD - pC >= 0.30,
0.20 <= pC <= 0.62).  It targets noisy TFT/Joss and probe-style
opponents, and leaves clean TFT, always-defect, and scheduled
late-harvest opponents on the original Final logic.  No proactive probe
is added, so the dev_family guardrail is unchanged.
"""
from __future__ import annotations

import math
import random

from ...core.bayes_plus import BayesPlusMachine
from ...core.machine import Machine
from ...core.opponents.reciprocal import Responsive3
from ...core.simulator import C, D

KIND = "triple-v4"
DISPLAY = "Triple-v4"
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

# Noise-tolerant branch thresholds.  The condition is intentionally
# conservative: it only fires after enough observations and only for an
# opponent that retaliates much more strongly than it probes.
NOISE_PD = 0.70
NOISE_PC_MIN = 0.20
NOISE_PC_MAX = 0.62
NOISE_GAP = 0.30
NOISE_RATE = 0.15
NOISE_FORGIVE = 0.75


class TolerantFinal(Machine):
    """Final state machine with a narrow noisy-reciprocator branch."""

    name = "Triple-v4-Final"

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        if day >= 100:
            return D
        if day == 1:
            return C
        n_c, n_d, d_c, d_d = self._stats(mine, theirs)
        p_c = d_c / n_c if n_c else 0.0
        p_d = d_d / n_d if n_d else 0.0
        d_rate = sum(theirs) / len(theirs)
        if (n_c >= self.MIN_C and n_d >= self.MIN_D
                and p_d >= NOISE_PD
                and NOISE_PC_MIN <= p_c <= NOISE_PC_MAX
                and p_d - p_c >= NOISE_GAP
                and d_rate >= NOISE_RATE):
            if theirs[-1] == C:
                return C
            return C if self.rng.random() < NOISE_FORGIVE else D
        return super().move(day, mine, theirs)


class _TunedBayes(BayesPlusMachine):
    name = "Triple-v4-Bayes"

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


class TripleV4Policy:
    """Final, Responsive3 and BayesPlus fused by majority voting."""

    name = DISPLAY

    def __init__(self, seed: int = SEED):
        self.f = TolerantFinal(seed=20261024)
        for key, value in FINAL_TUNED.items():
            setattr(self.f, key, value)
        self.r = Responsive3(rng=random.Random(20261024), **RESP_TUNED)
        self.b = _TunedBayes()
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
        return C if (a + b + c) <= 1 else D


def build(*, seed: int = SEED, name: str | None = None,
          **_: object) -> TripleV4Policy:
    policy = TripleV4Policy(seed=seed)
    policy.name = name or DISPLAY
    return policy
