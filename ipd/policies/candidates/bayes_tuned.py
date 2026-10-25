"""Candidate: Bayes-tuned.

Embeds the tuned prior scales and thresholds.  The scaling is applied locally
after the core BayesPlus model bank is built, so this file does not import
another candidate.
"""
from __future__ import annotations

import math

from ...core.bayes_plus import BayesPlusMachine

KIND = "bayes-tuned"
DISPLAY = "Bayes-tuned"

COOP_SCALE = 1.8369823500713045
PUNITIVE_SCALE = 2.4705958305576154
LATE_SCALE = 0.5216017158776368
NOISE_SCALE = 0.5817914534720606
PROBE_SCALE = 3.315461421931959
NONRESP_THRESHOLD = 0.27974873408266854
PUNITIVE_THRESHOLD = 0.7531347429866336


class BayesTunedPolicy(BayesPlusMachine):
    name = DISPLAY

    def reset(self) -> None:
        super().reset()
        for i, (name, prior, fn) in enumerate(self.models):
            if name.startswith("LateD"):
                scale = LATE_SCALE
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                scale = PUNITIVE_SCALE
            elif name.startswith(("Joss", "NoisyTFT", "Random")):
                scale = NOISE_SCALE
            elif name.startswith(("Tester", "Prober", "Remorseful")):
                scale = PROBE_SCALE
            else:
                scale = COOP_SCALE
            self.models[i] = (name, prior * scale, fn)
        self._priors = [x[1] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0
        self.NONRESP_THRESHOLD = NONRESP_THRESHOLD
        self.PUNITIVE_THRESHOLD = PUNITIVE_THRESHOLD


def build(*, seed: int = 20261024, name: str | None = None,
          **_: object) -> BayesTunedPolicy:
    policy = BayesTunedPolicy()
    policy.name = name or DISPLAY
    return policy
