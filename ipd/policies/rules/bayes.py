"""BayesPlus-based submission policies."""
from __future__ import annotations

import math
from typing import Any, Mapping

from ...core.bayes_plus import BayesPlusMachine


class ParamBayes(BayesPlusMachine):
    name = "Bayes-param"

    def __init__(self, seed: int = 20261024, **params: Any):
        for key, value in params.items():
            setattr(self, key, value)
        self.reset()


class ParamBayesPrior(BayesPlusMachine):
    """BayesPlus with scaled model priors (tuned by ``train_bayes_priors``)."""

    name = "Bayes-tuned"

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = dict(cfg)
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
