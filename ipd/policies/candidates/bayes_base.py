"""Candidate: Bayes-base.

Self-contained BayesPlus candidate with the base prior scales.
"""
from __future__ import annotations

from ...core.bayes_plus import BayesPlusMachine

KIND = "bayes-base"
DISPLAY = "Bayes-base"


class BayesBasePolicy(BayesPlusMachine):
    name = DISPLAY


def build(*, seed: int = 20261024, name: str | None = None,
          **_: object) -> BayesBasePolicy:
    policy = BayesBasePolicy()
    policy.name = name or DISPLAY
    return policy
