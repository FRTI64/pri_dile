"""Our own rule policies wrapped as simulator opponents."""
from __future__ import annotations

import random

from ...core.early import EarlyProbeMachine
from ...core.harvest import HarvestMachine
from ...core.machine import Machine
from ...core.opponents import Responsive3
from ...core.opponents.spec import TypeSpec
from ...core.simulator import Strategy
from ..rules import GapStartPolicy, ParamHarvest, ParamPolicy, ParamResp
from ..tuned import DEFAULT_FINAL_TUNED, DEFAULT_RESP_TUNED


class SmartAdapter(Strategy):
    """Wrap a policy as a simulator `Strategy` opponent."""

    def __init__(self, name: str, inner):
        self.name = name
        self.inner = inner

    def reset(self) -> None:
        self.inner.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if callable(getattr(self.inner, "act", None)):
            return self.inner.act(t, mine, theirs)
        return self.inner.move(t + 1, mine, theirs)


def _final(r):
    return SmartAdapter("Final", Machine(seed=r.randrange(1 << 30)))


def _harvest45(r):
    return SmartAdapter("Harvest45", HarvestMachine(seed=r.randrange(1 << 30)))


def _harvest40(r):
    h = HarvestMachine(seed=r.randrange(1 << 30))
    h.HARVEST_START = 40
    return SmartAdapter("Harvest40", h)


def _early(r):
    return SmartAdapter("EarlyProbe", EarlyProbeMachine(seed=r.randrange(1 << 30)))


def _gap(r):
    return SmartAdapter("GapStart", GapStartPolicy(1, 0.5))


def _bayes(r):
    from ...core.bayes_plus import BayesPlusMachine
    return SmartAdapter("BayesPlus", BayesPlusMachine())


def _resp(r):
    return SmartAdapter(
        "Responsive3", Responsive3(rng=random.Random(r.randrange(1 << 30)))
    )


SMART: list[TypeSpec] = [
    TypeSpec("Final", "smart", 4, _final),
    TypeSpec("Harvest45", "smart", 3, _harvest45),
    TypeSpec("Harvest40", "smart", 2, _harvest40),
    TypeSpec("EarlyProbe", "smart", 2, _early),
    TypeSpec("GapStart", "smart", 3, _gap),
    TypeSpec("BayesPlus", "smart", 2, _bayes),
    TypeSpec("Responsive3", "smart", 2, _resp),
]
SMART_BY_NAME = {spec.name: spec for spec in SMART}
