"""Our own policies used as opponents / tuned copies for robustness tests."""
from __future__ import annotations

import random

from ..core.early import EarlyProbeMachine
from ..core.harvest import HarvestMachine
from ..core.machine import Machine
from ..core.opponents import Responsive3
from ..core.opponents_extended import TYPE_BANK, TypeSpec
from ..core.opponents_plus import PLUS as _PLUS_BASE
from ..core.simulator import Strategy
from .rules import GapStartPolicy, ParamHarvest, ParamPolicy, ParamResp
from .tuned import DEFAULT_FINAL_TUNED, DEFAULT_RESP_TUNED


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
    from ..core.bayes_plus import BayesPlusMachine
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

# Tuned copies used as strong opponents in the "wide" distribution.
TUNED_SMART: list[TypeSpec] = [
    TypeSpec(
        "FinalTuned", "smart", 1.2,
        lambda r: SmartAdapter(
            "FinalTuned",
            ParamPolicy(seed=r.randrange(1 << 30), **DEFAULT_FINAL_TUNED),
        ),
    ),
    TypeSpec(
        "RespTuned", "smart", 1.0,
        lambda r: SmartAdapter(
            "RespTuned",
            ParamResp(seed=r.randrange(1 << 30), **DEFAULT_RESP_TUNED),
        ),
    ),
]
for _start in (21, 35, 50, 75):
    TUNED_SMART.append(TypeSpec(
        f"Harvest{_start}", "smart", 0.7,
        lambda r, start=_start: SmartAdapter(
            f"Harvest{start}",
            ParamHarvest(seed=r.randrange(1 << 30), HARVEST_START=start),
        ),
    ))
for _th in (0.45, 0.55, 0.64):
    TUNED_SMART.append(TypeSpec(
        f"Gap{_th}", "smart", 0.6,
        lambda r, th=_th: SmartAdapter(f"Gap{th}",
                                      GapStartPolicy(1, th)),
    ))

# Full opponent bank in the historical order: 85-type extended bank, our smart
# strategies, extra parameterised types, then the tuned copies.
PLUS: list[TypeSpec] = list(_PLUS_BASE) + list(TUNED_SMART)
ALL_SPECS: list[TypeSpec] = (
    list(TYPE_BANK) + list(SMART) + list(_PLUS_BASE) + list(TUNED_SMART)
)
_by_name = {spec.name: spec for spec in ALL_SPECS}
ALL_SPECS = list(_by_name.values())
ALL_BY_NAME = {spec.name: spec for spec in ALL_SPECS}
