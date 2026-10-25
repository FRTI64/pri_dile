"""Tuned copies of our policies used as strong ``wide`` opponents."""
from __future__ import annotations

from ...core.opponents.spec import TypeSpec
from ..rules import GapStartPolicy, ParamHarvest, ParamPolicy, ParamResp
from ..tuned import DEFAULT_FINAL_TUNED, DEFAULT_RESP_TUNED
from .smart import SmartAdapter

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
