"""Typed parameter sets for each rule-based policy.

Keeping one parameter object per strategy makes it explicit which knobs belong
to which model.  The historical dictionaries in :mod:`ipd.policies.tuned` are
still accepted by the policy constructors for backwards compatibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from ..tuned import (
    DEFAULT_BAYES_BASE,
    DEFAULT_BAYES_TUNED,
    DEFAULT_FINAL_TUNED,
    DEFAULT_RESP_TUNED,
)


def _pick(data: Mapping[str, Any], cls: type) -> dict[str, Any]:
    known = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in known}


@dataclass(frozen=True)
class FinalParams:
    """Thresholds for the Final state machine."""

    P_UNPROVOKED: float = 0.25
    P_MUTUAL: float = 0.45
    P_RESPONSIVE: float = 0.95
    MIN_C: int = 5
    MIN_D: int = 3
    RESP_GAP: float = 0.38
    DEFECT_RATE: float = 0.18
    PEACE_TESTS: int = 2
    PEACE_COOLDOWN: int = 8

    @classmethod
    def tuned(cls) -> "FinalParams":
        return cls(**_pick(DEFAULT_FINAL_TUNED, cls))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FinalParams":
        return cls(**_pick(data, cls))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Resp3Params:
    """Thresholds for Responsive3."""

    g_unprov: float = 0.55
    g_mutual: float = 0.40
    g_resp: float = 0.95
    min_c: int = 5
    min_d: int = 3
    resp: float = 0.38
    rate: float = 0.18
    max_tests: int = 2
    cooldown: int = 8

    @classmethod
    def tuned(cls) -> "Resp3Params":
        return cls(**_pick(DEFAULT_RESP_TUNED, cls))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Resp3Params":
        return cls(**_pick(data, cls))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BayesParams:
    """Prior scales and thresholds for BayesPlus."""

    coop_scale: float = 1.0
    punitive_scale: float = 1.0
    late_scale: float = 1.0
    noise_scale: float = 1.0
    probe_scale: float = 1.0
    nonresp: float = 0.40
    punitive_thr: float = 0.55

    @classmethod
    def base(cls) -> "BayesParams":
        return cls(**_pick(DEFAULT_BAYES_BASE, cls))

    @classmethod
    def tuned(cls) -> "BayesParams":
        return cls(**_pick(DEFAULT_BAYES_TUNED, cls))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BayesParams":
        return cls(**_pick(data, cls))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HarvestParams(FinalParams):
    """Final thresholds plus the harvest start day."""

    HARVEST_START: int = 45

    @classmethod
    def tuned(cls) -> "HarvestParams":
        merged = dict(DEFAULT_FINAL_TUNED)
        merged["HARVEST_START"] = 45
        return cls(**_pick(merged, cls))


@dataclass(frozen=True)
class HybridParams:
    final: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FINAL_TUNED))
    resp: dict[str, Any] = field(default_factory=dict)
    p: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HybridParams":
        return cls(
            final=dict(data.get("final") or DEFAULT_FINAL_TUNED),
            resp=dict(data.get("resp") or {}),
            p=float(data.get("p", 0.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TripleParams:
    final: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FINAL_TUNED))
    resp: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RESP_TUNED))
    bayes: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_BAYES_TUNED))
    mode: str = "or"
    conf: float = 0.65

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TripleParams":
        return cls(
            final=dict(data.get("final") or DEFAULT_FINAL_TUNED),
            resp=dict(data.get("resp") or DEFAULT_RESP_TUNED),
            bayes=dict(data.get("bayes") or DEFAULT_BAYES_TUNED),
            mode=str(data.get("mode", "majority")),
            conf=float(data.get("conf", 0.65)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GapStartParams:
    start: int | None = 1
    threshold: float = 0.55

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GapStartParams":
        return cls(
            start=data.get("start", 1),
            threshold=float(data.get("threshold", 0.55)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "FinalParams", "Resp3Params", "BayesParams", "HarvestParams",
    "HybridParams", "TripleParams", "GapStartParams",
]
