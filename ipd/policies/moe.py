"""Three-way safe/risky/unknown mixture-of-experts policy.

``ThreeWayMoE`` is an offline research wrapper for the P2 C3 screen.  A
three-way gate routes every round:

- ``safe``   -> ``safe_expert`` (default: always defect);
- ``risky``  -> ``risky_expert`` (default: RL-Repair fallback);
- ``unknown`` -> ``unknown_expert`` (default: RL-Repair fallback).

Only the selected expert is called, so a planner expert can wrap its own base
policy without advancing the fallback RNG stream.
"""
from __future__ import annotations

from typing import Any

from ipd.core.primitives import D


class DefectExpert:
    """Always defect, the maximal exploitation expert."""

    name = "DefectExpert"

    def reset(self) -> None:
        pass

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        return D


class ThreeWayMoE:
    """Route between a safe exploit expert, a risky expert and a fallback."""

    name = "ThreeWayMoE"

    def __init__(self, base, gate, *, safe_expert=None, risky_expert=None,
                 unknown_expert=None, name: str | None = None):
        self.base = base
        self.gate = gate
        self.safe_expert = safe_expert if safe_expert is not None else base
        self.risky_expert = risky_expert if risky_expert is not None else base
        self.unknown_expert = (unknown_expert if unknown_expert is not None
                               else base)
        if name:
            self.name = name
        self.stats = self._empty_stats()
        self.reset()

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "rounds": 0,
            "safe": 0,
            "risky": 0,
            "unknown": 0,
            "safe_actions": 0,
            "risky_actions": 0,
            "unknown_actions": 0,
        }

    def reset(self) -> None:
        seen: set[int] = set()
        for expert in (self.base, self.safe_expert, self.risky_expert,
                       self.unknown_expert):
            if id(expert) in seen:
                continue
            seen.add(id(expert))
            reset = getattr(expert, "reset", None)
            if reset is not None:
                reset()
        reset_gate = getattr(self.gate, "reset", None)
        if reset_gate is not None:
            reset_gate()
        self.stats = self._empty_stats()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        route, _info = self.gate(t, mine, theirs)
        if route == "safe":
            expert = self.safe_expert
            key = "safe"
        elif route == "risky":
            expert = self.risky_expert
            key = "risky"
        else:
            route = "unknown"
            expert = self.unknown_expert
            key = "unknown"
        action = int(expert.act(t, mine, theirs))
        self.stats["rounds"] += 1
        self.stats[key] += 1
        self.stats[f"{key}_actions"] += 1
        return action

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def diagnostics(self) -> dict[str, Any]:
        rounds = self.stats["rounds"]
        result = dict(self.stats)
        result.update({
            "safe_rate": self.stats["safe"] / rounds if rounds else 0.0,
            "risky_rate": self.stats["risky"] / rounds if rounds else 0.0,
            "unknown_rate": self.stats["unknown"] / rounds if rounds else 0.0,
            "safe_action_rate": (self.stats["safe_actions"] / rounds
                                 if rounds else 0.0),
            # Compatibility keys consumed by planner_eval.evaluate_policy.
            "overrides": self.stats["safe_actions"],
            "fallback_low_confidence": 0,
            "fallback_margin": 0,
            "gate_checks": rounds,
            "fallback_gate": self.stats["risky"] + self.stats["unknown"],
        })
        gate_diag = getattr(self.gate, "diagnostics", None)
        if gate_diag is not None:
            result["gate"] = gate_diag()
        return result

    def __repr__(self) -> str:
        return self.name


__all__ = ["DefectExpert", "ThreeWayMoE"]
