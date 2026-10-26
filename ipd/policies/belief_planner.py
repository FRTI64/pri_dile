"""Bayesian opponent belief + finite-horizon planner (P2 C1 prototype).

The planner maintains one particle per opponent type in
:data:`ipd.core.opponents.bank.TYPE_BANK`.  On every round it updates the
belief from the opponent's latest observed action, then evaluates a short
lookahead for ``C`` vs ``D`` using the top hypotheses.  The wrapped base policy
(usually RL-Repair) supplies the fallback action and the future rollout policy.

This is deliberately an **offline research module**.  It is not registered as a
submission candidate and does not change any existing weights.  The intended
use is:

    base = NeuralPolicy("weights/rl_policy_weights_repair.json")
    planner = BeliefPlanner(base, horizon=6, top_k=8)
    action = planner.act(t, mine, theirs)
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

from ipd.core.opponents.bank import TYPE_BANK
from ipd.core.primitives import C, D, payoff

CURATED_NAMES = (
    "AllC", "TFT", "sTFT", "TFT2", "Pavlov", "GTFT.05", "GTFT.10",
    "GTFT.20", "NoisyTFT.05", "NoisyTFT.10", "Joss.05", "Joss.10",
    "Grim", "GrimForgive.05", "Mistrust", "Bully", "AllD", "EarlyD1",
    "EarlyD10", "EarlyD25", "LateD10", "LateD25", "LateD50", "Tester",
    "Prober", "RemorsefulProber", "Random.10", "Random.30", "Random.50",
    "Random.70", "Random.90", "Alternator", "Periodic3", "Rand3->TFT",
    "Responsive3", "ProbeBot", "Extort1.5", "Extort2", "Extort3",
)
_CURATED = tuple(spec for spec in TYPE_BANK if spec.name in CURATED_NAMES)


@dataclass
class _Particle:
    name: str
    family: str
    model: Any
    log_weight: float = 0.0
    prediction: int = C
    weight: float = 1.0


@dataclass
class PlannerStats:
    """Diagnostics for offline evaluation."""

    rounds: int = 0
    overrides: int = 0
    fallback_low_confidence: int = 0
    fallback_margin: int = 0
    gate_checks: int = 0
    fallback_gate: int = 0
    gate_label: str = ""
    gate_confidence: float = 0.0
    top_name: str = ""
    top_weight: float = 0.0
    action_probability_d: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "overrides": self.overrides,
            "override_rate": (self.overrides / self.rounds
                              if self.rounds else 0.0),
            "fallback_low_confidence": self.fallback_low_confidence,
            "fallback_margin": self.fallback_margin,
            "gate_checks": self.gate_checks,
            "fallback_gate": self.fallback_gate,
            "gate_label": self.gate_label,
            "gate_confidence": self.gate_confidence,
            "top_name": self.top_name,
            "top_weight": self.top_weight,
            "action_probability_d": self.action_probability_d,
        }


class BeliefPlanner:
    """Wrap a base policy with a Bayesian opponent model and short planning."""

    name = "BeliefPlanner"

    def __init__(
        self,
        base,
        *,
        horizon: int = 6,
        top_k: int = 8,
        max_hypotheses: int = 48,
        fallback_confidence: float = 0.75,
        match_prob: float = 0.85,
        margin: float = 0.0,
        seed: int = 20261024,
        name: str | None = None,
        gate=None,
    ):
        self.base = base
        self.gate = gate
        self.horizon = max(1, int(horizon))
        self.top_k = max(1, int(top_k))
        self.max_hypotheses = max(1, int(max_hypotheses))
        self.fallback_confidence = float(fallback_confidence)
        self.match_prob = min(1.0, max(0.5, float(match_prob)))
        self.margin = float(margin)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        if name:
            self.name = name
        specs = _CURATED or tuple(TYPE_BANK)
        if self.max_hypotheses < len(specs):
            specs = tuple(sorted(specs, key=lambda spec: spec.weight,
                                 reverse=True)[:self.max_hypotheses])
        self.specs = specs
        self.stats = PlannerStats()
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        gate_reset = getattr(self.gate, "reset", None)
        if gate_reset is not None:
            gate_reset()
        self.particles: list[_Particle] = []
        for i, spec in enumerate(self.specs):
            rng = random.Random(self.rng.randrange(1 << 30) + i)
            model = spec.factory(rng)
            model.reset()
            prediction = int(model.act(0, [], []))
            self.particles.append(_Particle(
                name=spec.name, family=spec.family, model=model,
                log_weight=math.log(max(1e-9, spec.weight)),
                prediction=prediction, weight=1.0))
        self._normalise()
        self.stats = PlannerStats()
        self._last_observed = 0

    # ------------------------------------------------------------------
    # Belief update
    # ------------------------------------------------------------------
    def _normalise(self) -> None:
        if not self.particles:
            return
        top = max(particle.log_weight for particle in self.particles)
        total = 0.0
        for particle in self.particles:
            particle.weight = math.exp(particle.log_weight - top)
            total += particle.weight
        if total <= 0.0:
            for particle in self.particles:
                particle.weight = 1.0 / len(self.particles)
            return
        for particle in self.particles:
            particle.weight /= total

    def _update_belief(self, observed: int) -> None:
        match = self.match_prob
        miss = 1.0 - match
        for particle in self.particles:
            likelihood = match if particle.prediction == observed else miss
            particle.log_weight += math.log(max(1e-12, likelihood))
        self._normalise()

    def _advance_models(self, t: int, mine: Sequence[int],
                        theirs: Sequence[int]) -> None:
        our = list(mine)
        their = list(theirs)
        for particle in self.particles:
            particle.prediction = int(particle.model.act(t, their, our))

    def predicted_action_probability(self) -> float:
        total = sum(particle.weight for particle in self.particles)
        if total <= 0.0:
            return 0.5
        p_d = sum(particle.weight for particle in self.particles
                  if particle.prediction == D) / total
        return p_d

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _rollout(self, particle: _Particle, t: int, mine: Sequence[int],
                 theirs: Sequence[int], first: int) -> float:
        model = copy.deepcopy(particle.model)
        base = None
        if not hasattr(self.base, "predict_proba"):
            base = copy.deepcopy(self.base)
        our = list(mine)
        their = list(theirs)
        their.append(particle.prediction)
        our.append(first)
        value = float(payoff(first, particle.prediction))
        end = min(99, t + self.horizon - 1)
        for u in range(t + 1, end + 1):
            if u == 99:
                action = D
            elif base is None:
                # Stateless neural policy: use the deterministic argmax
                # continuation instead of consuming the live RNG stream.
                if (getattr(self.base, "safe", False) and D not in their):
                    action = C
                else:
                    p_defect = float(self.base.predict_proba(our, their))
                    action = D if p_defect >= 0.5 else C
            else:
                action = int(base.act(u, our, their))
            reply = int(model.act(u, their, our))
            value += payoff(action, reply)
            our.append(action)
            their.append(reply)
        return value

    def _plan(self, t: int, mine: Sequence[int], theirs: Sequence[int],
              base_action: int) -> tuple[int, float]:
        ranked = sorted(self.particles, key=lambda p: p.weight, reverse=True)
        top = ranked[:self.top_k]
        total = sum(particle.weight for particle in top)
        if total <= 0.0:
            return base_action, 0.0
        values = {C: 0.0, D: 0.0}
        for particle in top:
            weight = particle.weight / total
            values[C] += weight * self._rollout(
                particle, t, mine, theirs, C)
            values[D] += weight * self._rollout(
                particle, t, mine, theirs, D)
        best = max(values, key=lambda action: values[action])
        return best, values[best] - values[base_action]

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        self.stats.rounds += 1
        if t == 0:
            base_action = int(self.base.act(t, mine, theirs))
            return base_action
        self._update_belief(int(theirs[-1]))
        self._advance_models(t, mine, theirs)
        p_d = self.predicted_action_probability()
        confidence = max(p_d, 1.0 - p_d)
        self.stats.action_probability_d = p_d
        base_action = int(self.base.act(t, mine, theirs))
        if t >= 99:
            return D
        if self.gate is not None:
            self.stats.gate_checks += 1
            allowed, info = self.gate(t, mine, theirs)
            self.stats.gate_label = str(info.get("label", ""))
            self.stats.gate_confidence = float(info.get("confidence", 0.0))
            if not allowed:
                self.stats.fallback_gate += 1
                return base_action
        if confidence < self.fallback_confidence:
            self.stats.fallback_low_confidence += 1
            return base_action
        chosen, gain = self._plan(t, mine, theirs, base_action)
        if chosen != base_action and gain <= self.margin:
            self.stats.fallback_margin += 1
            return base_action
        if chosen != base_action:
            self.stats.overrides += 1
        return chosen

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def diagnostics(self) -> dict[str, Any]:
        top = max(self.particles, key=lambda p: p.weight, default=None)
        if top is not None:
            self.stats.top_name = top.name
            self.stats.top_weight = top.weight
        return self.stats.to_dict()


__all__ = ["BeliefPlanner", "PlannerStats", "CURATED_NAMES"]
