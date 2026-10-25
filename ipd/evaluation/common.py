"""Shared evaluation primitives: one match, one field, aggregation.

All harnesses (benchmark, blind, suite, matchup) call these helpers.  Candidate
policies are always wrapped through :func:`ipd.policies.base.ensure_policy`, so
both 0-based ``act`` and legacy 1-based ``move`` implementations work.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..core.simulator import C, D, payoff
from ..policies.base import PolicyAdapter, ensure_policy
from .distributions import (
    DISTS, build_dist_field, build_train_dist_field, sample_blind_field,
    sample_suite_field,
)


@dataclass
class MatchStats:
    score: int
    opponent_score: int
    mine: list[int] = field(default_factory=list)
    theirs: list[int] = field(default_factory=list)
    opponent_name: str = ""


@dataclass
class FieldStats:
    total: int
    scores: list[int]
    names: list[str]

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


def run_match(policy: Any, opponent: Any, rounds: int = 100,
              reset: bool = True) -> MatchStats:
    """Play one match with the canonical policy interface."""
    p: PolicyAdapter = ensure_policy(policy)
    if reset:
        p.reset()
        reset_fn = getattr(opponent, "reset", None)
        if reset_fn is not None:
            reset_fn()

    mine: list[int] = []
    theirs: list[int] = []
    score = opponent_score = 0
    for t in range(rounds):
        x = p.act(t, mine, theirs)
        if hasattr(opponent, "act"):
            y = opponent.act(t, theirs, mine)
        else:
            y = opponent.move(t + 1, theirs, mine)
        mine.append(x)
        theirs.append(y)
        score += payoff(x, y)
        opponent_score += payoff(y, x)
    return MatchStats(score, opponent_score, mine, theirs,
                      getattr(opponent, "name", type(opponent).__name__))


def evaluate_field(policy: Any, opponents: Sequence[Any],
                   rounds: int = 100) -> FieldStats:
    """Play one match against every opponent in a field."""
    p = ensure_policy(policy)
    scores: list[int] = []
    names: list[str] = []
    for opponent in opponents:
        result = run_match(p, opponent, rounds=rounds)
        scores.append(result.score)
        names.append(result.opponent_name)
    return FieldStats(sum(scores), scores, names)


def evaluate_spec_on_dist(spec: Any, dist: str, seeds: Sequence[int],
                          n: int, rounds: int = 100,
                          training: bool = False) -> list[float]:
    """Worker helper: mean score per opponent for one candidate/distribution."""
    builder = build_train_dist_field if training else build_dist_field
    values: list[float] = []
    for seed in seeds:
        field = builder(dist, seed, n)
        policy = spec.build() if hasattr(spec, "build") else ensure_policy(spec)
        values.append(evaluate_field(policy, field, rounds).mean)
    return values


def eval_task(task: dict[str, Any]) -> tuple[str, str, list[float]]:
    """Serialisable worker task used by benchmark/suite harnesses."""
    spec = task["spec"]
    if not hasattr(spec, "build"):
        from ..policies.registry import CandidateSpec
        spec = CandidateSpec.from_dict(spec)
    values = evaluate_spec_on_dist(
        spec, task["dist"], task["seeds"], int(task["n"]),
        int(task.get("rounds", 100)), bool(task.get("training", False)),
    )
    return task["name"], task["dist"], values


def quantile(values: Iterable[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def aggregate(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "p05": quantile(values, 0.05),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def multi_objective_fitness(by_dist: dict[str, Sequence[float]]) -> float:
    """Historical selection metric used by the rule-search trainings."""
    all_values = [x for values in by_dist.values() for x in values]
    mean = statistics.fmean(all_values)
    q05 = quantile(all_values, 0.05)
    worst_dist = min(statistics.fmean(values) for values in by_dist.values())
    return 0.45 * mean + 0.25 * worst_dist + 0.20 * q05 + 0.10 * min(all_values)


def seed_map(seed: int, dists: Sequence[str], fields: int) -> dict[str, list[int]]:
    """Deterministic per-distribution seeds used by every harness."""
    import random
    rng = random.Random(seed)
    return {d: [rng.randrange(1 << 30) for _ in range(fields)] for d in dists}


__all__ = [
    "C", "D", "DISTS", "MatchStats", "FieldStats", "run_match",
    "evaluate_field", "evaluate_spec_on_dist", "eval_task", "quantile",
    "aggregate", "multi_objective_fitness", "seed_map", "sample_blind_field",
    "sample_suite_field",
]
