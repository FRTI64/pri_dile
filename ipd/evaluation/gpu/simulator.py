"""Lock-step batched protocol simulator.

The simulator evaluates one candidate against a whole field (or several fields
concatenated) using a single tensor batch.  Per round it:

1. runs the blue policy on every row (batched GPU inference),
2. routes opponent rows to batched learned-policy groups or their original
   CPU objects,
3. records both actions and scores.

Field order, per-match scores and opponent names are preserved, so callers can
reuse the existing Protocol v5 aggregation schema unchanged.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .blue import GpuBlueField
from .policy_opponents import BatchOpponentField

C, D = 0, 1


@dataclass
class FieldSimulation:
    """Per-field result returned by :func:`evaluate_fields_gpu`."""

    scores: list[int] = field(default_factory=list)
    opponent_scores: list[int] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    by_opponent: dict[str, list[int]] = field(default_factory=dict)
    family_seconds: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    blue_seconds: float = 0.0
    opponent_seconds: float = 0.0
    mine: list[list[int]] = field(default_factory=list)
    theirs: list[list[int]] = field(default_factory=list)


def _payoff(mine: torch.Tensor, theirs: torch.Tensor) -> torch.Tensor:
    cc = (mine == C) & (theirs == C)
    dc = (mine == D) & (theirs == C)
    dd = (mine == D) & (theirs == D)
    return cc.long() * 6 + dc.long() * 10 + dd.long()


@torch.no_grad()
def _simulate_chunk(policy: Any, opponents: Sequence[Any],
                    rows: Sequence[int], *, device: torch.device,
                    dtype: torch.dtype, rounds: int,
                    profile: bool = False,
                    fast_rules: bool | str = "auto",
                    fast_rules_threshold: float = 0.90,
                    blue_mode: str = "argmax",
                    blue_uniforms: Any = None,
                    policy_opponent_mode: str = "argmax"
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                               torch.Tensor, float, float, dict[str, float]]:
    n = len(rows)
    blue = GpuBlueField(policy, list(range(n)), device, dtype, rounds,
                        mode=blue_mode, uniforms=blue_uniforms)
    opp = BatchOpponentField(opponents, list(range(n)), device=device,
                             dtype=dtype, rounds=rounds, profile=profile,
                             fast_rules=fast_rules,
                             fast_rules_threshold=fast_rules_threshold,
                             policy_mode=policy_opponent_mode)
    mine = torch.zeros(n, rounds, dtype=torch.int8, device=device)
    theirs = torch.zeros(n, rounds, dtype=torch.int8, device=device)
    score = torch.zeros(n, dtype=torch.long, device=device)
    opp_score = torch.zeros(n, dtype=torch.long, device=device)
    blue_seconds = opp_seconds = 0.0

    for t in range(rounds):
        started = time.perf_counter()
        x = blue.act(t)
        blue_seconds += time.perf_counter() - started
        started = time.perf_counter()
        y = opp.act(t, x)
        opp_seconds += time.perf_counter() - started
        mine[:, t] = x
        theirs[:, t] = y
        score += _payoff(x, y)
        opp_score += _payoff(y, x)
        blue.observe(x, y)
        opp.observe(x, y)

    family_seconds = opp.finish_profile()
    if device.type == "cuda" and profile:
        torch.cuda.synchronize(device)
    return (mine, theirs, score, opp_score, blue_seconds, opp_seconds,
            family_seconds)


def evaluate_fields_gpu(
    policy: Any,
    fields: Sequence[Sequence[Any]],
    *,
    rounds: int = 100,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    max_batch: int = 32768,
    profile: bool = False,
    keep_histories: bool = False,
    fast_rules: bool | str = "auto",
    fast_rules_threshold: float = 0.90,
    blue_mode: str = "argmax",
    blue_uniforms: Any = None,
    policy_opponent_mode: str = "argmax",
) -> list[FieldSimulation]:
    """Evaluate ``policy`` against every field with batched inference.

    ``fields`` is a sequence of opponent sequences.  They are concatenated in
    order and simulated in chunks of at most ``max_batch`` rows.  The returned
    list has one entry per input field, preserving match order.
    """
    if not fields:
        return []
    dev = torch.device(device)
    rounds = int(rounds)
    total = sum(len(field) for field in fields)
    if total <= 0:
        return [FieldSimulation() for _ in fields]
    if blue_mode == "sample":
        if blue_uniforms is None:
            raise ValueError("sample mode requires blue_uniforms")
        if int(blue_uniforms.shape[0]) != total:
            raise ValueError("blue_uniforms rows do not match field size")

    # Flatten metadata; row offsets are contiguous per field.
    all_opponents: list[Any] = []
    all_names: list[str] = []
    for index, field in enumerate(fields):
        for opponent in field:
            all_opponents.append(opponent)
            all_names.append(str(getattr(opponent, "name",
                                         type(opponent).__name__)))

    scores = torch.zeros(total, dtype=torch.long, device=dev)
    opp_scores = torch.zeros(total, dtype=torch.long, device=dev)
    started = time.perf_counter()
    blue_total = opp_total = 0.0
    family_total: dict[str, float] = defaultdict(float)
    chunk_histories: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for start in range(0, total, max(1, int(max_batch))):
        stop = min(total, start + max(1, int(max_batch)))
        rows = list(range(start, stop))
        chunk_uniforms = (blue_uniforms[start:stop]
                          if blue_mode == "sample" else None)
        mine_c, theirs_c, s, os_, blue_s, opp_s, family_s = _simulate_chunk(
            policy, all_opponents[start:stop], rows, device=dev,
            dtype=dtype, rounds=rounds, profile=profile,
            fast_rules=fast_rules,
            fast_rules_threshold=fast_rules_threshold,
            blue_mode=blue_mode,
            blue_uniforms=chunk_uniforms,
            policy_opponent_mode=policy_opponent_mode)
        scores[start:stop] = s
        opp_scores[start:stop] = os_
        blue_total += blue_s
        opp_total += opp_s
        for family, seconds in family_s.items():
            family_total[family] += float(seconds)
        chunk_histories.append(
            (mine_c.detach().cpu(), theirs_c.detach().cpu())
            if keep_histories else None)
    wall = time.perf_counter() - started

    score_values = scores.detach().cpu().tolist()
    opp_values = opp_scores.detach().cpu().tolist()
    mine_all = theirs_all = None
    if keep_histories:
        mine_all = torch.cat([h[0] for h in chunk_histories], dim=0).tolist()
        theirs_all = torch.cat([h[1] for h in chunk_histories],
                               dim=0).tolist()
    result: list[FieldSimulation] = []
    cursor = 0
    for index, field_rows in enumerate(fields):
        count = len(field_rows)
        names = all_names[cursor:cursor + count]
        values = score_values[cursor:cursor + count]
        opp_values_field = opp_values[cursor:cursor + count]
        by_opponent: dict[str, list[int]] = defaultdict(list)
        for name, value in zip(names, values):
            by_opponent[name].append(int(value))
        # Blue inference is shared by the whole chunk, so attribute it
        # proportionally to the field's share of chunk rows.  Opponent time is
        # also proportionally attributed; the exact per-family split is
        # collected separately by the CLI when profiling is requested.
        share = count / total if total else 0.0
        result.append(FieldSimulation(
            scores=[int(v) for v in values],
            opponent_scores=[int(v) for v in opp_values_field],
            names=names,
            by_opponent=dict(by_opponent),
            family_seconds={family: seconds * share
                            for family, seconds in family_total.items()},
            seconds=wall * share,
            blue_seconds=blue_total * share,
            opponent_seconds=opp_total * share,
            mine=(mine_all[cursor:cursor + count] if mine_all else []),
            theirs=(theirs_all[cursor:cursor + count] if theirs_all else []),
        ))
        cursor += count
    return result


def evaluate_field_gpu(
    policy: Any,
    opponents: Sequence[Any],
    *,
    rounds: int = 100,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    max_batch: int = 32768,
    profile: bool = False,
    keep_histories: bool = False,
    fast_rules: bool | str = "auto",
    fast_rules_threshold: float = 0.90,
    blue_mode: str = "argmax",
    blue_uniforms: Any = None,
    policy_opponent_mode: str = "argmax",
) -> FieldSimulation:
    """Convenience wrapper for a single field."""
    return evaluate_fields_gpu(
        policy, [opponents], rounds=rounds, device=device, dtype=dtype,
        max_batch=max_batch, profile=profile,
        keep_histories=keep_histories, fast_rules=fast_rules,
        fast_rules_threshold=fast_rules_threshold,
        blue_mode=blue_mode, blue_uniforms=blue_uniforms,
        policy_opponent_mode=policy_opponent_mode)[0]


__all__ = ["FieldSimulation", "evaluate_field_gpu", "evaluate_fields_gpu"]
