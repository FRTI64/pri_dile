"""Meta-game utilities for the E0 league harness.

The IPD payoff is not zero-sum, so the meta-game is analysed through the
antisymmetric *score-difference* matrix::

    A[i][j] = payoff(i beats j) - payoff(j beats i)

``fictitious_play`` then approximates a Nash mixture of this symmetric
zero-sum game.  A mixture with low ``exploitability`` is hard for any pure
strategy in the pool to counter.  This is a pool-level approximation, not a
proof of game-theoretic exploitability against an arbitrary best response.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Sequence

Matrix = Sequence[Sequence[float]]


def _validate(matrix: Matrix) -> int:
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must not be empty")
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    return n


def _normalise(weights: Sequence[float]) -> list[float]:
    values = [max(0.0, float(x)) for x in weights]
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def antisymmetrize(matrix: Matrix) -> list[list[float]]:
    """Return ``(M - M.T) / 2``.

    The result is antisymmetric and its diagonal is zero.  Passing an already
    antisymmetric score-difference matrix is a no-op.
    """
    n = _validate(matrix)
    return [[0.5 * (float(matrix[i][j]) - float(matrix[j][i]))
             for j in range(n)] for i in range(n)]


def _row_payoffs(matrix: Matrix, weights: Sequence[float]) -> list[float]:
    n = len(matrix)
    return [sum(matrix[i][j] * weights[j] for j in range(n))
            for i in range(n)]


def exploitability(matrix: Matrix, weights: Sequence[float]) -> dict:
    """Best pure-strategy gain against a mixed strategy.

    ``matrix`` is a score-difference matrix (or any matrix; it is
    antisymmetrised first).  For an antisymmetric zero-sum game the game value
    of the symmetric mixture is zero, so the gap is
    ``max_i (A @ weights)_i``.
    """
    n = _validate(matrix)
    if len(weights) != n:
        raise ValueError("weights length does not match matrix")
    a = antisymmetrize(matrix)
    p = _normalise(weights)
    payoffs = _row_payoffs(a, p)
    best = max(range(n), key=lambda i: payoffs[i])
    value = sum(p[i] * payoffs[i] for i in range(n))
    return {
        "gap": max(0.0, payoffs[best]),
        "best_response": best,
        "value": value,
        "payoffs": payoffs,
        "weights": p,
    }


def _best_response(values: Sequence[float], rng: random.Random,
                   maximise: bool = True) -> int:
    target = max(values) if maximise else min(values)
    candidates = [i for i, value in enumerate(values)
                  if abs(value - target) <= 1e-12]
    return rng.choice(candidates)


def fictitious_play(matrix: Matrix, iterations: int = 20000,
                    seed: int = 0, tol: float = 0.0) -> dict:
    """Approximate a Nash mixture for the antisymmetric meta-game.

    Returns the row player's average strategy (used as the meta-strategy),
    its exploitability, and the empirical game value.  Both players' averages
    are maintained because the meta-game is not required to be symmetric in
    noisy cross-play data.
    """
    n = _validate(matrix)
    if iterations <= 0:
        weights = [1.0 / n] * n
        result = exploitability(matrix, weights)
        return {"weights": weights, "iterations": 0,
                "value": result["value"],
                "exploitability": result["gap"]}
    a = antisymmetrize(matrix)
    rng = random.Random(seed)
    p_avg = [1.0 / n] * n
    q_avg = [1.0 / n] * n
    p_sum = [0.0] * n
    q_sum = [0.0] * n
    used = 0
    for step in range(1, iterations + 1):
        row_br = _best_response(_row_payoffs(a, q_avg), rng, maximise=True)
        col_values = [sum(p_avg[i] * a[i][j] for i in range(n))
                      for j in range(n)]
        col_br = _best_response(col_values, rng, maximise=False)
        p_sum[row_br] += 1.0
        q_sum[col_br] += 1.0
        p_avg = [x / step for x in p_sum]
        q_avg = [x / step for x in q_sum]
        used = step
        if tol > 0.0 and step % 100 == 0:
            gap = exploitability(a, p_avg)["gap"]
            if gap <= tol:
                break
    gap = exploitability(a, p_avg)["gap"]
    value = sum(p_avg[i] * a[i][j] * q_avg[j]
                for i in range(n) for j in range(n))
    return {
        "weights": _normalise(p_avg),
        "iterations": used,
        "value": value,
        "exploitability": gap,
    }


def nash_average(matrix: Matrix, iterations: int = 20000,
                 seed: int = 0) -> dict:
    """Alias for :func:`fictitious_play` used by PSRO/Nash-averaging reports."""
    return fictitious_play(matrix, iterations=iterations, seed=seed)


def elo_ratings(expected_scores: Matrix, iterations: int = 500,
                base: float = 1000.0, scale: float = 400.0,
                eps: float = 0.01) -> list[float]:
    """Bradley-Terry / Elo ratings from a pairwise expected-score matrix.

    ``expected_scores[i][j]`` is the fraction of the score i obtains against j
    (win + 0.5 * draw).  Perfect records are clipped to ``eps`` so the ratings
    stay finite.  Ratings are anchored at ``base`` for the geometric mean.
    """
    n = _validate(expected_scores)
    if n == 1:
        return [float(base)]
    scores = []
    for i in range(n):
        row = []
        for j in range(n):
            value = float(expected_scores[i][j])
            # Symmetrise noisy observations: i's expected score plus j's
            # expected score should be one.
            value = 0.5 * (value + (1.0 - float(expected_scores[j][i])))
            row.append(min(1.0 - eps, max(eps, value)))
        scores.append(row)
    strength = [1.0] * n
    for _ in range(max(1, iterations)):
        updated = [0.0] * n
        for i in range(n):
            wins = sum(scores[i][j] for j in range(n) if j != i)
            denom = 0.0
            for j in range(n):
                if j != i:
                    denom += 1.0 / (strength[i] + strength[j])
            updated[i] = wins / denom if denom > 0 else strength[i]
        log_mean = statistics.fmean(math.log(max(1e-12, x)) for x in updated)
        updated = [x / math.exp(log_mean) for x in updated]
        change = max(abs(updated[i] - strength[i]) / max(1e-12, strength[i])
                     for i in range(n))
        strength = updated
        if change < 1e-10:
            break
    return [base + scale * math.log10(max(1e-12, value)) for value in strength]


__all__ = [
    "Matrix", "antisymmetrize", "exploitability", "fictitious_play",
    "nash_average", "elo_ratings",
]
