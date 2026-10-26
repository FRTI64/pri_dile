"""Risk-sensitive PPO helpers.

``cvar_episode_weights`` is a practical tail-risk surrogate: it raises the
advantage weight of the worst ``quantile`` of episodes while keeping the mean
weight at one.  It is not an exact CVaR policy-gradient estimator, but it
directly targets the low-return tail that dominates E4-style exploitability.

``dirichlet_weights`` randomizes the training mixture around a base prior so
the policy does not overfit one fixed distribution composition.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Mapping, Sequence


def cvar_episode_weights(returns: Sequence[float], quantile: float = 0.2,
                         coef: float = 1.0) -> list[float]:
    """Return per-episode weights with the worst ``quantile`` boosted.

    The returned weights have mean 1.0, so the overall PPO loss scale is
    preserved.  If all returns are equal (or ``coef <= 0``), weights are all
    one.
    """
    n = len(returns)
    if n == 0:
        return []
    if coef <= 0.0 or n == 1:
        return [1.0] * n
    values = [float(x) for x in returns]
    if max(values) - min(values) <= 1e-12:
        return [1.0] * n
    q = min(1.0, max(0.0, float(quantile)))
    k = max(1, int(math.ceil(q * n)))
    order = sorted(range(n), key=lambda i: values[i])
    worst = set(order[:k])
    raw = [1.0 + float(coef) if i in worst else 1.0 for i in range(n)]
    mean = sum(raw) / n
    return [x / mean for x in raw]


def dirichlet_weights(base: Mapping[str, float], rng: random.Random,
                      concentration: float = 20.0) -> dict[str, float]:
    """Sample a normalized mixture around ``base``.

    ``concentration <= 0`` returns the normalized base weights exactly.
    Positive concentrations use a Dirichlet distribution with mean equal to
    the normalized base prior.
    """
    names = list(base)
    if not names:
        return {}
    total = sum(max(0.0, float(base[name])) for name in names)
    if total <= 0.0:
        prior = {name: 1.0 / len(names) for name in names}
    else:
        prior = {name: max(0.0, float(base[name])) / total for name in names}
    if concentration <= 0.0:
        return prior
    gammas = []
    for name in names:
        shape = max(1e-9, prior[name] * float(concentration))
        gammas.append(rng.gammavariate(shape, 1.0))
    total_gamma = sum(gammas)
    if total_gamma <= 0.0:
        return prior
    return {name: g / total_gamma for name, g in zip(names, gammas)}


def competitive_reward(own: float, opponent: float,
                       coef: float = 0.0) -> float:
    """Own payoff minus a penalty on the opponent's payoff.

    ``coef=0`` is the normal IPD return; ``coef=1`` optimises the score
    difference, which is useful for training an anti-exploitation blue
    response rather than a pure own-score best response.
    """
    return float(own) - float(coef) * float(opponent)


def dro_update_weights(weights: Mapping[str, float],
                       means: Mapping[str, float], *, eta: float = 0.5,
                       quantile: float = 0.5, blend: float = 0.5,
                       floor: float = 0.1) -> dict[str, float]:
    """Exponentiated-gradient / worst-family update of a mixture.

    Low-scoring distributions receive more weight.  ``blend`` mixes the EG
    update with a uniform distribution over the worst ``quantile`` fraction;
    ``floor`` prevents any distribution from disappearing.
    """
    names = list(weights)
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: 1.0}
    raw = [max(0.0, float(weights[name])) for name in names]
    total = sum(raw)
    if total <= 0.0:
        base = [1.0 / len(names)] * len(names)
    else:
        base = [x / total for x in raw]
    values = [float(means.get(name, 0.0)) for name in names]
    mu = statistics.fmean(values)
    sigma = statistics.pstdev(values) if len(values) > 1 else 0.0
    scale = max(1.0, sigma)
    eg = [base[i] * math.exp(-float(eta) * (values[i] - mu) / scale)
          for i in range(len(names))]
    eg_total = sum(eg)
    eg = ([1.0 / len(names)] * len(names) if eg_total <= 0.0
          else [x / eg_total for x in eg])
    q = min(1.0, max(0.0, float(quantile)))
    k = max(1, int(math.ceil(q * len(names))))
    worst = set(sorted(range(len(names)), key=lambda i: values[i])[:k])
    target = [1.0 / k if i in worst else 0.0 for i in range(len(names))]
    b = min(1.0, max(0.0, float(blend)))
    mixed = [(1.0 - b) * eg[i] + b * target[i] for i in range(len(names))]
    floored = [max(float(floor), x) for x in mixed]
    floored_total = sum(floored)
    return {name: floored[i] / floored_total
            for i, name in enumerate(names)}


__all__ = [
    "cvar_episode_weights", "dirichlet_weights", "dro_update_weights",
    "competitive_reward",
]
