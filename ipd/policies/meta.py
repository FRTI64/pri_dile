"""Small meta-policy that switches from RL-Wide160 to RL-Repair on an
early mutual-defection signal.

The switch is intentionally narrow: it only fires after the first six rounds
when both sides have already defected at least twice and the opponent's last
move is cooperation.  In the full opponent scan this pattern is common for
zero-determinant extortion and noisy/Joss opponents, where RL-Repair is more
robust, while TFT/Grim/AllC remain on the safe RL-Wide160 path.
"""
from __future__ import annotations

import json
import math
from typing import Sequence

from ..core.primitives import C, D
from ..paths import weights_path
from .neural import NeuralPolicy
from .rl import RLPolicy


def _load(path: str, seed: int, mode: str):
    with open(path, encoding="utf-8") as fh:
        fmt = json.load(fh).get("format")
    cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
    return cls(path, seed=seed, mode=mode)


class RLSwitchPolicy:
    """Base RL-Wide160 with a narrow early switch to RL-Repair."""

    def __init__(self, base_weights: str, alt_weights: str,
                 seed: int = 20261024, mode: str = "sample",
                 name: str = "RL-Switch"):
        self.base = _load(base_weights, seed, mode)
        self.alt = _load(alt_weights, seed + 1, mode)
        self.mode = mode
        self.name = name
        self.switched = False

    def reset(self) -> None:
        self.base.reset()
        self.alt.reset()
        self.switched = False

    def set_rounds(self, rounds: int) -> None:
        for policy in (self.base, self.alt):
            fn = getattr(policy, "set_rounds", None)
            if callable(fn):
                fn(rounds)

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if not self.switched and t >= 6:
            first_opp = theirs[:6]
            first_mine = mine[:6]
            if (first_opp.count(1) >= 2 and first_mine.count(1) >= 2
                    and theirs[-1] == 0):
                self.switched = True
        policy = self.alt if self.switched else self.base
        return policy.act(t, mine, theirs)

    def observe(self, mine: list[int], theirs: list[int]) -> None:
        """No internal observation state; histories are passed per action."""

    def __repr__(self) -> str:
        return self.name


GATE_FEATURES = 25


def _period_rate(seq: Sequence[int], period: int, window: int = 8) -> float:
    recent = list(seq[-window:])
    n = len(recent)
    if n <= period:
        return 0.0
    return sum(recent[i] == recent[i - period]
               for i in range(period, n)) / (n - period)


def _run_length(seq: Sequence[int]) -> int:
    if not seq:
        return 0
    n = 1
    for value in reversed(seq[:-1]):
        if value != seq[-1]:
            break
        n += 1
    return n


def _payoff(me: int, them: int) -> int:
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1


def gate_features(mine: Sequence[int], theirs: Sequence[int]) -> list[float]:
    """Compact opponent-state vector for the learned match-level gate."""
    t = len(theirs)
    if t != len(mine):
        raise ValueError("history lengths differ")
    f = [t / 100.0,
         (sum(mine) / t) if t else 0.0,
         (sum(theirs) / t) if t else 0.0]
    for seq in (mine, theirs):
        for lag in range(1, 5):
            j = t - lag
            f.append(0.0 if j < 0 else (1.0 if seq[j] == D else -1.0))
    nc = nd = dc = dd = 0
    for i in range(1, t):
        if mine[i - 1] == C:
            nc += 1
            dc += theirs[i] == D
        else:
            nd += 1
            dd += theirs[i] == D
    pc = (dc + 1.0) / (nc + 2.0)
    pd = (dd + 1.0) / (nd + 2.0)
    f += [pc, pd, pd - pc]
    for seq in (mine, theirs):
        f.append((sum(seq[i] != seq[i - 1] for i in range(1, t))
                  / max(1, t - 1)) if t > 1 else 0.0)
    f += [_period_rate(theirs, p) for p in (2, 3, 4)]
    f += [_period_rate(mine, p) for p in (2, 3, 4)]
    f.append(min(_run_length(theirs), 10) / 10.0)
    f.append(min(_run_length(mine), 10) / 10.0)
    score = sum(_payoff(a, b) for a, b in zip(mine, theirs))
    opp_score = sum(_payoff(b, a) for a, b in zip(mine, theirs))
    f.append((score - opp_score) / 1000.0)
    if len(f) != GATE_FEATURES:  # pragma: no cover
        raise AssertionError(f"gate_features produced {len(f)} features")
    return f


class LearnedSwitchPolicy:
    """Match-level learned gate: RL-Wide160 prefix, then A/B for the rest."""

    def __init__(self, base_weights: str, alt_weights: str, gate: str,
                 seed: int = 20261024, mode: str = "sample",
                 name: str = "RL-Gate"):
        self.base = _load(base_weights, seed, mode)
        self.alt = _load(alt_weights, seed + 1, mode)
        with open(gate, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("format") != "ipd-meta-gate-v1":
            raise ValueError(f"unsupported gate format: {data.get('format')}")
        if int(data["n_features"]) != GATE_FEATURES:
            raise ValueError("gate feature count mismatch")
        self.w = [float(x) for x in data["w"]]
        self.b = float(data["b"])
        self.mean = [float(x) for x in data["mean"]]
        self.std = [float(x) if abs(float(x)) > 1e-8 else 1.0
                    for x in data["std"]]
        self.k = int(data.get("k", 10))
        self.threshold = float(data.get("threshold", 0.5))
        self.mode = mode
        self.name = name
        self.decided = False
        self.use_alt = False

    def reset(self) -> None:
        self.base.reset()
        self.alt.reset()
        self.decided = False
        self.use_alt = False

    def set_rounds(self, rounds: int) -> None:
        for policy in (self.base, self.alt):
            fn = getattr(policy, "set_rounds", None)
            if callable(fn):
                fn(rounds)

    def _decide(self, mine: list[int], theirs: list[int]) -> None:
        k = min(self.k, len(theirs))
        x = gate_features(mine[:k], theirs[:k])
        z = self.b + sum(w * (v - m) / s
                         for w, v, m, s in zip(self.w, x, self.mean, self.std))
        self.use_alt = z >= math.log(self.threshold / (1.0 - self.threshold))
        self.decided = True

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if not self.decided and t >= self.k:
            self._decide(mine, theirs)
        return (self.alt if self.use_alt else self.base).act(t, mine, theirs)

    def observe(self, mine: list[int], theirs: list[int]) -> None:
        """No internal observation state; histories are passed per action."""

    def __repr__(self) -> str:
        return self.name


__all__ = ["RLSwitchPolicy", "LearnedSwitchPolicy", "gate_features",
           "GATE_FEATURES"]
