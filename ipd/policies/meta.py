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


class ProbeGuardPolicy:
    """Base policy with a reactive response to early unprovoked defection.

    The guard only fires after the opponent has a confirmed defection streak
    and the first defection was unprovoked (we had not defected before it):

    * the opponent's first ``D`` happens in an enabled trigger window;
    * the next ``confirm`` / ``late_confirm`` moves are also ``D``;
    * our history before that first ``D`` contains only ``C``.

    It then plays a short response sequence: ``response`` for the probe /
    generic window and ``late_response`` for the late window.  If the opponent
    plays ``C`` at any point, the guard releases and the base policy resumes.
    If the opponent keeps defecting, the guard plays ``D`` until the opponent
    cooperates again.  This separates a one-off probe from a permanently
    defecting opponent without sacrificing more than a few points against the
    latter.  The trigger is purely reactive: it never defects before the
    opponent does and it does not change the base safety rules.
    """

    def __init__(self, base_weights: str,
                 seed: int = 20261024, mode: str = "sample",
                 min_first: int = 1, max_first: int = 7,
                 exact_first: tuple[int, ...] | None = None,
                 late_min_first: int | None = None,
                 observe_rounds: int = 3,
                 confirm: int = 2,
                 late_confirm: int = 2,
                 response: tuple[int, ...] | None = None,
                 late_response: tuple[int, ...] | None = None,
                 name: str = "RL-ProbeGuard"):
        self.base = _load(base_weights, seed, mode)
        self.mode = mode
        self.min_first = int(min_first)
        self.max_first = int(max_first)
        self.exact_first = (tuple(int(x) for x in exact_first)
                            if exact_first is not None else None)
        self.late_min_first = (None if late_min_first is None
                               else int(late_min_first))
        self.observe_rounds = max(1, int(observe_rounds))
        self.confirm = max(1, int(confirm))
        self.late_confirm = max(1, int(late_confirm))
        self.response = (tuple(int(x) for x in response)
                         if response is not None
                         else (C,) * self.observe_rounds)
        self.late_response = (tuple(int(x) for x in late_response)
                              if late_response is not None
                              else self.response)
        self.name = name
        self.state = "base"
        self.idx = 0
        self.current: tuple[int, ...] = self.response

    def reset(self) -> None:
        self.base.reset()
        self.state = "base"
        self.idx = 0
        self.current = self.response

    def set_rounds(self, rounds: int) -> None:
        fn = getattr(self.base, "set_rounds", None)
        if callable(fn):
            fn(rounds)

    def _streak_trigger(self, mine: Sequence[int], theirs: Sequence[int],
                        n: int, first_ok) -> bool:
        if len(theirs) < n or len(mine) < n:
            return False
        if list(theirs[-n:]) != [D] * n:
            return False
        first = len(theirs) - n
        if D in theirs[:first]:
            return False
        if not first_ok(first):
            return False
        # No proactive defection: before the opponent's first D we only
        # cooperated.  The safety rule guarantees this for the base policy.
        return D not in mine[:first]

    def _triggered(self, mine: Sequence[int], theirs: Sequence[int]) -> bool:
        if self.exact_first is not None:
            if self._streak_trigger(
                    mine, theirs, self.confirm,
                    lambda first: first in self.exact_first):
                self.current = self.response
                return True
        if self.late_min_first is not None:
            if self._streak_trigger(
                    mine, theirs, self.late_confirm,
                    lambda first: first >= self.late_min_first):
                self.current = self.late_response
                return True
        if self.exact_first is None and self.late_min_first is None:
            if self._streak_trigger(
                    mine, theirs, self.confirm,
                    lambda first: self.min_first <= first <= self.max_first):
                self.current = self.response
                return True
        return False

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if self.state == "base":
            if self._triggered(mine, theirs):
                self.state = "observe"
                self.idx = 0
                return self.current[0]
            return self.base.act(t, mine, theirs)
        if self.state == "observe":
            if theirs[-1] == C:
                self.state = "base"
                return self.base.act(t, mine, theirs)
            self.idx += 1
            if self.idx < len(self.current):
                return self.current[self.idx]
            self.state = "punish"
            return D
        # Punish until the opponent cooperates again.
        if theirs[-1] == C:
            self.state = "base"
            return self.base.act(t, mine, theirs)
        return D

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


__all__ = ["RLSwitchPolicy", "LearnedSwitchPolicy", "ProbeGuardPolicy",
           "gate_features", "GATE_FEATURES"]
