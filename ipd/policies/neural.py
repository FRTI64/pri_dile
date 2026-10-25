#!/usr/bin/env python3
"""Dependency-free inference for the GPU/NPU-trained neural IPD policy.

The policy is a small MLP over hand-built history features.  Training lives in
``ipd/training/neural.py`` and requires PyTorch; this file only uses the Python
standard library, so the exported strategy can be dropped into a contest
harness without installing a deep-learning stack.

Action convention is the same as the rest of the project:

    C = 0, D = 1
    day = 1..100
    mine/theirs contain the actions of all previous days

Usage::

    from ipd.policies.neural import NeuralPolicy
    p = NeuralPolicy("weights/neural_policy_weights.json")
    p.reset()
    action = p.act(t, mine, theirs)        # t is zero based
    action = p.move(day, mine, theirs)     # day is one based

The network outputs P(defect).  In ``sample`` mode (default) the action is
sampled from that probability, matching the stochastic experts used as
teachers.  ``argmax`` mode always defects when P(defect) >= 0.5.
"""
from __future__ import annotations

import json
import math
import random
from operator import mul
from typing import Sequence

C = 0
D = 1

FEATURE_VERSION = "ipd-neural-features-v1"


def _rate(seq: Sequence[int]) -> float:
    return sum(seq) / len(seq) if seq else 0.0


def _run_length(seq: Sequence[int]) -> int:
    if not seq:
        return 0
    last = seq[-1]
    n = 0
    for x in reversed(seq):
        if x != last:
            break
        n += 1
    return n


def _alternation(seq: Sequence[int], window: int) -> float:
    if len(seq) < 2:
        return 0.0
    tail = seq[-window:]
    if len(tail) < 2:
        return 0.0
    return sum(tail[i] != tail[i - 1] for i in range(1, len(tail))) / (len(tail) - 1)


def _exact_alternation(seq: Sequence[int], window: int) -> float:
    if len(seq) < window:
        return 0.0
    tail = seq[-window:]
    return float(all(tail[i] != tail[i - 1] for i in range(1, len(tail))))


def _payoff(me: int, them: int) -> int:
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1


def _last_moves(seq: Sequence[int], k: int) -> list[float]:
    """Most recent first, D=+1, C=-1, missing history=0."""
    out: list[float] = []
    n = len(seq)
    for i in range(k):
        j = n - 1 - i
        out.append(0.0 if j < 0 else (1.0 if seq[j] == D else -1.0))
    return out


def extract_features(mine: Sequence[int], theirs: Sequence[int]) -> list[float]:
    """Build the state vector consumed by the MLP.

    Everything is bounded roughly to [-1, 1], so the exported weights remain
    portable even if the feature scaler is ignored.
    """
    t = len(theirs)
    if len(mine) != t:
        raise ValueError("history lengths differ")

    f: list[float] = []

    # Time / endgame.
    f += [
        t / 99.0,
        (99 - t) / 99.0,
        1.0 if t == 0 else 0.0,
        1.0 if t >= 99 else 0.0,
        1.0 if t >= 95 else 0.0,
        1.0 if t >= 90 else 0.0,
    ]

    # Recent actions, newest first.
    f += _last_moves(mine, 8)
    f += _last_moves(theirs, 8)

    # Cooperation / defection rates.
    f += [
        _rate(mine),
        _rate(theirs),
        _rate(mine[-10:]),
        _rate(theirs[-3:]),
        _rate(theirs[-5:]),
        _rate(theirs[-10:]),
        _rate(theirs[-20:]),
    ]

    # Conditional response statistics: the core signal of the hand-written
    # Final/Responsive strategies.
    n_c = n_d = d_c = d_d = 0
    for i in range(1, t):
        if mine[i - 1] == C:
            n_c += 1
            d_c += theirs[i] == D
        else:
            n_d += 1
            d_d += theirs[i] == D
    p_c_raw = d_c / n_c if n_c else 0.0
    p_d_raw = d_d / n_d if n_d else 0.0
    p_c = (d_c + 1.0) / (n_c + 2.0)
    p_d = (d_d + 1.0) / (n_d + 2.0)
    f += [
        p_c_raw,
        p_d_raw,
        p_c,
        p_d,
        p_d - p_c,
        p_d_raw - p_c_raw,
        1.0 if (n_c >= 5 and n_d >= 3) else 0.0,
        min(n_c, 20) / 20.0,
        min(n_d, 20) / 20.0,
        math.log1p(n_c) / math.log(101.0),
        math.log1p(n_d) / math.log(101.0),
    ]

    # Global evidence.
    f += [
        1.0 if D in theirs else 0.0,
        1.0 if D in mine else 0.0,
        1.0 if C in theirs else 0.0,
        sum(theirs) / 100.0,
        sum(mine) / 100.0,
        t / 99.0,
    ]

    # Runs and switching.
    f += [
        min(_run_length(theirs), 10) / 10.0,
        min(_run_length(mine), 10) / 10.0,
        _alternation(theirs, 10),
        _alternation(theirs, 20),
        _alternation(mine, 10),
        _exact_alternation(theirs, 10),
        sum(theirs[i] != theirs[i - 1] for i in range(1, t)) / max(1, t - 1),
        sum(mine[i] != mine[i - 1] for i in range(1, t)) / max(1, t - 1),
        1.0 if len(theirs) >= 5 and all(x == C for x in theirs[-5:]) else 0.0,
        1.0 if len(theirs) >= 5 and all(x == D for x in theirs[-5:]) else 0.0,
    ]

    # Last joint outcome and joint-action frequencies.
    if t:
        a, b = mine[-1], theirs[-1]
        f += [
            float(a == C and b == C),
            float(a == C and b == D),
            float(a == D and b == C),
            float(a == D and b == D),
        ]
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    cc = cd = dc = dd = 0
    for a, b in zip(mine, theirs):
        if a == C:
            if b == C:
                cc += 1
            else:
                cd += 1
        elif b == C:
            dc += 1
        else:
            dd += 1
    denom = max(1, t)
    f += [cc / denom, cd / denom, dc / denom, dd / denom]

    cc10 = cd10 = dc10 = dd10 = 0
    for a, b in zip(mine[-10:], theirs[-10:]):
        if a == C:
            if b == C:
                cc10 += 1
            else:
                cd10 += 1
        elif b == C:
            dc10 += 1
        else:
            dd10 += 1
    d10 = max(1, min(10, t))
    f += [cc10 / d10, cd10 / d10, dc10 / d10, dd10 / d10]

    # Recency / first-defection timing (late harvesters are a key opponent).
    try:
        last_opp_d = t - 1 - theirs[::-1].index(D)
    except ValueError:
        last_opp_d = -1
    try:
        last_my_d = t - 1 - mine[::-1].index(D)
    except ValueError:
        last_my_d = -1
    try:
        first_opp_d = theirs.index(D)
    except ValueError:
        first_opp_d = -1
    try:
        first_my_d = mine.index(D)
    except ValueError:
        first_my_d = -1
    f += [
        (t - last_opp_d) / 100.0 if last_opp_d >= 0 else 1.0,
        (t - last_my_d) / 100.0 if last_my_d >= 0 else 1.0,
        (first_opp_d + 1) / 100.0 if first_opp_d >= 0 else 1.0,
        (first_my_d + 1) / 100.0 if first_my_d >= 0 else 1.0,
        (t - first_opp_d) / 100.0 if first_opp_d >= 0 else 0.0,
        ((first_my_d - first_opp_d) / 100.0
         if first_my_d >= 0 and first_opp_d >= 0 and first_my_d >= first_opp_d
         else 0.0),
    ]

    # Score context.
    score = sum(_payoff(a, b) for a, b in zip(mine, theirs))
    f += [
        score / 1000.0,
        (_payoff(mine[-1], theirs[-1]) / 10.0) if t else 0.0,
        (1.0 if mine[-1] == C else -1.0) if t else 0.0,
        (1.0 if theirs[-1] == D else -1.0) if t else 0.0,
        1.0 if (t >= 3 and all((mine[i] == D and theirs[i] == D) for i in range(t - 3, t))) else 0.0,
    ]

    # Response in the most recent round(s): useful for distinguishing a
    # retaliator from a scheduled defector immediately after a probe.
    if t:
        f += [
            float(mine[-1] == D and theirs[-1] == C),
            float(mine[-1] == C and theirs[-1] == D),
            float(mine[-1] == D and theirs[-1] == D),
        ]
    else:
        f += [0.0, 0.0, 0.0]

    # Recent conditional response (last 10 opportunities), less diluted by
    # early noise than the lifetime estimate.
    nc10 = nd10 = dc10r = dd10r = 0
    for i in range(max(1, t - 10), t):
        if mine[i - 1] == C:
            nc10 += 1
            dc10r += theirs[i] == D
        else:
            nd10 += 1
            dd10r += theirs[i] == D
    f += [
        (dc10r + 1.0) / (nc10 + 2.0),
        (dd10r + 1.0) / (nd10 + 2.0),
        (dd10r + 1.0) / (nd10 + 2.0) - (dc10r + 1.0) / (nc10 + 2.0),
    ]

    return f


FEATURE_NAMES: tuple[str, ...] = tuple(f"f{i}" for i in range(len(extract_features([], []))))


class NeuralPolicy:
    """Small MLP policy with a standard-library-only forward pass."""

    name = "Neural"

    def __init__(
        self,
        weights: str | dict,
        seed: int = 20261024,
        mode: str = "sample",
        name: str | None = None,
        safe: bool | None = None,
    ):
        if isinstance(weights, str):
            with open(weights, encoding="utf-8") as fh:
                weights = json.load(fh)
        self.weights = weights
        self.mode = mode
        self.rng = random.Random(seed)
        if name:
            self.name = name
        # Behaviour cloning checkpoints and all older policies use the no-probe
        # safety rules.  Reward-trained checkpoints may explicitly disable them.
        self.safe = bool(weights.get("safe", True)) if safe is None else bool(safe)

        self.n_features = int(weights["n_features"])
        self.temperature = float(weights.get("temperature", 1.0))
        self.activation = weights.get("activation", "tanh")
        if self.activation != "tanh":
            raise ValueError(f"unsupported activation: {self.activation}")

        mean = [float(x) for x in weights["feature_mean"]]
        std = [float(x) if abs(float(x)) > 1e-8 else 1.0 for x in weights["feature_std"]]
        layers = []
        for i, layer in enumerate(weights["layers"]):
            w = [[float(x) for x in row] for row in layer["w"]]
            b = [float(x) for x in layer["b"]]
            if i == 0:
                # Fold feature standardization into the first affine layer.
                w = [[v / std[j] for j, v in enumerate(row)] for row in w]
                b = [v - sum(wj * m for wj, m in zip(row, mean)) for v, row in zip(b, w)]
            layers.append((w, b))
        self.layers = layers

        expected = self.n_features
        if len(self.layers[0][0][0]) != expected:
            raise ValueError("weight file and feature extractor disagree")

    @staticmethod
    def _affine(x: list[float], layer: tuple[list[list[float]], list[float]]) -> list[float]:
        w, b = layer
        return [sum(map(mul, row, x)) + bias for row, bias in zip(w, b)]

    def predict_proba(self, mine: Sequence[int], theirs: Sequence[int]) -> float:
        x = extract_features(mine, theirs)
        for i, layer in enumerate(self.layers[:-1]):
            x = self._affine(x, layer)
            x = [math.tanh(v) for v in x]
        logit = self._affine(x, self.layers[-1])[0]
        if self.mode == "argmax":
            return 1.0 if logit >= 0 else 0.0
        z = max(-40.0, min(40.0, logit / max(1e-6, self.temperature)))
        return 1.0 / (1.0 + math.exp(-z))

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        # Boundary and safety rules.  The teachers in this project never make
        # a proactive probe, so an opponent that has never defected cannot yet
        # be classified as exploitable.  Cooperating until its first D avoids
        # catastrophic false positives against Grim-style opponents.
        if t >= 99:
            return D
        if self.safe and (t == 0 or D not in theirs):
            return C
        p_defect = self.predict_proba(mine, theirs)
        if self.mode == "argmax":
            return D if p_defect >= 0.5 else C
        return D if self.rng.random() < p_defect else C

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def reset(self) -> None:
        # The policy is stateless; history is supplied by the harness.  Keep
        # the RNG stream across matches, like the stochastic hand-written
        # strategies in this repository.
        pass

    def observe(self, mine: list[int], theirs: list[int]) -> None:
        """Advance any internal state without choosing an action.

        The MLP has no state; this hook exists so wrapper policies can keep a
        recurrent base policy in sync when they override an action.
        """
        pass

    def __repr__(self) -> str:
        return self.name


# Alias matching the ``Machine`` interface used by several experiment scripts.
NeuralMachine = NeuralPolicy


def _self_test(path: str | None = None) -> None:
    if path is None:
        print("feature count:", len(extract_features([], [])))
        print("start:", extract_features([], [])[:10])
        print("after 3 rounds:", extract_features([C, D, C], [C, D, D])[:10])
        return
    p = NeuralPolicy(path)
    mine: list[int] = []
    theirs: list[int] = []
    for t in range(100):
        x = p.act(t, mine, theirs)
        y = C if t < 90 else D
        mine.append(x)
        theirs.append(y)
    print("actions C/D:", mine.count(C), mine.count(D))
    print("score:", sum(_payoff(a, b) for a, b in zip(mine, theirs)))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default=None)
    args = ap.parse_args()
    _self_test(args.weights)
