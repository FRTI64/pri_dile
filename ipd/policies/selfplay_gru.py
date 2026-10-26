#!/usr/bin/env python3
"""Dependency-free inference for the self-play GRU policy.

The deployable format (``ipd-selfplay-gru-v1``) stores the two-layer GRU and
the actor/critic heads as plain lists.  This module only uses the standard
library, so a contest harness can run the policy without PyTorch.

Example::

    from ipd.policies.selfplay_gru import SelfPlayGRUPolicy
    p = SelfPlayGRUPolicy("weights/selfplay_gru_policy.json")
    p.reset()
    action = p.act(t, mine, theirs)
"""
from __future__ import annotations

import json
import math
import random
from typing import Sequence

from ipd.core.primitives import C, D

from . import selfplay_gru_obs as obs


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(x, 40.0)))
    z = math.exp(max(x, -40.0))
    return z / (1.0 + z)


def _matvec(w: list[list[float]], x: Sequence[float],
            b: Sequence[float] | None = None) -> list[float]:
    out = [sum(a * c for a, c in zip(row, x)) for row in w]
    if b is not None:
        out = [v + bias for v, bias in zip(out, b)]
    return out


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x * y for x, y in zip(a, b)]


def _linear(layer, x: Sequence[float]) -> list[float]:
    if isinstance(layer, dict):
        return _matvec(layer["w"], x, layer["b"])
    w, b = layer
    return _matvec(w, x, b)


class SelfPlayGRUPolicy:
    name = "SelfPlay-GRU"

    def __init__(self, weights: str | dict, seed: int = 20261026,
                 mode: str = "sample", name: str | None = None):
        if isinstance(weights, str):
            with open(weights, encoding="utf-8") as fh:
                weights = json.load(fh)
        if weights.get("format") != "ipd-selfplay-gru-v1":
            raise ValueError(
                f"unsupported self-play checkpoint format: {weights.get('format')}")
        self.observation_version = str(
            weights.get("observation_version", obs.OBSERVATION_VERSION))
        if self.observation_version not in (obs.OBSERVATION_VERSION,
                                            "selfplay-21-v2"):
            raise ValueError("unsupported observation version")
        self.weights = weights
        self.n_features = int(weights["n_features"])
        self.hidden = int(weights["hidden"])
        self.layers = int(weights["layers"])
        self.mode = mode
        self.rng = random.Random(seed)
        self.name = name or self.name
        self.gru = []
        for layer in weights["gru"]:
            self.gru.append((
                [[float(x) for x in row] for row in layer["w_ih"]],
                [float(x) for x in layer["b_ih"]],
                [[float(x) for x in row] for row in layer["w_hh"]],
                [float(x) for x in layer["b_hh"]],
            ))
        self.actor = [
            ([[float(x) for x in row] for row in layer["w"]],
             [float(x) for x in layer["b"]])
            for layer in weights["actor"]
        ]
        self.critic = [
            ([[float(x) for x in row] for row in layer["w"]],
             [float(x) for x in layer["b"]])
            for layer in weights["critic"]
        ]
        self.h = [[0.0] * self.hidden for _ in range(self.layers)]

    def reset(self) -> None:
        self.h = [[0.0] * self.hidden for _ in range(self.layers)]

    def _gru_layer(self, x: Sequence[float], h: Sequence[float],
                   layer) -> list[float]:
        w_ih, b_ih, w_hh, b_hh = layer
        gi = _matvec(w_ih, x, b_ih)
        gh = _matvec(w_hh, h, b_hh)
        n = self.hidden
        r = [_sigmoid(gi[i] + gh[i]) for i in range(n)]
        z = [_sigmoid(gi[n + i] + gh[n + i]) for i in range(n)]
        # n gate uses r * (W_hn h + b_hn)
        nn = [math.tanh(gi[2 * n + i] + r[i] * gh[2 * n + i])
              for i in range(n)]
        return [(1.0 - z[i]) * nn[i] + z[i] * h[i] for i in range(n)]

    def _forward(self, features: Sequence[float]) -> tuple[float, float]:
        if len(features) != self.n_features:
            raise ValueError("feature count mismatch")
        x: list[float] = [float(v) for v in features]
        new_h = []
        for i, layer in enumerate(self.gru):
            x = self._gru_layer(x, self.h[i], layer)
            new_h.append(x)
        self.h = new_h
        h = x
        for layer in self.actor[:-1]:
            h = [math.tanh(v) for v in _linear(layer, h)]
        logit = _linear(self.actor[-1], h)[0]

        c = x
        for layer in self.critic[:-1]:
            c = [math.tanh(v) for v in _linear(layer, c)]
        value = _linear(self.critic[-1], c)[0]
        return logit, value

    def logit_value(self, t: int, mine: Sequence[int],
                    theirs: Sequence[int]) -> tuple[float, float]:
        fn = (obs.features_v2 if self.observation_version == "selfplay-21-v2"
              else obs.features)
        return self._forward(fn(t, mine, theirs))

    def predict_proba(self, t: int, mine: Sequence[int],
                      theirs: Sequence[int]) -> float:
        logit, _ = self.logit_value(t, mine, theirs)
        return _sigmoid(logit)

    def predict_value(self, t: int, mine: Sequence[int],
                      theirs: Sequence[int]) -> float:
        _, value = self.logit_value(t, mine, theirs)
        return value

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        logit, _ = self.logit_value(t, mine, theirs)
        if self.mode == "argmax":
            return D if logit >= 0.0 else C
        return D if self.rng.random() < _sigmoid(logit) else C

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)


class SelfPlayGRUAdapter:
    """Compatibility adapter for the public policy interface."""

    def __init__(self, policy: SelfPlayGRUPolicy):
        self.policy = policy
        self.name = policy.name

    def reset(self) -> None:
        self.policy.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        return self.policy.act(t, mine, theirs)

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.policy.move(day, mine, theirs)


__all__ = ["SelfPlayGRUPolicy", "SelfPlayGRUAdapter"]
