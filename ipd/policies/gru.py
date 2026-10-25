#!/usr/bin/env python3
"""Dependency-free inference for the GPU/NPU-trained GRU-Net IPD policy.

The GRU-Net consumes the same 85 history features as ``ipd/policies/neural.py`` but
maintains a recurrent hidden state instead of being stateless.  Training lives
in ``ipd/training/gru.py`` and requires PyTorch; inference only uses the Python
standard library.

Usage::

    from ipd.policies.gru import GRUPolicy
    p = GRUPolicy("weights/gru_policy_weights.json")
    p.reset()                               # call before every match
    action = p.act(t, mine, theirs)         # t is zero based
    action = p.move(day, mine, theirs)      # day is one based

The network outputs P(defect).  ``sample`` mode samples from that probability;
``argmax`` mode defects when P(defect) >= 0.5.
"""
from __future__ import annotations

import json
import math
import random
from operator import mul
from typing import Sequence

from ipd.policies.neural import C, D, extract_features


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 40.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -40.0))
    return z / (1.0 + z)


class GRUPolicy:
    """One- or multi-layer GRU followed by a small MLP head."""

    name = "GRU-Net"

    def __init__(
        self,
        weights: str | dict,
        seed: int = 20261024,
        mode: str = "sample",
        name: str | None = None,
    ):
        if isinstance(weights, str):
            with open(weights, encoding="utf-8") as fh:
                weights = json.load(fh)
        if weights.get("format") != "ipd-gru-policy-v1":
            raise ValueError(f"unsupported weights format: {weights.get('format')}")
        self.weights = weights
        self.mode = mode
        self.rng = random.Random(seed)
        if name:
            self.name = name

        self.n_features = int(weights["n_features"])
        self.hidden = int(weights["hidden"])
        self.num_layers = int(weights.get("num_layers", len(weights["gru"])))
        self.temperature = float(weights.get("temperature", 1.0))
        self.mean = [float(x) for x in weights["feature_mean"]]
        self.std = [float(x) if abs(float(x)) > 1e-8 else 1.0
                    for x in weights["feature_std"]]

        self.gru = []
        for layer in weights["gru"]:
            self.gru.append({
                "weight_ih": [[float(v) for v in row] for row in layer["weight_ih"]],
                "weight_hh": [[float(v) for v in row] for row in layer["weight_hh"]],
                "bias_ih": [float(v) for v in layer["bias_ih"]],
                "bias_hh": [float(v) for v in layer["bias_hh"]],
            })

        self.head = []
        for layer in weights["head"]:
            self.head.append((
                [[float(v) for v in row] for row in layer["w"]],
                [float(v) for v in layer["b"]],
            ))

        self.hidden_state: list[list[float]] = []
        self.reset()

    @staticmethod
    def _affine(x: Sequence[float], w: Sequence[Sequence[float]],
                b: Sequence[float]) -> list[float]:
        return [sum(map(mul, row, x)) + bias for row, bias in zip(w, b)]

    def reset(self) -> None:
        self.hidden_state = [[0.0] * self.hidden for _ in range(self.num_layers)]

    def observe(self, mine: Sequence[int], theirs: Sequence[int]) -> None:
        """Advance the GRU hidden state without choosing an action.

        Used by wrapper policies that override an action but still need the
        recurrent state to stay consistent with the full history.
        """
        self._step(mine, theirs)

    def _step(self, mine: Sequence[int], theirs: Sequence[int]) -> float:
        """Advance the recurrent state and return the raw output logit."""
        f = extract_features(mine, theirs)
        if len(f) != self.n_features:
            raise ValueError("feature count mismatch")
        x = [(v - m) / s for v, m, s in zip(f, self.mean, self.std)]

        for layer_idx, layer in enumerate(self.gru):
            gi = self._affine(x, layer["weight_ih"], layer["bias_ih"])
            gh = self._affine(self.hidden_state[layer_idx],
                              layer["weight_hh"], layer["bias_hh"])
            h = self.hidden_state[layer_idx]
            new_h = [0.0] * self.hidden
            for j in range(self.hidden):
                r = _sigmoid(gi[j] + gh[j])
                z = _sigmoid(gi[self.hidden + j] + gh[self.hidden + j])
                n = math.tanh(gi[2 * self.hidden + j] +
                              r * gh[2 * self.hidden + j])
                new_h[j] = (1.0 - z) * n + z * h[j]
            self.hidden_state[layer_idx] = new_h
            x = new_h

        for i, (w, b) in enumerate(self.head):
            x = self._affine(x, w, b)
            if i + 1 < len(self.head):
                x = [math.tanh(v) for v in x]
        return x[0]

    def predict_proba(self, mine: Sequence[int],
                      theirs: Sequence[int]) -> float:
        """Advance the state and return P(defect) for the current step."""
        logit = self._step(mine, theirs)
        if self.mode == "argmax":
            return 1.0 if logit >= 0.0 else 0.0
        z = max(-40.0, min(40.0, logit / max(1e-6, self.temperature)))
        return 1.0 / (1.0 + math.exp(-z))

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        # Always advance the GRU on the current observation, even when a hard
        # rule decides the action; otherwise the hidden state would diverge
        # from the training-time recurrence.
        p_defect = self.predict_proba(mine, theirs)

        if t >= 99:
            return D
        if t == 0 or D not in theirs:
            return C
        if self.mode == "argmax":
            return D if p_defect >= 0.5 else C
        return D if self.rng.random() < p_defect else C

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def __repr__(self) -> str:
        return self.name


GRUNetPolicy = GRUPolicy


def _self_test(path: str | None = None) -> None:
    if path is None:
        print("GRU-Net inference module; pass a weights file for a self-test.")
        return
    p = GRUPolicy(path)
    mine: list[int] = []
    theirs: list[int] = []
    for t in range(100):
        x = p.act(t, mine, theirs)
        y = C if t < 90 else D
        mine.append(x)
        theirs.append(y)
    print("actions C/D:", mine.count(C), mine.count(D))
    print("score:", sum(6 if (a, b) == (C, C) else
                        (0 if a == C else (10 if b == C else 1))
                        for a, b in zip(mine, theirs)))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default=None)
    args = ap.parse_args()
    _self_test(args.weights)
