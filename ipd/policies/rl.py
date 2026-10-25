#!/usr/bin/env python3
"""Dependency-free inference for the reward-trained (PPO) IPD policy.

The policy uses the same 85 history features and MLP shape as
``neural_policy.py``, but it was trained with a policy-gradient objective
instead of behavioural cloning.  Training lives in ``train_rl_policy.py`` and
requires PyTorch; this module only uses the Python standard library so the
exported policy can run in a contest harness without a deep-learning stack.

Checkpoint format (``ipd-rl-policy-v1``) contains a shared tanh trunk plus two
linear heads (policy logit and value).  The deployable policy is exported in
the existing ``ipd-neural-policy-v1`` format, so ``NeuralPolicy`` and the C++
implementation can consume it unchanged.

Usage::

    from ipd.policies.rl import RLPolicy
    p = RLPolicy("weights/rl_policy_checkpoint.json")
    p.reset()
    action = p.act(t, mine, theirs)        # t is zero based
    action = p.move(day, mine, theirs)     # day is one based
"""
from __future__ import annotations

import json
import math
import random
from operator import mul
from typing import Sequence

from ipd.policies.neural import (
    C,
    D,
    FEATURE_VERSION,
    _payoff,
    extract_features,
)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(x, 40.0)))
    z = math.exp(max(x, -40.0))
    return z / (1.0 + z)


def _log_sigmoid(x: float) -> float:
    """Numerically stable log(sigmoid(x))."""
    if x >= 0.0:
        return -math.log1p(math.exp(-min(x, 40.0)))
    return x - math.log1p(math.exp(max(x, -40.0)))


class RLPolicy:
    """Actor-critic checkpoint with a standard-library-only forward pass."""

    name = "RL-PPO"

    def __init__(
        self,
        weights: str | dict,
        seed: int = 20261024,
        mode: str = "sample",
        safe: bool | None = None,
        name: str | None = None,
    ):
        if isinstance(weights, str):
            with open(weights, encoding="utf-8") as fh:
                weights = json.load(fh)
        if weights.get("format") != "ipd-rl-policy-v1":
            raise ValueError(f"unsupported checkpoint format: {weights.get('format')}")
        self.weights = weights
        self.mode = mode
        self.rng = random.Random(seed)
        if name:
            self.name = name

        self.n_features = int(weights["n_features"])
        self.temperature = float(weights.get("temperature", 1.0))
        self.activation = weights.get("activation", "tanh")
        if self.activation != "tanh":
            raise ValueError(f"unsupported activation: {self.activation}")
        self.safe = bool(weights.get("safe", True)) if safe is None else bool(safe)

        mean = [float(x) for x in weights["feature_mean"]]
        std = [float(x) if abs(float(x)) > 1e-8 else 1.0
               for x in weights["feature_std"]]
        if len(mean) != self.n_features or len(std) != self.n_features:
            raise ValueError("checkpoint normalization has the wrong size")

        def load_trunk(layers):
            out = []
            for i, layer in enumerate(layers):
                w = [[float(x) for x in row] for row in layer["w"]]
                b = [float(x) for x in layer["b"]]
                if i == 0:
                    # Fold feature standardization into the first affine layer.
                    w = [[v / std[j] for j, v in enumerate(row)] for row in w]
                    b = [v - sum(wj * m for wj, m in zip(row, mean))
                         for v, row in zip(b, w)]
                out.append((w, b))
            return out

        self.trunk = load_trunk(weights["trunk"])
        value_trunk = weights.get("value_trunk")
        self.value_trunk = load_trunk(value_trunk) if value_trunk else None

        self.policy_head = (
            [[float(x) for x in row] for row in weights["policy_head"]["w"]],
            [float(x) for x in weights["policy_head"]["b"]],
        )
        value = weights.get("value_head")
        self.value_head = None
        if value is not None:
            self.value_head = (
                [[float(x) for x in row] for row in value["w"]],
                [float(x) for x in value["b"]],
            )

        if len(self.policy_head[0][0]) != len(self.trunk[-1][1]):
            raise ValueError("policy head and trunk disagree")
        if self.trunk and len(self.trunk[0][0][0]) != self.n_features:
            raise ValueError("checkpoint and feature extractor disagree")
        if (self.value_trunk
                and len(self.value_trunk[0][0][0]) != self.n_features):
            raise ValueError("value trunk and feature extractor disagree")

    @staticmethod
    def _affine(x: Sequence[float], layer) -> list[float]:
        w, b = layer
        return [sum(map(mul, row, x)) + bias for row, bias in zip(w, b)]

    def _trunk_forward(self, trunk, features: Sequence[float]) -> list[float]:
        x = [float(v) for v in features]
        for layer in trunk:
            x = [math.tanh(v) for v in self._affine(x, layer)]
        return x

    def logit_value(self, features: Sequence[float]) -> tuple[float, float]:
        """Return ``(policy logit, value)`` for a precomputed feature vector."""
        if len(features) != self.n_features:
            raise ValueError("feature count mismatch")
        h = self._trunk_forward(self.trunk, features)
        logit = self._affine(h, self.policy_head)[0]
        if self.value_head is None:
            return logit, 0.0
        hv = (h if self.value_trunk is None
              else self._trunk_forward(self.value_trunk, features))
        value = self._affine(hv, self.value_head)[0]
        return logit, value

    def predict_proba(self, mine: Sequence[int], theirs: Sequence[int]) -> float:
        logit, _ = self.logit_value(extract_features(mine, theirs))
        if self.mode == "argmax":
            return 1.0 if logit >= 0.0 else 0.0
        return _sigmoid(logit / max(1e-6, self.temperature))

    def predict_value(self, mine: Sequence[int], theirs: Sequence[int]) -> float:
        _, value = self.logit_value(extract_features(mine, theirs))
        return value

    def select(self, t: int, theirs: Sequence[int], logit: float
               ) -> tuple[int, float, bool]:
        """Apply hard rules and sample/argmax the policy head.

        Returns ``(action, log_prob, forced)``.  ``forced`` marks actions chosen
        by a hard rule rather than the learned policy; those transitions are
        masked out of the PPO policy loss.
        """
        if t >= 99:
            return D, 0.0, True
        if self.safe and (t == 0 or D not in theirs):
            return C, 0.0, True

        z = logit / max(1e-6, self.temperature)
        if self.mode == "argmax":
            action = D if z >= 0.0 else C
            logprob = _log_sigmoid(z) if action == D else _log_sigmoid(-z)
            return action, logprob, False

        p = _sigmoid(z)
        if self.rng.random() < p:
            return D, _log_sigmoid(z), False
        return C, _log_sigmoid(-z), False

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        logit, _ = self.logit_value(extract_features(mine, theirs))
        action, _, _ = self.select(t, theirs, logit)
        return action

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def reset(self) -> None:
        # Stateless; the RNG stream intentionally persists across matches, like
        # the other stochastic policies in this repository.
        pass

    def observe(self, mine: list[int], theirs: list[int]) -> None:
        pass

    def export_policy_weights(self, name: str | None = None) -> dict:
        """Convert the actor-critic checkpoint to the deployable MLP format."""
        # Export the *raw* trunk from the checkpoint.  ``self.trunk`` has the
        # feature standardization folded in for fast inference; NeuralPolicy
        # applies the stored mean/std itself, so exporting the folded weights
        # would normalise twice.
        layers = [
            {"w": layer["w"], "b": layer["b"]}
            for layer in self.weights["trunk"]
        ]
        layers.append({
            "w": self.weights["policy_head"]["w"],
            "b": self.weights["policy_head"]["b"],
        })
        meta = dict(self.weights.get("meta", {}))
        if name:
            meta["name"] = name
        meta["trained_with"] = "ppo"
        return {
            "format": "ipd-neural-policy-v1",
            "feature_version": FEATURE_VERSION,
            "n_features": self.n_features,
            "activation": "tanh",
            "layers": layers,
            "feature_mean": list(self.weights["feature_mean"]),
            "feature_std": list(self.weights["feature_std"]),
            "temperature": self.temperature,
            "safe": self.safe,
            "meta": meta,
        }

    def __repr__(self) -> str:
        return self.name


def _self_test(path: str) -> None:
    p = RLPolicy(path)
    print(f"features={p.n_features} safe={p.safe} temperature={p.temperature}")
    for name, opponent in (
        ("AllC", lambda t, m, o: C),
        ("AllD", lambda t, m, o: D),
        ("TFT", lambda t, m, o: C if t == 0 else o[-1]),
        ("Grim", lambda t, m, o: D if D in o else C),
    ):
        p.reset()
        mine: list[int] = []
        theirs: list[int] = []
        value = 0.0
        score = 0
        for t in range(100):
            value += p.predict_value(mine, theirs)
            a = p.act(t, mine, theirs)
            b = opponent(t, mine, theirs)
            mine.append(a)
            theirs.append(b)
            score += _payoff(a, b)
        print(f"{name:8} C={mine.count(C):3d} D={mine.count(D):3d} "
              f"score={score:4d} mean_value={value / 100:.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?",
                    default="weights/rl_policy_checkpoint.json")
    args = ap.parse_args()
    _self_test(args.weights)
