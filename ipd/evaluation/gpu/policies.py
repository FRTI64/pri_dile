"""Batched torch inference for the deployable learned policies.

The loaders accept the same JSON formats as the standard-library policy
classes, so no parameters are duplicated:

* ``ipd-neural-policy-v1`` -- MLP used by :class:`~ipd.policies.neural.NeuralPolicy`
* ``ipd-rl-policy-v1`` -- actor-critic checkpoint used by
  :class:`~ipd.policies.rl.RLPolicy`
* ``ipd-gru-policy-v1`` -- 85-feature GRU used by
  :class:`~ipd.policies.gru.GRUPolicy`
* ``ipd-selfplay-gru-v1`` -- 21-feature two-layer GRU used by
  :class:`~ipd.policies.selfplay_gru.SelfPlayGRUPolicy`

Only the policy/actor heads are loaded; training-only critic/value heads are
ignored.  ``argmax`` is the only supported action mode.  Sample-mode evaluation
deliberately stays on the CPU path (Protocol v4 RNG contract).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .features import C, D

MODE_ARGMAX = "argmax"
MODE_SAMPLE = "sample"


def _sample_from_logits(logits: torch.Tensor, temperature: float,
                        uniforms: torch.Tensor,
                        draw_idx: torch.Tensor,
                        need_draw: torch.Tensor,
                        hard_action: torch.Tensor) -> tuple[
                            torch.Tensor, torch.Tensor]:
    """Bernoulli-sample one action per row and advance the draw pointer."""
    b = logits.shape[0]
    rows = torch.arange(b, device=logits.device)
    index = draw_idx.clamp(min=0, max=uniforms.shape[1] - 1)
    p_defect = torch.sigmoid(logits / max(1e-6, float(temperature)))
    draw = uniforms[rows, index] < p_defect
    action = torch.where(need_draw, draw.to(torch.int8), hard_action)
    draw_idx = draw_idx + need_draw.to(torch.long)
    return action, draw_idx


def _load_json(weights: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(weights, Mapping):
        return dict(weights)
    with open(weights, encoding="utf-8") as fh:
        return json.load(fh)


def _tensor(data: Any, device: torch.device, dtype: torch.dtype
            ) -> torch.Tensor:
    return torch.tensor(data, device=device, dtype=dtype)


def _fold_mlp_layers(data: Mapping[str, Any], device: torch.device,
                     dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
    mean = [float(x) for x in data["feature_mean"]]
    std = [float(x) if abs(float(x)) > 1e-8 else 1.0
           for x in data["feature_std"]]
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, layer in enumerate(data["layers"]):
        w = _tensor(layer["w"], device, dtype)
        b = _tensor(layer["b"], device, dtype)
        if index == 0:
            w = w / _tensor(std, device, dtype).unsqueeze(0)
            b = b - (w * _tensor(mean, device, dtype).unsqueeze(0)).sum(1)
        layers.append((w, b))
    return layers


class BatchMLP:
    """Batched MLP for the deployable neural and RL-checkpoint formats."""

    n_features = 85

    def __init__(self, weights: str | Path | Mapping[str, Any],
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 mode: str = MODE_ARGMAX,
                 safe: bool | None = None):
        data = _load_json(weights)
        fmt = data.get("format")
        self.device = torch.device(device)
        self.dtype = dtype
        self.mode = mode
        if mode not in (MODE_ARGMAX, MODE_SAMPLE):
            raise ValueError(f"unsupported batch policy mode: {mode!r}")
        if fmt == "ipd-neural-policy-v1":
            self.layers = _fold_mlp_layers(data, self.device, dtype)
            self.temperature = float(data.get("temperature", 1.0))
            self.safe = bool(data.get("safe", True)) if safe is None else safe
        elif fmt == "ipd-rl-policy-v1":
            mean = [float(x) for x in data["feature_mean"]]
            std = [float(x) if abs(float(x)) > 1e-8 else 1.0
                   for x in data["feature_std"]]
            trunk = data["trunk"]
            self.layers = []
            for index, layer in enumerate(trunk):
                w = _tensor(layer["w"], self.device, dtype)
                b = _tensor(layer["b"], self.device, dtype)
                if index == 0:
                    w = w / _tensor(std, self.device, dtype).unsqueeze(0)
                    b = b - (w * _tensor(mean, self.device,
                                        dtype).unsqueeze(0)).sum(1)
                self.layers.append((w, b))
            head = data["policy_head"]
            self.layers.append((_tensor(head["w"], self.device, dtype),
                                _tensor(head["b"], self.device, dtype)))
            self.temperature = float(data.get("temperature", 1.0))
            self.safe = bool(data.get("safe", True)) if safe is None else safe
        else:
            raise ValueError(f"unsupported MLP policy format: {fmt!r}")
        self.n_features = int(data.get("n_features", self.n_features))
        self.num_layers = len(self.layers)

    def reset(self, batch: int | None = None) -> None:
        """Stateless; kept for the common batched-policy interface."""

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        h = features.to(dtype=self.dtype)
        for index, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if index + 1 < len(self.layers):
                h = torch.tanh(h)
        return h[:, 0]

    def act_from_logits(self, logits: torch.Tensor, t: int,
                        no_defect: torch.Tensor) -> torch.Tensor:
        """Apply the same hard rules as NeuralPolicy/RLPolicy.select."""
        b = logits.shape[0]
        action = (logits >= 0).to(torch.int8)
        if self.safe:
            action = torch.where(no_defect,
                                 torch.zeros(b, dtype=torch.int8,
                                             device=logits.device),
                                 action)
        if t >= 99:
            action = torch.ones(b, dtype=torch.int8, device=logits.device)
        return action

    def act_from_logits_sample(
            self, logits: torch.Tensor, t: int,
            no_defect: torch.Tensor, uniforms: torch.Tensor,
            draw_idx: torch.Tensor,
            need_mask: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the MLP head with the v5 per-match uniform stream."""
        b = logits.shape[0]
        final = torch.full((b,), t >= 99, dtype=torch.bool,
                           device=logits.device)
        safe_c = (self.safe & no_defect & (~final))
        hard_action = final.to(torch.int8)
        need = ~(safe_c | final)
        if need_mask is not None:
            need = need & need_mask
        action, draw_idx = _sample_from_logits(
            logits, self.temperature, uniforms, draw_idx, need, hard_action)
        action = torch.where(safe_c,
                             torch.zeros(b, dtype=torch.int8,
                                         device=logits.device),
                             action)
        return action, draw_idx


class BatchGRU:
    """Batched one-or-more-layer GRU over the 85-feature input."""

    n_features = 85

    def __init__(self, weights: str | Path | Mapping[str, Any],
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 mode: str = MODE_ARGMAX):
        data = _load_json(weights)
        if data.get("format") != "ipd-gru-policy-v1":
            raise ValueError(f"unsupported GRU policy format: {data.get('format')!r}")
        if mode not in (MODE_ARGMAX, MODE_SAMPLE):
            raise ValueError(f"unsupported batch policy mode: {mode!r}")
        self.device = torch.device(device)
        self.dtype = dtype
        self.mode = mode
        self.n_features = int(data["n_features"])
        self.hidden = int(data["hidden"])
        self.num_layers = int(data.get("num_layers", len(data["gru"])))
        self.temperature = float(data.get("temperature", 1.0))
        mean = _tensor(data["feature_mean"], self.device, dtype)
        std = _tensor([float(x) if abs(float(x)) > 1e-8 else 1.0
                       for x in data["feature_std"]], self.device, dtype)
        self.gru: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for index, layer in enumerate(data["gru"]):
            w_ih = _tensor(layer["weight_ih"], self.device, dtype)
            b_ih = _tensor(layer["bias_ih"], self.device, dtype)
            w_hh = _tensor(layer["weight_hh"], self.device, dtype)
            b_hh = _tensor(layer["bias_hh"], self.device, dtype)
            if index == 0:
                # Fold feature standardisation into the input transform.
                w_ih = w_ih / std.unsqueeze(0)
                b_ih = b_ih - (w_ih * mean.unsqueeze(0)).sum(1)
            self.gru.append((w_ih, b_ih, w_hh, b_hh))
        self.head: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in data["head"]:
            self.head.append((_tensor(layer["w"], self.device, dtype),
                              _tensor(layer["b"], self.device, dtype)))
        self.h = [torch.zeros(0, self.hidden, device=self.device,
                              dtype=dtype) for _ in range(self.num_layers)]

    def reset(self, batch: int) -> None:
        self.h = [torch.zeros(batch, self.hidden, device=self.device,
                              dtype=self.dtype) for _ in range(self.num_layers)]

    def step(self, features: torch.Tensor) -> torch.Tensor:
        x = features.to(dtype=self.dtype)
        n = self.hidden
        new_h: list[torch.Tensor] = []
        for index, (w_ih, b_ih, w_hh, b_hh) in enumerate(self.gru):
            gi = x @ w_ih.T + b_ih
            gh = self.h[index] @ w_hh.T + b_hh
            r = torch.sigmoid(gi[:, :n] + gh[:, :n])
            z = torch.sigmoid(gi[:, n:2 * n] + gh[:, n:2 * n])
            candidate = torch.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:])
            x = (1.0 - z) * candidate + z * self.h[index]
            new_h.append(x)
        self.h = new_h
        for index, (w, b) in enumerate(self.head):
            x = x @ w.T + b
            if index + 1 < len(self.head):
                x = torch.tanh(x)
        return x[:, 0]

    def act_from_logits(self, logits: torch.Tensor, t: int,
                        no_defect: torch.Tensor) -> torch.Tensor:
        """Apply the same hard rules as GRUPolicy.act."""
        b = logits.shape[0]
        action = (logits >= 0).to(torch.int8)
        action = torch.where(no_defect,
                             torch.zeros(b, dtype=torch.int8,
                                         device=logits.device),
                             action)
        if t >= 99:
            action = torch.ones(b, dtype=torch.int8, device=logits.device)
        return action

    def act_from_logits_sample(
            self, logits: torch.Tensor, t: int,
            no_defect: torch.Tensor, uniforms: torch.Tensor,
            draw_idx: torch.Tensor,
            need_mask: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the GRU head with the v5 per-match uniform stream."""
        b = logits.shape[0]
        final = torch.full((b,), t >= 99, dtype=torch.bool,
                           device=logits.device)
        safe_c = no_defect & (~final)
        hard_action = final.to(torch.int8)
        need = ~(safe_c | final)
        if need_mask is not None:
            need = need & need_mask
        action, draw_idx = _sample_from_logits(
            logits, self.temperature, uniforms, draw_idx, need, hard_action)
        action = torch.where(safe_c,
                             torch.zeros(b, dtype=torch.int8,
                                         device=logits.device),
                             action)
        return action, draw_idx


class BatchSelfPlayGRU:
    """Batched two-layer SelfPlay-GRU with the 21-feature observation."""

    n_features = 21

    def __init__(self, weights: str | Path | Mapping[str, Any],
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 mode: str = MODE_ARGMAX):
        data = _load_json(weights)
        if data.get("format") != "ipd-selfplay-gru-v1":
            raise ValueError(
                f"unsupported self-play format: {data.get('format')!r}")
        if mode not in (MODE_ARGMAX, MODE_SAMPLE):
            raise ValueError(f"unsupported batch policy mode: {mode!r}")
        self.device = torch.device(device)
        self.dtype = dtype
        self.mode = mode
        self.observation_version = str(
            data.get("observation_version", "selfplay-21-v1"))
        self.n_features = int(data["n_features"])
        self.hidden = int(data["hidden"])
        self.num_layers = int(data.get("layers", len(data["gru"])))
        self.gru: list[tuple[torch.Tensor, torch.Tensor,
                             torch.Tensor, torch.Tensor]] = []
        for layer in data["gru"]:
            self.gru.append((
                _tensor(layer["w_ih"], self.device, dtype),
                _tensor(layer["b_ih"], self.device, dtype),
                _tensor(layer["w_hh"], self.device, dtype),
                _tensor(layer["b_hh"], self.device, dtype),
            ))
        self.actor: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in data["actor"]:
            self.actor.append((_tensor(layer["w"], self.device, dtype),
                               _tensor(layer["b"], self.device, dtype)))
        self.h = [torch.zeros(0, self.hidden, device=self.device,
                              dtype=dtype) for _ in range(self.num_layers)]

    def reset(self, batch: int) -> None:
        self.h = [torch.zeros(batch, self.hidden, device=self.device,
                              dtype=self.dtype) for _ in range(self.num_layers)]

    def step(self, features: torch.Tensor) -> torch.Tensor:
        x = features.to(dtype=self.dtype)
        n = self.hidden
        new_h: list[torch.Tensor] = []
        for index, (w_ih, b_ih, w_hh, b_hh) in enumerate(self.gru):
            gi = x @ w_ih.T + b_ih
            gh = self.h[index] @ w_hh.T + b_hh
            r = torch.sigmoid(gi[:, :n] + gh[:, :n])
            z = torch.sigmoid(gi[:, n:2 * n] + gh[:, n:2 * n])
            candidate = torch.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:])
            x = (1.0 - z) * candidate + z * self.h[index]
            new_h.append(x)
        self.h = new_h
        for index, (w, b) in enumerate(self.actor):
            x = x @ w.T + b
            if index + 1 < len(self.actor):
                x = torch.tanh(x)
        return x[:, 0]

    def act_from_logits(self, logits: torch.Tensor, t: int,
                        no_defect: torch.Tensor) -> torch.Tensor:
        # SelfPlayGRUPolicy has no hard safety/endgame rule.
        return (logits >= 0).to(torch.int8)

    def act_from_logits_sample(
            self, logits: torch.Tensor, t: int,
            no_defect: torch.Tensor, uniforms: torch.Tensor,
            draw_idx: torch.Tensor,
            need_mask: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the self-play head; it has no hard action rules."""
        b = logits.shape[0]
        need = torch.ones(b, dtype=torch.bool, device=logits.device)
        if need_mask is not None:
            need = need & need_mask
        hard_action = torch.zeros(b, dtype=torch.int8, device=logits.device)
        return _sample_from_logits(
            logits, 1.0, uniforms, draw_idx, need, hard_action)


__all__ = ["BatchGRU", "BatchMLP", "BatchSelfPlayGRU", "MODE_ARGMAX",
           "MODE_SAMPLE"]
