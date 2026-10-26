"""Recurrent actor-critic networks and small inference wrappers."""
from __future__ import annotations

import copy
import math
import random
from typing import Any, Sequence

import torch
import torch.nn as nn

from ipd.core.primitives import C, D

from . import N_FEATURES, OBSERVATION_VERSION
from .obs import features


class GRUActorCritic(nn.Module):
    """Two-layer GRU trunk with separate 32-unit actor and critic heads."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64,
                 layers: int = 2, actor_hidden: int = 32,
                 value_bias: float = 5.5):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.gru = nn.GRU(n_features, hidden, num_layers=layers,
                          batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden, actor_hidden), nn.Tanh(),
            nn.Linear(actor_hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, actor_hidden), nn.Tanh(),
            nn.Linear(actor_hidden, 1),
        )
        nn.init.constant_(self.critic[-1].bias, float(value_bias))

    def init_hidden(self, batch: int, device: torch.device | str | None = None):
        return torch.zeros(self.layers, batch, self.hidden,
                           device=device or next(self.parameters()).device)

    def forward_sequence(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """x: [B, T, F] -> logits [B, T], values [B, T], h_final."""
        if h0 is None:
            h0 = self.init_hidden(x.shape[0], x.device)
        out, hn = self.gru(x, h0)
        logits = self.actor(out).squeeze(-1)
        values = self.critic(out).squeeze(-1)
        return logits, values, hn

    def step(self, x: torch.Tensor, h: torch.Tensor):
        """One round; x: [B, F] -> logits, values, h_next."""
        out, hn = self.gru(x.unsqueeze(1), h)
        out = out.squeeze(1)
        return (self.actor(out).squeeze(-1),
                self.critic(out).squeeze(-1), hn)


def make_model(config=None, value_bias: float = 5.5,
               device: str | torch.device = "cpu") -> GRUActorCritic:
    ppo = getattr(config, "ppo", config)
    model = GRUActorCritic(
        N_FEATURES,
        hidden=int(getattr(ppo, "hidden", 64)),
        layers=int(getattr(ppo, "gru_layers", 2)),
        actor_hidden=int(getattr(ppo, "actor_hidden", 32)),
        value_bias=value_bias,
    )
    return model.to(device)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.to(next(model.parameters()).device)
                           for k, v in state.items()})


def _linear_json(module: nn.Linear) -> dict[str, list]:
    return {"w": module.weight.detach().cpu().tolist(),
            "b": module.bias.detach().cpu().tolist()}


def _linear_layers(module: nn.Sequential) -> list[dict[str, list]]:
    return [_linear_json(layer) for layer in module
            if isinstance(layer, nn.Linear)]


def export_weights(model: GRUActorCritic, meta: dict[str, Any] | None = None
                   ) -> dict[str, Any]:
    """Deployable JSON with the recurrent policy (no PyTorch required)."""
    gru = []
    for layer in range(model.layers):
        gru.append({
            "w_ih": getattr(model.gru, f"weight_ih_l{layer}").detach().cpu().tolist(),
            "w_hh": getattr(model.gru, f"weight_hh_l{layer}").detach().cpu().tolist(),
            "b_ih": getattr(model.gru, f"bias_ih_l{layer}").detach().cpu().tolist(),
            "b_hh": getattr(model.gru, f"bias_hh_l{layer}").detach().cpu().tolist(),
        })
    return {
        "format": "ipd-selfplay-gru-v1",
        "observation_version": OBSERVATION_VERSION,
        "n_features": N_FEATURES,
        "hidden": model.hidden,
        "layers": model.layers,
        "activation": "tanh",
        "gru": gru,
        "actor": _linear_layers(model.actor),
        "critic": _linear_layers(model.critic),
        "meta": dict(meta or {}),
    }


class TorchPolicy:
    """Stateful single-episode wrapper around a :class:`GRUActorCritic`."""

    def __init__(self, model: GRUActorCritic, device: str | torch.device = "cpu",
                 mode: str = "sample", seed: int = 20261026,
                 name: str = "SelfPlay-GRU"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.mode = mode
        self.rng = random.Random(seed)
        self.name = name
        self.h: torch.Tensor | None = None

    def reset(self) -> None:
        self.h = self.model.init_hidden(1, self.device)

    @torch.no_grad()
    def logit_value(self, t: int, mine: Sequence[int],
                    theirs: Sequence[int]) -> tuple[float, float]:
        if self.h is None:
            self.reset()
        x = torch.tensor(features(t, mine, theirs), dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        logit, value, self.h = self.model.step(x, self.h)
        return float(logit[0]), float(value[0])

    @torch.no_grad()
    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        logit, _ = self.logit_value(t, mine, theirs)
        if self.mode == "argmax":
            return D if logit >= 0.0 else C
        p = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
        return D if self.rng.random() < p else C

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def export_policy_weights(self, meta: dict[str, Any] | None = None):
        return export_weights(self.model, meta)


class TorchOpponent(TorchPolicy):
    """Frozen snapshot used as an opponent inside a rollout."""

    def __init__(self, state: dict[str, torch.Tensor], config=None,
                 device: str | torch.device = "cpu", mode: str = "sample",
                 seed: int = 20261026, name: str = "red-snapshot"):
        model = make_model(config, value_bias=0.0, device="cpu")
        load_state(model, state)
        super().__init__(model, device=device, mode=mode, seed=seed,
                         name=name)


def snapshot(model: GRUActorCritic) -> dict[str, torch.Tensor]:
    return copy.deepcopy(clone_state(model))


__all__ = [
    "GRUActorCritic", "make_model", "clone_state", "load_state",
    "export_weights", "TorchPolicy", "TorchOpponent", "snapshot",
]
