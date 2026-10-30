"""GPU-resident lock-step IPD environment for self-play rollouts.

The CPU :class:`~ipd.training.selfplay.env.VectorEnv` is kept as the reference
implementation and as the fallback when CUDA is unavailable.  This module adds
the Plan 4 path:

* the blue observation history lives in one ``[B, T]`` tensor and is updated
  incrementally with :class:`~ipd.evaluation.gpu.features.FeatureState21`;
* native rule opponents use the Plan 3 C++ bank (one call per round);
* learned policy opponents are batched with the Plan 2 kernels;
* frozen self-play snapshots (:class:`~ipd.training.selfplay.models.TorchOpponent`)
  are grouped by source state and stepped as one recurrent batch;
* every other opponent (Axelrod, Final/Triple, RandomFSM, ...) keeps its
  original Python object and is stepped row by row.

The public surface mirrors ``VectorEnv``: ``n``, ``observations()``,
``step(actions)``, ``scores``, ``done``.  ``observations_tensor`` and
``step_tensor`` are the zero-copy variants used by the trainer.

RNG note
--------
Rule opponents inherit the Plan 3 contract: the native bank captures each
CPython MT19937 state at construction, so deterministic and random rules are
bit-identical to the CPU path.  Learned opponents and frozen snapshots sample
from a torch generator, and the Python fallbacks keep their own RNG streams.
This is a training-only change; Protocol v5 evaluation remains on the Plan 2
argmax contract.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from ipd.core.primitives import C, D
from ipd.evaluation.fast.bank import RuleBank
from ipd.evaluation.fast.spec import get_fast_spec, infer_fast_spec
from ipd.evaluation.gpu.features import FeatureState21, FeatureState85
from ipd.evaluation.gpu.policy_opponents import (
    build_batch_policy, policy_kind, policy_weights_key,
)

from . import OBSERVATION_VERSION
from .env import ROUNDS, OpponentFactory


def _sigmoid_sample(logits: torch.Tensor, temperature: float,
                    generator: torch.Generator | None) -> torch.Tensor:
    """Sample Bernoulli actions from policy logits without a Distribution."""
    temperature = max(1e-6, float(temperature))
    p = torch.sigmoid(logits / temperature)
    if generator is None:
        u = torch.rand_like(p)
    else:
        u = torch.rand(p.shape, dtype=p.dtype, device=p.device,
                       generator=generator)
    return (u < p).to(torch.int8)


def _apply_hard_rules(actions: torch.Tensor, t: int, safe: bool,
                      no_defect: torch.Tensor,
                      final_round: int) -> torch.Tensor:
    """Mirror NeuralPolicy/RLPolicy/GRUPolicy hard rules."""
    if safe:
        actions = torch.where(no_defect,
                              torch.zeros_like(actions), actions)
    if t >= final_round:
        actions = torch.ones_like(actions)
    return actions


class _TorchBatchPolicy:
    """Batched recurrent policy backed by one frozen ``GRUActorCritic``."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.h: torch.Tensor | None = None

    def reset(self, batch: int) -> None:
        self.h = self.model.init_hidden(batch, self.device)

    def step(self, features: torch.Tensor) -> torch.Tensor:
        logits, _values, self.h = self.model.step(features, self.h)
        return logits


@dataclass
class _PolicyGroup:
    rows: list[int]
    kind: str
    policy: Any
    state: Any
    mode: str
    temperature: float = 1.0
    safe: bool = False
    rounds: int = ROUNDS
    generator: torch.Generator | None = None
    generator_seed: int = 0
    rows_np: np.ndarray = field(init=False)
    rows_t: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.rows_np = np.asarray(self.rows, dtype=np.int64)
        self.rows_t = torch.as_tensor(
            self.rows_np, dtype=torch.long, device=self.state.device)
        self.policy.reset(len(self.rows))
        if self.generator is not None:
            self.generator.manual_seed(self.generator_seed)

    @property
    def device(self) -> torch.device:
        return self.state.device

    def reset(self) -> None:
        self.state.reset()
        self.policy.reset(len(self.rows))
        if self.generator is not None:
            self.generator.manual_seed(self.generator_seed)

    @torch.no_grad()
    def act(self, t: int) -> torch.Tensor:
        features = self.state.features()
        logits = (self.policy.logits(features)
                  if self.kind == "mlp"
                  else self.policy.step(features))
        if self.mode == "sample":
            action = _sigmoid_sample(logits, self.temperature, self.generator)
        else:
            action = (logits >= 0.0).to(torch.int8)
        if self.kind in ("mlp", "gru"):
            no_defect = ~self.state.any_defect("theirs")
            action = _apply_hard_rules(action, t, self.safe, no_defect,
                                       self.rounds - 1)
        return action

    def append(self, opponent: torch.Tensor, blue: torch.Tensor) -> None:
        self.state.append(opponent, blue)


def _torch_policy_kind(obj: Any) -> str | None:
    """Return ``"torch"`` for a frozen ``TorchOpponent``/``TorchPolicy``."""
    if type(obj).__name__ not in ("TorchOpponent", "TorchPolicy"):
        return None
    model = getattr(obj, "model", None)
    if model is not None and hasattr(model, "forward_sequence"):
        return "torch"
    return None


def _torch_group_key(obj: Any) -> Any:
    state = getattr(obj, "state_ref", None)
    if state is not None:
        return ("torch", id(state))
    return ("torch", id(getattr(obj, "model", obj)))


def _identity_key(obj: Any, kind: str) -> Any:
    if kind == "torch":
        return _torch_group_key(obj)
    return tuple(policy_weights_key(obj, kind))


def _ensure_fast_spec(obj: Any) -> Any:
    spec = get_fast_spec(obj)
    if spec is None:
        spec = infer_fast_spec(obj)
        if spec is not None:
            try:
                obj._fast_spec = spec
            except Exception:  # pragma: no cover - slotted policy wrapper
                pass
    return spec


def _payoff_tensors(me: torch.Tensor, them: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    cc = (me == C) & (them == C)
    cd = (me == C) & (them == D)
    dc = (me == D) & (them == C)
    dd = (me == D) & (them == D)
    own = (cc.to(torch.float32) * 6.0 + cd.to(torch.float32) * 0.0
           + dc.to(torch.float32) * 10.0 + dd.to(torch.float32) * 1.0)
    other = (cc.to(torch.float32) * 6.0 + cd.to(torch.float32) * 10.0
             + dc.to(torch.float32) * 0.0 + dd.to(torch.float32) * 1.0)
    return own, other


class GPUEnv:
    """Lock-step environment whose histories and learner features live on GPU."""

    def __init__(self, factories: Sequence[OpponentFactory],
                 families: Sequence[str] | None = None, *,
                 device: str | torch.device = "cuda",
                 rounds: int = ROUNDS,
                 fast_rules: bool = True,
                 fast_rules_min_envs: int = 1,
                 mode: str = "sample",
                 seed: int = 20261026,
                 dtype: torch.dtype = torch.float32,
                 policy_gpu_min: int = 256,
                 auto_reset: bool = False):
        if mode not in ("sample", "argmax"):
            raise ValueError("mode must be 'sample' or 'argmax'")
        self.device = torch.device(device)
        self.dtype = dtype
        self.rounds = int(rounds)
        self.mode = mode
        self.seed = int(seed)
        self.fast_rules = bool(fast_rules)
        self.fast_rules_min_envs = max(1, int(fast_rules_min_envs))
        self.policy_gpu_min = max(1, int(policy_gpu_min))
        self.auto_reset = bool(auto_reset)
        self._pending_reset = False
        self._build(list(factories), families)

    # ------------------------------------------------------------------
    # Construction and reset
    # ------------------------------------------------------------------

    def _build(self, factories: list[OpponentFactory],
               families: Sequence[str] | None) -> None:
        self.factories = list(factories)
        self.families = list(families or ["unknown"] * len(factories))
        if len(self.families) != len(factories):
            raise ValueError("family count does not match factory count")
        self.n = len(factories)
        self.opponents = [factory() for factory in factories]
        self.opponent_names = [
            str(getattr(obj, "name", type(obj).__name__))
            for obj in self.opponents
        ]

        grouped: dict[Any, list[tuple[int, Any]]] = {}
        self._groups: list[_PolicyGroup] = []
        self._rule_rows: list[int] = []
        self._rule_objects: list[Any] = []
        self._cpu_rows: list[Any] = []
        for index, obj in enumerate(self.opponents):
            spec = _ensure_fast_spec(obj)
            kind = _torch_policy_kind(obj) or policy_kind(obj)
            if kind is not None:
                grouped.setdefault(_identity_key(obj, kind), []).append(
                    (index, obj))
                continue
            if spec is not None:
                self._rule_rows.append(index)
                self._rule_objects.append(obj)
            else:
                self._cpu_rows.append([index, obj, [], [], None])

        self._build_groups(grouped)
        enable = (self.fast_rules
                  and len(self._rule_rows) >= self.fast_rules_min_envs)
        self.rules: RuleBank | None = None
        if self._rule_objects:
            bank = RuleBank(self._rule_objects, rounds=self.rounds,
                            enable=enable)
            if bank.enabled:
                self.rules = bank
            else:
                # The native bank declined (unavailable build or too few
                # rows).  Keep the exact Python objects instead of losing
                # them behind a disabled bank.
                for index, obj in zip(self._rule_rows, self._rule_objects):
                    self._cpu_rows.append([index, obj, [], [], None])
                self._rule_rows = []
                self._rule_objects = []
            if self.rules is None:
                bank.close()

        for row in self._cpu_rows:
            obj = row[1]
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()
            setter = getattr(obj, "set_rounds", None)
            if callable(setter):
                setter(self.rounds)
            rng = getattr(obj, "rng", None)
            if isinstance(rng, random.Random):
                row[4] = rng.getstate()

        self.state = FeatureState21(
            self.n, device=self.device, dtype=self.dtype,
            rounds=self.rounds, version=OBSERVATION_VERSION)
        self.t = 0
        self._own_sum = torch.zeros(self.n, dtype=torch.int64,
                                    device=self.device)
        self._other_sum = torch.zeros(self.n, dtype=torch.int64,
                                      device=self.device)

    def _build_groups(self, grouped: dict[Any, list[tuple[int, Any]]]
                      ) -> None:
        for key, members in grouped.items():
            kind = key[0]
            rows = [row for row, _ in members]
            template = members[0][1]
            use_gpu = (kind in ("torch", "selfplay")
                       or len(rows) >= self.policy_gpu_min)
            policy_device = self.device if use_gpu else torch.device("cpu")
            if kind in ("mlp", "gru"):
                state = FeatureState85(
                    len(rows), device=policy_device, dtype=self.dtype,
                    rounds=self.rounds)
            else:
                version = (getattr(template, "observation_version",
                                   OBSERVATION_VERSION)
                           if kind == "selfplay" else OBSERVATION_VERSION)
                state = FeatureState21(
                    len(rows), device=policy_device, dtype=self.dtype,
                    rounds=self.rounds, version=version)
            if kind == "torch":
                policy = _TorchBatchPolicy(template.model, policy_device)
            else:
                policy = build_batch_policy(template, policy_device, self.dtype)
            generator = torch.Generator(device=policy_device)
            generator.manual_seed(self.seed + rows[0])
            opponent_mode = str(getattr(template, "mode", self.mode))
            if opponent_mode not in ("sample", "argmax"):
                opponent_mode = self.mode
            self._groups.append(_PolicyGroup(
                rows=rows, kind=kind, policy=policy, state=state,
                mode=opponent_mode, rounds=self.rounds,
                temperature=float(getattr(template, "temperature", 1.0)),
                safe=bool(getattr(template, "safe", False)
                          if kind in ("mlp", "gru") else False),
                generator=generator, generator_seed=self.seed + rows[0],
            ))

    def reset(self, factories: Sequence[OpponentFactory] | None = None,
              families: Sequence[str] | None = None) -> None:
        if factories is not None:
            if self.rules is not None:
                self.rules.close()
            self._build(list(factories), families)
            return
        self.state.reset()
        self.t = 0
        self._own_sum.zero_()
        self._other_sum.zero_()
        self._pending_reset = False
        if self.rules is not None:
            self.rules.reset()
        for group in self._groups:
            group.reset()
        for row in self._cpu_rows:
            _index, obj, own, blue, initial = row
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()
            own.clear()
            blue.clear()
            if initial is not None:
                obj.rng.setstate(initial)

    def close(self) -> None:
        if self.rules is not None:
            self.rules.close()
            self.rules = None

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Observation and step API
    # ------------------------------------------------------------------

    @property
    def fast_enabled(self) -> bool:
        return self.rules is not None

    @property
    def fast_fraction(self) -> float:
        return len(self._rule_rows) / max(1, self.n)

    @property
    def done(self) -> bool:
        return self.t >= self.rounds

    def done_mask(self) -> torch.Tensor:
        return torch.full((self.n,), self.done, dtype=torch.bool,
                          device=self.device)

    def observations_tensor(self) -> torch.Tensor:
        return self.state.features()

    def observations(self) -> np.ndarray:
        return self.observations_tensor().detach().cpu().numpy()

    def _opponent_actions(self, blue: torch.Tensor) -> torch.Tensor:
        out = np.zeros(self.n, dtype=np.int8)
        for group in self._groups:
            action = group.act(self.t)
            out[group.rows_np] = action.detach().cpu().numpy()
        blue_cpu: list[int] | None = None
        if self.rules is not None:
            blue_cpu = blue.detach().cpu().tolist()
            replies = self.rules.step(
                self.t, [blue_cpu[index] for index in self._rule_rows])
            out[self._rule_rows] = np.asarray(replies, dtype=np.int8)
        if self._cpu_rows:
            if blue_cpu is None:
                blue_cpu = blue.detach().cpu().tolist()
            for row in self._cpu_rows:
                index, obj, own, theirs, _initial = row
                action = obj.act(self.t, own, theirs)
                if action not in (C, D):
                    raise ValueError(
                        f"{self.opponent_names[index]} returned {action!r}")
                own.append(int(action))
                theirs.append(int(blue_cpu[index]))
                out[index] = int(action)
        return torch.as_tensor(out, dtype=torch.int8, device=self.device)

    def _observe_groups(self, blue: torch.Tensor,
                        opponent: torch.Tensor) -> None:
        # One host copy serves every CPU policy group; GPU groups slice the
        # device tensors directly.  Avoid one synchronising slice per group.
        opponent_cpu = blue_cpu = None
        for group in self._groups:
            if group.device.type == "cpu":
                if opponent_cpu is None:
                    opponent_cpu = opponent.detach().cpu()
                    blue_cpu = blue.detach().cpu()
                group.append(opponent_cpu[group.rows_t],
                             blue_cpu[group.rows_t])
            else:
                group.append(opponent[group.rows_t], blue[group.rows_t])
        # Rule rows were already committed by RuleBank.step; the CPU fallback
        # histories were appended inside _opponent_actions.

    @torch.no_grad()
    def step_tensor(self, actions: torch.Tensor | Sequence[int]
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._pending_reset:
            self.reset()
        if self.t >= self.rounds:
            raise RuntimeError("episode already finished")
        blue = (actions if isinstance(actions, torch.Tensor)
                else torch.as_tensor(actions, dtype=torch.int8,
                                     device=self.device))
        blue = blue.to(device=self.device, dtype=torch.int8)
        if blue.numel() != self.n:
            raise ValueError("action count does not match environment count")
        opponent = self._opponent_actions(blue)
        self._observe_groups(blue, opponent)
        self.state.append(blue, opponent)
        own, other = _payoff_tensors(blue, opponent)
        self._own_sum += own.to(torch.int64)
        self._other_sum += other.to(torch.int64)
        self.t += 1
        done = torch.full((self.n,), self.done, dtype=torch.bool,
                          device=self.device)
        if self.done and self.auto_reset:
            # Caller still sees done=True once; the next call starts fresh.
            self._pending_reset = True
        return own, other, done

    def step(self, actions: Sequence[int]) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]:
        own, other, done = self.step_tensor(actions)
        return (own.detach().cpu().numpy(), other.detach().cpu().numpy(),
                done.detach().cpu().numpy())

    @property
    def scores(self) -> list[int]:
        return [int(x) for x in self._own_sum.detach().cpu().tolist()]

    @property
    def opponent_scores(self) -> list[int]:
        return [int(x) for x in self._other_sum.detach().cpu().tolist()]


__all__ = ["GPUEnv", "OpponentFactory"]
