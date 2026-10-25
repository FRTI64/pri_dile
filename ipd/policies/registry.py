"""Policy registry: names, specs and construction for every candidate.

Evaluation code should never import concrete policy classes directly.  It asks
this module for a :class:`CandidateSpec`, sends the spec to a worker, and calls
``spec.build()`` there.  That keeps candidate files, test harnesses and
training scripts decoupled.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..paths import config_path, weights_path
from .adversarial import AdversarialPolicy
from .base import PolicyAdapter, ensure_policy
from .gru import GRUPolicy
from .neural import NeuralPolicy
from .rl import RLPolicy
from .rules import (
    GapStartPolicy, Hybrid, ParamBayesPrior, ParamHarvest, ParamPolicy,
    ParamResp, Triple,
)
from .tuned import DEFAULT_BAYES_BASE, TunedConfigs, load_tuned_configs

DEFAULT_WEIGHTS = {
    "neural": "neural_policy_weights.json",
    "gru": "gru_policy_weights.json",
    "rl": "rl_policy_checkpoint.json",
    "adv-mlp": "neural_policy_weights.json",
    "adv-gru": "gru_policy_weights.json",
}
DEFAULT_CONFIG = {
    "adv-mlp": "adversarial_policy_config.json",
    "adv-gru": "adversarial_gru_policy_config.json",
    "adv-rl": "adversarial_rl_policy_config_aggressive.json",
}

DISPLAY = {
    "final-base": "Final-base",
    "final-tuned": "Final-tuned",
    "resp3-base": "Resp3-base",
    "resp3-tuned": "Resp3-tuned",
    "bayes-base": "Bayes-base",
    "bayes-tuned": "Bayes-tuned",
    "harvest45": "Harvest45",
    "harvest-tuned": "Harvest-tuned",
    "hybrid": "Final+Resp3",
    "triple-majority": "Triple-majority",
    "triple-or": "Triple-or",
    "triple-and": "Triple-and",
    "gap-start": "GapStart",
    "neural": "Neural",
    "gru": "GRU",
    "rl": "RL-PPO",
    "adv-mlp": "Adv-MLP",
    "adv-gru": "Adv-GRU",
    "adv-rl": "Adv-RL",
}

ALIASES = {
    "final": "final-base",
    "finalbase": "final-base",
    "finaltuned": "final-tuned",
    "final-param": "final-tuned",
    "resp": "resp3-base",
    "respbase": "resp3-base",
    "resp3": "resp3-base",
    "resp3base": "resp3-base",
    "resptuned": "resp3-tuned",
    "resp3tuned": "resp3-tuned",
    "bayes": "bayes-tuned",
    "bayesbase": "bayes-base",
    "bayestuned": "bayes-tuned",
    "harvest": "harvest-tuned",
    "harvest45": "harvest45",
    "harvesttuned": "harvest-tuned",
    "final+resp3": "hybrid",
    "finalresp3": "hybrid",
    "hybrid-best": "hybrid",
    "triple": "triple-majority",
    "triplemajority": "triple-majority",
    "gap": "gap-start",
    "gapstart": "gap-start",
    "mlp": "neural",
    "neural-sample": "neural",
    "neural-argmax": "neural",
    "gru-sample": "gru",
    "gru-argmax": "gru",
    "rl-ppo": "rl",
    "rl-sample": "rl",
    "rl-argmax": "rl",
    "adv": "adv-mlp",
    "adversarial": "adv-mlp",
    "advmlp": "adv-mlp",
    "advgru": "adv-gru",
    "advrl": "adv-rl",
}

CORE_NAMES = ("final-tuned", "resp3-tuned", "bayes-tuned", "triple-majority")
BASE_NAMES = (
    "final-base", "final-tuned", "resp3-base", "resp3-tuned",
    "bayes-base", "bayes-tuned", "hybrid",
)
LEGACY_NAMES = BASE_NAMES
CLASSIC_NAMES = BASE_NAMES + ("harvest45", "harvest-tuned", "gap-start")
SUITE_NAMES = BASE_NAMES + ("harvest45", "triple-majority")
LEARNED_NAMES = ("neural", "gru", "rl", "adv-mlp", "adv-gru", "adv-rl")
ALL_NAMES = tuple(dict.fromkeys(
    SUITE_NAMES + CLASSIC_NAMES + LEARNED_NAMES))

SET_ALIASES = {
    "core": CORE_NAMES,
    "base": BASE_NAMES,
    "legacy": LEGACY_NAMES,
    "classic": CLASSIC_NAMES,
    "suite": SUITE_NAMES,
    "learned": LEARNED_NAMES,
    "all": ALL_NAMES,
}


def canonical_name(value: str) -> str:
    """Normalise a candidate label (``Final-tuned``, ``resp3_tuned``, ...)."""
    key = value.strip().lower().replace("_", "-").replace(" ", "-")
    key = ALIASES.get(key, key)
    key = ALIASES.get(key.replace("-", ""), key)
    if key not in DISPLAY:
        raise ValueError(
            f"unknown candidate {value!r}; choose from: {', '.join(ALL_NAMES)}"
        )
    return key


def _load_rl_base(weights: str, seed: int, mode: str, name: str):
    """Accept either an actor-critic checkpoint or a deployable MLP export."""
    with open(weights, encoding="utf-8") as fh:
        fmt = json.load(fh).get("format")
    cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
    return cls(weights, seed=seed, mode=mode, name=name)


def _adv_params(config: str) -> tuple[dict[str, Any], str]:
    with open(config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    params = {
        "gap": cfg.get("gap"),
        "min_c": cfg.get("min_c", 8),
        "min_d": cfg.get("min_d", 3),
        "late_day": cfg.get("late_day"),
        "repair": cfg.get("repair", False),
    }
    base = cfg.get("base_weights", "")
    return params, base


def make_policy(
    kind: str,
    *,
    seed: int = 20261024,
    weights: str | None = None,
    config: str | None = None,
    mode: str = "sample",
    name: str | None = None,
    tuned: TunedConfigs | None = None,
) -> PolicyAdapter:
    """Construct one policy by canonical kind.

    ``kind`` is one of the names in :data:`DISPLAY`.  Rule policies read tuned
    thresholds from :func:`load_tuned_configs` unless explicit weights/config
    paths are supplied for a learned policy.
    """
    kind = canonical_name(kind)
    tuned = tuned or load_tuned_configs()
    label = name or DISPLAY[kind]

    if kind == "final-base":
        policy = ParamPolicy(seed=seed)
    elif kind == "final-tuned":
        policy = ParamPolicy(seed=seed, **tuned.final)
    elif kind == "resp3-base":
        policy = ParamResp(seed=seed)
    elif kind == "resp3-tuned":
        policy = ParamResp(seed=seed, **tuned.resp)
    elif kind == "bayes-base":
        policy = ParamBayesPrior(DEFAULT_BAYES_BASE)
    elif kind == "bayes-tuned":
        policy = ParamBayesPrior(tuned.bayes)
    elif kind == "harvest45":
        policy = ParamHarvest(seed=seed, HARVEST_START=45)
    elif kind == "harvest-tuned":
        policy = ParamHarvest(seed=seed, **tuned.harvest)
    elif kind == "hybrid":
        policy = Hybrid(
            tuned.hybrid.get("final", tuned.final),
            tuned.hybrid.get("resp", {}),
            tuned.hybrid.get("p", 0.0),
        )
    elif kind in ("triple-majority", "triple-or", "triple-and"):
        policy = Triple(
            tuned.final, tuned.resp, tuned.bayes,
            mode=kind.removeprefix("triple-"),
        )
    elif kind == "gap-start":
        policy = GapStartPolicy(1, 0.5)
    elif kind == "neural":
        policy = NeuralPolicy(
            weights or weights_path(DEFAULT_WEIGHTS["neural"]),
            seed=seed, mode=mode,
        )
    elif kind == "gru":
        policy = GRUPolicy(
            weights or weights_path(DEFAULT_WEIGHTS["gru"]),
            seed=seed, mode=mode,
        )
    elif kind == "rl":
        policy = _load_rl_base(
            weights or weights_path(DEFAULT_WEIGHTS["rl"]),
            seed=seed, mode=mode, name="RL-PPO",
        )
    elif kind in ("adv-mlp", "adv-gru"):
        config = config or config_path(DEFAULT_CONFIG[kind])
        policy = AdversarialPolicy.from_config(
            config, seed=seed, mode=mode, name=label,
        )
    elif kind == "adv-rl":
        config = config or config_path(DEFAULT_CONFIG["adv-rl"])
        params, base_weights = _adv_params(config)
        if weights is None:
            base_weights = base_weights or DEFAULT_WEIGHTS["rl"]
            weights = (base_weights if os.path.isabs(base_weights)
                       else weights_path(base_weights))
        base = _load_rl_base(weights, seed=seed, mode=mode, name=label)
        policy = AdversarialPolicy(base, name=label, **params)
    else:  # pragma: no cover - canonical_name already validates
        raise ValueError(kind)

    policy.name = label
    return ensure_policy(policy, name=label)


@dataclass(frozen=True)
class CandidateSpec:
    """Serialisable description of one candidate for worker processes."""

    kind: str
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def build(self) -> PolicyAdapter:
        return make_policy(self.kind, name=self.name, **dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name,
                "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateSpec":
        return cls(kind=str(data["kind"]), name=str(data["name"]),
                   params=dict(data.get("params", {})))


def _params_for(kind: str, weights: str | None, config: str | None,
                mode: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if kind in ("neural", "gru", "rl"):
        if weights:
            params["weights"] = weights
        params["mode"] = mode
    elif kind in ("adv-mlp", "adv-gru", "adv-rl"):
        if weights:
            params["weights"] = weights
        if config:
            params["config"] = config
        params["mode"] = mode
    return params


def resolve_names(values: str | Iterable[str] | None,
                  default: Iterable[str] = CORE_NAMES) -> list[str]:
    """Expand ``core``/``all`` tokens and comma-separated labels."""
    if values is None:
        return list(default)
    if isinstance(values, str):
        raw = [x for x in values.replace(";", ",").split(",") if x.strip()]
    else:
        raw = [x for x in values if str(x).strip()]
    if not raw:
        return list(default)
    out: list[str] = []
    for item in raw:
        key = str(item).strip().lower()
        if key in SET_ALIASES:
            out.extend(SET_ALIASES[key])
        else:
            out.append(canonical_name(item))
    return list(dict.fromkeys(out))


def _display_for(kind: str, config: str | None) -> str:
    """Use the config's base architecture for adversarial display labels."""
    if kind in ("adv-mlp", "adv-gru", "adv-rl") and config:
        try:
            with open(config, encoding="utf-8") as fh:
                arch = json.load(fh).get("base_arch")
        except (OSError, ValueError):
            arch = None
        if arch:
            return {"mlp": "Adv-MLP", "gru": "Adv-GRU",
                    "rl": "Adv-RL"}.get(arch, DISPLAY[kind])
    return DISPLAY[kind]


def candidate_specs(
    names: str | Iterable[str] | None = None,
    *,
    default: Iterable[str] = CORE_NAMES,
    weights: str | None = None,
    config: str | None = None,
    mode: str = "sample",
) -> list[CandidateSpec]:
    """Build serialisable specs for a candidate selection."""
    resolved = resolve_names(names, default=default)
    specs = []
    for kind in resolved:
        specs.append(CandidateSpec(
            kind=kind,
            name=_display_for(kind, config),
            params=_params_for(kind, weights, config, mode),
        ))
    return specs


def legacy_arch_names(arch: str) -> list[str]:
    """Map the old ``--arch`` flag values to candidate names."""
    return {
        "mlp": ["neural"],
        "neural": ["neural"],
        "gru": ["gru"],
        "rl": ["rl"],
        "adversarial": ["adv-mlp"],
        "adv-mlp": ["adv-mlp"],
        "adv-gru": ["adv-gru"],
        "adv-rl": ["adv-rl"],
    }.get(arch, [canonical_name(arch)])
