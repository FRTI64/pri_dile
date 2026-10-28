"""Policy registry: names, specs and construction for every candidate.

Evaluation code should never import concrete policy classes directly.  It asks
this module for a :class:`CandidateSpec`, sends the spec to a worker, and calls
``spec.build()`` there.  That keeps candidate files, test harnesses and
training scripts decoupled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .base import PolicyAdapter, ensure_policy
from .candidates import build_candidate
from .tuned import TunedConfigs

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
    "rl-hard60": "RL-PPO-Hard60",
    "rl-repair": "RL-Repair",
    "rl-pattern": "RL-Pattern",
    "psro": "PSRO-Meta",
    "psro-br": "PSRO-BR",
    "psro-scratch": "PSRO-Scratch",
    "psro-scratch-br": "PSRO-Scratch-BR",
    "adv-mlp": "Adv-MLP",
    "adv-gru": "Adv-GRU",
    "adv-rl": "Adv-RL",
    "selfplay": "SelfPlay-GRU",
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
    "rl-hard": "rl-hard60",
    "rl-hard60": "rl-hard60",
    "rl-ppo-hard60": "rl-hard60",
    "hard60": "rl-hard60",
    "rl-repair": "rl-repair",
    "repair": "rl-repair",
    "rl-robust-repair": "rl-repair",
    "rl-pattern": "rl-pattern",
    "pattern": "rl-pattern",
    "rl-pb": "rl-pattern",
    "psro": "psro",
    "psro-meta": "psro",
    "psro-br": "psro-br",
    "psro-response": "psro-br",
    "psro-scratch": "psro-scratch",
    "psro-zero": "psro-scratch",
    "psro-scratch-br": "psro-scratch-br",
    "psro-zero-br": "psro-scratch-br",
    "adv": "adv-mlp",
    "adversarial": "adv-mlp",
    "advmlp": "adv-mlp",
    "advgru": "adv-gru",
    "advrl": "adv-rl",
    "selfplay": "selfplay",
    "self-play": "selfplay",
    "selfplay-gru": "selfplay",
    "sp": "selfplay",
}

CORE_NAMES = ("final-tuned", "resp3-tuned", "bayes-tuned", "triple-majority")
BASE_NAMES = (
    "final-base", "final-tuned", "resp3-base", "resp3-tuned",
    "bayes-base", "bayes-tuned", "hybrid",
)
LEGACY_NAMES = BASE_NAMES
CLASSIC_NAMES = BASE_NAMES + ("harvest45", "harvest-tuned", "gap-start")
SUITE_NAMES = BASE_NAMES + ("harvest45", "triple-majority")
LEARNED_NAMES = ("neural", "gru", "rl", "rl-hard60", "rl-repair",
                 "rl-pattern",
                 "psro", "psro-br", "psro-scratch", "psro-scratch-br",
                 "adv-mlp", "adv-gru", "adv-rl", "selfplay")
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

    Each candidate is implemented in its own module under
    :mod:`ipd.policies.candidates`.  Candidate modules do not import each other;
    they only use core simulation/model code and their own constants.  ``tuned``
    is accepted for backwards compatibility but candidate files embed the
    selected parameter values directly.
    """
    kind = canonical_name(kind)
    label = name or DISPLAY[kind]
    kwargs: dict[str, Any] = {
        "seed": seed,
        "weights": weights,
        "config": config,
        "mode": mode,
        "name": label,
        "tuned": tuned,
    }
    if kind.startswith("triple-"):
        kwargs["voting"] = kind.removeprefix("triple-")
    policy = build_candidate(kind, **kwargs)
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
    if kind in ("neural", "gru", "rl", "rl-hard60", "rl-repair",
                "rl-pattern",
                "psro", "psro-br", "psro-scratch", "psro-scratch-br",
                "selfplay"):
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
