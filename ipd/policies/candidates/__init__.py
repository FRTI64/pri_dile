"""One self-contained file per submission/evaluation candidate.

Each module exposes ``KIND``, ``DISPLAY`` and ``build(...)``.  Candidate modules
must not import each other; they may use core simulation/model modules and their
own parameter constants.  This keeps every candidate easy to translate on its
own.
"""
from __future__ import annotations

import importlib

MODULES = {
    "final-base": "final_base",
    "final-tuned": "final_tuned",
    "resp3-base": "resp3_base",
    "resp3-tuned": "resp3_tuned",
    "bayes-base": "bayes_base",
    "bayes-tuned": "bayes_tuned",
    "hybrid": "hybrid",
    "harvest45": "harvest45",
    "harvest-tuned": "harvest_tuned",
    "triple-majority": "triple_majority",
    "gap-start": "gap_start",
    "neural": "neural",
    "gru": "gru",
    "rl": "rl",
    "rl-hard60": "rl_hard60",
    "rl-repair": "rl_repair",
    "psro": "psro",
    "psro-br": "psro_br",
    "psro-scratch": "psro_scratch",
    "psro-scratch-br": "psro_scratch_br",
    "adv-mlp": "adv_mlp",
    "adv-gru": "adv_gru",
    "adv-rl": "adv_rl",
    "selfplay": "selfplay",
}


def build_candidate(kind: str, **kwargs):
    """Import exactly one candidate module and call its ``build`` function."""
    key = kind
    if key in ("triple-or", "triple-and"):
        kwargs.setdefault("voting", key.removeprefix("triple-"))
        key = "triple-majority"
    module_name = MODULES.get(key)
    if module_name is None:
        raise ValueError(f"unknown candidate kind: {kind!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return module.build(**kwargs)


__all__ = ["MODULES", "build_candidate"]
