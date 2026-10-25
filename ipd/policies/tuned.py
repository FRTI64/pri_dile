"""Tuned rule-policy configurations.

The historical evaluation scripts read these values from ``results/*.json``.
A policy should still be usable in a minimal submit bundle, so every lookup has
an embedded fallback with the values documented in ``HANDOVER.md``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ..paths import result_path

DEFAULT_FINAL_TUNED: dict[str, Any] = {
    "P_UNPROVOKED": 0.2395699546614404,
    "P_MUTUAL": 0.6399155368157192,
    "P_RESPONSIVE": 0.9544467968114831,
    "MIN_C": 5,
    "MIN_D": 2,
    "RESP_GAP": 0.3871088255996472,
    "DEFECT_RATE": 0.1170465108152342,
    "PEACE_TESTS": 1,
    "PEACE_COOLDOWN": 9,
}

DEFAULT_RESP_TUNED: dict[str, Any] = {
    "g_unprov": 0.31711381663242244,
    "g_mutual": 0.26173130084075913,
    "g_resp": 0.9804446329081643,
    "min_c": 5,
    "min_d": 2,
    "resp": 0.46788262082576104,
    "rate": 0.197064176521839,
    "max_tests": 3,
    "cooldown": 14,
}

DEFAULT_BAYES_BASE: dict[str, Any] = {
    "coop_scale": 1.0,
    "punitive_scale": 1.0,
    "late_scale": 1.0,
    "noise_scale": 1.0,
    "probe_scale": 1.0,
    "nonresp": 0.40,
    "punitive_thr": 0.55,
}

DEFAULT_BAYES_TUNED: dict[str, Any] = {
    "coop_scale": 1.8369823500713045,
    "punitive_scale": 2.4705958305576154,
    "late_scale": 0.5216017158776368,
    "noise_scale": 0.5817914534720606,
    "probe_scale": 3.315461421931959,
    "nonresp": 0.27974873408266854,
    "punitive_thr": 0.7531347429866336,
}


@dataclass
class TunedConfigs:
    final: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FINAL_TUNED))
    resp: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RESP_TUNED))
    bayes: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_BAYES_TUNED))
    hybrid: dict[str, Any] = field(default_factory=lambda: {
        "final": dict(DEFAULT_FINAL_TUNED),
        "resp": {},
        "p": 0.0,
    })
    harvest: dict[str, Any] = field(default_factory=lambda: {
        **DEFAULT_FINAL_TUNED,
        "HARVEST_START": 45,
    })


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def load_tuned_configs() -> TunedConfigs:
    """Load the selected configs, falling back to embedded tuned defaults.

    Cached per process: benchmark workers build many policies but the tuned
    JSON files are immutable during a run.  Call ``load_tuned_configs.cache_clear()``
    after retraining if the same process must reload them.
    """
    cfg = TunedConfigs()

    multi = _read_json(result_path("train_multiobj_results.json"))
    if multi:
        rows = multi.get("validation") or []
        for kind, attr in (("final", "final"), ("resp", "resp"),
                           ("harvest", "harvest")):
            row = next((r for r in rows if r.get("kind") == kind), None)
            if row and isinstance(row.get("cfg"), dict):
                getattr(cfg, attr).update(row["cfg"])

    bayes = _read_json(result_path("train_bayes_priors_results.json"))
    if bayes:
        rows = bayes.get("validation") or []
        if rows:
            best = max(rows, key=lambda r: r.get("val_fit", float("-inf")))
            if isinstance(best.get("cfg"), dict):
                cfg.bayes.update(best["cfg"])

    joint = _read_json(result_path("train_joint_results.json"))
    if joint:
        rows = joint.get("validation") or []
        if rows:
            best = max(rows, key=lambda r: r.get("val_fit", float("-inf")))
            if all(k in best for k in ("final", "resp", "p")):
                cfg.hybrid = {
                    "final": dict(best["final"]),
                    "resp": dict(best["resp"]),
                    "p": float(best["p"]),
                }
    return cfg
