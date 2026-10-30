#!/usr/bin/env python3
"""Policy-Space Response Oracles (PSRO) for the 100-day IPD.

The module implements the full PSRO loop rather than the earlier
maximin-only "PSRO-lite" helper:

1. keep a finite population of frozen policies;
2. estimate the cross-play meta-game (score difference);
3. solve a meta-strategy with fictitious play;
4. train a best response to that meta-strategy;
5. add the best response to the population and repeat.

The best-response oracle is pluggable.  The default oracle shells out to
``python3 -m ipd train rl --opponent-pool ...`` so the existing 85-dim PPO
trainer, safety rules and deployable JSON export are reused unchanged.  A
function oracle can be supplied in tests, which keeps the meta-game and
population logic testable without PyTorch.

Deployment convention
---------------------
The final best response (or the meta-best blue policy, configurable) is copied
to ``weights/psro/psro_policy.json``.  The population, the full payoff matrix
and the meta-strategy are written next to the report for reproducibility.  A
PSRO result is **experimental** until it passes the same Protocol v5 gates as
every other candidate; this module never promotes the main weights.

Usage::

    python3 -m ipd train psro --config configs/psro/psro_20261027.json
    python3 -m ipd train psro --config ... --dry-run     # meta-game only
"""
from __future__ import annotations

if __package__ in (None, ""):  # allow: python ipd/training/psro.py
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ipd.evaluation.metagame import (
    antisymmetrize, fictitious_play, maximin_strategy,
)
from ipd.evaluation.policy_pool import build_policy_entry
from ipd.paths import ROOT, result_path
from ipd.training.league import write_opponent_manifest

POPULATION_SCHEMA = "ipd-psro-population-v1"
REPORT_SCHEMA = "ipd-psro-report-v1"
CONFIG_SCHEMA = "ipd-psro-config-v1"
_PATH_KEYS = ("path", "weights", "config", "checkpoint")
_DEPLOYABLE_KINDS = ("rl", "neural", "gru", "selfplay")


# ---------------------------------------------------------------------------
# Population entries
# ---------------------------------------------------------------------------


def _portable_path(value: Any) -> str:
    """Prefer a repository-relative path in committed artifacts."""
    text = str(value)
    try:
        return str(Path(text).resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return text


def _portable_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    if isinstance(out.get("path"), str):
        out["path"] = _portable_path(out["path"])
    params = out.get("params")
    if isinstance(params, Mapping):
        params = dict(params)
        for key in _PATH_KEYS:
            if isinstance(params.get(key), str):
                params[key] = _portable_path(params[key])
        out["params"] = params
    spec = out.get("spec")
    if isinstance(spec, Mapping):
        spec = dict(spec)
        spec_params = spec.get("params")
        if isinstance(spec_params, Mapping):
            spec_params = dict(spec_params)
            for key in _PATH_KEYS:
                if isinstance(spec_params.get(key), str):
                    spec_params[key] = _portable_path(spec_params[key])
            spec["params"] = spec_params
        out["spec"] = spec
    return out


@dataclass(frozen=True)
class PopulationEntry:
    """One frozen policy in the PSRO population.

    ``entry`` is a policy-pool entry: either ``{"path": "..."}`` or a
    serialisable candidate spec ``{"kind": ..., "params": {...}}``.  Metadata
    (``role`` / ``generation`` / ``parent``) is kept out of the policy entry so
    it can be passed to the RL oracle unchanged.
    """

    name: str
    entry: Mapping[str, Any]
    role: str = "blue"
    generation: int = 0
    parent: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("population entry needs a name")
        if not self.entry:
            raise ValueError(f"population entry {self.name!r} is empty")
        if not any(key in self.entry for key in ("path", "spec", "kind")):
            raise ValueError(
                f"population entry {self.name!r} needs path/spec/kind")

    @property
    def deploy_path(self) -> str | None:
        """Return a deployable JSON policy path, if this entry has one."""
        path = self.entry.get("path")
        if path:
            return str(path)
        kind = str(self.entry.get("kind", ""))
        if kind in _DEPLOYABLE_KINDS:
            params = self.entry.get("params") or {}
            if isinstance(params, Mapping) and params.get("weights"):
                return str(params["weights"])
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entry": _portable_entry(self.entry),
            "role": self.role,
            "generation": int(self.generation),
            "parent": self.parent,
            "tags": list(self.tags),
        }

    def to_pool_entry(self) -> dict[str, Any]:
        """Return the manifest-shaped entry consumed by ``--opponent-pool``."""
        item = _portable_entry(self.entry)
        item["name"] = self.name
        item["role"] = self.role
        item["generation"] = int(self.generation)
        return item

    def build(self, *, seed: int = 20261024, mode: str = "sample"):
        entry = dict(self.entry)
        entry.setdefault("name", self.name)
        return build_policy_entry(entry, seed=seed, mode=mode,
                                  name=self.name)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PopulationEntry":
        if not isinstance(data, Mapping):
            raise TypeError(f"population entry must be a mapping: {data!r}")
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("population entry needs name")
        entry: dict[str, Any] = {}
        if isinstance(data.get("entry"), Mapping):
            entry = dict(data["entry"])
        elif data.get("path"):
            entry["path"] = str(data["path"])
        elif data.get("spec"):
            entry["spec"] = dict(data["spec"])
        elif data.get("kind"):
            entry["kind"] = str(data["kind"])
            if data.get("params"):
                entry["params"] = dict(data["params"])
        elif data.get("weights"):
            # Convenience form: {"name": ..., "weights": "..."}.
            entry["kind"] = str(data.get("arch") or "rl")
            entry["params"] = {"weights": str(data["weights"])}
        else:
            raise ValueError(
                f"population entry {name!r} needs path/spec/kind/weights")
        return cls(
            name=name,
            entry=entry,
            role=str(data.get("role") or "blue"),
            generation=int(data.get("generation") or 0),
            parent=(str(data["parent"]) if data.get("parent") else None),
            tags=tuple(str(x) for x in data.get("tags") or ()),
        )


def make_scratch_policy(path: str | Path, *, seed: int = 20261027,
                        template: str | Path | None = None,
                        hidden: Sequence[int] = (64, 32)) -> dict[str, Any]:
    """Create a random 85-dim MLP policy for a from-scratch PSRO population.

    This is the rough equivalent of OpenSpiel PSRO's random initial policy.
    Only the feature normalisation is copied from the template checkpoint; all
    policy weights are random.  ``safe=True`` keeps the project's day-1 /
    day-100 boundary rules.
    """
    template_path = Path(template or "weights/neural_policy_weights.json")
    if not template_path.is_absolute():
        template_path = ROOT / template_path
    base: dict[str, Any] = {}
    if template_path.is_file():
        base = json.loads(template_path.read_text(encoding="utf-8"))
    n_features = int(base.get("n_features", 85))
    mean = [float(x) for x in (base.get("feature_mean")
                               or [0.0] * n_features)]
    std = [float(x) if abs(float(x)) > 1e-8 else 1.0
           for x in (base.get("feature_std") or [1.0] * n_features)]
    dims = [n_features] + [int(x) for x in hidden] + [1]
    rng = random.Random(int(seed))
    layers = []
    for n_in, n_out in zip(dims, dims[1:]):
        scale = 1.0 / max(1.0, float(n_in)) ** 0.5
        layers.append({
            "w": [[rng.uniform(-scale, scale) for _ in range(n_in)]
                  for _ in range(n_out)],
            "b": [rng.uniform(-scale, scale) for _ in range(n_out)],
        })
    data = {
        "format": "ipd-neural-policy-v1",
        "feature_version": base.get("feature_version",
                                    "ipd-neural-features-v1"),
        "n_features": n_features,
        "activation": "tanh",
        "layers": layers,
        "feature_mean": mean,
        "feature_std": std,
        "temperature": 1.0,
        "safe": True,
        "meta": {
            "algorithm": "psro-scratch-init",
            "seed": int(seed),
            "note": "random policy; only feature normalisation is inherited",
        },
    }
    _save_json(path, data)
    return data


def load_population(path: str | Path) -> list[PopulationEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        rows = data.get("population", data.get("entries"))
    else:
        rows = data
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"no population entries in {path}")
    entries = [PopulationEntry.from_dict(row) for row in rows]
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate population names in {path}")
    return entries


def save_population(path: str | Path, entries: Sequence[PopulationEntry],
                    meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "schema": POPULATION_SCHEMA,
        "entries": [entry.to_dict() for entry in entries],
    }
    if meta is not None:
        payload["meta"] = dict(meta)
    _save_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# Meta-game evaluation
# ---------------------------------------------------------------------------


def _seeds(base: int, count: int, step: int = 7919) -> tuple[int, ...]:
    return tuple(int(base) + i * step for i in range(int(count)))


def cross_play_pair_task(task: dict[str, Any]) -> dict[str, Any]:
    """Play both orientations of one population pair.

    The task is deliberately self-contained so it can run under ``spawn``.
    """
    from ipd.evaluation.common import run_match

    i = int(task["i"])
    j = int(task["j"])
    mode = task.get("mode", "sample")
    rounds = int(task.get("rounds", 100))
    seeds = tuple(int(x) for x in task["seeds"])
    i_seeds = (int(task["policy_seed_base"]) + i * 10007 + j,
               int(task["policy_seed_base"]) + i * 10007 + j + 500001)
    j_seeds = (int(task["policy_seed_base"]) + j * 10007 + i,
               int(task["policy_seed_base"]) + j * 10007 + i + 500001)
    i_entry = task["i_entry"]
    j_entry = task["j_entry"]
    i_scores: list[int] = []
    j_scores: list[int] = []
    diffs: list[int] = []
    for pair_index in range(2):
        a = build_policy_entry(i_entry, seed=i_seeds[pair_index], mode=mode,
                               name=task["i_name"])
        b = build_policy_entry(j_entry, seed=j_seeds[pair_index], mode=mode,
                               name=task["j_name"])
        for _seed in seeds:
            ab = run_match(a, b, rounds=rounds)
            ba = run_match(b, a, rounds=rounds)
            i_scores.extend((int(ab.score), int(ba.opponent_score)))
            j_scores.extend((int(ab.opponent_score), int(ba.score)))
            diffs.extend((
                int(ab.score) - int(ab.opponent_score),
                int(ba.opponent_score) - int(ba.score),
            ))
    return {
        "i": i, "j": j,
        "i_scores": i_scores, "j_scores": j_scores, "diffs": diffs,
    }


def _run_tasks(tasks: Sequence[dict[str, Any]], workers: int,
               label: str) -> list[dict[str, Any]]:
    if not tasks:
        return []
    if workers <= 1:
        return [cross_play_pair_task(task) for task in tasks]
    rows: list[dict[str, Any]] = []
    started = time.time()
    total = len(tasks)
    every = max(1, total // 10)
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        for done, result in enumerate(
                pool.map(cross_play_pair_task, tasks, chunksize=1), 1):
            rows.append(result)
            if done % every == 0 or done == total:
                print(f"  {label} {done}/{total} pairs "
                      f"({time.time() - started:.1f}s)", flush=True)
    return rows


class MetaGame:
    """Incrementally evaluated PSRO meta-game."""

    def __init__(self, entries: Sequence[PopulationEntry], *,
                 seeds: Sequence[int], rounds: int = 100,
                 workers: int = 1, mode: str = "sample",
                 policy_seed_base: int = 20461026):
        if not entries:
            raise ValueError("meta-game needs at least one entry")
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("population names must be unique")
        self.entries: list[PopulationEntry] = list(entries)
        self.seeds = tuple(int(x) for x in seeds)
        self.rounds = int(rounds)
        self.workers = int(workers)
        self.mode = mode
        self.policy_seed_base = int(policy_seed_base)
        n = len(self.entries)
        self._sums = [[0.0] * n for _ in range(n)]
        self._counts = [[0] * n for _ in range(n)]
        self._i_scores = [[0.0] * n for _ in range(n)]
        self._j_scores = [[0.0] * n for _ in range(n)]

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return [entry.name for entry in self.entries]

    def evaluate(self, pairs: Sequence[tuple[int, int]] | None = None,
                 *, label: str = "meta") -> None:
        n = len(self.entries)
        if pairs is None:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        tasks = []
        for i, j in pairs:
            tasks.append({
                "i": i, "j": j,
                "i_entry": self.entries[i].entry,
                "j_entry": self.entries[j].entry,
                "i_name": self.entries[i].name,
                "j_name": self.entries[j].name,
                "seeds": list(self.seeds),
                "rounds": self.rounds,
                "mode": self.mode,
                "policy_seed_base": self.policy_seed_base,
            })
        for row in _run_tasks(tasks, self.workers, label):
            self._absorb(row)

    def add(self, entry: PopulationEntry, *,
            opponents: Sequence[int] | None = None,
            label: str = "meta") -> None:
        if entry.name in self.names:
            raise ValueError(f"duplicate population name: {entry.name!r}")
        index = len(self.entries)
        self.entries.append(entry)
        for matrix, zero in ((self._sums, 0.0), (self._counts, 0),
                             (self._i_scores, 0.0), (self._j_scores, 0.0)):
            for line in matrix:
                line.append(zero)
            matrix.append([zero] * (index + 1))
        if index == 0:
            return
        indices = (list(range(index)) if opponents is None
                   else [int(j) for j in opponents])
        pairs = [(index, j) for j in indices if 0 <= j < index]
        if pairs:
            self.evaluate(pairs, label=label)

    def _absorb(self, row: Mapping[str, Any]) -> None:
        i = int(row["i"])
        j = int(row["j"])
        diffs = [float(x) for x in row["diffs"]]
        i_scores = [float(x) for x in row["i_scores"]]
        j_scores = [float(x) for x in row["j_scores"]]
        self._sums[i][j] += sum(diffs)
        self._sums[j][i] -= sum(diffs)
        self._counts[i][j] += len(diffs)
        self._counts[j][i] += len(diffs)
        self._i_scores[i][j] += sum(i_scores)
        self._i_scores[j][i] += sum(j_scores)
        self._j_scores[i][j] += sum(j_scores)
        self._j_scores[j][i] += sum(i_scores)

    def _mean(self, sums: Sequence[Sequence[float]],
              counts: Sequence[Sequence[int]], i: int, j: int) -> float:
        count = counts[i][j]
        return sums[i][j] / count if count else 0.0

    def matrix(self) -> list[list[float]]:
        n = len(self.entries)
        raw = [[self._mean(self._sums, self._counts, i, j)
                for j in range(n)] for i in range(n)]
        # Both orientations were sampled, so this is already antisymmetric up
        # to floating-point noise; the projection makes the property exact.
        return antisymmetrize(raw)

    def payoff(self) -> list[list[float]]:
        n = len(self.entries)
        return [[self._mean(self._i_scores, self._counts, i, j)
                 for j in range(n)] for i in range(n)]

    def opponent_payoff(self) -> list[list[float]]:
        n = len(self.entries)
        return [[self._mean(self._j_scores, self._counts, i, j)
                 for j in range(n)] for i in range(n)]

    def matches(self) -> list[list[int]]:
        return [list(row) for row in self._counts]

    def solve(self, *, iterations: int = 20000, seed: int = 0,
              uniform_mix: float = 0.0) -> dict[str, Any]:
        matrix = self.matrix()
        result = fictitious_play(matrix, iterations=iterations, seed=seed)
        weights = [max(0.0, float(x)) for x in result["weights"]]
        mix = min(1.0, max(0.0, float(uniform_mix)))
        n = len(weights)
        if mix > 0.0:
            weights = [(1.0 - mix) * value + mix / n
                       for value in weights]
        total = sum(weights)
        if total <= 0.0:
            weights = [1.0 / n] * n
        else:
            weights = [value / total for value in weights]
        row_values = [sum(matrix[i][j] * weights[j] for j in range(n))
                      for i in range(n)]
        best = max(range(n), key=lambda i: row_values[i])
        value = sum(weights[i] * row_values[i] for i in range(n))
        return {
            "weights": weights,
            "weights_by_name": dict(zip(self.names, weights)),
            "utility": dict(zip(self.names, row_values)),
            "value": value,
            "exploitability": max(0.0, row_values[best]),
            "best_response": self.names[best],
            "iterations": int(result["iterations"]),
            "uniform_mix": mix,
        }

    def report(self, **solve_kwargs: Any) -> dict[str, Any]:
        meta = self.solve(**solve_kwargs)
        return {
            "names": self.names,
            "roles": [entry.role for entry in self.entries],
            "matrix": self.matrix(),
            "payoff": self.payoff(),
            "opponent_payoff": self.opponent_payoff(),
            "matches": self.matches(),
            "meta": meta,
        }

    def best_init(self, weights: Sequence[float]) -> str | None:
        """Return the best deployable blue policy path for the oracle."""
        n = len(self.entries)
        matrix = self.matrix()
        values = [sum(matrix[i][j] * weights[j] for j in range(n))
                  for i in range(n)]
        candidates = [
            (values[i], self.entries[i].generation, i)
            for i in range(n)
            if self.entries[i].deploy_path and self.entries[i].role == "blue"
        ]
        if not candidates and any(entry.deploy_path for entry in self.entries):
            candidates = [
                (values[i], self.entries[i].generation, i)
                for i in range(n) if self.entries[i].deploy_path
            ]
        if not candidates:
            return None
        _, _, index = max(candidates)
        return self.entries[index].deploy_path

    def submatrix(self, rows: Sequence[int],
                  cols: Sequence[int]) -> list[list[float]]:
        matrix = self.matrix()
        return [[matrix[int(i)][int(j)] for j in cols] for i in rows]


# ---------------------------------------------------------------------------
# Blue-vs-red meta-game (the deployable player is blue)
# ---------------------------------------------------------------------------


def split_roles(entries: Sequence[PopulationEntry]
                ) -> tuple[list[int], list[int]]:
    """Return ``(blue_indices, red_indices)``.

    A PSRO run needs both sides.  Non-blue entries are treated as red/opponent
    policies, which keeps the format tolerant of future labels such as
    ``exploiter``.
    """
    blue = [i for i, entry in enumerate(entries) if entry.role == "blue"]
    red = [i for i, entry in enumerate(entries) if entry.role != "blue"]
    if not blue:
        raise ValueError("PSRO population needs at least one blue policy")
    if not red:
        raise ValueError("PSRO population needs at least one red/opponent policy")
    return blue, red


def _normalise(weights: Sequence[float]) -> list[float]:
    values = [max(0.0, float(x)) for x in weights]
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def solve_blue_red(game: MetaGame, blue_indices: Sequence[int],
                   red_indices: Sequence[int], *, iterations: int = 20000,
                   seed: int = 0, uniform_mix: float = 0.0
                   ) -> dict[str, Any]:
    """Solve the rectangular blue-vs-red meta-game.

    Rows are blue policies; columns are red exploiters.  The blue meta-strategy
    maximises the worst-case score difference, and the red meta-strategy
    maximises the red-minus-blue difference.  ``uniform_mix`` is applied only
    to the red training mixture so every exploiter remains represented in the
    oracle's opponent pool; it is not used for evaluation.
    """
    blues = [int(x) for x in blue_indices]
    reds = [int(x) for x in red_indices]
    if not blues or not reds:
        raise ValueError("both blue and red populations must be non-empty")
    matrix = game.submatrix(blues, reds)
    blue = maximin_strategy(matrix, iterations=iterations, seed=seed)
    red_matrix = [[-matrix[i][j] for i in range(len(blues))]
                  for j in range(len(reds))]
    red = maximin_strategy(red_matrix, iterations=iterations, seed=seed + 1)
    blue_weights = _normalise(blue["weights"])
    red_weights = _normalise(red["weights"])
    mix = min(1.0, max(0.0, float(uniform_mix)))
    if mix > 0.0:
        red_training_weights = [
            (1.0 - mix) * value + mix / len(red_weights)
            for value in red_weights
        ]
        red_training_weights = _normalise(red_training_weights)
    else:
        red_training_weights = list(red_weights)
    blue_names = [game.entries[i].name for i in blues]
    red_names = [game.entries[i].name for i in reds]
    blue_expected = [
        sum(matrix[i][j] * red_weights[j] for j in range(len(reds)))
        for i in range(len(blues))
    ]
    red_expected = [
        sum(red_matrix[j][i] * blue_weights[i] for i in range(len(blues)))
        for j in range(len(reds))
    ]
    worst_blue = float(blue["worst_case"])
    worst_red = float(red["worst_case"])
    # ``blue["best_column"]`` is the red that hurts the blue mixture most;
    # ``red["best_column"]`` is the blue that hurts the red mixture most.
    best_red_index = int(blue["best_column"])
    best_blue_index = int(red["best_column"])
    return {
        "blue_indices": blues,
        "red_indices": reds,
        "blue_meta": {
            "names": blue_names,
            "weights": blue_weights,
            "weights_by_name": dict(zip(blue_names, blue_weights)),
            "expected": dict(zip(blue_names, blue_expected)),
            "value": float(blue["value"]),
            "worst_case": float(blue["worst_case"]),
            "best_red": red_names[best_red_index] if red_names else None,
            "iterations": int(blue["iterations"]),
        },
        "red_meta": {
            "names": red_names,
            "weights": red_weights,
            "weights_by_name": dict(zip(red_names, red_weights)),
            "training_weights": red_training_weights,
            "training_weights_by_name": dict(
                zip(red_names, red_training_weights)),
            "expected": dict(zip(red_names, red_expected)),
            "value": float(red["value"]),
            "worst_case": float(red["worst_case"]),
            "best_blue": blue_names[best_blue_index] if blue_names else None,
            "iterations": int(red["iterations"]),
        },
        # Red's best response gain against the blue mixture.  Lower is better.
        "exploitability": max(0.0, -worst_blue),
        "blue_worst_case": worst_blue,
        "red_exploitability": max(0.0, -worst_red),
        "best_red": red_names[best_red_index] if red_names else None,
        "best_blue": blue_names[best_blue_index] if blue_names else None,
        "uniform_mix": mix,
    }


def two_population_report(game: MetaGame, blue_indices: Sequence[int],
                          red_indices: Sequence[int], *,
                          iterations: int = 20000, seed: int = 0,
                          uniform_mix: float = 0.0) -> dict[str, Any]:
    meta = solve_blue_red(
        game, blue_indices, red_indices, iterations=iterations, seed=seed,
        uniform_mix=uniform_mix)
    return {
        "names": game.names,
        "roles": [entry.role for entry in game.entries],
        "blue": [game.entries[i].name for i in blue_indices],
        "red": [game.entries[i].name for i in red_indices],
        "matrix": game.matrix(),
        "payoff": game.payoff(),
        "opponent_payoff": game.opponent_payoff(),
        "matches": game.matches(),
        **meta,
    }


def blue_expected_values(matrix: Sequence[Sequence[float]],
                         blue_indices: Sequence[int],
                         red_indices: Sequence[int],
                         red_weights: Sequence[float]) -> list[float]:
    return [
        sum(float(matrix[i][j]) * float(red_weights[k])
            for k, j in enumerate(red_indices))
        for i in blue_indices
    ]


def choose_blue_init(game: MetaGame, blue_indices: Sequence[int],
                     red_indices: Sequence[int],
                     red_weights: Sequence[float]) -> str | None:
    matrix = game.matrix()
    values = blue_expected_values(matrix, blue_indices, red_indices,
                                  red_weights)
    candidates = [
        (values[k], game.entries[i].generation, i)
        for k, i in enumerate(blue_indices)
        if game.entries[i].deploy_path
    ]
    if not candidates:
        return None
    _, _, index = max(candidates)
    return game.entries[index].deploy_path


def choose_blue_deploy(game: MetaGame, blue_indices: Sequence[int],
                       meta: Mapping[str, Any], mode: str
                       ) -> PopulationEntry | None:
    if mode == "last":
        for i in reversed([int(x) for x in blue_indices]):
            entry = game.entries[i]
            if entry.deploy_path:
                return entry
    expected = (meta.get("blue_meta") or {}).get("expected") or {}
    candidates = [
        (float(expected.get(game.entries[i].name, -1e18)),
         game.entries[i].generation, i)
        for i in blue_indices
        if game.entries[i].deploy_path
    ]
    if not candidates:
        return None
    _, _, index = max(candidates)
    return game.entries[index]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleConfig:
    init_weights: str | None = None
    iterations: int = 12
    fields_per_dist: int = 4
    n: int = 64
    workers: int = 20
    lr: float = 1e-4
    bc_kl_coef: float = 0.05
    opponent_fraction: float = 0.6
    opponent_penalty: float = 1.0
    ent_coef: float = 0.0001
    target_kl: float = 0.02
    value_bias: float | None = None
    allow_probe: bool = False
    device: str = "auto"
    seed: int = 20261030
    train_dists: tuple[str, ...] = ()
    dist_weights: tuple[float, ...] = ()
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class PSROConfig:
    name: str
    population: tuple[PopulationEntry, ...]
    output_dir: str
    config_path: str = ""
    rounds: int = 2
    seeds: tuple[int, ...] = field(default_factory=lambda: _seeds(20561026, 8))
    match_rounds: int = 100
    workers: int = 20
    mode: str = "sample"
    policy_seed_base: int = 20461026
    nash_iterations: int = 20000
    nash_seed: int = 20261027
    uniform_mix: float = 0.15
    deploy_path: str = "weights/psro/psro_policy.json"
    deploy_policy: str = "last"          # last | meta_best
    target_exploitability: float = 0.0
    oracle: OracleConfig = field(default_factory=OracleConfig)


def _resolve_path(value: str, config_dir: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    root_path = ROOT / path
    if root_path.exists():
        return str(root_path)
    config_path = config_dir / path
    if config_path.exists():
        return str(config_path)
    return str(root_path)


def _normalise_entry_paths(entry: Mapping[str, Any],
                           config_dir: Path) -> dict[str, Any]:
    out = dict(entry)
    if isinstance(out.get("entry"), Mapping):
        out["entry"] = _normalise_entry_paths(out["entry"], config_dir)
    if isinstance(out.get("path"), str):
        out["path"] = _resolve_path(out["path"], config_dir)
    params = out.get("params")
    if isinstance(params, Mapping):
        params = dict(params)
        for key in _PATH_KEYS:
            if isinstance(params.get(key), str):
                params[key] = _resolve_path(params[key], config_dir)
        out["params"] = params
    spec = out.get("spec")
    if isinstance(spec, Mapping):
        spec = dict(spec)
        spec_params = spec.get("params")
        if isinstance(spec_params, Mapping):
            spec_params = dict(spec_params)
            for key in _PATH_KEYS:
                if isinstance(spec_params.get(key), str):
                    spec_params[key] = _resolve_path(spec_params[key],
                                                     config_dir)
            spec["params"] = spec_params
        out["spec"] = spec
    return out


def load_config(path: str | Path) -> PSROConfig:
    path = Path(path)
    if not path.is_absolute():
        probe = ROOT / path
        path = probe if probe.exists() else path
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"PSRO config must be a mapping: {path}")
    config_dir = path.parent
    rows = data.get("population")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"PSRO config has no population: {path}")
    population = []
    for row in rows:
        item = dict(row)
        item = _normalise_entry_paths(item, config_dir)
        population.append(PopulationEntry.from_dict(item))
    names = [entry.name for entry in population]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate population names in {path}")

    cross = data.get("cross_play") or {}
    default_seeds = _seeds(
        int(cross.get("seed_base", 20561026)),
        int(cross.get("seeds", 8))
        if not isinstance(cross.get("seeds"), list)
        else len(cross["seeds"]))
    seeds = (tuple(int(x) for x in cross["seeds"])
             if isinstance(cross.get("seeds"), list) else default_seeds)
    meta = data.get("meta") or {}
    oracle_raw = data.get("oracle") or {}
    oracle_raw = dict(oracle_raw)
    if isinstance(oracle_raw.get("init_weights"), str):
        oracle_raw["init_weights"] = _resolve_path(
            oracle_raw["init_weights"], config_dir)
    extra = oracle_raw.get("extra_args") or ()
    if isinstance(extra, str):
        extra = shlex.split(extra)
    oracle_raw["extra_args"] = tuple(str(x) for x in extra)
    for key in ("train_dists", "dist_weights"):
        if key in oracle_raw and isinstance(oracle_raw[key], (list, tuple)):
            oracle_raw[key] = tuple(oracle_raw[key])
    output_dir = str(data.get("output_dir")
                     or result_path(f"psro_{data.get('name') or path.stem}"))
    output_dir = _resolve_path(output_dir, config_dir)
    deploy_path = str(data.get("deploy_path") or "weights/psro/psro_policy.json")
    deploy_path = _resolve_path(deploy_path, config_dir)
    return PSROConfig(
        name=str(data.get("name") or path.stem),
        population=tuple(population),
        output_dir=output_dir,
        config_path=str(path.resolve()),
        rounds=int(data.get("rounds", 2)),
        seeds=seeds,
        match_rounds=int(cross.get("rounds", 100)),
        workers=int(cross.get("workers", 20)),
        mode=str(cross.get("mode", "sample")),
        policy_seed_base=int(cross.get("policy_seed_base", 20461026)),
        nash_iterations=int(meta.get("iterations", 20000)),
        nash_seed=int(meta.get("seed", 20261027)),
        uniform_mix=float(meta.get("uniform_mix", 0.15)),
        deploy_path=deploy_path,
        deploy_policy=str(data.get("deploy_policy", "last")),
        target_exploitability=float(data.get("target_exploitability", 0.0)),
        oracle=OracleConfig(**oracle_raw),
    )


# ---------------------------------------------------------------------------
# Best-response oracle
# ---------------------------------------------------------------------------


def oracle_command(config: PSROConfig, *, manifest: str, out: str,
                   checkpoint: str, history: str, seed: int,
                   init_weights: str | None) -> list[str]:
    """Build the ``ipd train rl`` command for one best response."""
    oracle = config.oracle
    args = [
        sys.executable, "-m", "ipd", "train", "rl",
        "--opponent-pool", manifest,
        "--opponent-fraction", str(oracle.opponent_fraction),
        "--opponent-penalty", str(oracle.opponent_penalty),
        "--fields-per-dist", str(oracle.fields_per_dist),
        "--n", str(oracle.n),
        "--workers", str(oracle.workers),
        "--iterations", str(oracle.iterations),
        "--lr", str(oracle.lr),
        "--bc-kl-coef", str(oracle.bc_kl_coef),
        "--ent-coef", str(oracle.ent_coef),
        "--target-kl", str(oracle.target_kl),
        "--seed", str(seed),
        "--device", str(oracle.device),
        "--no-validation",
        "--out", out,
        "--checkpoint-out", checkpoint,
        "--history-out", history,
    ]
    if init_weights:
        args += ["--init-weights", init_weights]
    else:
        args += ["--from-scratch"]
    if oracle.train_dists:
        args += ["--train-dists", ",".join(oracle.train_dists)]
    if oracle.dist_weights:
        args += ["--dist-weights",
                 ",".join(str(x) for x in oracle.dist_weights)]
    if oracle.value_bias is not None:
        args += ["--value-bias", str(oracle.value_bias)]
    if oracle.allow_probe:
        args += ["--allow-probe"]
    args += list(oracle.extra_args)
    return args


def train_rl_oracle(context: OracleContext) -> PopulationEntry:
    config = context.config
    seed = int(config.oracle.seed) + context.round_index * 1009
    cmd = oracle_command(
        config, manifest=context.manifest_path, out=context.out_path,
        checkpoint=context.checkpoint_path, history=context.history_path,
        seed=seed, init_weights=context.init_weights)
    print("  oracle: " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not Path(context.out_path).is_file():
        raise RuntimeError(f"oracle did not write {context.out_path}")
    return PopulationEntry(
        name=f"PSRO-BR-{Path(context.out_path).stem}",
        entry={"path": str(context.out_path)},
        role="blue",
    )


@dataclass(frozen=True)
class OracleContext:
    round_index: int
    config: PSROConfig
    population: tuple[PopulationEntry, ...]
    weights: Mapping[str, float]
    manifest_path: str
    out_path: str
    checkpoint_path: str
    history_path: str
    init_weights: str | None


OracleFn = Callable[[OracleContext], PopulationEntry]


# ---------------------------------------------------------------------------
# PSRO loop
# ---------------------------------------------------------------------------


def _sha256_file(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _save_json(path: str | Path, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, out)


def _copy_deploy(policy_path: str, deploy_path: str,
                 meta: Mapping[str, Any]) -> str:
    source = Path(policy_path)
    if not source.is_file():
        raise FileNotFoundError(policy_path)
    target = Path(deploy_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        data = json.loads(target.read_text(encoding="utf-8"))
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
    data.setdefault("meta", {})
    data["meta"]["psro"] = dict(meta)
    _save_json(target, data)
    return str(target)


def run_psro(config: PSROConfig, *, oracle: OracleFn | None = None,
             dry_run: bool = False, progress: bool = True) -> dict[str, Any]:
    """Run a PSRO experiment and return the report payload.

    Blue and red populations are kept separate.  The blue meta-strategy is a
    maximin mixture over blues; the red meta-strategy is the corresponding
    maximin mixture over exploiters.  Each oracle round trains a new blue best
    response against the red training mixture, then re-solves both sides.
    """
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    game = MetaGame(
        config.population, seeds=config.seeds, rounds=config.match_rounds,
        workers=config.workers, mode=config.mode,
        policy_seed_base=config.policy_seed_base)
    blue_indices, red_indices = split_roles(game.entries)
    if progress:
        print(f"PSRO {config.name}: blue={len(blue_indices)} "
              f"red={len(red_indices)} seeds={len(config.seeds)} "
              f"rounds={config.rounds} workers={config.workers}", flush=True)
    game.evaluate([(i, j) for i in blue_indices for j in red_indices],
                  label="meta")
    initial = two_population_report(
        game, blue_indices, red_indices,
        iterations=config.nash_iterations, seed=config.nash_seed,
        uniform_mix=config.uniform_mix)
    _save_json(output / "meta_initial.json", initial)
    if progress:
        print(f"  initial blue worst-case={initial['blue_worst_case']:.3f} "
              f"exploitability={initial['exploitability']:.3f} "
              f"best_red={initial['best_red']}", flush=True)

    history: list[dict[str, Any]] = []
    oracle_fn = oracle or train_rl_oracle
    for round_index in range(1, int(config.rounds) + 1):
        before = solve_blue_red(
            game, blue_indices, red_indices,
            iterations=config.nash_iterations,
            seed=config.nash_seed + round_index,
            uniform_mix=config.uniform_mix)
        red_entries = [game.entries[j] for j in red_indices]
        manifest = str(output / f"oracle_manifest_{round_index:02d}.json")
        write_opponent_manifest(
            manifest, [entry.to_pool_entry() for entry in red_entries],
            before["red_meta"]["training_weights_by_name"],
            name=f"{config.name}-round{round_index}")
        init_weights = config.oracle.init_weights or choose_blue_init(
            game, blue_indices, red_indices,
            before["red_meta"]["training_weights"])
        stem = f"br_{round_index:02d}"
        out_path = str(output / f"{stem}.json")
        checkpoint = str(output / f"{stem}_checkpoint.json")
        history_path = str(output / f"{stem}_history.json")
        context = OracleContext(
            round_index=round_index,
            config=config,
            population=tuple(game.entries),
            weights=dict(before["red_meta"]["training_weights_by_name"]),
            manifest_path=manifest,
            out_path=out_path,
            checkpoint_path=checkpoint,
            history_path=history_path,
            init_weights=init_weights,
        )
        if dry_run:
            row = {
                "round": round_index,
                "dry_run": True,
                "before": before,
                "manifest": manifest,
                "init_weights": init_weights,
            }
            history.append(row)
            _save_json(output / f"round_{round_index:02d}.json", row)
            break
        entry = oracle_fn(context)
        if not isinstance(entry, PopulationEntry):
            entry = PopulationEntry.from_dict(entry)
        entry = replace(entry, role="blue",
                        generation=max(entry.generation, round_index),
                        parent=entry.parent or (
                            Path(init_weights).stem if init_weights else None))
        game.add(entry, opponents=red_indices, label=f"meta+{round_index}")
        blue_indices = list(blue_indices) + [len(game.entries) - 1]
        after = solve_blue_red(
            game, blue_indices, red_indices,
            iterations=config.nash_iterations,
            seed=config.nash_seed + round_index,
            uniform_mix=config.uniform_mix)
        row = {
            "round": round_index,
            "dry_run": False,
            "before": before,
            "after": after,
            "entry": entry.to_dict(),
            "manifest": manifest,
            "init_weights": init_weights,
            "exploitability_delta": (after["exploitability"]
                                     - before["exploitability"]),
        }
        history.append(row)
        _save_json(output / f"round_{round_index:02d}.json", row)
        save_population(
            output / "population.json", game.entries,
            meta={"round": round_index, "name": config.name,
                  "blue_weights": after["blue_meta"]["weights_by_name"],
                  "red_weights": after["red_meta"]["weights_by_name"]})
        if progress:
            print(f"  round {round_index}: added {entry.name}; "
                  f"exploitability {before['exploitability']:.3f} -> "
                  f"{after['exploitability']:.3f}", flush=True)
        if after["exploitability"] <= config.target_exploitability:
            break

    final = two_population_report(
        game, blue_indices, red_indices,
        iterations=config.nash_iterations, seed=config.nash_seed,
        uniform_mix=config.uniform_mix)
    save_population(
        output / "population.json", game.entries,
        meta={"round": len(history), "name": config.name,
              "blue_weights": final["blue_meta"]["weights_by_name"],
              "red_weights": final["red_meta"]["weights_by_name"]})
    deploy_meta = {
        "schema": REPORT_SCHEMA,
        "name": config.name,
        "population": [entry.to_dict() for entry in game.entries],
        "blue": final["blue"],
        "red": final["red"],
        "blue_meta": final["blue_meta"],
        "red_meta": final["red_meta"],
        "config": config.config_path,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    deploy_info: dict[str, Any] = {}
    if not dry_run:
        deploy_entry = choose_blue_deploy(game, blue_indices, final,
                                          config.deploy_policy)
        if deploy_entry is not None and deploy_entry.deploy_path:
            deploy_meta["deploy_entry"] = deploy_entry.name
            deploy_meta["deploy_source"] = deploy_entry.deploy_path
            try:
                target = _copy_deploy(deploy_entry.deploy_path,
                                      config.deploy_path, deploy_meta)
                deploy_info = {
                    "entry": deploy_entry.name,
                    "source": deploy_entry.deploy_path,
                    "path": target,
                    "sha256": _sha256_file(target),
                }
            except Exception as exc:  # pragma: no cover - filesystem edge
                deploy_info = {
                    "entry": deploy_entry.name,
                    "source": deploy_entry.deploy_path,
                    "error": str(exc),
                }
        _save_json(Path(config.deploy_path).parent / "meta_strategy.json",
                   deploy_meta)
    report = {
        "schema": REPORT_SCHEMA,
        "name": config.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "config": config.config_path,
        "config_sha256": _sha256_file(config.config_path),
        "population": [entry.to_dict() for entry in game.entries],
        "initial": initial,
        "final": final,
        "rounds": history,
        "deploy": deploy_info,
        "dry_run": bool(dry_run),
        "meta": {
            "seeds": list(config.seeds),
            "match_rounds": config.match_rounds,
            "policy_seed_base": config.policy_seed_base,
            "nash_iterations": config.nash_iterations,
            "uniform_mix": config.uniform_mix,
            "deploy_policy": config.deploy_policy,
        },
    }
    _save_json(output / "report.json", report)
    (output / "README.md").write_text(build_markdown(report),
                                      encoding="utf-8")
    if progress:
        print(f"final blue worst-case={final['blue_worst_case']:.3f} "
              f"exploitability={final['exploitability']:.3f} "
              f"best_red={final['best_red']}", flush=True)
        if deploy_info:
            print(f"deployed {deploy_info.get('entry')} -> "
                  f"{deploy_info.get('path')} "
                  f"({deploy_info.get('sha256', '')[:12]})", flush=True)
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(report: Mapping[str, Any]) -> str:
    final = report.get("final") or {}
    blue_meta = final.get("blue_meta") or {}
    red_meta = final.get("red_meta") or {}
    matrix = final.get("matrix") or []
    names = final.get("names") or []
    index = {name: i for i, name in enumerate(names)}
    blue_names = final.get("blue") or []
    red_names = final.get("red") or []
    lines = [
        f"# PSRO — {report.get('name', 'experiment')}",
        "",
        f"- **Status:** experimental; not promoted",
        f"- **Created:** {report.get('created', '')}",
        f"- **Config:** `{report.get('config', '')}`",
        f"- **Initial exploitability:** "
        f"**{_fmt((report.get('initial') or {}).get('exploitability'))}**",
        f"- **Final exploitability:** **{_fmt(final.get('exploitability'))}**",
        f"- **Final blue worst-case:** {_fmt(final.get('blue_worst_case'))}",
        f"- **Best red response:** `{final.get('best_red', '-')}`",
        f"- **Deploy:** `{(report.get('deploy') or {}).get('path', '-')}` "
        f"({(report.get('deploy') or {}).get('entry', '-')})",
        "",
        "## Blue population",
        "",
        "| Blue | Gen | Meta weight | Expected vs red | Worst case vs red | Standalone JSON |",
        "|---|---:|---:|---:|---:|---|",
    ]
    population = {row.get("name"): row for row in report.get("population") or []}
    blue_weights = blue_meta.get("weights_by_name") or {}
    blue_expected = blue_meta.get("expected") or {}
    for name in blue_names:
        i = index.get(name)
        worst = (min(float(matrix[i][index[red]]) for red in red_names)
                 if i is not None and red_names else None)
        row = population.get(name) or {}
        lines.append(
            f"| {name} | {row.get('generation', 0)} | "
            f"{_fmt(blue_weights.get(name), 4)} | "
            f"{_fmt(blue_expected.get(name), 2)} | {_fmt(worst, 2)} | "
            f"{'yes' if (row.get('entry') or {}).get('path') else '-'} |")
    lines += [
        "",
        "A `-` in **Standalone JSON** means a rule candidate that is deployed",
        "through the meta-policy rather than a single learned JSON file.",
        "",
        "## Red population",
        "",
        "| Red | Meta weight | Training weight | Expected vs blue |",
        "|---|---:|---:|---:|",
    ]
    red_weights = red_meta.get("weights_by_name") or {}
    red_training = red_meta.get("training_weights_by_name") or {}
    red_expected = red_meta.get("expected") or {}
    for name in red_names:
        lines.append(
            f"| {name} | {_fmt(red_weights.get(name), 4)} | "
            f"{_fmt(red_training.get(name), 4)} | "
            f"{_fmt(red_expected.get(name), 2)} |")
    lines += [
        "",
        "## Rounds",
        "",
        "| Round | Before exploitability | After exploitability | Delta | Added |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rounds") or []:
        before = (row.get("before") or {}).get("exploitability")
        after = (row.get("after") or {}).get("exploitability")
        added = (row.get("entry") or {}).get("name", "dry-run")
        lines.append(
            f"| {row.get('round')} | {_fmt(before)} | {_fmt(after)} | "
            f"{_fmt(row.get('exploitability_delta'))} | {added} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- The meta-game is estimated from a finite seed set, not solved exactly.",
        "- The best-response oracle is the existing PPO trainer; it is not a",
        "  proof of a true best response.",
        "- Blue exploitability is the red best-response gain against the maximin",
        "  blue mixture.  Lower is better; a positive delta means the new",
        "  best-response policy did not enter the maximin support.",
        "- Red-v5 / Red-v6 and Protocol v5 `test_v8` remain held out.  A PSRO",
        "  candidate is only promotable after the normal dev/test and C++ checks.",
        "- The main weights (`weights/rl_policy_weights.json`) are unchanged.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Policy-Space Response Oracles (PSRO) for IPD.")
    ap.add_argument("--config", default="configs/psro/psro_20261027.json")
    ap.add_argument("--out", default=None,
                    help="override output directory")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the number of cross-play seeds")
    ap.add_argument("--nash-iterations", type=int, default=None)
    ap.add_argument("--uniform-mix", type=float, default=None)
    ap.add_argument("--oracle-iterations", type=int, default=None)
    ap.add_argument("--oracle-fields", type=int, default=None)
    ap.add_argument("--oracle-n", type=int, default=None)
    ap.add_argument("--oracle-workers", type=int, default=None)
    ap.add_argument("--device", default=None,
                    choices=("auto", "cuda", "npu", "mps", "cpu"))
    ap.add_argument("--deploy-path", default=None)
    ap.add_argument("--deploy-policy", default=None,
                    choices=("last", "meta_best"))
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate the initial meta-game without training")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    changes: dict[str, Any] = {}
    if args.out:
        changes["output_dir"] = str(Path(args.out).expanduser())
    if args.rounds is not None:
        changes["rounds"] = int(args.rounds)
    if args.workers is not None:
        changes["workers"] = int(args.workers)
    if args.seeds is not None:
        changes["seeds"] = _seeds(config.seeds[0] if config.seeds
                                  else 20561026, args.seeds)
    if args.nash_iterations is not None:
        changes["nash_iterations"] = int(args.nash_iterations)
    if args.uniform_mix is not None:
        changes["uniform_mix"] = float(args.uniform_mix)
    if args.deploy_path:
        changes["deploy_path"] = str(Path(args.deploy_path).expanduser())
    if args.deploy_policy:
        changes["deploy_policy"] = args.deploy_policy
    oracle_changes: dict[str, Any] = {}
    if args.oracle_iterations is not None:
        oracle_changes["iterations"] = int(args.oracle_iterations)
    if args.oracle_fields is not None:
        oracle_changes["fields_per_dist"] = int(args.oracle_fields)
    if args.oracle_n is not None:
        oracle_changes["n"] = int(args.oracle_n)
    if args.oracle_workers is not None:
        oracle_changes["workers"] = int(args.oracle_workers)
    if args.device is not None:
        oracle_changes["device"] = args.device
    if oracle_changes:
        changes["oracle"] = replace(config.oracle, **oracle_changes)
    if changes:
        config = replace(config, **changes)
    report = run_psro(config, dry_run=args.dry_run)
    print(f"written {Path(config.output_dir) / 'report.json'}")
    print(f"written {Path(config.output_dir) / 'README.md'}")
    return 0


__all__ = [
    "POPULATION_SCHEMA", "REPORT_SCHEMA", "CONFIG_SCHEMA",
    "PopulationEntry", "MetaGame", "OracleConfig", "PSROConfig",
    "OracleContext", "OracleFn", "cross_play_pair_task", "load_population",
    "make_scratch_policy", "save_population", "load_config", "oracle_command",
    "train_rl_oracle",
    "run_psro", "build_markdown", "main",
]
