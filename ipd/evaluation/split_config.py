"""Configuration for split evaluation runs.

A run config gives every candidate its own evaluation parameters.  This is
useful because the recurrent GRU policy is much slower than the MLP/RL policies
and because quick experiments may want fewer fields for one model without
changing the rest.  The loader is deliberately JSON-only so configs can be
committed next to results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..paths import ROOT
from ..policies.registry import CandidateSpec
from .distributions import DISTS, EVAL_DISTS

BENCHMARK_DEFAULTS = {
    "fields": 20,
    "n": 299,
    "seed": 20261026,
    "workers": 20,
    "rounds": 100,
    "dists": list(DISTS),
}
BLIND_DEFAULTS = {
    "fields": 10,
    "n": 299,
    "seed": 20261025,
    "workers": 10,
    "rounds": 100,
    "dists": list(DISTS),
}
_PATH_KEYS = ("weights", "config", "checkpoint")


@dataclass(frozen=True)
class EvalParams:
    fields: int
    n: int
    seed: int
    workers: int
    rounds: int
    dists: tuple[str, ...] = DISTS

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvalParams":
        dists = tuple(str(x) for x in data.get("dists", DISTS))
        params = cls(
            fields=int(data["fields"]),
            n=int(data["n"]),
            seed=int(data["seed"]),
            workers=int(data["workers"]),
            rounds=int(data.get("rounds", 100)),
            dists=dists,
        )
        unknown = [d for d in params.dists if d not in EVAL_DISTS]
        if unknown:
            raise ValueError(f"unknown distributions: {unknown}")
        if params.fields <= 0 or params.n <= 0:
            raise ValueError("fields and n must be positive")
        return params

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": self.fields,
            "n": self.n,
            "seed": self.seed,
            "workers": self.workers,
            "rounds": self.rounds,
            "dists": list(self.dists),
        }

    def with_overrides(self, **overrides: Any) -> "EvalParams":
        clean = {key: value for key, value in overrides.items()
                 if value is not None}
        if "dists" in clean:
            clean["dists"] = tuple(clean["dists"])
        return replace(self, **clean)


@dataclass(frozen=True)
class CandidateRun:
    spec: CandidateSpec
    benchmark: EvalParams
    blind: EvalParams
    tags: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.spec.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "blind": self.blind.to_dict(),
            "tags": list(self.tags),
            "note": self.note,
        }


@dataclass(frozen=True)
class SplitConfig:
    name: str
    path: Path
    candidates: tuple[CandidateRun, ...]
    benchmark_defaults: EvalParams
    blind_defaults: EvalParams
    note: str = ""

    def selected(self, only: Iterable[str] | None = None) -> list[CandidateRun]:
        if not only:
            return list(self.candidates)
        wanted = {str(x).strip().lower() for x in only if str(x).strip()}
        rows = [row for row in self.candidates
                if row.spec.name.lower() in wanted
                or row.spec.kind.lower() in wanted]
        missing = wanted - {row.spec.name.lower() for row in rows}
        missing -= {row.spec.kind.lower() for row in rows}
        if missing:
            raise ValueError(f"unknown candidates: {sorted(missing)}")
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipd-eval-split-config-v1",
            "name": self.name,
            "note": self.note,
            "defaults": {
                "benchmark": self.benchmark_defaults.to_dict(),
                "blind": self.blind_defaults.to_dict(),
            },
            "candidates": [row.to_dict() for row in self.candidates],
        }


def _merge(base: Mapping[str, Any], override: Mapping[str, Any] | None):
    return {**base, **dict(override or {})}


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


def _normalise_params(params: Mapping[str, Any], config_dir: Path):
    out = dict(params)
    for key in _PATH_KEYS:
        if isinstance(out.get(key), str):
            out[key] = _resolve_path(out[key], config_dir)
    return out


def load_split_config(path: str | Path) -> SplitConfig:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    defaults = data.get("defaults") or {}
    benchmark_defaults = EvalParams.from_mapping(
        _merge(BENCHMARK_DEFAULTS, defaults.get("benchmark")))
    blind_defaults = EvalParams.from_mapping(
        _merge(BLIND_DEFAULTS, defaults.get("blind")))

    rows: list[CandidateRun] = []
    for entry in data.get("candidates") or []:
        if entry.get("enabled", True) is False:
            continue
        if not entry.get("kind") or not entry.get("name"):
            raise ValueError("every candidate needs name and kind")
        params = _normalise_params(entry.get("params") or {}, path.parent)
        benchmark = benchmark_defaults.with_overrides(
            **_merge({}, entry.get("benchmark")))
        blind = blind_defaults.with_overrides(
            **_merge({}, entry.get("blind")))
        rows.append(CandidateRun(
            spec=CandidateSpec(
                kind=str(entry["kind"]),
                name=str(entry["name"]),
                params=params,
            ),
            benchmark=benchmark,
            blind=blind,
            tags=tuple(str(x) for x in entry.get("tags") or ()),
            note=str(entry.get("note", "")),
        ))
    if not rows:
        raise ValueError(f"no enabled candidates in {path}")
    return SplitConfig(
        name=str(data.get("name") or path.stem),
        path=path,
        candidates=tuple(rows),
        benchmark_defaults=benchmark_defaults,
        blind_defaults=blind_defaults,
        note=str(data.get("note", "")),
    )


def override_params(rows: Iterable[CandidateRun], *,
                    fields: int | None = None, n: int | None = None,
                    seed: int | None = None, workers: int | None = None,
                    blind_fields: int | None = None,
                    blind_n: int | None = None,
                    blind_seed: int | None = None,
                    blind_workers: int | None = None,
                    ) -> list[CandidateRun]:
    """Apply global command-line overrides to every candidate."""
    out: list[CandidateRun] = []
    for row in rows:
        benchmark = row.benchmark.with_overrides(
            fields=fields, n=n, seed=seed, workers=workers)
        blind = row.blind.with_overrides(
            fields=blind_fields, n=blind_n, seed=blind_seed,
            workers=blind_workers)
        out.append(replace(row, benchmark=benchmark, blind=blind))
    return out


__all__ = [
    "BENCHMARK_DEFAULTS", "BLIND_DEFAULTS", "EvalParams", "CandidateRun",
    "SplitConfig", "load_split_config", "override_params",
]
