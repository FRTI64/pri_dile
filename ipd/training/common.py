"""Shared training utilities: CLI conventions, seeds and artifact metadata.

Every ``ipd.training`` command follows the same conventions:

* ``--seed``              deterministic seed
* ``--workers``           parallel rollout workers
* ``--n``                 opponents per field (where applicable)
* ``--out``               main artifact (weights/config/results JSON)
* ``--history-out``       optional training-curve / search-history JSON
* ``--device auto``       learned-policy trainers only
* ``main(argv=None)``     importable entry point

Training code may ignore options that do not apply to a parameter search, but
the names stay consistent.
"""
from __future__ import annotations

if __package__ in (None, ""):  # allow: python ipd/training/<file>.py
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break


import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from ipd.paths import ensure_parent


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str | os.PathLike[str], obj: Any) -> None:
    """Atomically write JSON, creating parent directories as needed."""
    target = ensure_parent(path)
    tmp = str(target) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def seed_map(seed: int, dists: Sequence[str], fields: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    return {d: [rng.randrange(1 << 30) for _ in range(fields)] for d in dists}


def add_common_args(parser, *, seed: int = 20261025, workers: int = 20,
                    n: int = 60, out: str | None = None,
                    history_out: str | None = None,
                    device: bool = False) -> None:
    """Add the standard training flags to an ``ArgumentParser``."""
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--workers", type=int, default=workers)
    parser.add_argument("--n", type=int, default=n)
    if out is not None:
        parser.add_argument("--out", default=out)
    if history_out is not None:
        parser.add_argument("--history-out", default=history_out)
    if device:
        parser.add_argument("--device", default="auto",
                            choices=("auto", "cuda", "npu", "mps", "cpu"))


@runtime_checkable
class TrainingTask(Protocol):
    """Every training entry point accepts an optional argv list."""

    def main(self, argv: Sequence[str] | None = None) -> int | None:
        ...


@dataclass
class TrainResult:
    """Uniform return value for training/search entry points."""

    task: str
    out: str | None = None
    history_out: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    created: str = field(default_factory=lambda: time.strftime(
        "%Y-%m-%dT%H:%M:%S"))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "ipd-training-result-v1",
            "task": self.task,
            "out": self.out,
            "history_out": self.history_out,
            "metrics": self.metrics,
            "seconds": round(self.seconds, 2),
            "created": self.created,
        }

    def write_manifest(self, path: str | os.PathLike[str]) -> None:
        save_json(path, self.manifest())


def print_result(result: TrainResult) -> None:
    if result.out:
        print(f"written {result.out}")
    if result.history_out:
        print(f"written {result.history_out}")
