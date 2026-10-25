"""Backward-compatible entry point.

Canonical implementation: :mod:`ipd.experiments.grim`.
"""
from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ipd.experiments.grim import *  # noqa: F401,F403


def _run(argv=None) -> None:
    warnings.filterwarnings(
        "ignore", message=".*found in sys.modules.*", category=RuntimeWarning)
    if argv is not None:
        sys.argv = [sys.argv[0], *map(str, argv)]
    runpy.run_module("ipd.experiments.grim", run_name="__main__")


if __name__ == "__main__":
    _run()
