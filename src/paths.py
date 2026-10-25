"""Filesystem layout for the 100-day IPD project.

All paths are resolved relative to the repository root, so source files can be
executed from any working directory::

    python3 src/blind_test.py

Keeping the layout in one module avoids scattering ``"../results/..."``
literals across the scripts.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT / "src"
DOCS_DIR = ROOT / "docs"
CPP_DIR = ROOT / "cpp"
CONFIG_DIR = ROOT / "configs"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"


def config_path(name: str) -> str:
    """Return the absolute path of a file in ``configs/``."""
    return str(CONFIG_DIR / name)


def weights_path(name: str) -> str:
    """Return the absolute path of a file in ``weights/``."""
    return str(WEIGHTS_DIR / name)


def result_path(name: str) -> str:
    """Return the absolute path of a file in ``results/``."""
    return str(RESULTS_DIR / name)


def cpp_path(name: str) -> str:
    """Return the absolute path of a file in ``cpp/``."""
    return str(CPP_DIR / name)
