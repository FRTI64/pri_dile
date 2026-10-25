"""Training entry points and shared CLI conventions."""
from __future__ import annotations

from .common import (
    TrainResult, TrainingTask, add_common_args, load_json, print_result,
    save_json, seed_map,
)
from .cli import ALIASES, TASKS, main

__all__ = [
    "TrainResult", "TrainingTask", "add_common_args", "load_json",
    "save_json", "seed_map",
    "print_result", "TASKS", "ALIASES", "main",
]
