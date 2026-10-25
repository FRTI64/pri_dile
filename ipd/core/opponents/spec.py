"""Opponent type metadata used by the distribution samplers."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from ..primitives import Strategy


@dataclass(frozen=True)
class TypeSpec:
    name: str
    family: str
    weight: float
    factory: Callable[[random.Random], Strategy]


__all__ = ["TypeSpec"]
