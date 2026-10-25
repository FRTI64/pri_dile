"""Opponent wrappers built from our own policies."""
from __future__ import annotations

from .bank import ALL_BY_NAME, ALL_SPECS, PLUS
from .smart import SMART, SMART_BY_NAME, SmartAdapter
from .tuned_smart import TUNED_SMART

__all__ = [
    "SmartAdapter", "SMART", "SMART_BY_NAME", "TUNED_SMART",
    "PLUS", "ALL_SPECS", "ALL_BY_NAME",
]
