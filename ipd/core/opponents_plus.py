"""Compatibility shim for the historical extra-opponent module.

The implementation now lives in :mod:`ipd.core.opponents.plus`.
"""
from __future__ import annotations

from .opponents.plus import ALL_SPECS, PLUS, add

__all__ = ["PLUS", "ALL_SPECS", "add"]
