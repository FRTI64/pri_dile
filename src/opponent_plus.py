"""Backward-compatible opponent-bank entry point."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ipd.core.opponents_plus import add  # noqa: F401
from ipd.policies.opponents import (  # noqa: F401
    ALL_SPECS, PLUS, SMART, SMART_BY_NAME, TUNED_SMART, SmartAdapter,
)
