"""Backward-compatible entry point for Triple and its validation search."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ipd.policies.rules import Triple  # noqa: F401
from ipd.training.search.triple import (  # noqa: F401
    eval_task, fit, load_cfgs, main, make,
)


if __name__ == "__main__":
    raise SystemExit(main())
