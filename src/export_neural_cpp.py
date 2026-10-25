"""Backward-compatible MLP C++ exporter."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ipd.tools.export_cpp import array, fmt, matrix, tensor  # noqa: F401
from ipd.tools.export_neural_cpp import main


if __name__ == "__main__":
    raise SystemExit(main())
