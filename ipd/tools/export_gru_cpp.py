#!/usr/bin/env python3
"""Legacy name for :mod:`ipd.tools.export_cpp` (GRU exporter)."""
from __future__ import annotations

import sys
from typing import Sequence

from ipd.tools.export_cpp import export_gru, main as _export_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return _export_main(["--arch", "gru", *args])


if __name__ == "__main__":
    raise SystemExit(main())
