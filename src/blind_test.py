"""Backward-compatible blind test entry point.

New command: ``python3 -m ipd blind``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ipd.evaluation.blind import main as _main


def _with_defaults(argv):
    if not any(x in argv for x in ("--candidates", "--arch", "--neural")):
        argv = ["--candidates", "classic", *argv]
    return argv


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    return _main(_with_defaults(args))


if __name__ == "__main__":
    raise SystemExit(main())
