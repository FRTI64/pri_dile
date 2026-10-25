"""Backward-compatible nine-suite benchmark entry point.

New command: ``python3 -m ipd suite``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ipd.evaluation.suite import main as _main


def _with_defaults(argv):
    legacy = ("final-base,final-tuned,resp3-base,resp3-tuned,"
              "bayes-tuned,hybrid,triple-majority,harvest45")
    if "--candidates" not in argv:
        argv = ["--candidates", legacy, *argv]
    return argv


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    return _main(_with_defaults(args))


if __name__ == "__main__":
    raise SystemExit(main())
