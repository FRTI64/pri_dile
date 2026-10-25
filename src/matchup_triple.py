"""Backward-compatible Triple matchup entry point.

New command: ``python3 -m ipd matchup``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ipd.evaluation.matchup import main as _main


def _with_defaults(argv):
    candidates = ("final-tuned,resp3-tuned,bayes-tuned,hybrid,"
                  "triple-majority")
    if "--seed" not in argv:
        argv = ["--seed", "900000", *argv]
    if "--out" not in argv:
        from ipd.paths import result_path
        argv = [*argv, "--out", result_path("matchup_triple_results.csv")]
    return ["--candidates", candidates, *argv]


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    return _main(_with_defaults(args))


if __name__ == "__main__":
    raise SystemExit(main())
