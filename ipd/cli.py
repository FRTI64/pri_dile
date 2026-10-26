"""Unified command-line interface.

Run from the repository root (no installation required)::

    python3 -m ipd list-policies
    python3 -m ipd self-test --candidates all
    python3 -m ipd benchmark --candidates core,neural --fields 8
    python3 -m ipd blind --candidates core --fields 10
    python3 -m ipd suite --fields 12
    python3 -m ipd matchup --candidates base --reps 8
    python3 -m ipd train neural --device auto
    python3 -m ipd export --arch mlp
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from .paths import CONFIG_DIR, CPP_DIR, RESULTS_DIR, ROOT, WEIGHTS_DIR
from .policies.registry import ALL_NAMES, DISPLAY, SET_ALIASES

COMMANDS = {
    "benchmark": "four-distribution benchmark",
    "blind": "blind ranking on unseen fields",
    "eval-split": "per-candidate parameters + per-opponent result files",
    "suite": "nine family-weighted suites",
    "matchup": "per-opponent-type matchup matrix",
    "league": "E0 cross-play / Nash / exploitability harness",
    "meta": "P3 red maximin weights / opponent manifest",
    "planner-eval": "P2 C1 belief planner offline evaluation",
    "moe-eval": "P2 C3 three-way gate + MoE offline evaluation",
    "train": "train a learned policy or search rule parameters",
    "export": "export weights to C++ headers",
    "self-test": "fast policy-interface check",
    "test": "run the unit test suite",
    "list-policies": "list registered candidate names",
    "paths": "show project directories",
}


def _list_policies() -> int:
    print(f"ipd {__version__}\n")
    print(f"{'name':20}{'display':20}sets")
    for name in ALL_NAMES:
        sets = [key for key, members in SET_ALIASES.items() if name in members]
        print(f"{name:20}{DISPLAY[name]:20}{','.join(sets)}")
    print("\nSet aliases: " + ", ".join(SET_ALIASES))
    return 0


def _run_tests(argv: Sequence[str]) -> int:
    import unittest

    pattern = argv[0] if argv else "test_*.py"
    suite = unittest.TestLoader().discover(
        str(ROOT / "tests"), pattern=pattern, top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _paths() -> int:
    for label, path in (("root", ROOT), ("configs", CONFIG_DIR),
                        ("weights", WEIGHTS_DIR), ("results", RESULTS_DIR),
                        ("cpp", CPP_DIR)):
        print(f"{label:10} {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(f"ipd {__version__} - IPD tournament toolkit\n")
        print("usage: python3 -m ipd <command> [options]\n")
        for name, description in COMMANDS.items():
            print(f"  {name:14} {description}")
        print("\nRun 'python3 -m ipd <command> --help' for options.")
        return 0

    if args[0] in ("-V", "--version", "version"):
        print(__version__)
        return 0

    command, rest = args[0], args[1:]
    if command == "benchmark":
        from .evaluation.benchmark import main as run
    elif command == "blind":
        from .evaluation.blind import main as run
    elif command in ("eval-split", "split-eval"):
        from .evaluation.split import main as run
    elif command == "suite":
        from .evaluation.suite import main as run
    elif command == "matchup":
        from .evaluation.matchup import main as run
    elif command in ("league", "e0"):
        from .evaluation.league import main as run
    elif command in ("meta", "meta-weights"):
        from .training.league import main as run
    elif command in ("planner-eval", "planner"):
        from .evaluation.planner_eval import main as run
    elif command in ("moe-eval", "moe"):
        from .evaluation.moe_eval import main as run
    elif command in ("train", "training"):
        from .training.cli import main as run
    elif command in ("export", "export-cpp"):
        from .tools.export_cpp import main as run
    elif command in ("self-test", "self_test", "selftest"):
        from .self_test import main as run
    elif command in ("list-policies", "list"):
        return _list_policies()
    elif command == "paths":
        return _paths()
    elif command in ("test", "tests"):
        return _run_tests(rest)
    else:
        print(f"unknown command: {command!r}\n", file=sys.stderr)
        print("commands: " + ", ".join(COMMANDS), file=sys.stderr)
        return 2
    return int(run(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
