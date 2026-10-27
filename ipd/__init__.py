"""Canonical 100-day iterated prisoner's dilemma package.

The package is deliberately split by responsibility:

``ipd.core``
    Simulation primitives, opponent banks and hand-written state machines.
``ipd.policies``
    Submission policies (the files that take part in a tournament) and their
    unified interface/registry.
``ipd.evaluation``
    Benchmark, blind-test, suite and matchup harnesses.  Evaluation code only
    talks to policies through :mod:`ipd.policies.base`.
``ipd.training``
    Training entry points for learned policies and parameter searches.
``ipd.tools``
    Deployment helpers such as C++ weight exporters.

All code and command-line entry points live in this package.
"""
from __future__ import annotations

__version__ = "2.3.0"

from .core.simulator import C, D, Strategy, payoff, play_match  # noqa: F401
from .policies.base import Policy, PolicyAdapter, ensure_policy  # noqa: F401

__all__ = [
    "C", "D", "Strategy", "payoff", "play_match",
    "Policy", "PolicyAdapter", "ensure_policy", "__version__",
]
