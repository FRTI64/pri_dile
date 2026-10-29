"""Unified training dispatcher.

Usage::

    python3 -m ipd train --list
    python3 -m ipd train neural --device auto --epochs 40
    python3 -m ipd train rl --device auto --iterations 100
    python3 -m ipd train multiobj --workers 24
"""
from __future__ import annotations

if __package__ in (None, ""):  # allow: python ipd/training/<file>.py
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break


import importlib
import sys
from typing import Sequence

TASKS = {
    "neural": ("ipd.training.neural", "MLP behaviour cloning + DAgger"),
    "gru": ("ipd.training.gru", "GRU-Net behaviour cloning + DAgger"),
    "rl": ("ipd.training.rl", "PPO reinforcement learning"),
    "expand": ("ipd.training.expand", "expand MLP hidden width (net2wider)"),
    "psro": ("ipd.training.psro", "policy-space response oracles"),
    "opponent-model": ("ipd.training.opponent_model",
                       "offline opponent classifier / gate"),
    "selfplay": ("ipd.training.selfplay.train",
                 "asymmetric self-play PPO (blue/red GRU)"),
    "selfplay-bc": ("ipd.training.selfplay.bc",
                    "optional BC warm start for self-play GRU"),
    "selfplay-compare": ("ipd.training.selfplay.compare",
                         "fair held-out comparison of self-play/BC/RL"),
    "selfplay-red-eval": ("ipd.training.selfplay.red_eval",
                          "evaluate a frozen red best response"),
    "adversarial": ("ipd.training.adversarial",
                    "constrained adversarial exploitation search"),
    "multiobj": ("ipd.training.search.multiobj",
                 "four-distribution rule parameter search"),
    "bayes": ("ipd.training.search.bayes_priors",
              "BayesPlus prior-scale search"),
    "joint": ("ipd.training.search.joint", "Final + Responsive3 hybrid search"),
    "wide": ("ipd.training.search.wide", "wide-bank continuous parameter search"),
    "triple": ("ipd.training.search.triple", "Triple voting-mode validation"),
    "p1": ("ipd.training.search.p1_parameter", "P1 threshold search"),
    "p1-grim": ("ipd.training.search.p1_grim", "P1 high-Grim threshold search"),
    "p2-gp": ("ipd.training.search.p2_gp", "P2 genetic-programming trees"),
    "p2-robust": ("ipd.training.search.p2_robust", "P2 robust GP"),
    "p2-fsm": ("ipd.training.search.p2_fsm", "P2 finite-state machines"),
    "p2-skeleton": ("ipd.training.search.p2_skeleton", "P2 skeleton search"),
}

ALIASES = {
    "mlp": "neural",
    "bc": "neural",
    "adv": "adversarial",
    "rules": "multiobj",
    "bayes-priors": "bayes",
    "final-resp": "joint",
}

TASK_ALIASES = ALIASES


def _print_tasks() -> None:
    print("Training tasks:")
    for name, (_, description) in TASKS.items():
        print(f"  {name:12} {description}")
    print("\nAliases: " + ", ".join(f"{k}->{v}" for k, v in ALIASES.items()))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help", "--list", "list"):
        _print_tasks()
        return 0
    task = ALIASES.get(args[0].lower().replace("_", "-"), args[0].lower())
    if task not in TASKS:
        print(f"unknown training task: {args[0]!r}\n", file=sys.stderr)
        _print_tasks()
        return 2
    module_name, _ = TASKS[task]
    module = importlib.import_module(module_name)
    result = module.main(args[1:])
    return int(result or 0)


__all__ = ["TASKS", "ALIASES", "TASK_ALIASES", "main"]
