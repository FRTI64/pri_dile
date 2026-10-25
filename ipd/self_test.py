#!/usr/bin/env python3
"""Fast policy-interface self-test (no PyTorch, no evaluation fields)."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .core.simulator import C, D, TFT, Grim, payoff
from .policies.registry import CORE_NAMES, DISPLAY, candidate_specs


def _play(policy, opponent_factory, rounds: int = 100):
    policy.reset()
    opponent = opponent_factory()
    mine: list[int] = []
    theirs: list[int] = []
    score = 0
    for t in range(rounds):
        x = policy.act(t, mine, theirs)
        y = opponent.act(t, theirs, mine)
        mine.append(x)
        theirs.append(y)
        score += payoff(x, y)
    return mine, score


def run(candidate: str, rounds: int = 100) -> dict:
    spec = candidate_specs([candidate])[0]
    policy = spec.build()
    mine_tft, score_tft = _play(policy, TFT, rounds)
    mine_grim, score_grim = _play(policy, Grim, rounds)
    # GapStart is an experimental search skeleton that may probe immediately;
    # all submission candidates must still start with C and defect on day 100.
    requires_boundaries = candidate != "gap-start"
    ok = (mine_tft[0] == C and mine_tft[-1] == D) if requires_boundaries else True
    return {
        "candidate": candidate,
        "name": spec.name,
        "first": mine_tft[0],
        "last": mine_tft[-1],
        "defections_vs_tft": mine_tft.count(D),
        "score_vs_tft": score_tft,
        "score_vs_grim": score_grim,
        "ok": ok,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Policy interface self-test.")
    ap.add_argument("--candidates", default="core",
                    help="comma-separated names or core|base|suite|learned|all")
    ap.add_argument("--rounds", type=int, default=100)
    args = ap.parse_args(argv)

    from .policies.registry import resolve_names
    names = resolve_names(args.candidates, default=CORE_NAMES)
    print(f"{'candidate':20}{'first':>6}{'last':>6}{'D vs TFT':>10}"
          f"{'vs TFT':>8}{'vs Grim':>9}{'ok':>5}")
    failed = False
    for name in names:
        row = run(name, args.rounds)
        failed |= not row["ok"]
        print(f"{row['name']:20}{row['first']:6d}{row['last']:6d}"
              f"{row['defections_vs_tft']:10d}{row['score_vs_tft']:8d}"
              f"{row['score_vs_grim']:9d}{'yes' if row['ok'] else 'NO':>5}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
