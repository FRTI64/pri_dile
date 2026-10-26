#!/usr/bin/env python3
"""Evaluate a frozen red best-response policy against one or more blue policies."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Sequence

from ipd.core.primitives import C, D, payoff
from ipd.paths import weights_path
from ipd.policies.neural import NeuralPolicy
from ipd.policies.rl import RLPolicy
from ipd.policies.selfplay_gru import SelfPlayGRUPolicy


def _load_blue(path: str, seed: int):
    """Load a self-play GRU, an RL checkpoint, or a deployable MLP policy."""
    try:
        with open(path, encoding="utf-8") as fh:
            fmt = json.load(fh).get("format")
    except (OSError, ValueError):
        fmt = None
    if fmt == "ipd-rl-policy-v1":
        return RLPolicy(path, seed=seed, mode="sample")
    if fmt == "ipd-neural-policy-v1":
        return NeuralPolicy(path, seed=seed, mode="sample")
    return SelfPlayGRUPolicy(path, seed=seed, mode="sample")


def play_pair(blue, red, seed: int, rounds: int = 100) -> dict:
    blue.reset()
    red.reset()
    mine: list[int] = []
    theirs: list[int] = []
    blue_score = red_score = 0
    for t in range(rounds):
        a = int(blue.act(t, mine, theirs))
        b = int(red.act(t, theirs, mine))
        mine.append(a)
        theirs.append(b)
        blue_score += payoff(a, b)
        red_score += payoff(b, a)
    return {
        "seed": seed,
        "blue_score": blue_score,
        "red_score": red_score,
        "diff_blue_minus_red": blue_score - red_score,
        "red_win": red_score > blue_score,
        "draw": red_score == blue_score,
        "blue_defect_rate": sum(mine) / rounds,
        "red_defect_rate": sum(theirs) / rounds,
        "blue_tft_like": (
            sum(1 for a, b in zip(mine, theirs) if a == C and b == C)
            / max(1, sum(1 for b in theirs if b == C))),
    }


def summarize(rows: Sequence[dict]) -> dict:
    return {
        "n": len(rows),
        "blue_mean": statistics.fmean(r["blue_score"] for r in rows),
        "red_mean": statistics.fmean(r["red_score"] for r in rows),
        "diff_mean": statistics.fmean(r["diff_blue_minus_red"] for r in rows),
        "red_win_rate": statistics.fmean(1.0 if r["red_win"] else 0.0
                                        for r in rows),
        "draw_rate": statistics.fmean(1.0 if r["draw"] else 0.0
                                      for r in rows),
        "blue_defect_rate": statistics.fmean(r["blue_defect_rate"]
                                            for r in rows),
        "red_defect_rate": statistics.fmean(r["red_defect_rate"] for r in rows),
        "blue_tft_like": statistics.fmean(r["blue_tft_like"] for r in rows),
        "rows": list(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate a red best-response JSON against blue policies.")
    ap.add_argument("--red", required=True)
    ap.add_argument("--blue", action="append", default=None,
                    help="name=path; may be repeated")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=20561026)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--out", default="results/selfplay_improvement/red_eval.json")
    args = ap.parse_args(argv)

    blue_specs = args.blue or [
        f"bc={weights_path('selfplay_bc_policy.json')}",
        f"scratch={weights_path('selfplay_scratch_policy.json')}",
    ]
    blues = {}
    for item in blue_specs:
        name, path = item.split("=", 1)
        blues[name] = path
    red = SelfPlayGRUPolicy(args.red, seed=args.seed, mode="sample")
    seeds = [args.seed_base + i * 7919 for i in range(args.seeds)]
    out = {
        "schema": "ipd-selfplay-red-eval-v1",
        "red": args.red,
        "seeds": seeds,
        "blues": {},
    }
    for name, path in blues.items():
        blue = _load_blue(path, args.seed)
        rows = [play_pair(blue, red, seed) for seed in seeds]
        out["blues"][name] = {"weights": path, **summarize(rows)}
        s = out["blues"][name]
        print(f"{name:10} blue={s['blue_mean']:.2f} red={s['red_mean']:.2f} "
              f"diff={s['diff_mean']:+.2f} red_win={s['red_win_rate']:.2f} "
              f"blue_D={s['blue_defect_rate']:.2f} red_D={s['red_defect_rate']:.2f}",
              flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"written {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
