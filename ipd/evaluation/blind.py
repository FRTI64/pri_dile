#!/usr/bin/env python3
"""Unified blind ranking test on unseen 300-player fields.

The opponent-vs-opponent round robin is computed once per field and reused by
every candidate, so adding candidates is cheap.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from ..core.simulator import payoff, play_match
from ..paths import result_path
from ..policies.registry import (
    CORE_NAMES, candidate_specs, legacy_arch_names, resolve_names,
)
from .distributions import sample_blind_field

SCHEMA = "ipd-blind-v1"
_LEGACY_ARCHES = ("mlp", "gru", "rl", "adversarial", "adv-rl",
                  "adv-mlp", "adv-gru")
_ARCH_OUT = {
    "mlp": "blind_neural_results.json",
    "gru": "blind_gru_results.json",
    "rl": "blind_rl_results.json",
    "adversarial": "blind_adversarial_results.json",
    "adv-mlp": "blind_adversarial_results.json",
    "adv-gru": "blind_adversarial_gru_results.json",
    "adv-rl": "blind_adversarial_rl_results.json",
}


def _run_field(task: dict, detail: bool = False):
    """Evaluate every candidate on one unseen field with shared opponents."""
    from ..policies.registry import CandidateSpec

    seed = int(task["seed"])
    n = int(task["n"])
    specs = [CandidateSpec.from_dict(x) for x in task["specs"]]
    base_field = sample_blind_field(seed, n)
    n_opp = len(base_field)

    # Opponent round robin is candidate-independent.
    field = copy.deepcopy(base_field)
    opp_scores = [0] * n_opp
    for i in range(n_opp):
        for j in range(i + 1, n_opp):
            result = play_match(field[i], field[j])
            opp_scores[i] += result.score_a
            opp_scores[j] += result.score_b

    out: dict[str, dict[str, float]] = {}
    for spec in specs:
        candidate = spec.build()
        cand_score = 0
        boosted = list(opp_scores)
        field = copy.deepcopy(base_field)
        per_opponent_scores: list[int] = []
        per_opponent_names: list[str] = []
        for i, opponent in enumerate(field):
            candidate.reset()
            opponent.reset()
            mine: list[int] = []
            theirs: list[int] = []
            own = other = 0
            for t in range(100):
                x = candidate.act(t, mine, theirs)
                y = opponent.act(t, theirs, mine)
                mine.append(x)
                theirs.append(y)
                own += payoff(x, y)
                other += payoff(y, x)
            cand_score += own
            boosted[i] += other
            if detail:
                per_opponent_names.append(getattr(opponent, "name",
                                                  type(opponent).__name__))
                per_opponent_scores.append(own)
        rank = 1 + sum(score > cand_score for score in boosted)
        row = {"rank": rank, "score": cand_score}
        if detail:
            row["opponents"] = per_opponent_names
            row["opponent_scores"] = per_opponent_scores
        out[spec.name] = row
    return seed, out


def field_task(task: dict) -> tuple[int, dict[str, dict[str, float]]]:
    """Evaluate every candidate on one unseen field with shared opponents."""
    return _run_field(task, detail=False)


def field_task_details(task: dict) -> tuple[int, dict[str, dict[str, Any]]]:
    """Like :func:`field_task`, but includes per-opponent score arrays."""
    return _run_field(task, detail=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Blind ranking test on unseen wide/smart fields.")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated names or core|base|suite|learned|all")
    ap.add_argument("--arch", default=None, choices=_LEGACY_ARCHES)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--neural-mode", "--mode", dest="mode", default="sample",
                    choices=("sample", "argmax"))
    ap.add_argument("--neural", action="store_true",
                    help="legacy: include one learned policy")
    ap.add_argument("--neural-arch", default="mlp",
                    choices=_LEGACY_ARCHES)
    ap.add_argument("--neural-weights", default=None)
    ap.add_argument("--neural-config", default=None)
    ap.add_argument("--fields", type=int, default=10)
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", default=None)
    return ap


def _selected_names(args: argparse.Namespace) -> list[str]:
    names = (resolve_names(args.candidates, default=CORE_NAMES)
             if args.candidates else list(CORE_NAMES))
    extra: list[str] = []
    if args.arch:
        extra.extend(legacy_arch_names(args.arch))
    if args.neural:
        extra.extend(legacy_arch_names(args.neural_arch))
    return list(dict.fromkeys(names + extra))


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    names = _selected_names(args)
    if not names:
        ap.error("no candidates selected")
    specs = candidate_specs(names, weights=args.weights or args.neural_weights,
                            config=args.config or args.neural_config,
                            mode=args.mode)

    out = args.out
    if out is None:
        arch = args.arch or (args.neural_arch if args.neural else None)
        out = result_path(_ARCH_OUT.get(arch, "blind_results.json"))

    seeds = [args.seed + i * 7919 for i in range(args.fields)]
    tasks = [
        {"seed": seed, "n": args.n,
         "specs": [spec.to_dict() for spec in specs]}
        for seed in seeds
    ]
    started = time.time()
    rows: list[dict[str, dict[str, float]]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (seed, result) in enumerate(
                ex.map(field_task, tasks, chunksize=1), 1):
            rows.append(result)
            print(f"field {i}/{args.fields} seed={seed} "
                  + ", ".join(f"{name}:{v['rank']}"
                              for name, v in result.items()),
                  flush=True)

    print(f"\nBlind test over {args.fields} unseen {args.n}-player fields:")
    print(f"{'strategy':22}{'score':>10}{'mean rank':>12}{'rank1':>8}"
          f"{'top10':>8}{'worst':>8}")
    summary: dict[str, dict] = {}
    for spec in specs:
        name = spec.name
        scores = [row[name]["score"] / args.n for row in rows]
        ranks = [row[name]["rank"] for row in rows]
        row = {
            "kind": spec.kind,
            "score": statistics.fmean(scores),
            "mean_rank": statistics.fmean(ranks),
            "rank1": sum(x == 1 for x in ranks),
            "top10": sum(x <= 10 for x in ranks),
            "worst": max(ranks),
            "scores": scores,
            "ranks": ranks,
        }
        summary[name] = row
        print(f"{name:22}{row['score']:10.2f}{row['mean_rank']:12.2f}"
              f"{row['rank1']:5d}/{args.fields:<2d}"
              f"{row['top10']:5d}/{args.fields:<2d}{row['worst']:8d}")

    payload = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "seed": args.seed,
        "fields": args.fields,
        "n": args.n,
        "candidates": [spec.to_dict() for spec in specs],
        "results": summary,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
