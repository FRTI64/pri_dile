#!/usr/bin/env python3
"""Full per-opponent-type matchup matrix, plus suite-weighted aggregates.

The base matchup CSV scores every candidate against every opponent archetype.
Because the nine test suites are family-weighted mixtures over the same
archetypes, the expected suite score can be computed directly from the
per-type means:

    score_suite(candidate) = sum_type w[suite, family(type)] * mean[type]
                             / sum_type w[suite, family(type)]

where ``w`` uses the same family multipliers as
:func:`ipd.evaluation.distributions.sample_suite_field`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

from ..core.opponents import TYPE_BANK, TYPE_BY_NAME
from ..core.simulator import play_match
from ..paths import result_path
from ..policies.opponents import ALL_BY_NAME, ALL_SPECS
from ..policies.registry import BASE_NAMES, candidate_specs
from .distributions import FAMILY_MULT

SCHEMA = "ipd-matchup-v2"


def type_task(task: dict) -> list[tuple[str, str, str, float, float, float, int]]:
    from ..policies.registry import CandidateSpec

    type_name = task["type"]
    opponent_set = task.get("opponent_set", "core")
    spec = (ALL_BY_NAME if opponent_set == "all"
            else TYPE_BY_NAME)[type_name]
    reps = int(task["reps"])
    base_seed = int(task["seed"])
    specs = [CandidateSpec.from_dict(x) for x in task["candidates"]]
    rows = []
    for rep in range(reps):
        opp_seed = (base_seed + rep * 1000
                    + zlib.crc32(type_name.encode("utf-8")) % 997)
        for cand_spec in specs:
            opponent = spec.factory(random.Random(opp_seed))
            opponent.name = type_name
            candidate = cand_spec.build()
            result = play_match(candidate, opponent)
            rows.append((cand_spec.name, type_name, spec.family,
                         result.score_a, result.score_b,
                         result.score_a - result.score_b, 1))
    return rows


def _selected_specs(opponent_set: str):
    return list(ALL_SPECS if opponent_set == "all" else TYPE_BANK)


def _suite_weights(specs) -> dict[str, dict[str, float]]:
    """Return the normalized suite weight of every selected opponent type."""
    out: dict[str, dict[str, float]] = {}
    for suite, mult in FAMILY_MULT.items():
        raw = {spec.name: max(0.001, mult.get(spec.family, 0.5))
               for spec in specs}
        total = sum(raw.values())
        out[suite] = {name: value / total for name, value in raw.items()}
    return out


def _suite_scores(rows, specs) -> dict[str, dict[str, float]]:
    means: dict[tuple[str, str], float] = {
        (cand, typ): our for cand, typ, _family, our, _opp, _diff, _n
        in rows
    }
    weights = _suite_weights(specs)
    family = {spec.name: spec.family for spec in specs}
    candidates = sorted({cand for cand, _typ, *_ in rows})
    types = {typ for _cand, typ, *_ in rows}
    out: dict[str, dict[str, float]] = {}
    for cand in candidates:
        out[cand] = {}
        for suite in FAMILY_MULT:
            total_w = 0.0
            total = 0.0
            for typ in types:
                # A suite weight is defined per family, but types in the same
                # family share it; use the per-type normalized value.
                w = weights[suite].get(typ)
                if w is None:
                    w = max(0.001, FAMILY_MULT[suite].get(
                        family[typ], 0.5)) / sum(
                        max(0.001, FAMILY_MULT[suite].get(
                            family[t], 0.5)) for t in types)
                total_w += w
                total += w * means[(cand, typ)]
            out[cand][suite] = total / total_w if total_w else 0.0
    return out


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate", "type", "family", "our", "opponent",
                         "diff", "reps"])
        writer.writerows(rows)


def _write_suite_csv(path: Path, suite_scores: dict[str, dict[str, float]],
                     suites: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate", *suites, "avg", "worst"])
        for cand in sorted(suite_scores,
                           key=lambda c: statistics.fmean(
                               suite_scores[c].values()), reverse=True):
            values = [suite_scores[cand][suite] for suite in suites]
            writer.writerow([cand, *[f"{value:.4f}" for value in values],
                             f"{statistics.fmean(values):.4f}",
                             f"{min(values):.4f}"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Per-opponent-type matchup matrix plus family-weighted "
                    "nine-suite scores.")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated names (default: base rule candidates)")
    ap.add_argument("--opponent-set", choices=("core", "all"), default="core",
                    help="core = 85-type bank, all = 158-type full bank")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--out", default=result_path("matchup_results.csv"),
                    help="per-opponent CSV")
    ap.add_argument("--json-out", default=None,
                    help="structured JSON (default: same stem as --out)")
    ap.add_argument("--suite-out", default=None,
                    help="suite-weighted CSV (default: <out>_suite_weighted.csv)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    specs = candidate_specs(args.candidates, default=BASE_NAMES)
    opponent_specs = _selected_specs(args.opponent_set)
    tasks = [{"type": spec.name, "reps": args.reps, "seed": args.seed,
              "candidates": [x.to_dict() for x in specs],
              "opponent_set": args.opponent_set}
             for spec in opponent_specs]

    started = time.time()
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rows in ex.map(type_task, tasks, chunksize=1):
            for cand, typ, family, our, opp, diff, n in rows:
                key = (cand, typ, family)
                bucket = agg[key]
                bucket[0] += our
                bucket[1] += opp
                bucket[2] += diff
                bucket[3] += n

    rows = []
    for (cand, typ, family), (our, opp, diff, n) in agg.items():
        rows.append((cand, typ, family, our / n, opp / n, diff / n, n))
    rows.sort(key=lambda x: (x[0], x[2], x[1]))

    out_path = Path(args.out)
    json_path = (Path(args.json_out) if args.json_out
                 else out_path.with_suffix(".json"))
    suite_path = (Path(args.suite_out) if args.suite_out
                  else out_path.with_name(out_path.stem + "_suite_weighted.csv"))
    _write_csv(out_path, rows)

    suites = list(FAMILY_MULT)
    suite_scores = _suite_scores(rows, opponent_specs)
    suite_weights = _suite_weights(opponent_specs)
    _write_suite_csv(suite_path, suite_scores, suites)

    by_cand = defaultdict(list)
    by_family = defaultdict(list)
    for cand, typ, family, our, opp, diff, n in rows:
        by_cand[cand].append(our)
        by_family[(cand, family)].append(our)
    families = sorted({family for _, _, family, *_ in rows})

    print("Per-opponent mean")
    for cand, vals in sorted(by_cand.items(),
                             key=lambda kv: statistics.fmean(kv[1]),
                             reverse=True):
        print(f"{cand:24}{statistics.fmean(vals):8.2f}")

    print("\nFamily means")
    print(f"{'strategy':24}" + "".join(f"{f:>12}" for f in families))
    for spec in specs:
        print(f"{spec.name:24}"
              + "".join(f"{statistics.fmean(by_family[(spec.name, f)]):12.2f}"
                        for f in families))

    print("\nSuite-weighted means")
    print(f"{'strategy':24}" + "".join(f"{s:>11}" for s in suites)
          + f"{'avg':>9}{'worst':>9}")
    for cand in sorted(suite_scores,
                       key=lambda c: statistics.fmean(
                           suite_scores[c].values()), reverse=True):
        values = [suite_scores[cand][suite] for suite in suites]
        print(f"{cand:24}" + "".join(f"{v:11.2f}" for v in values)
              + f"{statistics.fmean(values):9.2f}{min(values):9.2f}")

    payload = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "seed": args.seed,
        "reps": args.reps,
        "opponent_set": args.opponent_set,
        "candidates": [spec.to_dict() for spec in specs],
        "opponents": [{"name": spec.name, "family": spec.family}
                      for spec in opponent_specs],
        "type_rows": [
            {"candidate": cand, "type": typ, "family": family,
             "our": our, "opponent": opp, "diff": diff, "reps": n}
            for cand, typ, family, our, opp, diff, n in rows
        ],
        "suite_weights": suite_weights,
        "suite_scores": suite_scores,
        "candidate_summary": {
            cand: {
                "equal_type_mean": statistics.fmean(by_cand[cand]),
                "suite_scores": suite_scores[cand],
                "suite_avg": statistics.fmean(suite_scores[cand].values()),
                "suite_worst": min(suite_scores[cand].values()),
            }
            for cand in by_cand
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print("\nwritten", out_path)
    print("written", suite_path)
    print("written", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
