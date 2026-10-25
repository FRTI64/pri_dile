#!/usr/bin/env python3
"""Split evaluation: one config/output per candidate, one file per opponent.

The normal ``ipd benchmark``/``ipd blind`` commands evaluate a whole candidate
list with one shared parameter set and write one combined JSON file.  This
driver is for the cases where you want:

* a separate result file for every strategy/model,
* different fields/n/seed/workers for different candidates, and
* per-opponent-type result files for error analysis.

Run from the repository root::

    python3 -m ipd eval-split --config configs/eval/retrain_20261025.json

Output layout::

    results/<run>/
      config.json
      summary.json
      README.md
      by_candidate/<candidate>/benchmark.json
      by_candidate/<candidate>/blind.json
      by_candidate/<candidate>/opponents.json
      by_opponent/<opponent>.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..paths import result_path
from ..policies.opponents import ALL_BY_NAME
from .blind import field_task_details
from .common import eval_task_details, quantile, seed_map
from .split_config import (
    CandidateRun, EvalParams, SplitConfig, load_split_config, override_params,
)

SCHEMA = "ipd-eval-split-v1"
CANDIDATE_OPPONENT_SCHEMA = "ipd-candidate-opponents-v1"
OPPONENT_SCHEMA = "ipd-opponent-v1"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "unnamed"


def _stats(values: Sequence[float], with_values: bool = True) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None, "p05": None,
                "count": 0, **({"values": []} if with_values else {})}
    row = {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p05": quantile(values, 0.05),
        "count": len(values),
    }
    if with_values:
        row["values"] = list(values)
    return row


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _without_details(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "per_opponent"}


def _run_benchmark(row: CandidateRun, detail: bool,
                   progress: bool = True) -> dict[str, Any]:
    params = row.benchmark
    seeds = seed_map(params.seed, params.dists, params.fields)
    tasks = [
        {
            "spec": row.spec.to_dict(),
            "name": row.spec.name,
            "dist": dist,
            "seeds": [seed],
            "n": params.n,
            "rounds": params.rounds,
        }
        for dist in params.dists
        for seed in seeds[dist]
    ]
    started = time.time()
    results: dict[str, list[float]] = {dist: [] for dist in params.dists}
    per_opponent: dict[str, dict[str, list[float]]] = {
        dist: defaultdict(list) for dist in params.dists
    }
    total = len(tasks)
    progress_every = max(1, total // 10)
    with ProcessPoolExecutor(max_workers=max(1, params.workers)) as ex:
        if detail:
            iterator = ex.map(eval_task_details, tasks, chunksize=1)
            for done, (_name, dist, values, details) in enumerate(iterator, 1):
                results[dist].extend(values)
                for opp, scores in details.items():
                    per_opponent[dist][opp].extend(scores)
                if progress and (done % progress_every == 0 or done == total):
                    print(f"  benchmark {done}/{total} field tasks", flush=True)
        else:
            from .common import eval_task
            iterator = ex.map(eval_task, tasks, chunksize=1)
            for done, (_name, dist, values) in enumerate(iterator, 1):
                results[dist].extend(values)
                if progress and (done % progress_every == 0 or done == total):
                    print(f"  benchmark {done}/{total} field tasks", flush=True)

    dists: dict[str, Any] = {}
    for dist in params.dists:
        values = results[dist]
        dists[dist] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "p05": quantile(values, 0.05),
            "values": values,
        }
    means = [row["mean"] for row in dists.values()]
    summary = {
        "schema": "ipd-split-benchmark-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "candidate": row.spec.to_dict(),
        "params": params.to_dict(),
        "dists": dists,
        "avg": statistics.fmean(means),
        "worst": min(means),
        "min": min(row["min"] for row in dists.values()),
    }
    if detail:
        summary["per_opponent"] = {
            dist: {opp: _stats(scores)
                   for opp, scores in sorted(per_opponent[dist].items())}
            for dist in params.dists
        }
    return summary


def _blind_groups(rows: Iterable[CandidateRun]) -> dict[tuple, list[CandidateRun]]:
    groups: dict[tuple, list[CandidateRun]] = defaultdict(list)
    for row in rows:
        p = row.blind
        key = (p.fields, p.n, p.seed, p.workers, p.rounds)
        groups[key].append(row)
    return dict(groups)


def _run_blind_group(rows: list[CandidateRun], params: EvalParams,
                     detail: bool) -> dict[str, dict[str, Any]]:
    seeds = [params.seed + i * 7919 for i in range(params.fields)]
    tasks = [
        {
            "seed": seed,
            "n": params.n,
            "rounds": params.rounds,
            "specs": [row.spec.to_dict() for row in rows],
        }
        for seed in seeds
    ]
    names = [row.spec.name for row in rows]
    scores: dict[str, list[float]] = {name: [] for name in names}
    ranks: dict[str, list[int]] = {name: [] for name in names}
    per_opponent: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in names
    }
    started = time.time()
    with ProcessPoolExecutor(max_workers=max(1, params.workers)) as ex:
        for i, (_seed, result) in enumerate(
                ex.map(field_task_details if detail else _field_task,
                       tasks, chunksize=1), 1):
            for name in names:
                value = result[name]
                scores[name].append(value["score"] / params.n)
                ranks[name].append(value["rank"])
                if detail:
                    for opp, score in zip(value.get("opponents", []),
                                          value.get("opponent_scores", [])):
                        per_opponent[name][opp].append(float(score))
            print(f"  blind field {i}/{params.fields}", flush=True)

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.spec.name
        row_ranks = ranks[name]
        out[name] = {
            "schema": "ipd-split-blind-v1",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - started, 2),
            "candidate": row.spec.to_dict(),
            "params": params.to_dict(),
            "score": statistics.fmean(scores[name]),
            "mean_rank": statistics.fmean(row_ranks),
            "rank1": sum(x == 1 for x in row_ranks),
            "top10": sum(x <= 10 for x in row_ranks),
            "worst": max(row_ranks),
            "scores": scores[name],
            "ranks": row_ranks,
        }
        if detail:
            out[name]["per_opponent"] = {
                opp: _stats(values)
                for opp, values in sorted(per_opponent[name].items())
            }
    return out


def _field_task(task: dict):
    from .blind import field_task
    return field_task(task)


def _candidate_opponent_payload(
        benchmark: dict[str, Any] | None,
        blind: dict[str, Any] | None,
        name: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": CANDIDATE_OPPONENT_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate": name,
        "benchmark": {},
        "blind": {},
    }
    if benchmark and benchmark.get("per_opponent"):
        payload["benchmark"] = benchmark["per_opponent"]
    if blind and blind.get("per_opponent"):
        payload["blind"] = blind["per_opponent"]
    return payload


def _opponent_payloads(
        rows: Iterable[CandidateRun],
        benchmark_rows: dict[str, dict[str, Any]],
        blind_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_opponent: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.spec.name
        bench = benchmark_rows.get(name, {})
        blind = blind_rows.get(name, {})
        opp_names = set()
        for dist in bench.get("per_opponent", {}):
            opp_names.update(bench["per_opponent"][dist])
        opp_names.update(blind.get("per_opponent", {}))
        for opp in sorted(opp_names):
            spec = ALL_BY_NAME.get(opp)
            entry = by_opponent.setdefault(opp, {
                "schema": OPPONENT_SCHEMA,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "opponent": opp,
                "family": spec.family if spec else "unknown",
                "candidates": {},
            })
            bench_entry = {
                dist: {key: value for key, value in
                       bench["per_opponent"][dist][opp].items()
                       if key != "values"}
                for dist in bench.get("per_opponent", {})
                if opp in bench["per_opponent"][dist]
            }
            blind_entry = None
            if opp in blind.get("per_opponent", {}):
                blind_entry = {
                    key: value for key, value in
                    blind["per_opponent"][opp].items() if key != "values"
                }
            entry["candidates"][name] = {
                "kind": row.spec.kind,
                "benchmark": bench_entry,
                "blind": blind_entry,
            }
    return by_opponent


def _write_readme(out_dir: Path, config: SplitConfig,
                  benchmark_rows: dict[str, dict[str, Any]],
                  blind_rows: dict[str, dict[str, Any]]) -> None:
    lines = [
        f"# Split evaluation: {config.name}",
        "",
        f"- config: `{config.path}`",
        f"- created: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        "Each candidate has its own parameter set and result files.",
        "",
        "## Benchmark",
        "",
        "| candidate | params | avg | worst | files |",
        "|---|---|---:|---:|---|",
    ]
    for row in config.candidates:
        name = row.spec.name
        data = benchmark_rows.get(name)
        if not data:
            continue
        p = row.benchmark
        files = f"[result](by_candidate/{_safe_name(name)}/benchmark.json)"
        opp_file = (out_dir / "by_candidate" / _safe_name(name)
                    / "opponents.json")
        if opp_file.is_file():
            files += (f", [opponents](by_candidate/{_safe_name(name)}"
                      f"/opponents.json)")
        lines.append(
            f"| {name} | {p.fields} fields × {p.n}, seed {p.seed} | "
            f"{data['avg']:.2f} | {data['worst']:.2f} | {files} |"
        )
    lines += [
        "",
        "## Blind",
        "",
        "| candidate | params | score | mean rank | files |",
        "|---|---|---:|---:|---|",
    ]
    for row in config.candidates:
        name = row.spec.name
        data = blind_rows.get(name)
        if not data:
            continue
        p = row.blind
        files = f"[result](by_candidate/{_safe_name(name)}/blind.json)"
        opp_file = (out_dir / "by_candidate" / _safe_name(name)
                    / "opponents.json")
        if opp_file.is_file():
            files += (f", [opponents](by_candidate/{_safe_name(name)}"
                      f"/opponents.json)")
        lines.append(
            f"| {name} | {p.fields} fields × {p.n}, seed {p.seed} | "
            f"{data['score']:.2f} | {data['mean_rank']:.2f} | {files} |"
        )
    lines += [
        "",
        "## Per-opponent files",
        "",
        "`by_opponent/<opponent>.json` contains every candidate's score against "
        "that opponent type.  `by_candidate/<candidate>/opponents.json` contains "
        "all opponent types for one candidate.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m ipd eval-split",
        description="Evaluate each candidate with its own parameters and write "
                    "separate per-candidate and per-opponent result files.")
    ap.add_argument("--config", default=str(
        Path("configs") / "eval" / "retrain_20261025.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated candidate names/kinds")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--skip-blind", action="store_true")
    ap.add_argument("--no-details", dest="details", action="store_false",
                    help="skip per-opponent breakdown files")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result directory")
    ap.add_argument("--fields", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--blind-fields", type=int)
    ap.add_argument("--blind-n", type=int)
    ap.add_argument("--blind-seed", type=int)
    ap.add_argument("--blind-workers", type=int)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    config = load_split_config(args.config)
    rows = config.selected(
        [x for x in (args.only or "").split(",") if x.strip()])
    rows = override_params(
        rows, fields=args.fields, n=args.n, seed=args.seed,
        workers=args.workers, blind_fields=args.blind_fields,
        blind_n=args.blind_n, blind_seed=args.blind_seed,
        blind_workers=args.blind_workers,
    )
    if not rows:
        ap.error("no candidates selected")

    out_dir = Path(args.out_dir) if args.out_dir else result_path(
        f"split_{config.name}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        ap.error(f"{out_dir} is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows: dict[str, dict[str, Any]] = {}
    blind_rows: dict[str, dict[str, Any]] = {}
    started = time.time()
    print(f"run: {config.name} -> {out_dir}")
    print(f"candidates: {', '.join(row.spec.name for row in rows)}")
    _write_json(out_dir / "config.json", config.to_dict())

    if not args.skip_benchmark:
        print("\n[benchmark] separate parameters per candidate")
        for row in rows:
            p = row.benchmark
            print(f"- {row.spec.name}: {p.fields} fields × {p.n}, "
                  f"seed {p.seed}, workers {p.workers}")
            data = _run_benchmark(row, detail=args.details)
            benchmark_rows[row.spec.name] = data
            cand_dir = out_dir / "by_candidate" / _safe_name(row.spec.name)
            _write_json(cand_dir / "benchmark.json", _without_details(data))

    if not args.skip_blind:
        print("\n[blind] grouped by identical blind parameters")
        for group in _blind_groups(rows).values():
            p = group[0].blind
            print(f"- {len(group)} candidate(s): {p.fields} fields × {p.n}, "
                  f"seed {p.seed}, workers {p.workers}")
            data = _run_blind_group(group, group[0].blind,
                                    detail=args.details)
            blind_rows.update(data)
            for row in group:
                cand_dir = out_dir / "by_candidate" / _safe_name(row.spec.name)
                _write_json(cand_dir / "blind.json",
                            _without_details(data[row.spec.name]))

    if args.details:
        for row in rows:
            name = row.spec.name
            payload = _candidate_opponent_payload(
                benchmark_rows.get(name), blind_rows.get(name), name)
            _write_json(out_dir / "by_candidate" / _safe_name(name)
                        / "opponents.json", payload)
        for opp, payload in _opponent_payloads(
                rows, benchmark_rows, blind_rows).items():
            _write_json(out_dir / "by_opponent" / f"{_safe_name(opp)}.json",
                        payload)

    summary = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "run": config.name,
        "config": str(config.path),
        "candidates": [row.to_dict() for row in rows],
        "benchmark": {name: _without_details(row)
                      for name, row in benchmark_rows.items()},
        "blind": {name: _without_details(row)
                  for name, row in blind_rows.items()},
    }
    _write_json(out_dir / "summary.json", summary)
    _write_readme(out_dir, config, benchmark_rows, blind_rows)
    print(f"\nwritten {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
