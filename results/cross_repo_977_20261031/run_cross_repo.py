#!/usr/bin/env python3
"""Cross-repo pairwise evaluation.

A side: the 30 standalone strategies in ``/home/frti64/frti64_ipd/``.
B side: the 977 unique Protocol v4/v5 opponent configs from this repo.

For every A x B pair we play ``--reps`` matches of ``--rounds`` turns under the
project's canonical 6/0/10/1 matrix.  Every match uses fresh player instances
and seeds derived from ``(base_seed, role, candidate, opponent, rep)``.

Outputs (in --out):
  raw_reps.csv          one row per A x B x rep
  matchups.csv          one row per A x B, 20-rep mean/std/min/max
  weighted_summary.csv  per-candidate means weighted by the v4/v5 manifest
                        entry weights (normalised over all 977 opponents)
  candidates.csv
  opponents.csv
  run_config.json
  README.md
  shards/               resumable per-opponent shards
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
import random
import statistics
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_A_ROOT = Path("/home/frti64/frti64_ipd")
DEFAULT_BASE_SEED = 20261024
DEFAULT_REPS = 20
DEFAULT_ROUNDS = 100
PAYOFF = {"T": 10, "R": 6, "P": 1, "S": 0}

# File stem -> class name in the standalone repository.
A_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("adv_gru", "AdvGRU"),
    ("adv_mlp", "AdvMLP"),
    ("adv_rl", "AdvRL"),
    ("bayes_base", "BayesBase"),
    ("bayes_tuned", "BayesTuned"),
    ("final_base", "FinalBase"),
    ("final_tuned", "FinalTuned"),
    ("gru", "GRU"),
    ("hybrid", "Hybrid"),
    ("neural", "Neural"),
    ("resp3_base", "Resp3Base"),
    ("resp3_tuned", "Resp3Tuned"),
    ("rl", "RL"),
    ("rl_dagger_gru", "GRUDagger"),
    ("rl_defection_guard", "RLDefectionGuard"),
    ("rl_full_guard", "RLFullGuard"),
    ("rl_full_guard_c", "RLFullGuardC"),
    ("rl_gate", "RLGate"),
    ("rl_hard60", "RLHard60"),
    ("rl_pattern", "RLPattern"),
    ("rl_probe_guard", "RLProbeGuard"),
    ("rl_repair", "RLRepair"),
    ("rl_switch", "RLSwitch"),
    ("rl_v5_ft", "RLV5FT"),
    ("rl_v5_ft_complex", "RLV5FTComplex"),
    ("rl_wide", "RLWide"),
    ("rl_wide160", "RLWide160"),
    ("selfplay", "SelfPlayGRU"),
    ("triple_majority", "TripleMajority"),
    ("triple_v4", "TripleV4"),
)
A_ORDER = tuple(stem for stem, _ in A_CANDIDATES)
A_CLASSES = dict(A_CANDIDATES)
A_REGISTRY = {stem: stem.replace("_", "-") for stem in A_ORDER}

RAW_FIELDS = [
    "candidate", "candidate_class", "registry_name",
    "opponent_index", "opponent_key", "family", "type",
    "tier_v4", "tier_v5", "anchor", "thinking_level", "complexity",
    "weight_v4", "weight_v5", "family_weight",
    "rep", "seed_a", "seed_b",
    "score_a", "score_b", "diff", "a_coop_rate", "b_coop_rate", "error",
]
SUMMARY_FIELDS = [
    "candidate", "candidate_class", "registry_name",
    "opponent_index", "opponent_key", "family", "type",
    "tier_v4", "tier_v5", "anchor", "thinking_level", "complexity",
    "weight_v4", "weight_v5", "family_weight",
    "reps", "reps_ok", "errors",
    "score_a_mean", "score_b_mean", "diff_mean",
    "score_a_std", "score_b_std",
    "score_a_min", "score_a_max", "score_b_min", "score_b_max",
    "a_coop_rate_mean", "b_coop_rate_mean",
    "a_wins", "b_wins", "ties",
]

# Worker globals.
_A_ROOT = DEFAULT_A_ROOT
_CATALOG: list[Any] | None = None
_B_META: list[dict[str, Any]] | None = None
_A_CACHE: dict[str, type] = {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _load_b_meta() -> tuple[list[dict[str, Any]], str]:
    from ipd.evaluation.protocol import FAMILY_WEIGHTS, build_catalog, catalog_sha256

    v4_manifest = json.loads(
        (ROOT / "configs/splits/protocol_v4/manifest.json").read_text())
    v4_by_key = {e["key"]: e for e in v4_manifest["entries"]}
    catalog = list(build_catalog())
    v5_keys = {e.key for e in catalog}
    if set(v4_by_key) != v5_keys:
        raise RuntimeError("v4/v5 catalog key sets differ")
    meta: list[dict[str, Any]] = []
    for idx, entry in enumerate(catalog):
        v4 = v4_by_key[entry.key]
        meta.append({
            "opponent_index": idx,
            "opponent_key": entry.key,
            "family": entry.family,
            "type": entry.type_name,
            "tier_v4": str(v4["tier"]),
            "tier_v5": str(entry.tier),
            "weight_v4": float(v4.get("weight", 1.0)),
            "weight_v5": float(entry.weight),
            "family_weight": float(FAMILY_WEIGHTS.get(entry.family, 1.0)),
            "anchor": bool(entry.anchor),
            "thinking_level": int(entry.thinking_level),
            "complexity": int(entry.complexity),
        })
    return meta, catalog_sha256(catalog)


def _derive_seed(base_seed: int, role: str, candidate: str,
                 opponent: str, rep: int) -> int:
    payload = f"{int(base_seed)}|{role}|{candidate}|{opponent}|{int(rep)}"
    return zlib.crc32(payload.encode("utf-8")) & 0x7FFFFFFF


class _SeededPlayerFactory:
    """Callable class factory for an exported Axelrod Player.

    ``AxelrodOpponent`` calls ``player_cls()`` on every reset.  This wrapper
    injects the per-match seed and the required ``sample`` mode where the
    exported class supports them.
    """

    __slots__ = ("cls", "seed", "__name__", "_sig")

    def __init__(self, cls: type, seed: int):
        self.cls = cls
        self.seed = int(seed)
        self.__name__ = cls.__name__
        self._sig = inspect.signature(cls.__init__)

    def __call__(self):
        kwargs: dict[str, Any] = {}
        if "seed" in self._sig.parameters:
            kwargs["seed"] = self.seed
        if "mode" in self._sig.parameters:
            kwargs["mode"] = "sample"
        return self.cls(**kwargs)


def _get_a_class(stem: str) -> type:
    cls = _A_CACHE.get(stem)
    if cls is not None:
        return cls
    path = _A_ROOT / f"{stem}.py"
    module_name = f"cross_repo_a_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    cls = getattr(module, A_CLASSES[stem])
    _A_CACHE[stem] = cls
    return cls


def _make_a(stem: str, seed: int, rounds: int):
    from ipd.core.opponents.axelrod_bank import AxelrodOpponent

    cls = _get_a_class(stem)
    factory = _SeededPlayerFactory(cls, seed)
    return AxelrodOpponent(factory, random.Random(seed), rounds=rounds)


def _make_b(index: int, seed: int):
    assert _CATALOG is not None
    return _CATALOG[index].build(random.Random(seed))


def _worker_init(b_meta: list[dict[str, Any]], a_root: str) -> None:
    global _B_META, _CATALOG, _A_ROOT
    _B_META = b_meta
    _A_ROOT = Path(a_root)
    from ipd.evaluation.protocol import build_catalog

    _CATALOG = list(build_catalog())


def _run_b_task(b_index: int, reps: int, rounds: int,
                base_seed: int) -> tuple[int, list[dict[str, Any]],
                                        list[dict[str, Any]], float]:
    from ipd.core.primitives import C
    from ipd.evaluation.common import run_match

    assert _B_META is not None
    meta = _B_META[b_index]
    common = {
        "opponent_index": meta["opponent_index"],
        "opponent_key": meta["opponent_key"],
        "family": meta["family"],
        "type": meta["type"],
        "tier_v4": meta["tier_v4"],
        "tier_v5": meta["tier_v5"],
        "weight_v4": meta["weight_v4"],
        "weight_v5": meta["weight_v5"],
        "family_weight": meta["family_weight"],
        "anchor": meta["anchor"],
        "thinking_level": meta["thinking_level"],
        "complexity": meta["complexity"],
    }
    raw: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()

    for stem in A_ORDER:
        score_a: list[int] = []
        score_b: list[int] = []
        coop_a: list[float] = []
        coop_b: list[float] = []
        diffs: list[int] = []
        errors = 0

        for rep in range(reps):
            seed_a = _derive_seed(base_seed, "A", stem,
                                  meta["opponent_key"], rep)
            seed_b = _derive_seed(base_seed, "B", stem,
                                  meta["opponent_key"], rep)
            row: dict[str, Any] = {
                "candidate": stem,
                "candidate_class": A_CLASSES[stem],
                "registry_name": A_REGISTRY[stem],
                **common,
                "rep": rep,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "score_a": "",
                "score_b": "",
                "diff": "",
                "a_coop_rate": "",
                "b_coop_rate": "",
                "error": "",
            }
            try:
                a = _make_a(stem, seed_a, rounds)
                b = _make_b(b_index, seed_b)
                stats = run_match(a, b, rounds=rounds, reset=False)
                a_coop = sum(1 for x in stats.mine if x == C) / rounds
                b_coop = sum(1 for x in stats.theirs if x == C) / rounds
                diff = int(stats.score) - int(stats.opponent_score)
                row.update({
                    "score_a": int(stats.score),
                    "score_b": int(stats.opponent_score),
                    "diff": diff,
                    "a_coop_rate": a_coop,
                    "b_coop_rate": b_coop,
                })
                score_a.append(int(stats.score))
                score_b.append(int(stats.opponent_score))
                coop_a.append(a_coop)
                coop_b.append(b_coop)
                diffs.append(diff)
            except Exception as exc:  # keep the run alive; report later
                errors += 1
                row["error"] = f"{type(exc).__name__}: {exc}"
            raw.append(row)

        ok = len(score_a)
        summaries.append({
            "candidate": stem,
            "candidate_class": A_CLASSES[stem],
            "registry_name": A_REGISTRY[stem],
            **common,
            "reps": reps,
            "reps_ok": ok,
            "errors": errors,
            "score_a_mean": statistics.fmean(score_a) if ok else "",
            "score_b_mean": statistics.fmean(score_b) if ok else "",
            "diff_mean": statistics.fmean(diffs) if ok else "",
            "score_a_std": statistics.stdev(score_a) if ok > 1 else 0.0,
            "score_b_std": statistics.stdev(score_b) if ok > 1 else 0.0,
            "score_a_min": min(score_a) if ok else "",
            "score_a_max": max(score_a) if ok else "",
            "score_b_min": min(score_b) if ok else "",
            "score_b_max": max(score_b) if ok else "",
            "a_coop_rate_mean": statistics.fmean(coop_a) if ok else "",
            "b_coop_rate_mean": statistics.fmean(coop_b) if ok else "",
            "a_wins": sum(1 for d in diffs if d > 0),
            "b_wins": sum(1 for d in diffs if d < 0),
            "ties": sum(1 for d in diffs if d == 0),
        })

    return b_index, raw, summaries, time.perf_counter() - started


def _write_csv(path: Path, fields: Iterable[str],
               rows: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _shard_done_path(shards: Path, index: int) -> Path:
    return shards / f"b_{index:04d}_done.json"


def _shard_valid(shards: Path, index: int, reps: int,
                 expected_a: int) -> bool:
    done = _shard_done_path(shards, index)
    raw = shards / f"b_{index:04d}_raw.csv"
    summary = shards / f"b_{index:04d}_summary.csv"
    if not (done.exists() and raw.exists() and summary.exists()):
        return False
    try:
        meta = json.loads(done.read_text())
    except Exception:
        return False
    if int(meta.get("reps", -1)) != reps:
        return False
    if int(meta.get("candidates", -1)) != expected_a:
        return False
    if meta.get("complete") is not True:
        return False
    try:
        with raw.open(newline="", encoding="utf-8") as fh:
            raw_rows = sum(1 for _ in csv.DictReader(fh))
        with summary.open(newline="", encoding="utf-8") as fh:
            summary_rows = sum(1 for _ in csv.DictReader(fh))
    except Exception:
        return False
    return (raw_rows == int(meta.get("raw_rows", -1))
            and summary_rows == expected_a)


def _merge_shards(shards: Path, out: Path, indices: list[int],
                  raw_fields: list[str],
                  summary_fields: list[str]) -> tuple[int, int]:
    raw_path = out / "raw_reps.csv"
    summary_path = out / "matchups.csv"
    raw_tmp = raw_path.with_name(raw_path.name + ".tmp")
    summary_tmp = summary_path.with_name(summary_path.name + ".tmp")
    raw_count = summary_count = 0
    with raw_tmp.open("w", newline="", encoding="utf-8") as rf, \
            summary_tmp.open("w", newline="", encoding="utf-8") as sf:
        rw = csv.DictWriter(rf, fieldnames=raw_fields, extrasaction="ignore")
        sw = csv.DictWriter(sf, fieldnames=summary_fields,
                            extrasaction="ignore")
        rw.writeheader()
        sw.writeheader()
        for index in indices:
            for row in _read_csv(shards / f"b_{index:04d}_raw.csv"):
                rw.writerow(row)
                raw_count += 1
            for row in _read_csv(shards / f"b_{index:04d}_summary.csv"):
                sw.writerow(row)
                summary_count += 1
    os.replace(raw_tmp, raw_path)
    os.replace(summary_tmp, summary_path)
    return raw_count, summary_count


def _weighted_summary(out: Path, meta_by_key: dict[str, dict[str, Any]],
                      expected_b: int, reps: int) -> list[dict[str, Any]]:
    rows = _read_csv(out / "matchups.csv")
    acc: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["candidate"]
        item = acc.setdefault(key, {
            "candidate": key,
            "candidate_class": row["candidate_class"],
            "registry_name": row["registry_name"],
            "n_opponents": 0,
            "w4_sum": 0.0,
            "w5_sum": 0.0,
            "s_eq": 0.0,
            "s4": 0.0,
            "s5": 0.0,
            "d_eq": 0.0,
            "d4": 0.0,
            "d5": 0.0,
        })
        if row.get("score_a_mean", "") == "":
            continue
        score = float(row["score_a_mean"])
        diff = float(row["diff_mean"])
        w4 = float(row["weight_v4"])
        w5 = float(row["weight_v5"])
        item["n_opponents"] += 1
        item["w4_sum"] += w4
        item["w5_sum"] += w5
        item["s_eq"] += score
        item["s4"] += score * w4
        item["s5"] += score * w5
        item["d_eq"] += diff
        item["d4"] += diff * w4
        item["d5"] += diff * w5

    result: list[dict[str, Any]] = []
    for item in acc.values():
        n = item["n_opponents"]
        if not n:
            continue
        w4 = item["w4_sum"]
        w5 = item["w5_sum"]
        result.append({
            "candidate": item["candidate"],
            "candidate_class": item["candidate_class"],
            "registry_name": item["registry_name"],
            "n_opponents": n,
            "expected_opponents": expected_b,
            "coverage": n / expected_b,
            "reps": reps,
            "equal_weight_score": item["s_eq"] / n,
            "v4_weighted_score": item["s4"] / w4 if w4 else "",
            "v5_weighted_score": item["s5"] / w5 if w5 else "",
            "equal_weight_diff": item["d_eq"] / n,
            "v4_weighted_diff": item["d4"] / w4 if w4 else "",
            "v5_weighted_diff": item["d5"] / w5 if w5 else "",
            "v4_weight_sum": w4,
            "v5_weight_sum": w5,
        })
    result.sort(key=lambda r: (
        -float(r["v5_weighted_score"])
        if r["v5_weighted_score"] != "" else float("inf")))
    for rank, row in enumerate(result, 1):
        row["rank_v5_weighted"] = rank
    by_v4 = sorted(result, key=lambda r: (
        -float(r["v4_weighted_score"])
        if r["v4_weighted_score"] != "" else float("inf")))
    for rank, row in enumerate(by_v4, 1):
        row["rank_v4_weighted"] = rank
    by_eq = sorted(result, key=lambda r: -float(r["equal_weight_score"]))
    for rank, row in enumerate(by_eq, 1):
        row["rank_equal_weight"] = rank
    fields = [
        "rank_v5_weighted", "rank_v4_weighted", "rank_equal_weight",
        "candidate", "candidate_class", "registry_name",
        "n_opponents", "expected_opponents", "coverage", "reps",
        "v5_weighted_score", "v4_weighted_score", "equal_weight_score",
        "v5_weighted_diff", "v4_weighted_diff", "equal_weight_diff",
        "v5_weight_sum", "v4_weight_sum",
    ]
    _write_csv(out / "weighted_summary.csv", fields, result)
    _atomic_json(out / "weighted_summary.json", result)
    return result


def _write_readme(out: Path, config: dict[str, Any],
                  weighted: list[dict[str, Any]],
                  raw_count: int, summary_count: int,
                  elapsed: float, failures: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-repo 977 evaluation",
        "",
        f"- Generated: {config['created']}",
        f"- A: `{config['a_root']}` — {config['a_count']} standalone strategies",
        f"- B: Protocol v4/v5 full catalog — {config['b_count']} unique opponents",
        f"- Matches: {config['a_count']} × {config['b_count']} × "
        f"{config['reps']} = {config['expected_matches']:,}",
        f"- Rounds per match: {config['rounds']}",
        f"- Payoff: T={PAYOFF['T']}, R={PAYOFF['R']}, "
        f"P={PAYOFF['P']}, S={PAYOFF['S']}",
        f"- Base seed: {config['base_seed']}",
        f"- Wall time: {elapsed:.1f}s",
        f"- Raw rows: {raw_count:,}; matchup rows: {summary_count:,}",
        f"- Catalog sha256: `{config['catalog_sha256']}`",
        f"- Git: `{config['git_commit']}`",
        "",
        "Every match uses fresh A/B instances.  Seeds are derived as",
        "`crc32(f\"{base_seed}|{A|B}|{candidate}|{opponent}|{rep}\")`.",
        "",
        "## Files",
        "",
        "| file | content |",
        "|---|---|",
        "| `raw_reps.csv` | local-only large file: one row per candidate × opponent × rep |",
        "| `matchups.csv` | main committed table: one row per candidate × opponent, 20-rep mean/std/min/max |",
        "| `matchups.json` | local-only large file: same as `matchups.csv`, JSON records |",
        "| `weighted_summary.csv` | per-candidate equal / protocol-v4 / protocol-v5 weighted scores |",
        "| `candidates.csv` | A-side metadata |",
        "| `opponents.csv` | B-side metadata, v4/v5 tier and entry weights |",
        "| `run_config.json` | run parameters and checksums |",
        "| `shards/` | resumable per-opponent shards |",
        "",
        "## Column notes",
        "",
        "`raw_reps.csv` / `matchups.csv`:",
        "",
        "- `candidate`, `candidate_class`, `registry_name`: A-side identity.",
        "- `opponent_key`, `family`, `type`: B-side identity.",
        "- `tier_v4`, `tier_v5`, `weight_v4`, `weight_v5`: protocol metadata.",
        "- `score_a` / `score_b`: 100-turn total score (A/B side).",
        "- `diff`: `score_a - score_b`.",
        "- `a_coop_rate` / `b_coop_rate`: fraction of C moves.",
        "- `score_*_mean/std/min/max`, `a_wins/b_wins/ties`: 20-rep summary.",
        "- `reps_ok`: successful reps; `errors`: failed reps for that pair.",
        "",
        "`matchups.csv` is the unaggregated per-pair table.  Max possible",
        "score is `100 × T = 1000`, not `100 × R`.",
        "",
        "## Weighting",
        "",
        "`weighted_summary.csv` uses each manifest entry's `weight` field:",
        "the family weight divided by the number of entries of that family in",
        "its v4/v5 tier.  Weights are normalised over all 977 entries.  The",
        "v4 and v5 columns therefore differ because the v4→v5 split moves 405",
        "entries between train/dev/test tiers, not because the opponent",
        "implementations differ.",
        "",
        "## Weighted ranking",
        "",
        "| # | candidate | v5 weighted | v4 weighted | equal weight |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in weighted:
        lines.append(
            f"| {row['rank_v5_weighted']} | `{row['registry_name']}` | "
            f"{float(row['v5_weighted_score']):.3f} | "
            f"{float(row['v4_weighted_score']):.3f} | "
            f"{float(row['equal_weight_score']):.3f} |")
    if failures:
        lines += ["", "## Failures", ""]
        for failure in failures:
            lines.append(
                f"- `{failure.get('opponent_key', '?')}`: "
                f"{failure.get('error', '?')}")
    lines += [
        "",
        "## Notes",
        "",
        "- B is the unique 977-entry union; v4/v5 copies are not duplicated.",
        "- Excluded from B: 19 long-run Axelrod strategies, `Darwin`, and",
        "  `Red-v5`/`Red-v6` red pools (not part of the 977 catalog).",
        "- `matchups.csv` is intentionally unaggregated; the only",
        "  cross-opponent aggregation is in `weighted_summary.csv`.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--a-root", type=Path, default=DEFAULT_A_ROOT)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="only first N B entries (smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="skip valid existing shards")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    shards = out / "shards"
    shards.mkdir(exist_ok=True)
    workers = args.workers or max(1, min(20, (os.cpu_count() or 4) - 2))

    b_meta, catalog_sha = _load_b_meta()
    if args.limit:
        b_meta = b_meta[:args.limit]
    b_count = len(b_meta)
    meta_by_key = {m["opponent_key"]: m for m in b_meta}
    expected_matches = len(A_ORDER) * b_count * args.reps

    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    config = {
        "schema": "cross-repo-977-v1",
        "created": created,
        "a_root": str(args.a_root.resolve()),
        "a_count": len(A_ORDER),
        "b_count": b_count,
        "base_seed": args.base_seed,
        "reps": args.reps,
        "rounds": args.rounds,
        "payoff": PAYOFF,
        "workers": workers,
        "expected_matches": expected_matches,
        "expected_raw_rows": expected_matches,
        "expected_summary_rows": len(A_ORDER) * b_count,
        "catalog_sha256": catalog_sha,
        "git_commit": _git_commit(),
        "seed_derivation": "crc32(base_seed|role|candidate|opponent|rep)",
        "weight_source": "v4 manifest entry.weight; v5 catalog entry.weight",
        "excluded": ["19 long-run Axelrod", "Darwin", "Red-v5", "Red-v6"],
    }
    _atomic_json(out / "run_config.json", config)

    _write_csv(out / "candidates.csv",
               ["candidate", "candidate_class", "registry_name"],
               [{"candidate": stem, "candidate_class": A_CLASSES[stem],
                 "registry_name": A_REGISTRY[stem]} for stem in A_ORDER])
    _write_csv(out / "opponents.csv",
               ["opponent_index", "opponent_key", "family", "type",
                "tier_v4", "tier_v5", "anchor", "thinking_level",
                "complexity", "weight_v4", "weight_v5", "family_weight"],
               b_meta)

    todo = []
    skipped = 0
    for meta in b_meta:
        index = meta["opponent_index"]
        if args.resume and _shard_valid(shards, index, args.reps,
                                        len(A_ORDER)):
            skipped += 1
        else:
            todo.append(index)
    # Mix heavy/light opponents to avoid a long tail.
    random.Random(args.base_seed).shuffle(todo)

    print(f"A={len(A_ORDER)} B={b_count} reps={args.reps} "
          f"matches={expected_matches:,} workers={workers} "
          f"resume_skipped={skipped} todo={len(todo)}", flush=True)
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    completed = 0
    total_raw = 0

    if todo:
        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(b_meta, str(args.a_root.resolve()))) as pool:
            futures = {
                pool.submit(_run_b_task, index, args.reps, args.rounds,
                            args.base_seed): index
                for index in todo
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    b_index, raw, summaries, task_elapsed = future.result()
                    _write_csv(shards / f"b_{b_index:04d}_raw.csv",
                               RAW_FIELDS, raw)
                    _write_csv(shards / f"b_{b_index:04d}_summary.csv",
                               SUMMARY_FIELDS, summaries)
                    _atomic_json(_shard_done_path(shards, b_index), {
                        "complete": True,
                        "b_index": b_index,
                        "opponent_key": b_meta[b_index]["opponent_key"],
                        "reps": args.reps,
                        "candidates": len(A_ORDER),
                        "raw_rows": len(raw),
                        "summary_rows": len(summaries),
                        "errors": sum(1 for r in raw if r.get("error")),
                        "elapsed": task_elapsed,
                    })
                    total_raw += len(raw)
                except Exception as exc:
                    failures.append({
                        "opponent_index": index,
                        "opponent_key": b_meta[index]["opponent_key"],
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                completed += 1
                if completed % 20 == 0 or completed == len(todo):
                    elapsed = time.perf_counter() - started
                    rate = completed / elapsed if elapsed else 0.0
                    eta = (len(todo) - completed) / rate if rate else 0.0
                    print(f"[{completed}/{len(todo)}] "
                          f"shards, {total_raw:,} raw rows, "
                          f"{elapsed:.1f}s, ETA {eta:.0f}s, "
                          f"failures={len(failures)}", flush=True)

    # Merge every valid shard, including ones skipped by resume.
    indices = [m["opponent_index"] for m in b_meta]
    valid_indices = [i for i in indices
                     if _shard_valid(shards, i, args.reps, len(A_ORDER))]
    missing = [i for i in indices if i not in set(valid_indices)]
    raw_count, summary_count = _merge_shards(
        shards, out, valid_indices, RAW_FIELDS, SUMMARY_FIELDS)
    weighted = _weighted_summary(out, meta_by_key, b_count, args.reps)
    elapsed = time.perf_counter() - started

    expected_raw = expected_matches
    expected_summary = len(A_ORDER) * b_count
    print(f"merged raw={raw_count:,} (expected {expected_raw:,}) "
          f"summary={summary_count:,} (expected {expected_summary:,})",
          flush=True)
    if missing:
        print(f"missing/invalid shards: {missing}", flush=True)
    if failures:
        print(f"task failures: {len(failures)}", flush=True)

    _write_readme(out, config, weighted, raw_count, summary_count,
                  elapsed, failures)
    _atomic_json(out / "run_result.json", {
        "raw_rows": raw_count,
        "summary_rows": summary_count,
        "expected_raw_rows": expected_raw,
        "expected_summary_rows": expected_summary,
        "missing_shards": missing,
        "failures": failures,
        "elapsed": elapsed,
        "complete": (not missing and not failures
                     and raw_count == expected_raw
                     and summary_count == expected_summary),
    })
    return 0 if (not missing and not failures
                 and raw_count == expected_raw
                 and summary_count == expected_summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
