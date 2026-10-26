#!/usr/bin/env python3
"""E0 league harness: frozen cross-play, suite metrics and red exploitability.

The harness is intentionally separate from training and from the historical
``final`` suite.  It freezes every candidate spec, seed and opponent suite in a
JSON config so that a comparison can be reproduced exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ipd.evaluation.common import quantile, run_match
from ipd.evaluation.distributions import DISTS, build_dist_field, sample_blind_field
from ipd.evaluation.metagame import (
    elo_ratings, exploitability, fictitious_play,
)
from ipd.evaluation.opponent_suite import build_suite
from ipd.policies.opponents import ALL_BY_NAME
from ipd.policies.registry import CandidateSpec
from ipd.paths import ROOT, result_path

SCHEMA = "ipd-league-report-v1"
_PATH_KEYS = ("weights", "config", "checkpoint")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateEntry:
    spec: CandidateSpec
    group: str = ""
    tags: tuple[str, ...] = ()
    policy_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "group": self.group,
            "tags": list(self.tags),
            "policy_seed": self.policy_seed,
        }


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    kind: str                      # official | blind | opponents
    suite: str | None = None
    fields: int = 0
    n: int = 299
    seeds: tuple[int, ...] = ()
    rounds: int = 100
    seed_base: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "suite": self.suite,
            "fields": self.fields,
            "n": self.n,
            "seeds": list(self.seeds),
            "rounds": self.rounds,
            "seed_base": self.seed_base,
        }


@dataclass(frozen=True)
class LeagueConfig:
    name: str
    path: Path
    candidates: tuple[CandidateEntry, ...]
    red_policies: tuple[CandidateEntry, ...]
    suites: Mapping[str, SuiteConfig]
    cross_play_seeds: tuple[int, ...]
    cross_play_seed_base: int
    cross_play_policy_seed_base: int
    red_seeds: tuple[int, ...]
    red_seed_base: int
    red_policy_seed_base: int
    workers: int = 20
    mode: str = "sample"
    baseline: str = "RL-PPO"
    nash_iterations: int = 20000
    note: str = ""

    def selected_candidates(self, only: Iterable[str] | None = None
                            ) -> list[CandidateEntry]:
        if not only:
            return list(self.candidates)
        wanted = {str(x).strip().lower() for x in only if str(x).strip()}
        rows = [row for row in self.candidates
                if row.spec.name.lower() in wanted
                or row.spec.kind.lower() in wanted]
        found = {row.spec.name.lower() for row in rows}
        found |= {row.spec.kind.lower() for row in rows}
        missing = wanted - found
        if missing:
            raise ValueError(f"unknown candidates: {sorted(missing)}")
        return rows


def _resolve_path(value: str, config_dir: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    root_path = ROOT / path
    if root_path.exists():
        return str(root_path)
    config_path = config_dir / path
    if config_path.exists():
        return str(config_path)
    return str(root_path)


def _normalise_params(params: Mapping[str, Any], config_dir: Path
                      ) -> dict[str, Any]:
    out = dict(params)
    for key in _PATH_KEYS:
        if isinstance(out.get(key), str):
            out[key] = _resolve_path(out[key], config_dir)
    return out


def _entry(data: Mapping[str, Any], config_dir: Path) -> CandidateEntry:
    if not data.get("name") or not data.get("kind"):
        raise ValueError("every candidate needs name and kind")
    params = _normalise_params(data.get("params") or {}, config_dir)
    policy_seed = data.get("policy_seed")
    return CandidateEntry(
        spec=CandidateSpec(
            kind=str(data["kind"]), name=str(data["name"]), params=params),
        group=str(data.get("group", "")),
        tags=tuple(str(x) for x in data.get("tags") or ()),
        policy_seed=int(policy_seed) if policy_seed is not None else None,
    )


def _seeds(base: int, count: int, step: int = 7919) -> tuple[int, ...]:
    return tuple(int(base) + i * step for i in range(int(count)))


def _suite_config(name: str, data: Mapping[str, Any],
                  defaults: Mapping[str, Any]) -> SuiteConfig:
    kind = str(data.get("kind", "opponents"))
    rounds = int(data.get("rounds", defaults.get("rounds", 100)))
    if kind == "official":
        return SuiteConfig(
            name=name, kind=kind, suite=None,
            fields=int(data.get("fields", 20)),
            n=int(data.get("n", 299)),
            rounds=rounds,
            seed_base=int(data.get("seed", data.get("seed_base", 20261026))),
        )
    if kind == "blind":
        fields = int(data.get("fields", 20))
        base = int(data.get("seed", data.get("seed_base", 20261025)))
        return SuiteConfig(
            name=name, kind=kind, suite=None, fields=fields,
            n=int(data.get("n", 299)), seeds=_seeds(base, fields),
            rounds=rounds, seed_base=base,
        )
    if kind == "opponents":
        suite = str(data.get("suite", name))
        base = int(data.get("seed_base", data.get("seed", 20961026)))
        count = int(data.get("seeds", 20))
        return SuiteConfig(
            name=name, kind=kind, suite=suite, seeds=_seeds(base, count),
            rounds=rounds, seed_base=base,
        )
    raise ValueError(f"unknown suite kind: {kind}")


def _default_suites(defaults: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "official": {
            "kind": "official", "fields": int(defaults.get("fields", 20)),
            "n": int(defaults.get("n", 299)), "seed": 20261026,
        },
        "blind": {
            "kind": "blind", "fields": int(defaults.get("blind_fields", 20)),
            "n": int(defaults.get("n", 299)), "seed": 20261025,
        },
        "validation": {
            "kind": "opponents", "suite": "validation",
            "seeds": int(defaults.get("opponent_seeds", 20)),
            "seed_base": 20261026,
        },
        "heldout_v2": {
            "kind": "opponents", "suite": "heldout_v2",
            "seeds": int(defaults.get("opponent_seeds", 20)),
            "seed_base": 20961026,
        },
    }


def load_league_config(path: str | Path) -> LeagueConfig:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = json.loads(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults") or {}
    candidates = tuple(_entry(x, path.parent)
                       for x in data.get("candidates") or []
                       if x.get("enabled", True) is not False)
    if not candidates:
        raise ValueError(f"no enabled candidates in {path}")
    red = tuple(_entry(x, path.parent)
                for x in data.get("red_policies") or []
                if x.get("enabled", True) is not False)
    suites_raw = data.get("suites") or _default_suites(defaults)
    suites = {name: _suite_config(name, value, defaults)
              for name, value in suites_raw.items()}
    cross = data.get("cross_play") or {}
    cross_seeds = int(cross.get("seeds", defaults.get("cross_play_seeds", 20)))
    cross_base = int(cross.get("seed_base", 20561026))
    red_cfg = data.get("red") or {}
    red_count = int(red_cfg.get("seeds", defaults.get("red_seeds", 20)))
    red_base = int(red_cfg.get("seed_base", 20561026))
    return LeagueConfig(
        name=str(data.get("name") or path.stem),
        path=path.resolve(),
        candidates=candidates,
        red_policies=red,
        suites=suites,
        cross_play_seeds=_seeds(cross_base, cross_seeds),
        cross_play_seed_base=cross_base,
        cross_play_policy_seed_base=int(
            cross.get("policy_seed_base", 20461026)),
        red_seeds=_seeds(red_base, red_count),
        red_seed_base=red_base,
        red_policy_seed_base=int(red_cfg.get("policy_seed_base", 20361026)),
        workers=int(defaults.get("workers", 20)),
        mode=str(defaults.get("mode", "sample")),
        baseline=str(data.get("baseline", "RL-PPO")),
        nash_iterations=int(data.get("nash_iterations", 20000)),
        note=str(data.get("note", "")),
    )


# ---------------------------------------------------------------------------
# Policy construction and workers
# ---------------------------------------------------------------------------


def _build(spec: CandidateSpec, seed: int, mode: str):
    params = dict(spec.params)
    params["seed"] = int(seed)
    params.setdefault("mode", mode)
    return CandidateSpec(spec.kind, spec.name, params).build()


@lru_cache(maxsize=None)
def _suite_items(name: str):
    return tuple(build_suite(name))


def _row(suite: str, family: str, opponent: str, seed: int,
         result, candidate: str) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "suite": suite,
        "family": family,
        "opponent": opponent,
        "seed": int(seed),
        "score": int(result.score),
        "opponent_score": int(result.opponent_score),
        "defect_rate": sum(result.mine) / max(1, len(result.mine)),
    }


def suite_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    spec = CandidateSpec.from_dict(task["spec"])
    policy = _build(spec, task["policy_seed"], task["mode"])
    item = _suite_items(task["suite"])[int(task["item_index"])]
    rows = []
    for seed in task["seeds"]:
        opponent = item.factory(random.Random(int(seed)))
        if hasattr(opponent, "name"):
            opponent.name = item.name
        result = run_match(policy, opponent, rounds=int(task["rounds"]))
        rows.append(_row(task["suite"], item.family, item.name, int(seed),
                         result, spec.name))
    return rows


def official_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    spec = CandidateSpec.from_dict(task["spec"])
    policy = _build(spec, task["policy_seed"], task["mode"])
    field = build_dist_field(task["dist"], int(task["seed"]), int(task["n"]))
    rows = []
    for opponent in field:
        result = run_match(policy, opponent, rounds=int(task["rounds"]))
        name = result.opponent_name or getattr(opponent, "name", "opponent")
        rows.append(_row("official", task["dist"], name, int(task["seed"]),
                         result, spec.name))
    return rows


def _blind_family(name: str) -> str:
    spec = ALL_BY_NAME.get(name)
    return spec.family if spec is not None else "blind"


def blind_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    spec = CandidateSpec.from_dict(task["spec"])
    policy = _build(spec, task["policy_seed"], task["mode"])
    field = sample_blind_field(int(task["seed"]), int(task["n"]))
    rows = []
    for opponent in field:
        result = run_match(policy, opponent, rounds=int(task["rounds"]))
        name = result.opponent_name or getattr(opponent, "name", "opponent")
        rows.append(_row("blind", _blind_family(name), name,
                         int(task["seed"]), result, spec.name))
    return rows


def cross_play_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    row_spec = CandidateSpec.from_dict(task["row_spec"])
    col_spec = CandidateSpec.from_dict(task["col_spec"])
    mode = task.get("mode", "sample")
    row = _build(row_spec, task["row_seed"], mode)
    col = _build(col_spec, task["col_seed"], mode)
    rows = []
    for seed in task["seeds"]:
        result = run_match(row, col, rounds=int(task["rounds"]))
        rows.append({
            "row": row_spec.name,
            "col": col_spec.name,
            "seed": int(seed),
            "row_score": int(result.score),
            "col_score": int(result.opponent_score),
            "row_defect_rate": sum(result.mine) / max(1, len(result.mine)),
            "col_defect_rate": sum(result.theirs) / max(1, len(result.theirs)),
        })
    return rows


def red_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    blue_spec = CandidateSpec.from_dict(task["blue_spec"])
    red_spec = CandidateSpec.from_dict(task["red_spec"])
    mode = task.get("mode", "sample")
    blue = _build(blue_spec, task["blue_seed"], mode)
    red = _build(red_spec, task["red_seed"], mode)
    rows = []
    for seed in task["seeds"]:
        result = run_match(blue, red, rounds=int(task["rounds"]))
        rows.append({
            "blue": blue_spec.name,
            "red": red_spec.name,
            "seed": int(seed),
            "blue_score": int(result.score),
            "red_score": int(result.opponent_score),
            "blue_defect_rate": sum(result.mine) / max(1, len(result.mine)),
            "red_defect_rate": sum(result.theirs) / max(1, len(result.theirs)),
        })
    return rows


# ---------------------------------------------------------------------------
# Summary and meta-game report
# ---------------------------------------------------------------------------


def _family_key(row: Mapping[str, Any]) -> str:
    return f"{row['suite']}:{row['family']}"


def summarize_rows(rows: Sequence[Mapping[str, Any]],
                   baseline: Mapping[str, float] | None = None) -> dict:
    if not rows:
        return {
            "n": 0, "mean": 0.0, "families": {}, "min_family_raw": 0.0,
            "argmin_family": None, "min_norm_family": None,
            "cvar10_family": 0.0, "cvar10_item": 0.0,
            "sigma_family": 0.0, "max_minus_min": 0.0,
            "top_family": None, "top_family_share": 0.0,
            "per_opponent": {}, "family_counts": {},
        }
    scores = [float(r["score"]) for r in rows]
    by_family: dict[str, list[float]] = {}
    by_opponent: dict[str, list[float]] = {}
    for row in rows:
        key = _family_key(row)
        by_family.setdefault(key, []).append(float(row["score"]))
        opp_key = f"{row['suite']}:{row.get('opponent', '')}"
        by_opponent.setdefault(opp_key, []).append(float(row["score"]))
    families = {key: statistics.fmean(vals)
                for key, vals in by_family.items()}
    family_sorted = sorted(families.values())
    k_family = max(1, int(math.ceil(0.1 * len(family_sorted))))
    cvar_family = statistics.fmean(family_sorted[:k_family])
    item_sorted = sorted(scores)
    k_item = max(1, int(math.ceil(0.1 * len(item_sorted))))
    cvar_item = statistics.fmean(item_sorted[:k_item])
    argmin = min(families, key=families.get) if families else None
    top = max(families, key=families.get) if families else None
    total = sum(scores)
    top_share = (sum(by_family[top]) / total
                 if top is not None and total else 0.0)
    norm = None
    missing: list[str] = []
    if baseline:
        values = []
        for key, value in families.items():
            base = baseline.get(key)
            if base:
                values.append(value / base)
            else:
                missing.append(key)
        if values:
            norm = min(values)
    return {
        "n": len(rows),
        "mean": statistics.fmean(scores),
        "families": families,
        "family_counts": {k: len(v) for k, v in by_family.items()},
        "min_family_raw": min(families.values()) if families else 0.0,
        "argmin_family": argmin,
        "min_norm_family": norm,
        "missing_baseline_families": missing,
        "cvar10_family": cvar_family,
        "cvar10_item": cvar_item,
        "p05_item": quantile(scores, 0.05),
        "sigma_family": (statistics.pstdev(list(families.values()))
                         if len(families) > 1 else 0.0),
        "max_minus_min": (max(families.values()) - min(families.values())
                          if families else 0.0),
        "top_family": top,
        "top_family_share": top_share,
        "per_opponent": {key: statistics.fmean(vals)
                         for key, vals in sorted(by_opponent.items())},
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def build_cross_play_report(rows: Sequence[Mapping[str, Any]],
                            names: Sequence[str], *,
                            nash_iterations: int = 20000,
                            seed: int = 0) -> dict:
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    pair: dict[tuple[int, int], list[tuple[int, int]]] = {
        (i, j): [] for i in range(n) for j in range(n)
    }
    for row in rows:
        i = index[row["row"]]
        j = index[row["col"]]
        pair[(i, j)].append((int(row["row_score"]), int(row["col_score"])))
    payoff = [[0.0] * n for _ in range(n)]
    win = [[0.5] * n for _ in range(n)]
    matches = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            values = pair[(i, j)]
            matches[i][j] = len(values)
            payoff[i][j] = _mean([a for a, _ in values]) if values else 0.0
            if i != j:
                win[i][j] = _mean([
                    1.0 if a > b else 0.5 if a == b else 0.0
                    for a, b in values
                ]) if values else 0.5
    diff = [[payoff[i][j] - payoff[j][i] for j in range(n)]
            for i in range(n)]
    nash = fictitious_play(diff, iterations=nash_iterations, seed=seed)
    nash_weights = nash["weights"]
    meta_exploit = exploitability(diff, nash_weights)
    elo = elo_ratings(win)
    pool_exploit = {}
    for i, name in enumerate(names):
        gains = [(diff[j][i], names[j]) for j in range(n) if j != i]
        best_gain, best_name = max(gains) if gains else (0.0, None)
        own = _mean([payoff[i][j] for j in range(n) if j != i])
        pool_exploit[name] = {
            "max_gain": best_gain,
            "best_response": best_name,
            "own_mean": own,
            "normalized": best_gain / max(1.0, own) if own else 0.0,
        }
    meta_score = sum(nash_weights[i] * nash_weights[j] * payoff[i][j]
                     for i in range(n) for j in range(n))
    return {
        "names": list(names),
        "payoff": payoff,
        "diff": diff,
        "win_rate": win,
        "matches": matches,
        "elo": {name: elo[i] for i, name in enumerate(names)},
        "nash": {
            "weights": {name: nash_weights[i]
                        for i, name in enumerate(names)},
            "iterations": nash["iterations"],
            "value": nash["value"],
            "exploitability": nash["exploitability"],
            "meta_score": meta_score,
            "best_response": (names[meta_exploit["best_response"]]
                              if names else None),
        },
        "pool_exploitability": pool_exploit,
        "n_matches": sum(len(v) for v in pair.values()),
    }


cross_play_matrices = build_cross_play_report


def build_red_report(rows: Sequence[Mapping[str, Any]],
                     blue_names: Sequence[str],
                     baseline_blue: Mapping[str, float] | None = None) -> dict:
    by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault((row["blue"], row["red"]), []).append(row)
    pairs = {}
    for (blue, red), values in sorted(by_pair.items()):
        blue_mean = _mean([float(x["blue_score"]) for x in values])
        red_mean = _mean([float(x["red_score"]) for x in values])
        wins = sum(1 for x in values if x["red_score"] > x["blue_score"])
        draws = sum(1 for x in values if x["red_score"] == x["blue_score"])
        pairs[f"{blue}|{red}"] = {
            "blue": blue, "red": red,
            "n": len(values),
            "blue_mean": blue_mean,
            "red_mean": red_mean,
            "diff_mean": red_mean - blue_mean,
            "red_win_rate": wins / len(values) if values else 0.0,
            "draw_rate": draws / len(values) if values else 0.0,
            "blue_min": min((float(x["blue_score"]) for x in values),
                            default=0.0),
        }
    blues = {}
    for blue in blue_names:
        values = [row for row in rows if row["blue"] == blue]
        if not values:
            continue
        blue_mean = _mean([float(x["blue_score"]) for x in values])
        red_mean = _mean([float(x["red_score"]) for x in values])
        wins = sum(1 for x in values if x["red_score"] > x["blue_score"])
        worst = max(
            (v for v in pairs.values() if v["blue"] == blue),
            key=lambda v: v["diff_mean"], default=None)
        normalized = None
        if baseline_blue and baseline_blue.get(blue):
            normalized = blue_mean / baseline_blue[blue]
        blues[blue] = {
            "n": len(values),
            "blue_mean": blue_mean,
            "red_mean": red_mean,
            "diff_mean": red_mean - blue_mean,
            "red_win_rate": wins / len(values) if values else 0.0,
            "blue_min": min((float(x["blue_score"]) for x in values),
                            default=0.0),
            "worst_red": worst["red"] if worst else None,
            "max_red_gain": worst["diff_mean"] if worst else 0.0,
            "normalized_vs_baseline": normalized,
        }
    return {"pairs": pairs, "blues": blues}


# ---------------------------------------------------------------------------
# Run driver and output
# ---------------------------------------------------------------------------


def _run_tasks(func, tasks: Sequence[dict], workers: int,
               label: str) -> list[dict]:
    if not tasks:
        return []
    rows: list[dict] = []
    started = time.time()
    total = len(tasks)
    progress_every = max(1, total // 10)
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for done, result in enumerate(pool.map(func, tasks, chunksize=1), 1):
            rows.extend(result)
            if done % progress_every == 0 or done == total:
                print(f"  {label} {done}/{total} tasks "
                      f"({time.time() - started:.1f}s)", flush=True)
    return rows


def _suite_tasks(cfg: LeagueConfig, suite: SuiteConfig,
                 candidates: Sequence[CandidateEntry]) -> list[dict]:
    tasks: list[dict] = []
    for ci, entry in enumerate(candidates):
        policy_seed = (entry.policy_seed if entry.policy_seed is not None
                       else 20261024 + ci * 1009)
        if suite.kind == "official":
            # Match the historical benchmark seed map: one independent seed per
            # distribution and field.
            from ipd.evaluation.common import seed_map

            seed_by_dist = seed_map(suite.seed_base, DISTS, suite.fields)
            for dist in DISTS:
                for seed in seed_by_dist[dist]:
                    tasks.append({
                        "spec": entry.spec.to_dict(), "dist": dist,
                        "seed": int(seed), "n": suite.n,
                        "rounds": suite.rounds, "mode": cfg.mode,
                        "policy_seed": policy_seed,
                    })
        elif suite.kind == "blind":
            for seed in suite.seeds:
                tasks.append({
                    "spec": entry.spec.to_dict(), "seed": int(seed),
                    "n": suite.n, "rounds": suite.rounds, "mode": cfg.mode,
                    "policy_seed": policy_seed,
                })
        else:
            count = len(_suite_items(suite.suite or suite.name))
            for index in range(count):
                tasks.append({
                    "spec": entry.spec.to_dict(), "suite": suite.suite or suite.name,
                    "item_index": index, "seeds": list(suite.seeds),
                    "rounds": suite.rounds, "mode": cfg.mode,
                    "policy_seed": policy_seed,
                })
    return tasks


def _cross_play_tasks(cfg: LeagueConfig,
                      candidates: Sequence[CandidateEntry]) -> list[dict]:
    tasks = []
    for i, row in enumerate(candidates):
        for j, col in enumerate(candidates):
            if i > j:
                continue
            tasks.append({
                "row_spec": row.spec.to_dict(), "col_spec": col.spec.to_dict(),
                "seeds": list(cfg.cross_play_seeds), "rounds": 100,
                "mode": cfg.mode,
                "row_seed": cfg.cross_play_policy_seed_base + i * 10007 + j,
                "col_seed": cfg.cross_play_policy_seed_base + j * 10007 + i,
            })
            if i != j:
                tasks.append({
                    "row_spec": col.spec.to_dict(), "col_spec": row.spec.to_dict(),
                    "seeds": list(cfg.cross_play_seeds), "rounds": 100,
                    "mode": cfg.mode,
                    "row_seed": cfg.cross_play_policy_seed_base + j * 10007 + i,
                    "col_seed": cfg.cross_play_policy_seed_base + i * 10007 + j,
                })
    return tasks


def _red_tasks(cfg: LeagueConfig, candidates: Sequence[CandidateEntry]
               ) -> list[dict]:
    tasks = []
    for bi, blue in enumerate(candidates):
        for ri, red in enumerate(cfg.red_policies):
            tasks.append({
                "blue_spec": blue.spec.to_dict(),
                "red_spec": red.spec.to_dict(),
                "seeds": list(cfg.red_seeds), "rounds": 100,
                "mode": cfg.mode,
                "blue_seed": cfg.red_policy_seed_base + bi * 10007 + ri,
                "red_seed": cfg.red_policy_seed_base + ri * 10007 + bi,
            })
    return tasks


def _fingerprint(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('name', 'League')} — cross-play / league report",
        "",
        f"- **Name:** {report['name']}",
        f"- **Created:** {report['created']}",
        f"- **Schema:** {report['schema']}",
        f"- **Baseline:** {report.get('baseline')}",
        f"- **Fingerprint:** `{report.get('fingerprint', '')}`",
        "",
    ]
    if report.get("note"):
        lines += [report["note"], ""]
    lines += [
        "## Candidate summary",
        "",
        "| Candidate | Group | Overall | Min norm | CVaR10 family | Red blue | Red win |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["candidates"].items():
        red = row.get("red") or {}
        lines.append(
            f"| {name} | {row.get('group', '')} | "
            f"{_fmt(row['overall']['mean'])} | "
            f"{_fmt(row['overall']['min_norm_family'], 4)} | "
            f"{_fmt(row['overall']['cvar10_family'])} | "
            f"{_fmt(red.get('blue_mean'))} | "
            f"{_fmt(red.get('red_win_rate'), 3)} |"
        )
    lines += ["", "## Per-suite means", "",
              "| Candidate | " + " | ".join(report["suite_names"]) + " |",
              "|---|" + "---:|" * len(report["suite_names"])]
    for name, row in report["candidates"].items():
        cells = [_fmt(row["suites"].get(s, {}).get("mean"))
                 for s in report["suite_names"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    cross = report.get("cross_play") or {}
    if cross.get("names"):
        names = cross["names"]
        lines += ["", "## Cross-play payoff matrix", "",
                  "| row \\ col | " + " | ".join(names) + " |",
                  "|---|" + "---:|" * len(names)]
        for i, name in enumerate(names):
            cells = [f"{cross['payoff'][i][j]:.1f}" for j in range(len(names))]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "## Meta-game", "",
                  "| Candidate | Elo | Nash weight | Pool exploitability | Best response |",
                  "|---|---:|---:|---:|---|"]
        for name in names:
            nash = cross["nash"]["weights"].get(name, 0.0)
            pool = cross["pool_exploitability"].get(name, {})
            lines.append(
                f"| {name} | {cross['elo'].get(name, 0):.1f} | "
                f"{nash:.4f} | {pool.get('normalized', 0):.4f} | "
                f"{pool.get('best_response')} |"
            )
        lines += [
            "",
            f"Nash exploitability (score-difference units): "
            f"**{cross['nash']['exploitability']:.4f}**; "
            f"iterations={cross['nash']['iterations']}.",
        ]
    red = report.get("red") or {}
    if red.get("blues"):
        lines += ["", "## Held-out red pressure", "",
                  "| Blue | Blue mean | Red mean | Diff | Red win | Worst red |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, row in red["blues"].items():
            lines.append(
                f"| {name} | {row['blue_mean']:.2f} | {row['red_mean']:.2f} | "
                f"{row['diff_mean']:+.2f} | {row['red_win_rate']:.3f} | "
                f"{row['worst_red']} |"
            )
    focus = [x for x in ("RL-PPO", "RL-PPO-Hard60", "Adv-RL",
                         "SelfPlay-GRU") if x in report["candidates"]]
    if focus:
        lines += ["", "## Held-out v2 per family", "",
                  "| Candidate | Family | Mean |", "|---|---|---:|"]
        for name in focus:
            families = report["candidates"][name]["suites"].get(
                "heldout_v2", {}).get("families", {})
            for key, value in families.items():
                lines.append(f"| {name} | {key} | {value:.2f} |")
    gate = report.get("gate") or {}
    lines += ["", "## Gate", "",
              f"- Data complete: **{gate.get('data_complete')}**",
              f"- Cross-play complete: **{gate.get('cross_play_complete')}**",
              f"- Red complete: **{gate.get('red_complete')}**",
              f"- Status: **{gate.get('status')}**"]
    if gate.get("missing"):
        lines.append(f"- Missing: {', '.join(gate['missing'])}")
    lines += [
        "",
        "## Caveats",
        "",
        "- This is a synthetic opponent pool; cross-play Elo/Nash is a pool-level",
        "  approximation, not proof against an arbitrary best response.",
        "- `heldout_v2` was created for E0 and must not be reused for iterative",
        "  model selection. The older `final` suite is now validation.",
        "- E4 red is a held-out stress test for non-BC candidates, but it was",
        "  trained against a BC blue. It does not solve anti-exploitation.",
        "- No candidate is promoted by this report. The main weights are unchanged.",
        "",
    ]
    return "\n".join(lines)


def _gate(report: Mapping[str, Any], selected_suites: Sequence[str],
          require_red: bool) -> dict:
    missing = []
    for name, row in report["candidates"].items():
        for suite in selected_suites:
            if not row["suites"].get(suite, {}).get("n"):
                missing.append(f"{name}:{suite}")
    cross = report.get("cross_play") or {}
    cross_complete = bool(cross.get("names")) and all(
        cross["matches"][i][j] > 0
        for i in range(len(cross.get("names", [])))
        for j in range(len(cross.get("names", [])))
    )
    red_complete = (not require_red) or all(
        row.get("red") for row in report["candidates"].values())
    complete = not missing
    status = "PASS" if complete and cross_complete and red_complete else "INCOMPLETE"
    return {
        "data_complete": complete,
        "cross_play_complete": cross_complete,
        "red_complete": red_complete,
        "missing": missing,
        "status": status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E0 cross-play / Nash / exploitability league harness.")
    ap.add_argument("--config", default="configs/eval/e0_league.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated candidate names/kinds")
    ap.add_argument("--suites", default=None,
                    help="comma-separated configured suite names")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cross-play-seeds", type=int, default=None)
    ap.add_argument("--red-seeds", type=int, default=None)
    ap.add_argument("--nash-iterations", type=int, default=None)
    ap.add_argument("--skip-suites", action="store_true")
    ap.add_argument("--skip-cross-play", action="store_true")
    ap.add_argument("--skip-red", action="store_true")
    ap.add_argument("--include-rows", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_league_config(args.config)
    if args.cross_play_seeds is not None:
        cfg = _replace_cfg(
            cfg, cross_play_seeds=_seeds(cfg.cross_play_seed_base,
                                         args.cross_play_seeds))
    if args.red_seeds is not None:
        cfg = _replace_cfg(
            cfg, red_seeds=_seeds(cfg.red_seed_base, args.red_seeds))
    if args.nash_iterations is not None:
        cfg = _replace_cfg(cfg, nash_iterations=args.nash_iterations)
    workers = args.workers or cfg.workers
    only = [x for x in (args.only or "").split(",") if x.strip()]
    candidates = cfg.selected_candidates(only)
    if not candidates:
        raise SystemExit("no candidates selected")

    configured_suites = list(cfg.suites)
    if args.suites:
        selected = [x.strip() for x in args.suites.split(",") if x.strip()]
        unknown = [x for x in selected if x not in cfg.suites]
        if unknown:
            raise SystemExit(f"unknown suites: {unknown}; choose from {configured_suites}")
    else:
        selected = [] if args.skip_suites else configured_suites

    out = Path(args.out or result_path("e0_league_20261026/report.json"))
    md_out = Path(args.md_out or out.with_name("README.md"))
    print(f"E0 league: candidates={[x.spec.name for x in candidates]} "
          f"suites={selected} workers={workers}", flush=True)
    started = time.time()

    suite_rows: dict[str, list[dict]] = {name: [] for name in selected}
    for name in selected:
        suite = cfg.suites[name]
        tasks = _suite_tasks(cfg, suite, candidates)
        func = {"official": official_task, "blind": blind_task,
                "opponents": suite_task}[suite.kind]
        suite_rows[name] = _run_tasks(func, tasks, workers, f"suite:{name}")

    cross_rows: list[dict] = []
    if not args.skip_cross_play:
        cross_rows = _run_tasks(
            cross_play_task, _cross_play_tasks(cfg, candidates), workers,
            "cross-play")

    red_rows: list[dict] = []
    if not args.skip_red and cfg.red_policies:
        red_rows = _run_tasks(
            red_task, _red_tasks(cfg, candidates), workers, "red")

    all_rows = [row for rows in suite_rows.values() for row in rows]
    baseline_name = cfg.baseline
    if baseline_name not in {x.spec.name for x in candidates}:
        baseline_name = candidates[0].spec.name
    baseline_overall = summarize_rows(
        [r for r in all_rows if r["candidate"] == baseline_name])
    baseline_families = baseline_overall["families"]
    baseline_by_suite = {}
    for suite in selected:
        base_rows = [r for r in suite_rows[suite]
                     if r["candidate"] == baseline_name]
        baseline_by_suite[suite] = summarize_rows(base_rows)

    red_payload = {}
    if red_rows:
        baseline_red = build_red_report(
            [r for r in red_rows if r["blue"] == baseline_name],
            [baseline_name])
        baseline_blue_mean = baseline_red.get("blues", {}).get(
            baseline_name, {}).get("blue_mean")
        red_base = ({x.spec.name: baseline_blue_mean for x in candidates}
                    if baseline_blue_mean else None)
        red_payload = build_red_report(
            red_rows, [x.spec.name for x in candidates], red_base)

    candidate_payload = {}
    for entry in candidates:
        name = entry.spec.name
        rows = [r for r in all_rows if r["candidate"] == name]
        per_suite = {}
        for suite in selected:
            subset = [r for r in rows if r["suite"] == suite]
            per_suite[suite] = summarize_rows(
                subset, baseline_by_suite[suite]["families"])
        overall = summarize_rows(rows, baseline_families)
        candidate_payload[name] = {
            "spec": entry.spec.to_dict(),
            "group": entry.group,
            "tags": list(entry.tags),
            "suites": per_suite,
            "overall": overall,
            "red": red_payload.get("blues", {}).get(name, {}),
            "params": _param_count(entry.spec),
        }

    cross_payload = {}
    if cross_rows:
        cross_payload = build_cross_play_report(
            cross_rows, [x.spec.name for x in candidates],
            nash_iterations=cfg.nash_iterations)

    report = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "name": cfg.name,
        "note": cfg.note,
        "config": str(cfg.path),
        "config_sha256": _sha256_file(cfg.path),
        "baseline": baseline_name,
        "suite_names": selected,
        "candidates": candidate_payload,
        "suites": {suite: {
            name: candidate_payload[name]["suites"].get(suite, {})
            for name in candidate_payload
        } for suite in selected},
        "cross_play": cross_payload,
        "red": red_payload,
        "seeds": {
            "cross_play": list(cfg.cross_play_seeds),
            "red": list(cfg.red_seeds),
            "suites": {name: cfg.suites[name].to_dict()
                       for name in selected},
        },
    }
    report["gate"] = _gate(report, selected,
                           require_red=bool(cfg.red_policies)
                           and not args.skip_red)
    report["fingerprint"] = _fingerprint({
        "baseline": report["baseline"],
        "candidates": {
            name: {
                "overall": row["overall"],
                "red": row["red"],
            }
            for name, row in report["candidates"].items()
        },
        "cross_play": report["cross_play"],
        "red": report["red"],
    })
    if args.include_rows:
        report["raw"] = {
            "suites": suite_rows,
            "cross_play": cross_rows,
            "red": red_rows,
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(build_markdown(report), encoding="utf-8")
    print(f"written {out}")
    print(f"written {md_out}")
    print(f"fingerprint {report['fingerprint']}")
    for name, row in report["candidates"].items():
        print(f"{name:20} mean={row['overall']['mean']:8.2f} "
              f"min_norm={_fmt(row['overall']['min_norm_family'], 4):>8} "
              f"red_blue={_fmt((row['red'] or {}).get('blue_mean')):>8}")
    print(f"gate: {report['gate']['status']}")
    return 0


def _replace_cfg(cfg: LeagueConfig, **changes: Any) -> LeagueConfig:
    data = {**cfg.__dict__, **changes}
    return LeagueConfig(**data)


def _sha256_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return hashlib.sha256(data).hexdigest()


def _param_count(spec: CandidateSpec) -> dict[str, int] | None:
    from ipd.training.selfplay.compare import _count_candidate

    try:
        return _count_candidate(spec)
    except Exception:
        return None


__all__ = [
    "SCHEMA", "CandidateEntry", "SuiteConfig", "LeagueConfig",
    "load_league_config", "suite_task", "official_task", "blind_task",
    "cross_play_task", "red_task", "summarize_rows",
    "build_cross_play_report", "build_red_report", "build_markdown", "main",
]
