#!/usr/bin/env python
"""Run the phased 33-candidate x 1,937-opponent parity study.

Phases
------
0. Build ``entities.csv`` and assert the frozen pool sizes.
1. CPU/GPU sample parity on a stratified 13 x 100 sample.  Hard gate.
2. Provisional parity branch: GPU for the 13 supported candidates, CPU for
   the other 20.
3. Authoritative pure-CPU sample branch for all 33 candidates.
4. Join Phase 2 and Phase 3 and report mismatches.

The script is resumable: every candidate / GPU shard is written atomically and
completed candidates are skipped on a rerun.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep each spawned worker single-threaded.  The heavy Axelrod Meta opponents
# are Python-bound; letting torch spin up many intra-op threads per worker
# saturates the box and slows the run down dramatically.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

OUT_ROOT = Path(__file__).resolve().parent
ENTITIES_CSV = OUT_ROOT / "entities.csv"
ENTITIES_COUNTS = OUT_ROOT / "entities_counts.json"

ROUNDS = 100
REPS = 1
BASE_SEED = 20261029
EXPECTED_ENTITIES = 1937
EXPECTED_PAIRS = 33 * EXPECTED_ENTITIES
CPU_WORKERS = 20
GPU_WORKERS = 8
GPU_SHARDS = 8
GPU_MAX_BATCH = 2048

GPU_CANDIDATES = (
    "neural", "gru", "rl", "rl-hard60", "rl-repair", "rl-pattern",
    "rl-wide", "rl-wide160", "rl-dagger-gru", "psro-br",
    "psro-scratch-br", "psro-v4-br", "selfplay",
)

LOG = logging.getLogger("candidates_vs_all_phased")


def setup_logging() -> None:
    """Timestamped, immediately flushed console + file logging."""
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    LOG.addHandler(console)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        OUT_ROOT / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    LOG.addHandler(file_handler)
CPU_CANDIDATES = tuple(
    kind for kind in (
        "final-base", "final-tuned", "resp3-base", "resp3-tuned",
        "bayes-base", "bayes-tuned", "hybrid", "harvest45",
        "triple-majority", "triple-v4", "harvest-tuned", "gap-start",
        "rl-switch", "rl-gate", "psro", "psro-scratch", "psro-v4",
        "adv-mlp", "adv-gru", "adv-rl",
    )
)

LONG_CSV_FIELDS = [
    "pair_id", "rep", "branch", "candidate_key", "candidate_name",
    "opponent_key", "opponent_name", "opponent_pool", "opponent_family",
    "seed_candidate", "seed_opponent", "candidate_score", "opponent_score",
    "diff_candidate_minus_opponent", "candidate_action_hash",
    "opponent_action_hash", "actual_name_candidate",
    "actual_name_opponent", "mode", "device",
]

EXPECTED_POOLS = {
    "candidate": 33,
    "legacy": 158,
    "axelrod": 243,
    "axelrod_cheating": 1,
    "protocol_v4": 977,
    "heldout_v2": 74,
    "heldout_v3": 83,
    "heldout_v4": 83,
    "validation": 51,
    "final": 86,
    "hard": 16,
    "hard2": 28,
    "periodic": 9,
    "robust": 13,
    "repair": 15,
    "extort": 10,
    "policy_weight": 57,
}

ACCEPTED_WEIGHT_FORMATS = {
    "ipd-rl-policy-v1", "ipd-neural-policy-v1",
    "ipd-gru-policy-v1", "ipd-selfplay-gru-v1",
}

# These are all in Axelrod's long_run_time_strategies set but are cheap enough
# for the Phase 1 smoke gate.  The full Phase 2/3 runs include all 19.
PHASE1_LONG_RUN = (
    "DBS", "MetaMajorityLongMemory", "MetaWinnerLongMemory",
    "MetaWinnerStochastic", "NMWELongMemory", "NMWEStochastic",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)


def action_hash(mine: Sequence[int], theirs: Sequence[int]) -> str:
    payload = bytes(int(x) for x in mine) + b"|" + \
        bytes(int(x) for x in theirs)
    return hashlib.sha256(payload).hexdigest()


def match_seed(base_seed: int, candidate_key: str, opponent_key: str,
               rep: int) -> int:
    payload = f"{base_seed}|{candidate_key}|{opponent_key}|{rep}".encode()
    return (base_seed + zlib.crc32(payload)) & 0x7FFFFFFF


def opponent_seed_for(seed: int) -> int:
    payload = f"{seed}|opponent".encode()
    return (seed + zlib.crc32(payload)) & 0x7FFFFFFF


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        return max(0, sum(1 for _ in fh) - 1)


# ---------------------------------------------------------------------------
# Phase 0: entities
# ---------------------------------------------------------------------------


def build_entities() -> tuple[list[dict[str, Any]], dict[str, int]]:
    from ipd.core.opponents.axelrod_bank import (
        axelrod_cheating_specs, axelrod_specs,
    )
    from ipd.evaluation.distributions import HARD_OPPONENTS
    from ipd.evaluation.extort import EXTORT_HARD_OPPONENTS
    from ipd.evaluation.hard2 import HARD2_OPPONENTS
    from ipd.evaluation.opponent_suite import (
        build_heldout_v2_suite, build_heldout_v3_suite,
        build_heldout_v4_suite,
    )
    from ipd.evaluation.periodic import PERIODIC_HARD_OPPONENTS
    from ipd.evaluation.protocol import build_catalog
    from ipd.evaluation.robust import REPAIR_OPPONENTS, ROBUST_OPPONENTS
    from ipd.policies.opponents import ALL_SPECS
    from ipd.policies.registry import candidate_specs
    from ipd.training.selfplay.opponents import (
        build_final_suite, build_validation_suite,
    )

    rows: list[dict[str, Any]] = []

    def add(*, key: str, name: str, display: str, source: str, pool: str,
            family: str, type_name: str, params: dict[str, Any],
            path: str = "") -> None:
        rows.append({
            "key": key,
            "name": name,
            "display": display,
            "source": source,
            "pool": pool,
            "family": family,
            "type": type_name,
            "params": json.dumps(params, sort_keys=True, separators=(",", ":")),
            "path": path,
        })

    for spec in candidate_specs("all"):
        add(key=f"C:{spec.kind}", name=spec.name, display=spec.name,
            source='candidate_specs("all")', pool="candidate",
            family="candidate", type_name=spec.kind,
            params=dict(spec.params))

    for spec in ALL_SPECS:
        add(key=f"L:{spec.name}", name=spec.name, display=spec.name,
            source="ipd.policies.opponents.ALL_SPECS", pool="legacy",
            family=spec.family, type_name=spec.name, params={})

    for spec in axelrod_specs(include_long_run=True):
        add(key=f"A:{spec.name}", name=spec.name, display=spec.name,
            source="axelrod_specs(include_long_run=True)", pool="axelrod",
            family=spec.family, type_name=spec.name, params={})

    for spec in axelrod_cheating_specs():
        add(key=f"X:{spec.name}", name=spec.name, display=spec.name,
            source="axelrod_cheating_specs()", pool="axelrod_cheating",
            family=spec.family, type_name=spec.name, params={})

    for entry in build_catalog():
        add(key=f"P:{entry.key}", name=entry.key,
            display=f"{entry.family}:{entry.type_name}",
            source="ipd.evaluation.protocol.build_catalog()",
            pool="protocol_v4", family=entry.family,
            type_name=entry.type_name, params=dict(entry.params))

    heldout = (
        ("heldout_v2", "H2", "build_heldout_v2_suite()",
         build_heldout_v2_suite()),
        ("heldout_v3", "H3", "build_heldout_v3_suite()",
         build_heldout_v3_suite()),
        ("heldout_v4", "H4", "build_heldout_v4_suite()",
         build_heldout_v4_suite()),
    )
    for pool, prefix, source, suite in heldout:
        for item in suite:
            add(key=f"{prefix}:{item.name}", name=item.name,
                display=item.name, source=source, pool=pool,
                family=item.family, type_name=item.name, params={})

    for pool, prefix, source, suite in (
        ("validation", "V", "build_validation_suite()",
         build_validation_suite()),
        ("final", "F", "build_final_suite()", build_final_suite()),
    ):
        for item in suite:
            add(key=f"{prefix}:{item.name}", name=item.name,
                display=item.name, source=source, pool=pool,
                family=item.family, type_name=item.name, params={})

    for name in HARD_OPPONENTS:
        add(key=f"T:hard:{name}", name=name, display=name,
            source="HARD_OPPONENTS", pool="hard", family="hard",
            type_name=name, params={"weight": HARD_OPPONENTS[name]})

    training_groups = (
        ("hard2", "HARD2_OPPONENTS", HARD2_OPPONENTS),
        ("periodic", "PERIODIC_HARD_OPPONENTS", PERIODIC_HARD_OPPONENTS),
        ("robust", "ROBUST_OPPONENTS", ROBUST_OPPONENTS),
        ("repair", "REPAIR_OPPONENTS", REPAIR_OPPONENTS),
        ("extort", "EXTORT_HARD_OPPONENTS", EXTORT_HARD_OPPONENTS),
    )
    for pool, source, group in training_groups:
        for name, weight, _factory in group:
            add(key=f"T:{pool}:{name}", name=name, display=name,
                source=source, pool=pool, family=pool, type_name=name,
                params={"weight": weight})

    for path in sorted((ROOT / "weights").glob("**/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fmt = data.get("format")
        rel = path.relative_to(ROOT).as_posix()
        if "checkpoint" in rel.lower() or fmt not in ACCEPTED_WEIGHT_FORMATS:
            continue
        add(key=f"W:{rel}", name=path.stem, display=path.stem,
            source="weights/**/*.json", pool="policy_weight",
            family="policy", type_name=str(fmt), params={"format": fmt},
            path=rel)

    counts = Counter(row["pool"] for row in rows)
    return rows, dict(counts)


def phase0(force: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    started = time.perf_counter()
    rows, counts = build_entities()
    if len(rows) != EXPECTED_ENTITIES:
        raise RuntimeError(
            f"entity count {len(rows)} != {EXPECTED_ENTITIES}; pools={counts}")
    if counts != EXPECTED_POOLS:
        raise RuntimeError(
            f"pool counts do not match frozen spec: {counts} != "
            f"{EXPECTED_POOLS}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if force or not ENTITIES_CSV.exists():
        tmp = ENTITIES_CSV.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["key", "name", "display", "source", "pool",
                                "family", "type", "params", "path"])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, ENTITIES_CSV)
    write_json(ENTITIES_COUNTS, {
        "entities": len(rows),
        "pools": counts,
        "expected_pools": EXPECTED_POOLS,
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": now(),
    })
    cand = [r for r in rows if r["pool"] == "candidate"]
    if len(cand) != 33:
        raise RuntimeError(f"candidate rows {len(cand)} != 33")
    return rows, counts


_ENTITY_CACHE: dict[str, tuple[dict[str, Any], ...]] = {}


def load_entities(path: str | Path = ENTITIES_CSV) -> tuple[dict[str, Any], ...]:
    key = str(path)
    cached = _ENTITY_CACHE.get(key)
    if cached is not None:
        return cached
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row = dict(row)
            row["params_obj"] = json.loads(row.get("params") or "{}")
            rows.append(row)
    _ENTITY_CACHE[key] = tuple(rows)
    return _ENTITY_CACHE[key]


# ---------------------------------------------------------------------------
# Runtime builders (also used inside spawned workers)
# ---------------------------------------------------------------------------

_LEGACY_MAP: dict[str, Any] | None = None
_AXELROD_MAP: dict[str, Any] | None = None
_CHEAT_MAP: dict[str, Any] | None = None
_PROTOCOL_MAP: dict[str, Any] | None = None
_SUITE_MAPS: dict[str, dict[str, Any]] = {}
_TRAINING_MAPS: dict[str, dict[str, Any]] = {}


def _maps() -> dict[str, Any]:
    global _LEGACY_MAP, _AXELROD_MAP, _CHEAT_MAP, _PROTOCOL_MAP
    if _LEGACY_MAP is None:
        from ipd.core.opponents.axelrod_bank import (
            axelrod_cheating_specs, axelrod_specs,
        )
        from ipd.evaluation.protocol import build_catalog
        from ipd.policies.opponents import ALL_SPECS

        _LEGACY_MAP = {spec.name: spec for spec in ALL_SPECS}
        _AXELROD_MAP = {
            spec.name: spec for spec in axelrod_specs(include_long_run=True)}
        _CHEAT_MAP = {
            spec.name: spec for spec in axelrod_cheating_specs()}
        _PROTOCOL_MAP = {entry.key: entry for entry in build_catalog()}
    return {
        "legacy": _LEGACY_MAP,
        "axelrod": _AXELROD_MAP,
        "axelrod_cheating": _CHEAT_MAP,
        "protocol": _PROTOCOL_MAP,
    }


def _suite_map(pool: str) -> dict[str, Any]:
    cached = _SUITE_MAPS.get(pool)
    if cached is not None:
        return cached
    from ipd.evaluation.opponent_suite import (
        build_heldout_v2_suite, build_heldout_v3_suite,
        build_heldout_v4_suite,
    )
    from ipd.training.selfplay.opponents import (
        build_final_suite, build_validation_suite,
    )
    builders = {
        "heldout_v2": build_heldout_v2_suite,
        "heldout_v3": build_heldout_v3_suite,
        "heldout_v4": build_heldout_v4_suite,
        "validation": build_validation_suite,
        "final": build_final_suite,
    }
    mapping = {item.name: item for item in builders[pool]()}
    _SUITE_MAPS[pool] = mapping
    return mapping


def _training_map(pool: str) -> dict[str, Any]:
    cached = _TRAINING_MAPS.get(pool)
    if cached is not None:
        return cached
    from ipd.core.opponents.bank import TYPE_BY_NAME
    from ipd.evaluation.distributions import HARD_OPPONENTS
    from ipd.evaluation.extort import EXTORT_HARD_OPPONENTS
    from ipd.evaluation.hard2 import HARD2_OPPONENTS
    from ipd.evaluation.periodic import PERIODIC_HARD_OPPONENTS
    from ipd.evaluation.robust import REPAIR_OPPONENTS, ROBUST_OPPONENTS

    if pool == "hard":
        mapping = {name: TYPE_BY_NAME[name]
                   for name in HARD_OPPONENTS if name in TYPE_BY_NAME}
    else:
        groups = {
            "hard2": HARD2_OPPONENTS,
            "periodic": PERIODIC_HARD_OPPONENTS,
            "robust": ROBUST_OPPONENTS,
            "repair": REPAIR_OPPONENTS,
            "extort": EXTORT_HARD_OPPONENTS,
        }
        mapping = {name: factory for name, _weight, factory
                   in groups[pool]}
    _TRAINING_MAPS[pool] = mapping
    return mapping


def _candidate_params(row: dict[str, Any]) -> dict[str, Any]:
    params = dict(row.get("params_obj") or {})
    for key in ("seed", "mode", "name"):
        params.pop(key, None)
    return params


def build_candidate(row: dict[str, Any], seed: int,
                    name: str | None = None) -> Any:
    from ipd.policies.registry import make_policy

    return make_policy(
        row["type"],
        seed=int(seed),
        mode="sample",
        name=name or row["display"],
        **_candidate_params(row),
    )


def build_opponent(row: dict[str, Any], seed: int) -> Any:
    pool = row["pool"]
    if pool == "candidate":
        return build_candidate(row, seed)
    if pool == "axelrod_cheating":
        # Darwin keeps a class-level genome that survives reset() and new
        # instances.  Reset it before every match so candidate workers can be
        # reused without leaking state between candidates / shards.
        try:
            from axelrod.strategies.darwin import Darwin
        except ImportError:  # pragma: no cover - axelrod is a dependency
            pass
        else:
            Darwin.reset_genome()
    if pool in ("legacy", "axelrod", "axelrod_cheating"):
        table = _maps()[
            "protocol" if pool == "protocol" else pool]
        spec = table[row["name"]]
        return spec.factory(random.Random(int(seed)))
    if pool == "protocol_v4":
        entry = _maps()["protocol"][row["name"]]
        return entry.build(random.Random(int(seed)))
    if pool in ("heldout_v2", "heldout_v3", "heldout_v4",
                "validation", "final"):
        item = _suite_map(pool)[row["name"]]
        return item.factory(random.Random(int(seed)))
    if pool in ("hard", "hard2", "periodic", "robust", "repair", "extort"):
        factory = _training_map(pool)[row["name"]]
        if pool == "hard":
            return factory.factory(random.Random(int(seed)))
        return factory(random.Random(int(seed)))
    if pool == "policy_weight":
        from ipd.evaluation.policy_pool import load_policy

        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        return load_policy(str(path), seed=int(seed), mode="sample")
    raise KeyError(f"unknown opponent pool {pool!r}")


# ---------------------------------------------------------------------------
# One matched row
# ---------------------------------------------------------------------------


def _row_payload(candidate: dict[str, Any], opponent: dict[str, Any],
                 rep: int, branch: str, device: str, seed_candidate: int,
                 seed_opponent: int, candidate_score: int,
                 opponent_score: int, mine: Sequence[int],
                 theirs: Sequence[int], actual_candidate: str,
                 actual_opponent: str) -> dict[str, Any]:
    return {
        "pair_id": f"{candidate['key']}|{opponent['key']}|{rep}",
        "rep": rep,
        "branch": branch,
        "candidate_key": candidate["key"],
        "candidate_name": candidate["display"],
        "opponent_key": opponent["key"],
        "opponent_name": opponent["name"],
        "opponent_pool": opponent["pool"],
        "opponent_family": opponent["family"],
        "seed_candidate": seed_candidate,
        "seed_opponent": seed_opponent,
        "candidate_score": int(candidate_score),
        "opponent_score": int(opponent_score),
        "diff_candidate_minus_opponent": int(candidate_score)
        - int(opponent_score),
        "candidate_action_hash": action_hash(mine, theirs),
        "opponent_action_hash": action_hash(theirs, mine),
        "actual_name_candidate": str(actual_candidate),
        "actual_name_opponent": str(actual_opponent),
        "mode": "sample",
        "device": device,
    }


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LONG_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Phase 1: stratified sample parity
# ---------------------------------------------------------------------------


def _pick(entities: Sequence[dict[str, Any]], pool: str,
          name: str) -> dict[str, Any] | None:
    for row in entities:
        if row["pool"] != pool:
            continue
        if (row["type"] if pool == "candidate" else row["name"]) == name:
            return row
    return None


def select_phase1_opponents(
        entities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    from ipd.core.opponents.axelrod_bank import _load_axelrod
    from ipd.evaluation.protocol import build_catalog

    catalog = {entry.key: entry for entry in build_catalog()}
    long_run = {c.__name__ for c in _load_axelrod().long_run_time_strategies}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    reasons: dict[str, str] = {}

    def add(row: dict[str, Any] | None, reason: str) -> bool:
        if row is None or row["key"] in seen:
            return False
        selected.append(row)
        seen.add(row["key"])
        reasons[row["key"]] = reason
        return True

    def first(pool: str, *, family: str | None = None,
              type_name: str | None = None,
              tier: str | None = None) -> dict[str, Any] | None:
        for row in entities:
            if row["pool"] != pool:
                continue
            if family is not None and row["family"] != family:
                continue
            if type_name is not None and row["type"] != type_name:
                continue
            if tier is not None:
                entry = catalog.get(row["name"])
                if entry is None or entry.tier != tier:
                    continue
            return row
        return None

    # Required anchors.
    for name in ("AllC", "AllD", "TFT", "Grim"):
        add(_pick(entities, "legacy", name), f"legacy anchor {name}")

    random_fsm = first("protocol_v4", type_name="RandomFSM")
    add(random_fsm, "protocol RandomFSM")
    probe_bot = first("protocol_v4", type_name="ProbeBot")
    add(probe_bot, "protocol ProbeBot")

    # At least five Axelrod long-run strategies (six cheap ones here).
    for name in PHASE1_LONG_RUN:
        add(_pick(entities, "axelrod", name), f"axelrod long-run {name}")

    # Protocol complexity / test-only coverage.
    complex_row = first("protocol_v4", tier="test_complex")
    add(complex_row, "protocol test_complex")
    test_only_row = first("protocol_v4", tier="test_only")
    add(test_only_row, "protocol test_only")

    # At least ten policy weights; cover every accepted format.
    weights = [row for row in entities if row["pool"] == "policy_weight"]
    for fmt in sorted(ACCEPTED_WEIGHT_FORMATS):
        add(next((row for row in weights if row["type"] == fmt), None),
            f"policy weight {fmt}")
    step = max(1, len(weights) // 12)
    for row in weights[::step]:
        if len([r for r in selected
                if r["pool"] == "policy_weight"]) >= 12:
            break
        add(row, "policy weight sample")
    for row in weights:
        if len([r for r in selected
                if r["pool"] == "policy_weight"]) >= 12:
            break
        add(row, "policy weight fill")

    # At least five training random factories.
    training_quotas = {
        "hard2": 2, "periodic": 2, "robust": 2, "repair": 2,
        "extort": 1, "hard": 1,
    }
    for pool, quota in training_quotas.items():
        added = 0
        for row in entities:
            if row["pool"] != pool:
                continue
            if add(row, f"training factory {pool}"):
                added += 1
            if added >= quota:
                break

    # Candidate-vs-candidate column, including an explicit self-play pair.
    for kind in ("selfplay", "neural", "gru", "final-tuned", "triple-v4",
                 "adv-mlp", "psro-scratch", "rl-wide", "bayes-tuned",
                 "harvest45"):
        add(_pick(entities, "candidate", kind), f"candidate column {kind}")

    # Heldout / validation / final coverage.
    for pool, count in (("heldout_v2", 2), ("heldout_v3", 2),
                        ("heldout_v4", 2), ("validation", 1), ("final", 1)):
        taken = 0
        for row in entities:
            if row["pool"] == pool:
                if add(row, f"{pool} sample"):
                    taken += 1
                if taken >= count:
                    break

    # Diverse legacy and protocol rows.
    for row in entities:
        if row["pool"] == "legacy":
            add(row, "legacy diversity")
        if len([r for r in selected if r["pool"] == "legacy"]) >= 16:
            break
    for row in entities:
        if row["pool"] == "protocol_v4":
            add(row, "protocol diversity")
        if len([r for r in selected if r["pool"] == "protocol_v4"]) >= 14:
            break

    # Ordinary Axelrod rows (avoid long-run to keep the gate quick).
    for row in entities:
        if row["pool"] == "axelrod" and row["name"] not in long_run:
            add(row, "axelrod ordinary")
        if len([r for r in selected if r["pool"] == "axelrod"]) >= 14:
            break

    # Fill to exactly 100 from the remaining frozen entities.
    fill_order = ("candidate", "legacy", "protocol_v4", "axelrod",
                  "heldout_v2", "heldout_v3", "heldout_v4", "validation",
                  "final", "hard", "hard2", "periodic", "robust", "repair",
                  "extort", "policy_weight", "axelrod_cheating")
    cursor = 0
    while len(selected) < 100:
        progressed = False
        for pool in fill_order:
            rows = [row for row in entities
                    if row["pool"] == pool and row["key"] not in seen]
            if rows:
                add(rows[cursor % len(rows)], "stratified fill")
                progressed = True
                if len(selected) >= 100:
                    break
        cursor += 1
        if not progressed:
            break
    if len(selected) != 100:
        raise RuntimeError(f"Phase 1 sample has {len(selected)} rows, want 100")

    # Hard coverage assertions.
    def count(pred) -> int:
        return sum(1 for row in selected if pred(row))

    assert count(lambda r: r["pool"] == "legacy" and
                 r["name"] in {"AllC", "AllD", "TFT", "Grim"}) >= 4
    assert count(lambda r: r["type"] in {"RandomFSM", "ProbeBot"}) >= 2
    assert count(lambda r: r["pool"] == "axelrod" and
                 r["name"] in long_run) >= 5
    assert count(lambda r: r["pool"] == "protocol_v4") >= 2
    assert count(lambda r: r["pool"] == "policy_weight") >= 10
    assert count(lambda r: r["pool"] in {
        "hard", "hard2", "periodic", "robust", "repair", "extort"}) >= 5
    assert count(lambda r: r["pool"] == "candidate") >= 2
    assert _pick(selected, "candidate", "selfplay") is not None
    return selected


def _phase1_candidate_task(task: dict[str, Any]) -> dict[str, Any]:
    import torch
    from ipd.evaluation.common import run_match
    from ipd.evaluation.gpu.simulator import evaluate_field_gpu
    from ipd.evaluation.protocol import sample_uniforms

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    started = time.perf_counter()
    entities = load_entities(task["entities_csv"])
    by_key = {row["key"]: row for row in entities}
    candidate = by_key[task["candidate_key"]]
    sample = [by_key[key] for key in task["sample_keys"]]
    reps = int(task["reps"])
    rounds = int(task["rounds"])

    cpu_rows: list[dict[str, Any]] = []
    for rep in range(reps):
        for opponent in sample:
            seed_c = match_seed(
                int(task["base_seed"]), candidate["key"], opponent["key"],
                rep)
            seed_o = opponent_seed_for(seed_c)
            cpu_policy = build_candidate(candidate, seed_c)
            cpu_opponent = build_opponent(opponent, seed_o)
            stats = run_match(cpu_policy, cpu_opponent, rounds=rounds)
            cpu_rows.append({
                "rep": rep,
                "opponent": opponent,
                "seed_candidate": seed_c,
                "seed_opponent": seed_o,
                "score": int(stats.score),
                "opponent_score": int(stats.opponent_score),
                "mine": list(stats.mine),
                "theirs": list(stats.theirs),
                "actual_candidate": str(getattr(
                    cpu_policy, "name", type(cpu_policy).__name__)),
                "actual_opponent": str(stats.opponent_name),
            })

    gpu_meta: list[tuple[int, dict[str, Any], int]] = []
    gpu_opponents: list[Any] = []
    uniform_rows: list[list[float]] = []
    for rep in range(reps):
        for opponent in sample:
            seed = match_seed(int(task["base_seed"]), candidate["key"],
                              opponent["key"], rep)
            gpu_meta.append((rep, opponent, seed))
            gpu_opponents.append(
                build_opponent(opponent, opponent_seed_for(seed)))
            uniform_rows.append(sample_uniforms(seed, rounds))
    uniforms = torch.tensor(uniform_rows, dtype=torch.float64)
    gpu_policy = build_candidate(candidate, int(task["base_seed"]))
    simulation = evaluate_field_gpu(
        gpu_policy, gpu_opponents, rounds=rounds, device="cuda",
        dtype=torch.float64, max_batch=int(task["max_batch"]),
        fast_rules=False, blue_mode="sample", blue_uniforms=uniforms,
        policy_opponent_mode="sample", keep_histories=True,
    )
    if len(cpu_rows) != len(gpu_meta):
        raise RuntimeError("phase1 CPU/GPU row count mismatch")

    mismatches: list[dict[str, Any]] = []
    passed = failed = 0
    score_mismatches = action_mismatches = name_mismatches = 0
    max_diff = 0
    for index, (cpu, meta) in enumerate(zip(cpu_rows, gpu_meta)):
        rep, opponent, _seed = meta
        gpu_score = int(simulation.scores[index])
        gpu_opp_score = int(simulation.opponent_scores[index])
        gpu_mine = [int(x) for x in simulation.mine[index]]
        gpu_theirs = [int(x) for x in simulation.theirs[index]]
        gpu_actual_candidate = str(getattr(
            gpu_policy, "name", type(gpu_policy).__name__))
        gpu_actual_opponent = str(getattr(
            gpu_opponents[index], "name", simulation.names[index]))

        score_bad = (cpu["score"] != gpu_score or
                     cpu["opponent_score"] != gpu_opp_score)
        cpu_hash_c = action_hash(cpu["mine"], cpu["theirs"])
        gpu_hash_c = action_hash(gpu_mine, gpu_theirs)
        cpu_hash_o = action_hash(cpu["theirs"], cpu["mine"])
        gpu_hash_o = action_hash(gpu_theirs, gpu_mine)
        action_bad = (cpu_hash_c != gpu_hash_c or
                      cpu_hash_o != gpu_hash_o)
        name_bad = (cpu["actual_candidate"] != gpu_actual_candidate or
                    cpu["actual_opponent"] != gpu_actual_opponent)
        diff = max(abs(cpu["score"] - gpu_score),
                   abs(cpu["opponent_score"] - gpu_opp_score))
        max_diff = max(max_diff, diff)
        score_mismatches += int(score_bad)
        action_mismatches += int(action_bad)
        name_mismatches += int(name_bad)
        if score_bad or action_bad or name_bad:
            failed += 1
            reasons = []
            if score_bad:
                reasons.append("score")
            if action_bad:
                reasons.append("action")
            if name_bad:
                reasons.append("name")
            mismatches.append({
                "candidate_key": candidate["key"],
                "candidate_kind": candidate["type"],
                "rep": rep,
                "opponent_key": opponent["key"],
                "opponent_pool": opponent["pool"],
                "seed_candidate": cpu["seed_candidate"],
                "seed_opponent": cpu["seed_opponent"],
                "cpu_candidate_score": cpu["score"],
                "gpu_candidate_score": gpu_score,
                "cpu_opponent_score": cpu["opponent_score"],
                "gpu_opponent_score": gpu_opp_score,
                "cpu_candidate_action_hash": cpu_hash_c,
                "gpu_candidate_action_hash": gpu_hash_c,
                "cpu_opponent_action_hash": cpu_hash_o,
                "gpu_opponent_action_hash": gpu_hash_o,
                "cpu_actual_name_candidate": cpu["actual_candidate"],
                "gpu_actual_name_candidate": gpu_actual_candidate,
                "cpu_actual_name_opponent": cpu["actual_opponent"],
                "gpu_actual_name_opponent": gpu_actual_opponent,
                "reasons": "+".join(reasons),
            })
        else:
            passed += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "candidate_kind": candidate["type"],
        "breakdown": {
            "pairs_checked": len(cpu_rows),
            "pairs_passed": passed,
            "pairs_failed": failed,
            "score_mismatches": score_mismatches,
            "action_mismatches": action_mismatches,
            "name_mismatches": name_mismatches,
            "max_score_diff": max_diff,
            "seconds": time.perf_counter() - started,
        },
        "mismatches": mismatches,
    }


def phase1(entities: Sequence[dict[str, Any]], *, force: bool = False,
           workers: int = GPU_WORKERS) -> bool:
    out = OUT_ROOT / "phase1_sample_parity"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "parity_sample_report.json"
    mismatch_path = out / "parity_sample_mismatches.csv"
    sample = select_phase1_opponents(entities)
    expected = len(GPU_CANDIDATES) * len(sample) * REPS
    if report_path.exists() and not force:
        report = read_json(report_path)
        if report.get("pairs_checked") == expected:
            LOG.info("phase1: cached report found, pass=%s",
                     report.get("pass"))
            return bool(report.get("pass"))

    candidate_rows = {row["type"]: row for row in entities
                      if row["pool"] == "candidate"}
    started = time.perf_counter()
    LOG.info("phase1: starting %d GPU candidates x %d stratified opponents "
             "x %d reps = %d matches; pools=%s", len(GPU_CANDIDATES),
             len(sample), REPS, expected,
             dict(Counter(row["pool"] for row in sample)))
    tasks = [{
        "entities_csv": str(ENTITIES_CSV),
        "candidate_key": candidate_rows[kind]["key"],
        "sample_keys": [row["key"] for row in sample],
        "base_seed": BASE_SEED,
        "reps": REPS,
        "rounds": ROUNDS,
        "max_batch": GPU_MAX_BATCH,
    } for kind in GPU_CANDIDATES]
    results: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_phase1_candidate_task, task):
                   task["candidate_key"] for task in tasks}
        for done_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            kind = result["candidate_kind"]
            results[kind] = result["breakdown"]
            mismatches.extend(result["mismatches"])
            elapsed = time.perf_counter() - started
            eta = elapsed / done_count * (len(tasks) - done_count)
            LOG.info("phase1: %d/%d (%5.1f%%) %s %d/%d pass in %.1fs; "
                     "elapsed=%.1fs eta=%.1fs", done_count, len(tasks),
                     100.0 * done_count / len(tasks), kind,
                     result["breakdown"]["pairs_passed"],
                     result["breakdown"]["pairs_checked"],
                     result["breakdown"]["seconds"], elapsed, eta)

    breakdown = {kind: results[kind] for kind in GPU_CANDIDATES}
    pairs_checked = sum(item["pairs_checked"] for item in breakdown.values())
    pairs_passed = sum(item["pairs_passed"] for item in breakdown.values())
    pairs_failed = sum(item["pairs_failed"] for item in breakdown.values())
    report = {
        "pairs_checked": pairs_checked,
        "pairs_passed": pairs_passed,
        "pairs_failed": pairs_failed,
        "score_mismatches": sum(item["score_mismatches"]
                                for item in breakdown.values()),
        "action_mismatches": sum(item["action_mismatches"]
                                 for item in breakdown.values()),
        "name_mismatches": sum(item["name_mismatches"]
                               for item in breakdown.values()),
        "max_score_diff": max(
            (item["max_score_diff"] for item in breakdown.values()),
            default=0),
        "pass": bool(pairs_checked == expected and pairs_failed == 0),
        "gate": "100% score + action-hash + actual-name equality",
        "gpu_candidates": list(GPU_CANDIDATES),
        "sampled_opponents": len(sample),
        "reps": REPS,
        "sample_pools": dict(Counter(row["pool"] for row in sample)),
        "sample_keys": [row["key"] for row in sample],
        "candidate_breakdown": breakdown,
        "rounds": ROUNDS,
        "base_seed": BASE_SEED,
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": now(),
    }
    write_json(report_path, report)
    fields = [
        "candidate_key", "candidate_kind", "rep", "opponent_key",
        "opponent_pool", "seed_candidate", "seed_opponent",
        "cpu_candidate_score", "gpu_candidate_score",
        "cpu_opponent_score", "gpu_opponent_score",
        "cpu_candidate_action_hash", "gpu_candidate_action_hash",
        "cpu_opponent_action_hash", "gpu_opponent_action_hash",
        "cpu_actual_name_candidate", "gpu_actual_name_candidate",
        "cpu_actual_name_opponent", "gpu_actual_name_opponent", "reasons",
    ]
    tmp = mismatch_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mismatches)
    os.replace(tmp, mismatch_path)
    LOG.info("phase1: finished %d/%d pass, failures=%d, pass=%s, "
             "elapsed=%.1fs", pairs_passed, pairs_checked, pairs_failed,
             report["pass"], report["elapsed_seconds"])
    return bool(report["pass"])


# ---------------------------------------------------------------------------
# Candidate-level CPU runner (Phases 2 and 3)
# ---------------------------------------------------------------------------


def _cpu_candidate_task(task: dict[str, Any]) -> dict[str, Any]:
    entities = load_entities(task["entities_csv"])
    by_key = {row["key"]: row for row in entities}
    candidate = by_key[task["candidate_key"]]
    opponents = entities[int(task["start"]):int(task["end"])]
    branch = str(task["branch"])
    device = "cpu"
    path = Path(task["output_csv"])
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    reps = int(task["reps"])

    from ipd.evaluation.common import run_match

    for rep in range(reps):
        for opponent in opponents:
            seed_c = match_seed(
                int(task["base_seed"]), candidate["key"], opponent["key"],
                rep)
            seed_o = opponent_seed_for(seed_c)
            policy = build_candidate(candidate, seed_c)
            opponent_obj = build_opponent(opponent, seed_o)
            stats = run_match(policy, opponent_obj,
                              rounds=int(task["rounds"]))
            rows.append(_row_payload(
                candidate, opponent, rep, branch, device,
                seed_c, seed_o, stats.score, stats.opponent_score,
                stats.mine, stats.theirs,
                getattr(policy, "name", type(policy).__name__),
                stats.opponent_name,
            ))
    _write_rows(path, rows)
    return {
        "candidate_key": candidate["key"],
        "rows": len(rows),
        "seconds": time.perf_counter() - started,
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# Phase 2: provisional parity branch
# ---------------------------------------------------------------------------


def _gpu_shard_task(task: dict[str, Any]) -> dict[str, Any]:
    import torch
    from ipd.evaluation.gpu.simulator import evaluate_field_gpu
    from ipd.evaluation.protocol import sample_uniforms

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    entities = load_entities(task["entities_csv"])
    by_key = {row["key"]: row for row in entities}
    candidate = by_key[task["candidate_key"]]
    indices = [int(index) for index in task["indices"]]
    opponents = [entities[index] for index in indices]
    reps = int(task["reps"])
    rounds = int(task["rounds"])

    flat_opponents: list[dict[str, Any]] = []
    flat_objects: list[Any] = []
    flat_seeds: list[int] = []
    flat_reps: list[int] = []
    for rep in range(reps):
        for opponent in opponents:
            seed = match_seed(int(task["base_seed"]), candidate["key"],
                              opponent["key"], rep)
            flat_opponents.append(opponent)
            flat_objects.append(
                build_opponent(opponent, opponent_seed_for(seed)))
            flat_seeds.append(seed)
            flat_reps.append(rep)
    uniforms = torch.tensor(
        [sample_uniforms(seed, rounds) for seed in flat_seeds],
        dtype=torch.float64,
    )
    policy = build_candidate(candidate, int(task["base_seed"]))
    started = time.perf_counter()
    simulation = evaluate_field_gpu(
        policy, flat_objects, rounds=rounds, device="cuda",
        dtype=torch.float64, max_batch=int(task["max_batch"]),
        fast_rules=False, blue_mode="sample", blue_uniforms=uniforms,
        policy_opponent_mode="sample", keep_histories=True,
    )
    rows: list[dict[str, Any]] = []
    for index, (opponent, seed, rep) in enumerate(
            zip(flat_opponents, flat_seeds, flat_reps)):
        rows.append(_row_payload(
            candidate, opponent, rep, "parity", "cuda", seed,
            opponent_seed_for(seed),
            int(simulation.scores[index]),
            int(simulation.opponent_scores[index]),
            [int(x) for x in simulation.mine[index]],
            [int(x) for x in simulation.theirs[index]],
            getattr(policy, "name", type(policy).__name__),
            str(getattr(flat_objects[index], "name",
                        simulation.names[index])),
        ))
    path = Path(task["output_csv"])
    _write_rows(path, rows)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "candidate_key": candidate["key"],
        "shard": int(task["shard"]),
        "rows": len(rows),
        "seconds": time.perf_counter() - started,
        "path": str(path),
    }


def _completed_path(directory: Path) -> Path:
    return directory / "completed.json"


def _update_completed(directory: Path, key: str, meta: dict[str, Any]) -> None:
    path = _completed_path(directory)
    data = read_json(path) if path.exists() else {}
    data[key] = {**meta, "completed_at": now()}
    write_json(path, data)


def _run_cpu_candidates(entities: Sequence[dict[str, Any]], *,
                        branch: str, out_dir: Path, candidate_rows:
                        Sequence[dict[str, Any]], max_workers: int,
                        force: bool = False) -> dict[str, Any]:
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_path(shard_dir)
    done = read_json(completed) if completed.exists() else {}
    tasks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    expected_rows = EXPECTED_ENTITIES * REPS
    for candidate in candidate_rows:
        key = candidate["key"]
        out_path = shard_dir / f"{safe_name(key)}.csv"
        if (not force and key in done and out_path.exists()
                and count_csv_rows(out_path) == expected_rows):
            results.append({"candidate_key": key, "rows": expected_rows,
                            "seconds": 0.0, "cached": True,
                            "path": str(out_path)})
            continue
        tasks.append({
            "entities_csv": str(ENTITIES_CSV),
            "candidate_key": key,
            "start": 0,
            "end": len(entities),
            "branch": branch,
            "base_seed": BASE_SEED,
            "reps": REPS,
            "rounds": ROUNDS,
            "output_csv": str(out_path),
        })
    if tasks:
        LOG.info("%s-cpu: scheduled %d candidates (%d cached), workers=%d",
                 branch, len(tasks), len(results), max_workers)
        batch_started = time.perf_counter()
        with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(_cpu_candidate_task, task)
                       for task in tasks]
            for done_count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                _update_completed(shard_dir, result["candidate_key"], result)
                elapsed = time.perf_counter() - batch_started
                eta = (elapsed / done_count) * (len(tasks) - done_count)
                LOG.info("%s-cpu: %d/%d (%5.1f%%) %s rows=%d in %.1fs; "
                         "elapsed=%.1fs eta=%.1fs", branch, done_count,
                         len(tasks), 100.0 * done_count / len(tasks),
                         result["candidate_key"], result["rows"],
                         result["seconds"], elapsed, eta)
    else:
        LOG.info("%s-cpu: all %d candidates already complete",
                 branch, len(candidate_rows))
    for candidate in candidate_rows:
        key = candidate["key"]
        out_path = shard_dir / f"{safe_name(key)}.csv"
        got = count_csv_rows(out_path)
        if got != expected_rows:
            raise RuntimeError(
                f"{branch}: {key} has {got} rows, want {expected_rows}")
    return {
        "rows": sum(item["rows"] for item in results),
        "seconds": sum(item["seconds"] for item in results),
        "candidates": len(candidate_rows),
        "cached": sum(1 for item in results if item.get("cached")),
    }


def _run_gpu_candidates(entities: Sequence[dict[str, Any]],
                        candidate_rows: Sequence[dict[str, Any]],
                        out_dir: Path, *, max_workers: int,
                        force: bool = False) -> dict[str, Any]:
    shard_root = out_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    # Interleave entities so the very slow Axelrod long-run opponents are
    # spread over every shard instead of all landing in part_00.
    shard_indices = {
        index: list(range(index, len(entities), GPU_SHARDS))
        for index in range(GPU_SHARDS)
    }
    for candidate in candidate_rows:
        key = candidate["key"]
        cand_dir = shard_root / safe_name(key)
        cand_dir.mkdir(parents=True, exist_ok=True)
        marker = cand_dir / "completed.json"
        expected = {
            index: cand_dir / f"part_{index:02d}.csv"
            for index in range(GPU_SHARDS)
        }
        if (not force and marker.exists()
                and all(count_csv_rows(path) == len(shard_indices[index])
                        * REPS for index, path in expected.items())):
            results.append({"candidate_key": key,
                            "rows": sum(count_csv_rows(p)
                                        for p in expected.values()),
                            "seconds": 0.0, "cached": True})
            continue
        for index, indices in shard_indices.items():
            out_path = expected[index]
            want = len(indices) * REPS
            if (not force and out_path.exists()
                    and count_csv_rows(out_path) == want):
                results.append({"candidate_key": key, "shard": index,
                                "rows": want, "seconds": 0.0,
                                "cached": True, "path": str(out_path)})
                continue
            tasks.append({
                "entities_csv": str(ENTITIES_CSV),
                "candidate_key": key,
                "shard": index,
                "indices": indices,
                "base_seed": BASE_SEED,
                "reps": REPS,
                "rounds": ROUNDS,
                "max_batch": GPU_MAX_BATCH,
                "output_csv": str(out_path),
            })
    if tasks:
        LOG.info("parity-gpu: scheduled %d shards (%d cached), workers=%d, "
                 "shards_per_candidate=%d", len(tasks),
                 len(results), max_workers, GPU_SHARDS)
        batch_started = time.perf_counter()
        with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(_gpu_shard_task, task) for task in tasks]
            for done_count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                elapsed = time.perf_counter() - batch_started
                eta = (elapsed / done_count) * (len(tasks) - done_count)
                LOG.info("parity-gpu: %d/%d (%5.1f%%) %s part %s rows=%d "
                         "in %.1fs; elapsed=%.1fs eta=%.1fs",
                         done_count, len(tasks),
                         100.0 * done_count / len(tasks),
                         result["candidate_key"], result.get("shard"),
                         result["rows"], result["seconds"], elapsed, eta)
    else:
        LOG.info("parity-gpu: all shards already complete")
    # Validate and mark each candidate.
    for cand_index, candidate in enumerate(candidate_rows, 1):
        key = candidate["key"]
        cand_dir = shard_root / safe_name(key)
        total = 0
        for index, indices in shard_indices.items():
            path = cand_dir / f"part_{index:02d}.csv"
            want = len(indices) * REPS
            got = count_csv_rows(path)
            if got != want:
                raise RuntimeError(
                    f"parity-gpu: {key} part {index} has {got} rows, "
                    f"want {want}")
            total += got
        expected_total = EXPECTED_ENTITIES * REPS
        if total != expected_total:
            raise RuntimeError(
                f"parity-gpu: {key} has {total} rows != {expected_total}")
        write_json(cand_dir / "completed.json", {
            "candidate_key": key, "rows": total, "completed_at": now(),
        })
        LOG.info("parity-gpu: candidate complete %d/%d %s rows=%d",
                 cand_index, len(candidate_rows), key, total)
    return {
        "rows": sum(item["rows"] for item in results),
        "seconds": sum(item["seconds"] for item in results),
        "candidates": len(candidate_rows),
        "cached": sum(1 for item in results if item.get("cached")),
    }


def _candidate_order(entities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in entities if row["pool"] == "candidate"]


def phase2(entities: Sequence[dict[str, Any]], *, cpu_workers: int,
           gpu_workers: int, force: bool = False) -> dict[str, Any]:
    out = OUT_ROOT / "phase2_parity_full"
    out.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_order(entities)
    by_kind = {row["type"]: row for row in candidates}
    gpu_rows = [by_kind[kind] for kind in GPU_CANDIDATES]
    cpu_rows = [by_kind[kind] for kind in CPU_CANDIDATES]
    if len(gpu_rows) != 13 or len(cpu_rows) != 20:
        raise RuntimeError(f"routing mismatch: {len(gpu_rows)} GPU, "
                           f"{len(cpu_rows)} CPU")
    started = time.perf_counter()
    LOG.info("phase2: starting provisional parity branch: %d GPU candidates "
             "+ %d CPU candidates x %d opponents x %d reps = %d matches",
             len(gpu_rows), len(cpu_rows), len(entities), REPS,
             len(candidates) * len(entities) * REPS)
    gpu_stats = _run_gpu_candidates(
        entities, gpu_rows, out, max_workers=gpu_workers, force=force)
    LOG.info("phase2-gpu: branch complete: %s", gpu_stats)
    cpu_stats = _run_cpu_candidates(
        entities, branch="parity", out_dir=out, candidate_rows=cpu_rows,
        max_workers=cpu_workers, force=force)
    LOG.info("phase2-cpu: branch complete: %s", cpu_stats)
    elapsed = time.perf_counter() - started
    meta = {
        "phase": 2,
        "branch": "parity",
        "status": "provisional",
        "authoritative": False,
        "created_at": now(),
        "elapsed_seconds": elapsed,
        "base_seed": BASE_SEED,
        "rounds": ROUNDS,
        "reps": REPS,
        "entities": len(entities),
        "candidates": len(candidates),
        "pairs_expected": EXPECTED_PAIRS,
        "gpu_candidates": list(GPU_CANDIDATES),
        "cpu_candidates": list(CPU_CANDIDATES),
        "gpu_stats": gpu_stats,
        "cpu_stats": cpu_stats,
        "settings": {
            "mode": "sample",
            "device": "cuda for GPU candidates, cpu otherwise",
            "dtype": "float64",
            "fast_rules": False,
            "gpu": "on for the 13 supported candidates, off otherwise",
            "max_batch": GPU_MAX_BATCH,
            "keep_histories": True,
            "blue_mode": "sample",
            "policy_opponent_mode": "sample",
        },
    }
    write_json(out / "parity_run.json", meta)
    return meta


# ---------------------------------------------------------------------------
# Merge, matrices, and validation
# ---------------------------------------------------------------------------


def _iter_candidate_files(entities: Sequence[dict[str, Any]], out_dir: Path,
                          *, gpu_split: bool) -> Iterable[Path]:
    shard_dir = out_dir / "shards"
    for candidate in _candidate_order(entities):
        key = candidate["key"]
        if gpu_split and candidate["type"] in GPU_CANDIDATES:
            cand_dir = shard_dir / safe_name(key)
            for index in range(GPU_SHARDS):
                path = cand_dir / f"part_{index:02d}.csv"
                if path.exists():
                    yield path
        else:
            yield shard_dir / f"{safe_name(key)}.csv"


def merge_and_write(
        entities: Sequence[dict[str, Any]], *, out_dir: Path,
        long_name: str, matrix_name: str, opponent_matrix_name: str,
        gpu_split: bool, expected_branch: str,
        expected_device: str | None,
) -> dict[str, Any]:
    LOG.info("merge: writing %s ...", out_dir / long_name)
    long_path = out_dir / long_name
    tmp = long_path.with_suffix(".csv.tmp")
    score_sum: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    opp_sum: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    diff_sum: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    diff_min: dict[str, dict[str, int]] = defaultdict(dict)
    diff_max: dict[str, dict[str, int]] = defaultdict(dict)
    pair_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    counts_by_candidate: Counter[str] = Counter()
    counts_by_device: Counter[str] = Counter()
    total = 0
    with tmp.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=LONG_CSV_FIELDS)
        writer.writeheader()
        for path in _iter_candidate_files(entities, out_dir,
                                          gpu_split=gpu_split):
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row["branch"] != expected_branch:
                        raise RuntimeError(
                            f"{path}: branch {row['branch']} != "
                            f"{expected_branch}")
                    if expected_device and row["device"] != expected_device:
                        raise RuntimeError(
                            f"{path}: device {row['device']} != "
                            f"{expected_device}")
                    writer.writerow(row)
                    total += 1
                    counts_by_candidate[row["candidate_key"]] += 1
                    counts_by_device[row["device"]] += 1
                    key = row["candidate_key"]
                    opponent_key = row["opponent_key"]
                    score = int(row["candidate_score"])
                    opp_score = int(row["opponent_score"])
                    diff = int(row["diff_candidate_minus_opponent"])
                    score_sum[key][opponent_key] += score
                    opp_sum[key][opponent_key] += opp_score
                    diff_sum[key][opponent_key] += diff
                    pair_counts[key][opponent_key] += 1
                    if opponent_key not in diff_min[key]:
                        diff_min[key][opponent_key] = diff
                        diff_max[key][opponent_key] = diff
                    else:
                        diff_min[key][opponent_key] = min(
                            diff_min[key][opponent_key], diff)
                        diff_max[key][opponent_key] = max(
                            diff_max[key][opponent_key], diff)
    os.replace(tmp, long_path)

    entity_keys = [row["key"] for row in entities]
    candidate_rows = _candidate_order(entities)
    means: dict[str, dict[str, float]] = defaultdict(dict)
    for key, values in score_sum.items():
        for opponent_key, value in values.items():
            means[key][opponent_key] = (
                value / pair_counts[key][opponent_key])

    pair_means_path = out_dir / (
        long_name.replace("_long.csv", "_pair_means.csv"))
    tmp = pair_means_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "candidate_key", "candidate_name", "opponent_key",
            "opponent_name", "opponent_pool", "opponent_family", "n_reps",
            "mean_candidate_score", "mean_opponent_score", "mean_diff",
            "min_diff", "max_diff",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for candidate in candidate_rows:
            key = candidate["key"]
            for entity in entities:
                opponent_key = entity["key"]
                n = pair_counts[key][opponent_key]
                if n == 0:
                    continue
                writer.writerow({
                    "candidate_key": key,
                    "candidate_name": candidate["display"],
                    "opponent_key": opponent_key,
                    "opponent_name": entity["name"],
                    "opponent_pool": entity["pool"],
                    "opponent_family": entity["family"],
                    "n_reps": n,
                    "mean_candidate_score": (
                        score_sum[key][opponent_key] / n),
                    "mean_opponent_score": (
                        opp_sum[key][opponent_key] / n),
                    "mean_diff": diff_sum[key][opponent_key] / n,
                    "min_diff": diff_min[key][opponent_key],
                    "max_diff": diff_max[key][opponent_key],
                })
    os.replace(tmp, pair_means_path)

    matrix_path = out_dir / matrix_name
    tmp = matrix_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_key", "candidate_name", *entity_keys])
        for candidate in candidate_rows:
            key = candidate["key"]
            writer.writerow([
                key, candidate["display"],
                *[means[key].get(opponent_key, "") for opponent_key in
                  entity_keys],
            ])
    os.replace(tmp, matrix_path)

    opponent_matrix_path = out_dir / opponent_matrix_name
    tmp = opponent_matrix_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "opponent_key", "opponent_name", "opponent_pool",
            "opponent_family", "n_candidates", "n_matches",
            "mean_candidate_score", "mean_opponent_score", "mean_diff",
            "min_diff", "max_diff",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for entity in entities:
            key = entity["key"]
            cand_sum = 0
            opp_total = 0
            diff_total = 0
            n_matches = 0
            dmin = None
            dmax = None
            for c in candidate_rows:
                n = pair_counts[c["key"]][key]
                if not n:
                    continue
                cand_sum += score_sum[c["key"]][key]
                opp_total += opp_sum[c["key"]][key]
                diff_total += diff_sum[c["key"]][key]
                n_matches += n
                lo = diff_min[c["key"]][key]
                hi = diff_max[c["key"]][key]
                dmin = lo if dmin is None else min(dmin, lo)
                dmax = hi if dmax is None else max(dmax, hi)
            writer.writerow({
                "opponent_key": key,
                "opponent_name": entity["name"],
                "opponent_pool": entity["pool"],
                "opponent_family": entity["family"],
                "n_candidates": sum(
                    1 for c in candidate_rows
                    if pair_counts[c["key"]][key]),
                "n_matches": n_matches,
                "mean_candidate_score": (
                    cand_sum / n_matches if n_matches else ""),
                "mean_opponent_score": (
                    opp_total / n_matches if n_matches else ""),
                "mean_diff": (
                    diff_total / n_matches if n_matches else ""),
                "min_diff": dmin if dmin is not None else "",
                "max_diff": dmax if dmax is not None else "",
            })
    os.replace(tmp, opponent_matrix_path)

    expected_candidate_rows = EXPECTED_ENTITIES * REPS
    if total != EXPECTED_PAIRS:
        raise RuntimeError(
            f"{long_path}: {total} rows != {EXPECTED_PAIRS}")
    if len(counts_by_candidate) != 33:
        raise RuntimeError(
            f"{long_path}: {len(counts_by_candidate)} candidates != 33")
    bad = {key: count for key, count in counts_by_candidate.items()
           if count != expected_candidate_rows}
    if bad:
        raise RuntimeError(f"{long_path}: bad candidate row counts {bad}")
    bad_pairs = {
        (cand, opp): n
        for cand, values in pair_counts.items()
        for opp, n in values.items() if n != REPS
    }
    if bad_pairs:
        raise RuntimeError(
            f"{long_path}: {len(bad_pairs)} pairs do not have {REPS} reps")

    LOG.info("merge: wrote %d rows across %d candidates (%d reps/pair); "
             "devices=%s", total, len(counts_by_candidate), REPS,
             dict(counts_by_device))
    return {
        "rows": total,
        "pairs": len(pair_counts) * REPS,
        "candidates": len(counts_by_candidate),
        "reps": REPS,
        "devices": dict(counts_by_device),
        "long_csv": str(long_path.resolve()),
        "pair_means_csv": str(pair_means_path.resolve()),
        "matrix_csv": str(matrix_path.resolve()),
        "opponent_matrix_csv": str(opponent_matrix_path.resolve()),
    }


def write_readme(out_dir: Path, *, phase: int, title: str, status: str,
                 meta: dict[str, Any], authoritative_path: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Status: `{status}`",
        f"- Branch: `{meta.get('branch')}`",
        f"- Pairs: `{meta.get('pairs_written', meta.get('pairs_expected'))}`",
        f"- Entities: `{meta.get('entities')}`",
        f"- Candidates: `{meta.get('candidates')}`",
        f"- Base seed: `{meta.get('base_seed')}`",
        f"- Rounds: `{meta.get('rounds')}` / reps: `{meta.get('reps')}`",
        f"- Elapsed: `{meta.get('elapsed_seconds', 0):.1f}s`",
        "",
        "Each matchup is repeated `reps` times with independent seeds. "
        "Matrices contain the mean candidate score per matchup; the long CSV "
        "keeps every rep.",
        "",
        "## Files",
        "",
        f"- `{meta.get('long_csv', '')}`",
        f"- `{meta.get('pair_means_csv', '')}`",
        f"- `{meta.get('matrix_csv', '')}`",
        f"- `{meta.get('opponent_matrix_csv', '')}`",
        "",
        "## Authoritative data",
        "",
        f"The authoritative main dataset is always the pure-CPU sample run: "
        f"`{authoritative_path}`.",
        "",
    ]
    if phase == 2:
        lines.insert(4, "- Provisional: this branch must be validated in "
                       "Phase 4 against the pure-CPU reference.")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 3: authoritative CPU sample branch
# ---------------------------------------------------------------------------


def phase3(entities: Sequence[dict[str, Any]], *, cpu_workers: int,
           force: bool = False) -> dict[str, Any]:
    out = OUT_ROOT / "phase3_sample_cpu_full"
    out.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_order(entities)
    started = time.perf_counter()
    LOG.info("phase3: starting authoritative pure-CPU sample branch: "
             "%d candidates x %d opponents x %d reps = %d matches; "
             "workers=%d", len(candidates), len(entities), REPS,
             len(candidates) * len(entities) * REPS, cpu_workers)
    stats = _run_cpu_candidates(
        entities, branch="cpu", out_dir=out, candidate_rows=candidates,
        max_workers=cpu_workers, force=force)
    merged = merge_and_write(
        entities, out_dir=out, long_name="sample_cpu_long.csv",
        matrix_name="sample_cpu_matrix.csv",
        opponent_matrix_name="sample_cpu_opponent_matrix.csv",
        gpu_split=False, expected_branch="cpu", expected_device="cpu")
    elapsed = time.perf_counter() - started
    meta = {
        "phase": 3,
        "branch": "cpu",
        "status": "authoritative",
        "authoritative": True,
        "created_at": now(),
        "elapsed_seconds": elapsed,
        "base_seed": BASE_SEED,
        "rounds": ROUNDS,
        "reps": REPS,
        "entities": len(entities),
        "candidates": len(candidates),
        "pairs_expected": EXPECTED_PAIRS,
        "pairs_written": merged["rows"],
        "cpu_stats": stats,
        "devices": merged["devices"],
        "long_csv": merged["long_csv"],
        "pair_means_csv": merged["pair_means_csv"],
        "matrix_csv": merged["matrix_csv"],
        "opponent_matrix_csv": merged["opponent_matrix_csv"],
        "settings": {
            "mode": "sample",
            "device": "cpu",
            "mode_argmax": False,
            "fast_rules": False,
            "gpu": False,
            "workers": cpu_workers,
            "rounds": ROUNDS,
            "reps": REPS,
        },
    }
    write_json(out / "sample_cpu_run.json", meta)
    write_readme(out, phase=3, title="Phase 3 - authoritative pure-CPU "
                 "sample", status="authoritative", meta=meta,
                 authoritative_path=merged["long_csv"])
    return meta


# ---------------------------------------------------------------------------
# Phase 4: join and compare
# ---------------------------------------------------------------------------


def _load_long(path: Path) -> dict[tuple[int, str, str], dict[str, str]]:
    data: dict[tuple[int, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["rep"]), row["candidate_key"],
                   row["opponent_key"])
            if key in data:
                raise RuntimeError(f"duplicate key in {path}: {key}")
            data[key] = row
    return data


def phase4(entities: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out = OUT_ROOT / "phase4_compare"
    out.mkdir(parents=True, exist_ok=True)
    p2_path = OUT_ROOT / "phase2_parity_full" / "parity_long.csv"
    p3_path = OUT_ROOT / "phase3_sample_cpu_full" / "sample_cpu_long.csv"
    if not p2_path.exists():
        LOG.error("phase4: missing %s", p2_path)
        report = {
            "pairs_checked": 0, "pairs_passed": 0,
            "pairs_failed": EXPECTED_PAIRS, "score_mismatches": 0,
            "action_mismatches": 0, "name_mismatches": 0,
            "candidate_mismatch_counts": {}, "max_score_diff": 0,
            "pass": False, "reason": "missing Phase 2 parity_long.csv",
            "created_at": now(),
        }
        write_json(out / "comparison_report.json", report)
        return report
    started = time.perf_counter()
    LOG.info("phase4: loading parity and CPU long CSVs ...")
    parity = _load_long(p2_path)
    cpu = _load_long(p3_path)
    LOG.info("phase4: joined %d parity + %d CPU rows",
             len(parity), len(cpu))
    missing_in_parity = set(cpu) - set(parity)
    missing_in_cpu = set(parity) - set(cpu)
    keys = sorted(set(parity) | set(cpu))
    mismatches: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    score_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    passed = 0
    max_diff = 0

    for key in keys:
        p = parity.get(key)
        c = cpu.get(key)
        if p is None or c is None:
            candidate_key = key[1]
            counts[candidate_key] += 1
            mismatches.append({
                "rep": key[0], "candidate_key": key[1],
                "opponent_key": key[2], "reasons": "missing_row",
                "parity_present": p is not None,
                "cpu_present": c is not None,
            })
            continue
        fields = {
            "candidate_score": (int(p["candidate_score"]),
                                int(c["candidate_score"])),
            "opponent_score": (int(p["opponent_score"]),
                               int(c["opponent_score"])),
            "candidate_action_hash": (
                p["candidate_action_hash"], c["candidate_action_hash"]),
            "opponent_action_hash": (
                p["opponent_action_hash"], c["opponent_action_hash"]),
            "actual_name_candidate": (
                p["actual_name_candidate"], c["actual_name_candidate"]),
            "actual_name_opponent": (
                p["actual_name_opponent"], c["actual_name_opponent"]),
        }
        score_bad = (fields["candidate_score"][0] !=
                     fields["candidate_score"][1] or
                     fields["opponent_score"][0] !=
                     fields["opponent_score"][1])
        action_bad = (fields["candidate_action_hash"][0] !=
                      fields["candidate_action_hash"][1] or
                      fields["opponent_action_hash"][0] !=
                      fields["opponent_action_hash"][1])
        name_bad = (fields["actual_name_candidate"][0] !=
                    fields["actual_name_candidate"][1] or
                    fields["actual_name_opponent"][0] !=
                    fields["actual_name_opponent"][1])
        diff = max(abs(fields["candidate_score"][0] -
                       fields["candidate_score"][1]),
                   abs(fields["opponent_score"][0] -
                       fields["opponent_score"][1]))
        max_diff = max(max_diff, diff)
        if score_bad:
            score_counts[key[1]] += 1
        if action_bad:
            action_counts[key[1]] += 1
        if name_bad:
            name_counts[key[1]] += 1
        if score_bad or action_bad or name_bad:
            counts[key[1]] += 1
            reasons = []
            if score_bad:
                reasons.append("score")
            if action_bad:
                reasons.append("action")
            if name_bad:
                reasons.append("name")
            mismatches.append({
                "rep": key[0], "candidate_key": key[1],
                "opponent_key": key[2],
                "parity_candidate_score": p["candidate_score"],
                "cpu_candidate_score": c["candidate_score"],
                "parity_opponent_score": p["opponent_score"],
                "cpu_opponent_score": c["opponent_score"],
                "parity_candidate_action_hash": p["candidate_action_hash"],
                "cpu_candidate_action_hash": c["candidate_action_hash"],
                "parity_opponent_action_hash": p["opponent_action_hash"],
                "cpu_opponent_action_hash": c["opponent_action_hash"],
                "parity_actual_name_candidate": p["actual_name_candidate"],
                "cpu_actual_name_candidate": c["actual_name_candidate"],
                "parity_actual_name_opponent": p["actual_name_opponent"],
                "cpu_actual_name_opponent": c["actual_name_opponent"],
                "reasons": "+".join(reasons),
            })
        else:
            passed += 1

    gpu_count = {kind: counts.get(f"C:{kind}", 0)
                 for kind in GPU_CANDIDATES}
    cpu_count = {kind: counts.get(f"C:{kind}", 0)
                 for kind in CPU_CANDIDATES}
    gpu_pass = all(value == 0 for value in gpu_count.values())
    cpu_pass = all(value == 0 for value in cpu_count.values())
    all_pairs_pass = (not missing_in_parity and not missing_in_cpu and
                      len(keys) == EXPECTED_PAIRS and
                      len(mismatches) == 0)
    report = {
        "pairs_checked": len(keys),
        "pairs_expected": EXPECTED_PAIRS,
        "pairs_passed": passed,
        "pairs_failed": len(keys) - passed,
        "reps": REPS,
        "score_mismatches": sum(score_counts.values()),
        "action_mismatches": sum(action_counts.values()),
        "name_mismatches": sum(name_counts.values()),
        "candidate_mismatch_counts": dict(sorted(counts.items())),
        "candidate_score_mismatch_counts": dict(sorted(
            score_counts.items())),
        "candidate_action_mismatch_counts": dict(sorted(
            action_counts.items())),
        "candidate_name_mismatch_counts": dict(sorted(
            name_counts.items())),
        "gpu_candidate_mismatch_counts": gpu_count,
        "cpu_candidate_mismatch_counts": cpu_count,
        "max_score_diff": max_diff,
        "pass": bool(gpu_pass and cpu_pass and all_pairs_pass),
        "parity_branch_pass": bool(gpu_pass),
        "gpu_candidates_pass": bool(gpu_pass),
        "cpu_candidates_pass": bool(cpu_pass),
        "missing_in_parity": len(missing_in_parity),
        "missing_in_cpu": len(missing_in_cpu),
        "gpu_candidates": list(GPU_CANDIDATES),
        "cpu_candidates": list(CPU_CANDIDATES),
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": now(),
        "parity_csv": str(p2_path.resolve()),
        "cpu_csv": str(p3_path.resolve()),
    }
    write_json(out / "comparison_report.json", report)
    LOG.info("phase4: pass=%s pairs=%d failed=%d score_mismatches=%d "
             "action_mismatches=%d name_mismatches=%d max_score_diff=%d",
             report["pass"], report["pairs_checked"],
             report["pairs_failed"], report["score_mismatches"],
             report["action_mismatches"], report["name_mismatches"],
             report["max_score_diff"])
    fields = [
        "rep", "candidate_key", "opponent_key", "reasons",
        "parity_candidate_score", "cpu_candidate_score",
        "parity_opponent_score", "cpu_opponent_score",
        "parity_candidate_action_hash", "cpu_candidate_action_hash",
        "parity_opponent_action_hash", "cpu_opponent_action_hash",
        "parity_actual_name_candidate", "cpu_actual_name_candidate",
        "parity_actual_name_opponent", "cpu_actual_name_opponent",
        "parity_present", "cpu_present",
    ]
    tmp = (out / "comparison_mismatches.csv").with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mismatches)
    os.replace(tmp, out / "comparison_mismatches.csv")
    return report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    global OUT_ROOT, ENTITIES_CSV, ENTITIES_COUNTS, REPS, EXPECTED_PAIRS
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="recompute Phase 1 and candidate shards")
    parser.add_argument("--cpu-workers", type=int, default=CPU_WORKERS)
    parser.add_argument("--gpu-workers", type=int, default=GPU_WORKERS)
    parser.add_argument("--phase1-workers", type=int, default=GPU_WORKERS)
    parser.add_argument("--output-root", type=str, default=str(OUT_ROOT))
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--skip-phase4", action="store_true")
    args = parser.parse_args()

    if args.reps <= 0:
        parser.error("--reps must be >= 1")
    OUT_ROOT = Path(args.output_root).expanduser().resolve()
    ENTITIES_CSV = OUT_ROOT / "entities.csv"
    ENTITIES_COUNTS = OUT_ROOT / "entities_counts.json"
    REPS = int(args.reps)
    EXPECTED_PAIRS = 33 * EXPECTED_ENTITIES * REPS
    _ENTITY_CACHE.clear()

    setup_logging()
    overall_started = time.perf_counter()
    LOG.info("=" * 72)
    LOG.info("candidates_vs_all_phased: START pid=%d output_root=%s "
             "base_seed=%d rounds=%d reps=%d", os.getpid(), OUT_ROOT,
             BASE_SEED, ROUNDS, REPS)
    LOG.info("routing: GPU=%d candidates, CPU=%d candidates; workers: "
             "cpu=%d gpu=%d phase1=%d; gpu_shards=%d",
             len(GPU_CANDIDATES), len(CPU_CANDIDATES), args.cpu_workers,
             args.gpu_workers, args.phase1_workers, GPU_SHARDS)
    entities, counts = phase0(force=args.force)
    LOG.info("phase0: %d entities built; per-pool counts=%s",
             len(entities), counts)

    phase1_pass = False
    if not args.skip_phase1:
        phase1_pass = phase1(entities, force=args.force,
                             workers=args.phase1_workers)
    else:
        cached = OUT_ROOT / "phase1_sample_parity" / \
            "parity_sample_report.json"
        phase1_pass = bool(cached.exists() and
                           read_json(cached).get("pass"))
        LOG.info("phase1: skipped; cached pass=%s", phase1_pass)

    phase2_meta = None
    if not args.skip_phase2:
        if not phase1_pass:
            out = OUT_ROOT / "phase2_parity_full"
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "parity_run.json", {
                "phase": 2, "branch": "parity",
                "status": "skipped_phase1_gate_failed",
                "authoritative": False, "phase1_pass": False,
                "created_at": now(),
            })
            LOG.error("phase2: skipped because the Phase 1 gate failed")
        else:
            phase2_meta = phase2(
                entities, cpu_workers=args.cpu_workers,
                gpu_workers=args.gpu_workers, force=args.force)
            merged = merge_and_write(
                entities, out_dir=OUT_ROOT / "phase2_parity_full",
                long_name="parity_long.csv",
                matrix_name="parity_matrix.csv",
                opponent_matrix_name="parity_opponent_matrix.csv",
                gpu_split=True, expected_branch="parity",
                expected_device=None)
            meta = read_json(
                OUT_ROOT / "phase2_parity_full" / "parity_run.json")
            meta.update(merged)
            meta["pairs_written"] = merged["rows"]
            meta["phase1_pass"] = bool(phase1_pass)
            write_json(OUT_ROOT / "phase2_parity_full" / "parity_run.json",
                       meta)
            write_readme(
                OUT_ROOT / "phase2_parity_full", phase=2,
                title="Phase 2 - provisional parity branch",
                status="provisional", meta=meta,
                authoritative_path=str(
                    (OUT_ROOT / "phase3_sample_cpu_full" /
                     "sample_cpu_long.csv").resolve()))
            LOG.info("phase2: complete: rows=%d status=provisional",
                     merged["rows"])

    phase3_meta = None
    if not args.skip_phase3:
        phase3_meta = phase3(
            entities, cpu_workers=args.cpu_workers, force=args.force)
        LOG.info("phase3: complete: authoritative rows=%d",
                 phase3_meta["pairs_written"])

    phase4_report = None
    if not args.skip_phase4:
        phase4_report = phase4(entities)
        LOG.info("phase4: complete: pass=%s pairs=%d failed=%d",
                 phase4_report["pass"], phase4_report["pairs_checked"],
                 phase4_report["pairs_failed"])

    elapsed = time.perf_counter() - overall_started
    summary = {
        "created_at": now(),
        "elapsed_seconds": elapsed,
        "entities": len(entities),
        "entity_pools": counts,
        "pairs_per_branch": EXPECTED_PAIRS,
        "phase1_pairs": len(GPU_CANDIDATES) * 100 * REPS,
        "total_pairs_three_relevant_runs": (
            EXPECTED_PAIRS * 2 + len(GPU_CANDIDATES) * 100 * REPS),
        "base_seed": BASE_SEED,
        "rounds": ROUNDS,
        "reps": REPS,
        "phase1_pass": bool(phase1_pass),
        "phase2_status": (phase2_meta or {}).get("status"),
        "phase3_status": (phase3_meta or {}).get("status"),
        "phase4_pass": (phase4_report or {}).get("pass"),
        "authoritative_main_data": str(
            (OUT_ROOT / "phase3_sample_cpu_full" /
             "sample_cpu_long.csv").resolve()),
        "phase2_long": str(
            (OUT_ROOT / "phase2_parity_full" / "parity_long.csv").resolve()),
        "phase1_report": str(
            (OUT_ROOT / "phase1_sample_parity" /
             "parity_sample_report.json").resolve()),
        "phase4_report": str(
            (OUT_ROOT / "phase4_compare" /
             "comparison_report.json").resolve()),
    }
    write_json(OUT_ROOT / "final_summary.json", summary)
    root_readme = OUT_ROOT / "README.md"
    root_readme.write_text(
        "# candidates_vs_all_phased_20261029\n\n"
        f"- Entities: **{len(entities)}** (33 candidates x 1,937 opponents).\n"
        f"- Phase 1 sample parity: **{'PASS' if phase1_pass else 'FAIL'}**\n"
        f"- Phase 2 provisional parity: "
        f"**{(phase2_meta or {}).get('status', 'not run')}**\n"
        f"- Phase 3 authoritative CPU sample: "
        f"**{(phase3_meta or {}).get('status', 'not run')}**\n"
        f"- Phase 4 comparison: "
        f"**{'PASS' if (phase4_report or {}).get('pass') else 'FAIL'}**\n\n"
        "## Main data\n\n"
        "The authoritative main dataset is the pure-CPU sample branch:\n\n"
        f"`{(OUT_ROOT / 'phase3_sample_cpu_full' / 'sample_cpu_long.csv').resolve()}`\n\n"
        "Phase 2 parity is provisional and must never overwrite or replace "
        "the Phase 3 CPU data.\n",
        encoding="utf-8")
    LOG.info("DONE elapsed=%.1fs summary=%s", elapsed,
             OUT_ROOT / "final_summary.json")
    LOG.info("authoritative_main_data=%s",
             summary["authoritative_main_data"])
    LOG.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
