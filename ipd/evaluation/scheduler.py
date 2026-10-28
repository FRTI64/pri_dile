"""Worker sizing, progress, profiling and field-chunk helpers.

Protocol evaluation used to submit one task per ``(candidate, tier, field)``.
The field builder is deterministic, so a worker can build one field and reuse
it for a small batch of candidates as long as RNG snapshots are restored
between candidates.  This module keeps the scheduling bookkeeping out of the
CLI so it can be tested without spawning a process pool.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence, TextIO

DEFAULT_MAX_WORKERS = 20
DEFAULT_WORKER_RAM_GB = 1.0
PARENT_RESERVE_GB = 2.0
_GIB = 1024.0 ** 3


def available_ram_gb() -> float | None:
    """Best-effort available-memory estimate on Linux/macOS."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return float(pages) * float(page_size) / _GIB
    except (AttributeError, OSError, ValueError):
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def worker_budget(
        requested: int | str | None = None, *,
        cpu_count: int | None = None,
        available_gb: float | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        worker_ram_gb: float = DEFAULT_WORKER_RAM_GB,
        parent_reserve_gb: float = PARENT_RESERVE_GB,
) -> dict[str, Any]:
    """Resolve ``--workers`` and return the inputs used for the decision.

    ``auto`` means ``min(20, cpu_count - 2, RAM budget)``.  The RAM budget
    reserves ``parent_reserve_gb`` for the parent process and allocates
    ``worker_ram_gb`` per worker.  Explicit integers are never capped, so an
    operator can still override the heuristic.
    """
    cpu = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    cpu_limit = max(1, cpu - 2)
    ram = available_ram_gb() if available_gb is None else float(available_gb)
    if ram is None or ram <= 0:
        ram_limit = int(max_workers)
    else:
        ram_limit = max(1, int(max(0.0, ram - parent_reserve_gb)
                               // max(0.1, worker_ram_gb)))
    text = "auto" if requested is None else str(requested).strip().lower()
    if text == "auto":
        resolved = max(1, min(int(max_workers), cpu_limit, ram_limit))
    else:
        try:
            resolved = int(text)
        except ValueError as exc:
            raise ValueError(
                f"workers must be an integer or 'auto', got {requested!r}"
            ) from exc
        if resolved <= 0:
            raise ValueError("workers must be positive")
    return {
        "requested": text,
        "resolved": resolved,
        "cpu_count": cpu,
        "cpu_limit": cpu_limit,
        "ram_available_gb": None if ram is None else round(float(ram), 2),
        "ram_limit": ram_limit,
        "max_workers": int(max_workers),
        "ram_per_worker_gb": float(worker_ram_gb),
        "parent_reserve_gb": float(parent_reserve_gb),
    }


def resolve_workers(requested: int | str | None = None, **kwargs: Any) -> int:
    """Return only the resolved worker count."""
    return int(worker_budget(requested, **kwargs)["resolved"])


def chunk_specs(specs: Sequence[Any], chunk_size: int) -> list[list[Any]]:
    """Split candidates into balanced chunks of at most ``chunk_size``.

    Interleaving ``specs[i::n_chunks]`` keeps expensive candidates from
    clustering in one chunk when the registry order is roughly cost-sorted.
    """
    if not specs:
        return []
    size = max(1, int(chunk_size))
    count = max(1, math.ceil(len(specs) / size))
    return [list(specs[index::count]) for index in range(count)]


def _rss_mb(usage: Any) -> float:
    # Linux reports KiB, macOS reports bytes.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return float(usage.ru_maxrss) / divisor


def peak_rss_mb() -> float:
    """Peak RSS of this process and any terminated children, in MiB."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Unix
        return 0.0
    values = [_rss_mb(resource.getrusage(resource.RUSAGE_SELF))]
    try:
        values.append(_rss_mb(resource.getrusage(resource.RUSAGE_CHILDREN)))
    except (AttributeError, ValueError):  # pragma: no cover - platform
        pass
    return round(max(values), 2)


def _process_tree(root: int) -> list[int]:
    """Return ``root`` and all of its descendants (Linux /proc)."""
    children: dict[int, list[int]] = defaultdict(list)
    try:
        entries = os.listdir("/proc")
    except OSError:  # pragma: no cover - non-Linux
        return [root]
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        # Skip "pid (comm)" so spaces in comm cannot shift the ppid field.
        fields = text[text.rfind(")") + 2:].split()
        if len(fields) > 1:
            children[int(fields[1])].append(pid)
    out: list[int] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(children.get(pid, ()))
    return out


def _smaps_mb(pid: int) -> tuple[float, float, float]:
    rss = pss = anon = 0.0
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("Rss", "Pss", "Anonymous"):
                    value = float(rest.split()[0]) / 1024.0
                    if key == "Rss":
                        rss += value
                    elif key == "Pss":
                        pss += value
                    else:
                        anon += value
    except (OSError, IndexError, ValueError):
        pass
    return rss, pss, anon


class MemorySampler:
    """Sample the process tree's PSS/RSS while a pool is running.

    ``RSS`` double-counts shared pages such as the torch libraries pulled in by
    Axelrod; PSS is the better estimate of physical memory.  Both are recorded
    so the audit can show why they differ.
    """

    def __init__(self, interval: float = 1.0):
        self.interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.processes_peak = 0
        self.peak_rss_mb = 0.0
        self.peak_pss_mb = 0.0
        self.peak_anon_mb = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ipd-memory-sampler")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)
        self.sample()

    def sample(self) -> None:
        pids = _process_tree(os.getpid())
        rss = pss = anon = 0.0
        for pid in pids:
            a, b, c = _smaps_mb(pid)
            rss += a
            pss += b
            anon += c
        self.samples += 1
        self.processes_peak = max(self.processes_peak, len(pids))
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        self.peak_pss_mb = max(self.peak_pss_mb, pss)
        self.peak_anon_mb = max(self.peak_anon_mb, anon)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 2))
            self._thread = None
        return {
            "samples": self.samples,
            "processes_peak": self.processes_peak,
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "peak_pss_mb": round(self.peak_pss_mb, 2),
            "peak_anon_mb": round(self.peak_anon_mb, 2),
        }


class ProgressTracker:
    """Print field/task progress with elapsed time and an ETA."""

    def __init__(self, tasks: Sequence[Mapping[str, Any]], *,
                 interval: float = 5.0,
                 stream: TextIO | None = None):
        self.total_tasks = len(tasks)
        self.total_fields = len({
            (str(task["tier"]), int(seed))
            for task in tasks
            for seed in (task.get("seeds") or (task["seed"],))
        })
        self.interval = max(0.0, float(interval))
        self.stream = stream if stream is not None else sys.stdout
        self.started = time.monotonic()
        self.last_print = float("-inf")
        self.done = 0
        self.fields_done = 0
        self._pending: Counter[tuple[str, int]] = Counter(
            (str(task["tier"]), int(seed)) for task in tasks
            for seed in (task.get("seeds") or (task["seed"],)))

    def update(self, task_result: Mapping[str, Any], *,
               force: bool = False) -> bool:
        self.done += 1
        for seed in (task_result.get("seeds") or (task_result["seed"],)):
            key = (str(task_result["tier"]), int(seed))
            self._pending[key] -= 1
            if self._pending[key] <= 0:
                self.fields_done += 1
        now = time.monotonic()
        if not force and now - self.last_print < self.interval:
            return False
        self.last_print = now
        elapsed = now - self.started
        eta = (elapsed / self.done * (self.total_tasks - self.done)
               if self.done else 0.0)
        results = task_result.get("results", [])
        candidate = results[-1].get("candidate", "?") if results else "?"
        print(f"progress fields {self.fields_done}/{self.total_fields} "
              f"tasks {self.done}/{self.total_tasks} "
              f"tier={task_result['tier']} candidate={candidate} "
              f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
              file=self.stream, flush=True)
        return True


class ProfileAccumulator:
    """Aggregate per-task timings into a JSON-friendly profile."""

    def __init__(self, *, workers: int, workers_requested: str,
                 candidate_chunk: int, field_reuse: str):
        self.workers = int(workers)
        self.workers_requested = str(workers_requested)
        self.candidate_chunk = int(candidate_chunk)
        self.field_reuse = str(field_reuse)
        self.field_build_seconds = 0.0
        self.field_tasks = 0
        self.reused_tasks = 0
        self.by_candidate: dict[str, dict[str, Any]] = {}
        self.by_tier: dict[str, dict[str, Any]] = {}
        self.by_family: dict[str, dict[str, Any]] = {}
        self._tier_seeds: dict[str, set[int]] = defaultdict(set)

    @staticmethod
    def _bucket(mapping: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        return mapping.setdefault(key, {
            "fields": 0,
            "eval_seconds": 0.0,
            "candidate_build_seconds": 0.0,
            "field_build_seconds_shared": 0.0,
            "family_seconds": defaultdict(float),
        })

    def add(self, task_result: Mapping[str, Any]) -> None:
        results = list(task_result.get("results", []))
        if not results:
            return
        field_seconds = float(task_result.get("field_build_seconds", 0.0))
        self.field_build_seconds += field_seconds
        seeds = [int(seed) for seed in
                 (task_result.get("seeds") or (task_result["seed"],))]
        self.field_tasks += len(seeds)
        if task_result.get("field_reuse"):
            self.reused_tasks += 1
        share = field_seconds / len(results)
        tier = str(task_result["tier"])
        self._tier_seeds[tier].update(seeds)
        tier_row = self.by_tier.setdefault(tier, {
            "fields": 0,
            "candidate_fields": 0,
            "tasks": 0,
            "eval_seconds": 0.0,
            "candidate_build_seconds": 0.0,
            "field_build_seconds": 0.0,
            "peak_rss_mb": 0.0,
        })
        tier_row["tasks"] += 1
        tier_row["field_build_seconds"] += field_seconds
        tier_row["peak_rss_mb"] = max(
            float(tier_row["peak_rss_mb"]),
            float(task_result.get("peak_rss_mb", 0.0)),
        )
        for result in results:
            name = str(result["candidate"])
            row = self._bucket(self.by_candidate, name)
            row["fields"] += 1
            row["eval_seconds"] += float(
                result.get("timings", {}).get("eval_seconds", 0.0))
            row["candidate_build_seconds"] += float(
                result.get("timings", {}).get("candidate_build_seconds", 0.0))
            row["field_build_seconds_shared"] += share
            family_seconds = result.get("timings", {}).get(
                "family_seconds", {})
            for family, seconds in family_seconds.items():
                row["family_seconds"][family] += float(seconds)
                family_row = self.by_family.setdefault(family, {
                    "seconds": 0.0, "matches": 0,
                })
                family_row["seconds"] += float(seconds)
                family_row["matches"] += int(
                    result.get("family_counts", {}).get(family, 0))
            tier_row["eval_seconds"] += float(
                result.get("timings", {}).get("eval_seconds", 0.0))
            tier_row["candidate_build_seconds"] += float(
                result.get("timings", {}).get("candidate_build_seconds", 0.0))
            tier_row["candidate_fields"] += 1

    @staticmethod
    def _finish_families(row: dict[str, Any]) -> None:
        row["family_seconds"] = {
            family: round(float(seconds), 6)
            for family, seconds in sorted(row["family_seconds"].items())
        }

    def payload(self, *, wall_seconds: float,
                peak_rss: float) -> dict[str, Any]:
        for row in self.by_candidate.values():
            self._finish_families(row)
            row["eval_seconds"] = round(row["eval_seconds"], 6)
            row["candidate_build_seconds"] = round(
                row["candidate_build_seconds"], 6)
            row["field_build_seconds_shared"] = round(
                row["field_build_seconds_shared"], 6)
        for tier, row in self.by_tier.items():
            row["fields"] = len(self._tier_seeds[tier])
            row["eval_seconds"] = round(row["eval_seconds"], 6)
            row["candidate_build_seconds"] = round(
                row["candidate_build_seconds"], 6)
            row["field_build_seconds"] = round(
                row["field_build_seconds"], 6)
        families = {}
        for family, row in sorted(self.by_family.items()):
            matches = int(row["matches"])
            families[family] = {
                "seconds": round(float(row["seconds"]), 6),
                "matches": matches,
                "seconds_per_match": round(
                    float(row["seconds"]) / matches, 6) if matches else None,
            }
        return {
            "schema": "ipd-protocol-profile-v1",
            "workers": self.workers,
            "workers_requested": self.workers_requested,
            "candidate_chunk": self.candidate_chunk,
            "field_reuse": self.field_reuse,
            "field_tasks": self.field_tasks,
            "field_tasks_reused": self.reused_tasks,
            "field_build_seconds": round(self.field_build_seconds, 6),
            "wall_seconds": round(float(wall_seconds), 6),
            "peak_rss_mb": round(float(peak_rss), 2),
            "by_candidate": dict(sorted(self.by_candidate.items())),
            "by_tier": dict(sorted(self.by_tier.items())),
            "by_family": families,
        }


__all__ = [
    "DEFAULT_MAX_WORKERS", "DEFAULT_WORKER_RAM_GB", "PARENT_RESERVE_GB",
    "available_ram_gb", "worker_budget", "resolve_workers", "chunk_specs",
    "peak_rss_mb", "MemorySampler", "ProgressTracker", "ProfileAccumulator",
]
