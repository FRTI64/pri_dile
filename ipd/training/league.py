"""P3 league utilities: red meta-game weights and Gate D checks.

This module is deliberately training-agnostic.  It turns the red/blue payoffs
from an evaluated red pool into a maximin (PSRO-lite) mixture, writes the
opponent manifest consumed by ``ipd train rl --opponent-pool``, and evaluates
the pre-registered Gate D conditions on a league report.

The payoff convention is ``red_score - blue_score``: larger means the red
exploiter is stronger.  The blue response therefore minimizes this matrix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ipd.evaluation.metagame import maximin_strategy

POOL_SCHEMA = "ipd-opponent-pool-v1"


def _pairs_from(report: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    red = report.get("red")
    if isinstance(red, Mapping) and isinstance(red.get("pairs"), Mapping):
        return red["pairs"]
    if isinstance(report.get("pairs"), Mapping):
        return report["pairs"]  # type: ignore[return-value]
    raise ValueError("report has no red.pairs mapping")


def red_payoff_matrix(pairs: Mapping[str, Mapping[str, Any]],
                      reds: Sequence[str],
                      blues: Sequence[str]) -> list[list[float]]:
    """Extract a ``reds x blues`` score-difference matrix."""
    matrix: list[list[float]] = []
    for red in reds:
        row = []
        for blue in blues:
            key = f"{blue}|{red}"
            if key not in pairs:
                raise KeyError(f"missing red pair {key!r}")
            row.append(float(pairs[key]["diff_mean"]))
        matrix.append(row)
    return matrix


def meta_weights_from_report(report: Mapping[str, Any],
                             train_reds: Sequence[str],
                             blues: Sequence[str] | None = None,
                             *,
                             iterations: int = 20000,
                             seed: int = 20261047,
                             uniform_mix: float = 0.2) -> dict[str, Any]:
    """Return a maximin red mixture plus a small uniform regulariser.

    ``uniform_mix`` prevents a single red from dominating the learner's
    opponent pool, which is important for PSRO-style diversity.  Set it to 0
    for the pure maximin solution.
    """
    pairs = _pairs_from(report)
    if blues is None:
        blues = sorted({str(value["blue"]) for value in pairs.values()})
    reds = list(train_reds)
    if not reds:
        raise ValueError("train_reds must not be empty")
    if not blues:
        raise ValueError("blues must not be empty")
    missing = [red for red in reds
               if not any(key.endswith(f"|{red}") for key in pairs)]
    if missing:
        raise KeyError(f"reds missing from report: {missing}")
    matrix = red_payoff_matrix(pairs, reds, blues)
    result = maximin_strategy(matrix, iterations=iterations, seed=seed)
    mix = min(1.0, max(0.0, float(uniform_mix)))
    n = len(reds)
    weights = [(1.0 - mix) * float(result["weights"][i]) + mix / n
               for i in range(n)]
    total = sum(weights)
    weights = [value / total for value in weights]
    by_col = [sum(weights[i] * matrix[i][j] for i in range(n))
              for j in range(len(blues))]
    worst_index = min(range(len(blues)), key=lambda j: by_col[j])
    return {
        "weights": dict(zip(reds, weights)),
        "matrix": matrix,
        "reds": reds,
        "blues": list(blues),
        "worst_case": by_col[worst_index],
        "worst_blue": blues[worst_index],
        "maximin_value": float(result["value"]),
        "maximin_weights": dict(zip(reds, result["weights"])),
        "uniform_mix": mix,
        "iterations": int(result["iterations"]),
    }


def write_opponent_manifest(path: str | Path,
                            entries: Sequence[Mapping[str, Any]],
                            weights: Mapping[str, float] | None = None,
                            *,
                            name: str = "") -> dict[str, Any]:
    """Write a weighted opponent manifest for ``--opponent-pool``."""
    payload_entries = []
    for entry in entries:
        item = dict(entry)
        if weights is not None:
            key = str(item.get("name") or item.get("kind"))
            if key not in weights:
                raise KeyError(f"missing weight for opponent {key!r}")
            item["weight"] = float(weights[key])
        payload_entries.append(item)
    payload = {
        "schema": POOL_SCHEMA,
        "name": name or Path(path).stem,
        "entries": payload_entries,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, out)
    return payload


def load_opponent_manifest(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    schema = data.get("schema") if isinstance(data, Mapping) else None
    if schema != POOL_SCHEMA:
        raise ValueError(f"unsupported opponent pool schema {schema!r}")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("opponent pool must contain a non-empty entries list")
    out = []
    for entry in entries:
        item = dict(entry)
        if not (item.get("path") or item.get("kind") or item.get("spec")):
            raise ValueError(f"invalid opponent entry: {item!r}")
        item["weight"] = max(0.0, float(item.get("weight", 1.0)))
        out.append(item)
    return out


def gate_d(report: Mapping[str, Any], baseline: str, candidate: str,
           heldout_reds: Sequence[str], *,
           min_reduction: float = 0.20,
           max_official_drop: float = 2.0,
           max_family_drop: float = 0.02,
           max_red_increase: float = 0.05) -> dict[str, Any]:
    """Evaluate the pre-registered Gate D on a league report.

    Conditions (all must hold):
    1. mean held-out red exploitability (red - blue) drops by >= 20%;
    2. official mean does not drop by more than 2 vs the baseline;
    3. no suite/family ratio drops by more than 2% vs the baseline;
    4. no single held-out red gets more than 5% easier than the baseline.
    """
    pairs = _pairs_from(report)
    blues = report.get("candidates") or {}
    if baseline not in blues or candidate not in blues:
        raise KeyError("baseline/candidate missing from report candidates")
    per_red = {}
    baseline_diffs = []
    candidate_diffs = []
    for red in heldout_reds:
        key = f"{baseline}|{red}"
        cand_key = f"{candidate}|{red}"
        if key not in pairs or cand_key not in pairs:
            raise KeyError(f"missing held-out red pair {red!r}")
        old = float(pairs[key]["diff_mean"])
        new = float(pairs[cand_key]["diff_mean"])
        baseline_diffs.append(old)
        candidate_diffs.append(new)
        increase = (new - old) / max(1.0, abs(old))
        per_red[red] = {
            "baseline_diff": old,
            "candidate_diff": new,
            "delta": new - old,
            "relative_increase": increase,
            "baseline_blue": float(pairs[key]["blue_mean"]),
            "candidate_blue": float(pairs[cand_key]["blue_mean"]),
        }
    old_mean = sum(baseline_diffs) / len(baseline_diffs)
    new_mean = sum(candidate_diffs) / len(candidate_diffs)
    reduction = (old_mean - new_mean) / max(1.0, old_mean)

    def official_mean(name: str) -> float | None:
        suites = blues[name].get("suites") or {}
        official = suites.get("official") or {}
        return official.get("mean")

    old_official = official_mean(baseline)
    new_official = official_mean(candidate)
    official_delta = (None if old_official is None or new_official is None
                      else new_official - old_official)

    baseline_families = (blues[baseline].get("overall") or {}).get(
        "families") or {}
    candidate_families = (blues[candidate].get("overall") or {}).get(
        "families") or {}
    family_deltas = {}
    for key, old in baseline_families.items():
        new = candidate_families.get(key)
        if new is None or abs(float(old)) < 1e-9:
            continue
        family_deltas[key] = float(new) / float(old) - 1.0
    worst_family = (min(family_deltas.items(), key=lambda item: item[1])
                    if family_deltas else (None, 0.0))

    checks = {
        "exploitability_reduction": reduction >= min_reduction,
        "official": (official_delta is not None
                     and official_delta >= -max_official_drop),
        "families": worst_family[1] >= -max_family_drop,
        "per_red": all(row["relative_increase"] <= max_red_increase
                       for row in per_red.values()),
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "heldout_reds": list(heldout_reds),
        "baseline_exploitability": old_mean,
        "candidate_exploitability": new_mean,
        "reduction": reduction,
        "official_delta": official_delta,
        "worst_family": worst_family[0],
        "worst_family_delta": worst_family[1],
        "per_red": per_red,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a PSRO-lite red opponent manifest from a league "
                    "red report.")
    ap.add_argument("--config", required=True,
                    help="league config that produced the red report")
    ap.add_argument("--report", required=True,
                    help="red evaluation report JSON")
    ap.add_argument("--train-reds", required=True,
                    help="comma-separated red names used by the blue response")
    ap.add_argument("--heldout-reds", default="",
                    help="comma-separated red names excluded from training")
    ap.add_argument("--blues", default="",
                    help="optional comma-separated blue columns")
    ap.add_argument("--uniform-mix", type=float, default=0.2,
                    help="blend weight for the uniform red mixture")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261047)
    ap.add_argument("--out", required=True,
                    help="weighted opponent manifest JSON")
    ap.add_argument("--summary-out", default=None,
                    help="optional meta-weight summary JSON")
    args = ap.parse_args(argv)

    from ipd.evaluation.league import load_league_config

    cfg = load_league_config(args.config)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    specs = {entry.spec.name: entry.spec.to_dict()
             for entry in cfg.red_policies}
    train_reds = [x.strip() for x in args.train_reds.split(",") if x.strip()]
    heldout = [x.strip() for x in args.heldout_reds.split(",") if x.strip()]
    overlaps = sorted(set(train_reds) & set(heldout))
    if overlaps:
        raise SystemExit(f"held-out reds must not be trained on: {overlaps}")
    missing = [red for red in train_reds + heldout if red not in specs]
    if missing:
        raise SystemExit(f"reds missing from {args.config}: {missing}")
    blues = ([x.strip() for x in args.blues.split(",") if x.strip()]
             or None)
    summary = meta_weights_from_report(
        report, train_reds, blues=blues, iterations=args.iterations,
        seed=args.seed, uniform_mix=args.uniform_mix)
    payload = write_opponent_manifest(
        args.out, [specs[red] for red in train_reds],
        weights=summary["weights"], name=Path(args.out).stem)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print("train reds: " + ", ".join(
        f"{red}={summary['weights'][red]:.3f}" for red in train_reds))
    print(f"held-out reds (excluded): {', '.join(heldout) or '-'}")
    print(f"worst-case diff {summary['worst_case']:.2f} "
          f"vs {summary['worst_blue']}")
    print(f"written {args.out} ({len(payload['entries'])} entries)")
    if args.summary_out:
        print(f"written {args.summary_out}")
    return 0


__all__ = [
    "POOL_SCHEMA", "red_payoff_matrix", "meta_weights_from_report",
    "write_opponent_manifest", "load_opponent_manifest", "gate_d", "main",
]
