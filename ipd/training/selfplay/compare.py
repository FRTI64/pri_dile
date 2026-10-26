#!/usr/bin/env python3
"""Fair held-out comparison for self-play / BC / RL candidates.

The script evaluates every candidate on exactly the same suite items and seeds,
reports raw and normalized per-family scores, per-opponent details, parameter
counts, and the score against Final/Triple.  It is intentionally separate from
the training loop so the final test is not used for checkpoint selection.
"""
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
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from ipd.evaluation.common import run_match
from ipd.paths import config_path, weights_path
from ipd.policies.registry import CandidateSpec

from .opponents import build_final_suite, build_validation_suite


def _candidate_table(args) -> dict[str, CandidateSpec]:
    common = {"mode": args.mode, "seed": args.policy_seed}
    table = {
        "bc": CandidateSpec("selfplay", "BC-only",
                            {**common, "weights": args.bc_weights}),
        "final": CandidateSpec("selfplay", "SelfPlay-final",
                               {**common, "weights": args.final_weights}),
        "scratch": CandidateSpec("selfplay", "SelfPlay-scratch",
                                 {**common, "weights": args.scratch_weights}),
        "rl-ppo": CandidateSpec("rl", "RL-PPO",
                                {**common, "weights": args.rl_weights}),
        "adv-rl": CandidateSpec("adv-rl", "Adv-RL",
                                {**common, "config": args.adv_config}),
    }
    for name, path in args.extra.items():
        table[name] = CandidateSpec("selfplay", name,
                                    {**common, "weights": path})
    for name, path in getattr(args, "extra_rl", {}).items():
        table[name] = CandidateSpec("rl", name,
                                    {**common, "weights": path})
    return {k: v for k, v in table.items() if k in args.names}


def _count_selfplay(path: str) -> dict[str, int]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    deploy = 0
    for layer in data.get("gru", []):
        deploy += sum(len(row) for row in layer["w_ih"])
        deploy += sum(len(row) for row in layer["w_hh"])
        deploy += len(layer["b_ih"]) + len(layer["b_hh"])
    for layer in data.get("actor", []):
        deploy += sum(len(row) for row in layer["w"]) + len(layer["b"])
    total = deploy
    for layer in data.get("critic", []):
        total += sum(len(row) for row in layer["w"]) + len(layer["b"])
    return {"deployable": deploy, "with_critic": total}


def _count_mlp(path: str) -> dict[str, int]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format") == "ipd-rl-policy-v1":
        layers = list(data.get("trunk", [])) + [data["policy_head"]]
    else:
        layers = list(data.get("layers", []))
    deploy = sum(sum(len(row) for row in layer["w"]) + len(layer["b"])
                 for layer in layers)
    return {"deployable": deploy, "with_critic": deploy}


def _count_candidate(spec: CandidateSpec) -> dict[str, int] | None:
    try:
        if spec.kind == "selfplay":
            return _count_selfplay(str(spec.params["weights"]))
        if spec.kind == "rl":
            return _count_mlp(str(spec.params["weights"]))
        if spec.kind == "adv-rl":
            cfg_path = spec.params.get("config") or config_path(
                "adversarial_rl_policy_config_aggressive.json")
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            base = cfg.get("base_weights")
            if base and not Path(base).is_absolute():
                base = str(weights_path(base))
            return _count_mlp(base) if base else None
    except (OSError, KeyError, ValueError):
        return None
    return None


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    spec = CandidateSpec.from_dict(task["spec"])
    suite_name = task["suite"]
    suite = (build_final_suite() if suite_name == "final"
             else build_validation_suite())
    item = suite[int(task["item_index"])]
    rows = []
    policy = spec.build()
    for seed in task["seeds"]:
        opponent = item.factory(random.Random(int(seed)))
        result = run_match(policy, opponent)
        mine = result.mine
        theirs = result.theirs
        rows.append({
            "candidate": spec.name,
            "family": item.family,
            "opponent": item.name,
            "seed": int(seed),
            "score": int(result.score),
            "opponent_score": int(result.opponent_score),
            "own_defect_rate": sum(mine) / max(1, len(mine)),
            "opp_defect_rate": sum(theirs) / max(1, len(theirs)),
        })
    return {"candidate": spec.name, "family": item.family,
            "item": item.name, "rows": rows}


def _summarize(rows: Sequence[dict], baseline: dict[str, float] | None,
               ceiling: dict[str, float] | None = None) -> dict:
    by_family: dict[str, list[float]] = {}
    by_opponent: dict[str, list[float]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(float(row["score"]))
        by_opponent.setdefault(row["opponent"], []).append(float(row["score"]))
    families = {k: statistics.fmean(v) for k, v in by_family.items()}
    opps = {k: statistics.fmean(v) for k, v in by_opponent.items()}
    mean = statistics.fmean(float(r["score"]) for r in rows)
    sigma = (statistics.pstdev(list(families.values()))
             if len(families) > 1 else 0.0)
    norm_base = {}
    norm_ceiling = {}
    if baseline:
        norm_base = {k: v / baseline[k] for k, v in families.items()
                     if baseline.get(k)}
    if ceiling:
        norm_ceiling = {k: v / ceiling[k] for k, v in families.items()
                        if ceiling.get(k)}
    total = sum(float(r["score"]) for r in rows)
    top_family = max(families, key=families.get) if families else None
    top_share = (sum(by_family[top_family]) / total
                 if top_family and total else 0.0)
    final_scores = [v for k, v in opps.items() if k.startswith("Final-tuned")]
    triple_scores = [v for k, v in opps.items()
                     if k.startswith("Triple-majority")]
    return {
        "mean": mean,
        "families": families,
        "family_counts": {k: len(v) for k, v in by_family.items()},
        "min_family_raw": min(families.values()) if families else 0.0,
        "min_family_norm_vs_baseline": (min(norm_base.values())
                                        if norm_base else None),
        "min_family_norm_vs_ceiling": (min(norm_ceiling.values())
                                       if norm_ceiling else None),
        "max_minus_min": (max(families.values()) - min(families.values())
                          if families else 0.0),
        "sigma": sigma,
        "top_family": top_family,
        "top_family_share": top_share,
        "vs_final": statistics.fmean(final_scores) if final_scores else 0.0,
        "vs_triple": statistics.fmean(triple_scores) if triple_scores else 0.0,
        "per_opponent": opps,
        "n": len(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fair held-out comparison of IPD candidates.")
    ap.add_argument("--candidates", default="bc,rl-ppo,adv-rl,scratch")
    ap.add_argument("--extra", default="",
                    help="comma-separated name=path self-play JSON candidates")
    ap.add_argument("--extra-rl", default="",
                    help="comma-separated name=path RL/MLP JSON candidates")
    ap.add_argument("--suite", default="final",
                    choices=("final", "validation"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--baseline", default="rl-ppo")
    ap.add_argument("--mode", default="sample",
                    choices=("sample", "argmax"))
    ap.add_argument("--policy-seed", type=int, default=20261024)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--bc-weights", default=str(
        weights_path("selfplay_bc_policy.json")))
    ap.add_argument("--final-weights", default=str(
        weights_path("selfplay_gru_policy.json")))
    ap.add_argument("--scratch-weights", default=str(
        weights_path("selfplay_scratch_policy.json")))
    ap.add_argument("--rl-weights", default=str(
        weights_path("rl_policy_checkpoint.json")))
    ap.add_argument("--adv-config", default=str(
        config_path("adversarial_rl_policy_config_aggressive.json")))
    ap.add_argument("--out", default="results/selfplay_improvement/compare.json")
    args = ap.parse_args(argv)

    args.names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    extra_raw = args.extra
    args.extra = {}
    if extra_raw:
        for item in extra_raw.split(","):
            if not item.strip():
                continue
            name, path = item.split("=", 1)
            args.extra[name.strip()] = path.strip()
    extra_rl_raw = args.extra_rl
    args.extra_rl = {}
    if extra_rl_raw:
        for item in extra_rl_raw.split(","):
            if not item.strip():
                continue
            name, path = item.split("=", 1)
            args.extra_rl[name.strip()] = path.strip()
    table = _candidate_table(args)
    if not table:
        raise SystemExit("no candidates selected")

    suite = (build_final_suite() if args.suite == "final"
             else build_validation_suite())
    base = args.seed_base
    if base is None:
        base = 20561026 if args.suite == "final" else 20261026
    seeds = [base + i * 7919 for i in range(args.seeds)]
    tasks = []
    for spec in table.values():
        for index in range(len(suite)):
            tasks.append({"spec": spec.to_dict(), "suite": args.suite,
                          "item_index": index, "seeds": seeds})
    print(f"candidates={list(table)} suite={args.suite} "
          f"items={len(suite)} seeds={len(seeds)} tasks={len(tasks)}",
          flush=True)
    started = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for done, result in enumerate(pool.map(_worker, tasks, chunksize=1), 1):
            rows.extend(result["rows"])
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} item-tasks", flush=True)

    candidates: dict[str, dict] = {}
    names = list(table.values())
    name_list = [s.name for s in names]
    for spec in names:
        subset = [r for r in rows if r["candidate"] == spec.name]
        candidates[spec.name] = _summarize(subset, None)
        candidates[spec.name]["params"] = _count_candidate(spec)

    baseline_name = table[args.baseline].name if args.baseline in table else None
    baseline = candidates[baseline_name]["families"] if baseline_name else None
    ceiling = {}
    for cand in candidates.values():
        for family, value in cand["families"].items():
            ceiling[family] = max(ceiling.get(family, -1e9), value)
    for spec in names:
        subset = [r for r in rows if r["candidate"] == spec.name]
        summary = _summarize(subset, baseline, ceiling)
        summary["params"] = candidates[spec.name]["params"]
        candidates[spec.name] = summary

    payload = {
        "schema": "ipd-selfplay-compare-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "suite": args.suite,
        "seeds": seeds,
        "baseline": baseline_name,
        "candidates": candidates,
        "per_opponent": {
            name: {
                opp: [r for r in rows
                      if r["candidate"] == name and r["opponent"] == opp]
                for opp in sorted({r["opponent"] for r in rows
                                   if r["candidate"] == name})
            }
            for name in name_list
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written {out}", flush=True)
    for name, summary in candidates.items():
        p = summary.get("params") or {}
        print(f"{name:18} mean={summary['mean']:7.2f} "
              f"min_raw={summary['min_family_raw']:6.1f} "
              f"min_norm={summary['min_family_norm_vs_baseline']} "
              f"params={p.get('deployable')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
