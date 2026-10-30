#!/usr/bin/env python3
"""Evaluate a narrow anti-probe rule overlay on Protocol v5 dev tiers."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.core.opponents import AllD  # noqa: E402
from ipd.core.primitives import C, D  # noqa: E402
from ipd.evaluation.common import evaluate_field_details  # noqa: E402
from ipd.evaluation.protocol import build_protocol_field  # noqa: E402
from ipd.policies.registry import make_policy  # noqa: E402


class RuleOverlay:
    """RL-Wide160 with a narrow probe=6/8 -> AllD switch."""

    def __init__(self, base, alt, firsts=(5, 7), need=2):
        self.base = base
        self.alt = alt
        self.firsts = tuple(firsts)
        self.need = int(need)
        self.switched = False
        self.name = "ProbeRule"

    def reset(self):
        self.base.reset()
        self.alt.reset()
        self.switched = False

    def set_rounds(self, rounds):
        for policy in (self.base, self.alt):
            fn = getattr(policy, "set_rounds", None)
            if callable(fn):
                fn(rounds)

    def act(self, t, mine, theirs):
        if not self.switched and theirs:
            first_d = next((i for i, x in enumerate(theirs) if x == D), None)
            if (first_d in self.firsts
                    and all(x == C for x in theirs[:first_d])
                    and sum(1 for x in theirs[first_d:] if x == D) >= self.need):
                self.switched = True
        if self.switched:
            return self.alt.act(t, mine, theirs)
        return self.base.act(t, mine, theirs)


def _worker(task):
    tier, seed, n, variant, base_weights, alt_weights = task
    field = build_protocol_field(tier, seed, n)
    if base_weights:
        base = make_policy("rl", weights=base_weights, mode="sample")
    else:
        base = make_policy("rl-wide160", mode="sample")
    if alt_weights:
        alt = make_policy("rl", weights=alt_weights, mode="sample")
    else:
        alt = AllD()
    if variant == "base":
        policy = base
    else:
        firsts = tuple(int(x) for x in variant.split("-"))
        policy = RuleOverlay(base, alt, firsts=firsts)
    stats, by = evaluate_field_details(policy, field, rounds=100)
    fam = {
        name.split(":", 1)[0]: statistics.fmean(scores)
        for name, scores in by.items()
    }
    # evaluate_field_details groups by full opponent name; collapse family.
    fam_scores: dict[str, list[int]] = {}
    for name, scores in by.items():
        fam_scores.setdefault(name.split(":", 1)[0], []).extend(scores)
    family_means = {k: statistics.fmean(v) for k, v in fam_scores.items()}
    return {"tier": tier, "seed": seed, "variant": variant,
            "mean": stats.mean, "family_means": family_means}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="dev_complex,dev_ood")
    ap.add_argument("--fields", type=int, default=30)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--base-weights", default=None,
                    help="optional actor-critic/deployable weights for the base")
    ap.add_argument("--alt-weights", default=None,
                    help="optional specialist checkpoint for the switched branch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    variants = ["base", "5", "7", "5-7"]
    seeds = [args.seed + i * 7919 for i in range(args.fields)]
    tasks = [(tier, seed, args.n, variant, args.base_weights,
              args.alt_weights)
             for tier in tiers for seed in seeds for variant in variants]
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for row in ex.map(_worker, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % max(1, len(tasks) // 20) == 0:
                print(f"progress {len(rows)}/{len(tasks)} "
                      f"elapsed={time.time()-started:.0f}s", flush=True)
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for tier in tiers:
            vals = [r for r in rows
                    if r["variant"] == variant and r["tier"] == tier]
            summary[variant][tier] = {
                "mean": statistics.fmean(r["mean"] for r in vals),
                "family_means": {
                    fam: statistics.fmean(r["family_means"][fam]
                                         for r in vals)
                    for fam in vals[0]["family_means"]
                },
            }
    payload = {
        "protocol": "protocol_v5",
        "seconds": round(time.time() - started, 2),
        "tiers": tiers,
        "fields": args.fields,
        "n": args.n,
        "seed": args.seed,
        "variants": variants,
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    for variant in variants:
        cells = "  ".join(
            f"{tier}={summary[variant][tier]['mean']:.2f}" for tier in tiers
        )
        probe = "  ".join(
            f"{tier}/probe={summary[variant][tier]['family_means'].get('probe_exploit', 0):.2f}"
            for tier in tiers
        )
        print(f"{variant:6} {cells}  {probe}")
    print("written", out)


if __name__ == "__main__":
    main()
