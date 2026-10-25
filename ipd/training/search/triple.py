#!/usr/bin/env python3
"""Validation search for Triple fusion modes and confidence gates."""
from __future__ import annotations

if __package__ in (None, ""):  # allow: python ipd/training/<file>.py
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
from concurrent.futures import ProcessPoolExecutor

from ipd.evaluation.distributions import DISTS, build_dist_field
from ipd.paths import result_path
from ipd.policies.rules import (
    ParamBayesPrior, ParamPolicy, ParamResp, Triple,
)
from ipd.policies.tuned import load_tuned_configs



def load_cfgs():
    """Return ``(final_cfg, resp_cfg, bayes_cfg)`` for legacy callers."""
    tuned = load_tuned_configs()
    return tuned.final, tuned.resp, tuned.bayes


def make(kind, cfg):
    if kind == "final":
        return ParamPolicy(seed=20261024, **cfg)
    if kind == "resp":
        return ParamResp(seed=20261024, **cfg)
    if kind == "bayes":
        return ParamBayesPrior(cfg)
    if kind == "triple":
        return Triple(cfg["final"], cfg["resp"], cfg["bayes"],
                      cfg["mode"], cfg["conf"])
    raise ValueError(kind)


def eval_task(task):
    name, kind, cfg, seed_map, n = task
    out = {}
    for dist, seeds in seed_map.items():
        vals = []
        for seed in seeds:
            field = build_dist_field(dist, seed, n)
            policy = make(kind, cfg)
            total = 0
            for opponent in field:
                policy.reset()
                opponent.reset()
                mine: list[int] = []
                theirs: list[int] = []
                score = 0
                for t in range(100):
                    x = (policy.act(t, mine, theirs)
                         if hasattr(policy, "act")
                         else policy.move(t + 1, mine, theirs))
                    y = opponent.act(t, theirs, mine)
                    mine.append(x)
                    theirs.append(y)
                    score += (6 if (x, y) == (0, 0) else
                              0 if x == 0 else (10 if y == 0 else 1))
                total += score
            vals.append(total / n)
        out[dist] = vals
    return name, out


def fit(out):
    vals = [x for values in out.values() for x in values]
    return (0.45 * statistics.fmean(vals)
            + 0.25 * min(statistics.fmean(v) for v in out.values())
            + 0.20 * sorted(vals)[int(0.05 * len(vals))]
            + 0.10 * min(vals))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--fields", type=int, default=12)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--out", default=result_path("triple_results.json"))
    args = ap.parse_args(argv)

    tuned = load_tuned_configs()
    f, r, bc = tuned.final, tuned.resp, tuned.bayes
    variants = []
    for mode in ("or", "and", "majority"):
        variants.append((mode, "triple", {"final": f, "resp": r,
                                          "bayes": bc, "mode": mode,
                                          "conf": 0.65}))
    for mode in ("bayes_conf", "bayes_punitive", "bayes_nonresp"):
        for conf in ((0.55, 0.75) if mode == "bayes_conf" else (0.65,)):
            variants.append((f"{mode}_{conf}", "triple",
                             {"final": f, "resp": r, "bayes": bc,
                              "mode": mode, "conf": conf}))
    variants.append(("final", "final", f))
    variants.append(("resp", "resp", r))
    variants.append(("bayes", "bayes", bc))

    rng = random.Random(args.seed)
    seed_map = {d: [rng.randrange(1 << 30) for _ in range(args.fields)]
                for d in DISTS}
    tasks = [(name, kind, cfg, seed_map, args.n)
             for name, kind, cfg in variants]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(eval_task, tasks, chunksize=1))

    rows = sorted(((fit(out), name, out) for name, out in results),
                  reverse=True, key=lambda x: x[0])
    print("Validation:")
    for score, name, out in rows:
        print(f"{name:22} fit={score:7.2f} "
              f"mean={statistics.fmean([x for v in out.values() for x in v]):7.2f} "
              "dist={" + ", ".join(
                  f"{k}:{statistics.fmean(v):.1f}" for k, v in out.items())
              + "}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump([{"name": n, "fit": s, "out": o} for s, n, o in rows],
                  fh, ensure_ascii=False, indent=2)
    print("written", args.out)
    return 0


if __name__ == "__main__":
    main()
