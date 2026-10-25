#!/usr/bin/env python3
"""Benchmark the exported neural policy against the hand-written candidates.

Uses exactly the same four distributions and seed convention as
``final_benchmark.py``, so the numbers are directly comparable to
``final_benchmark.json``.

Examples::

    python3 src/benchmark_neural_policy.py --fields 8 --workers 20
    python3 src/benchmark_neural_policy.py --neural-mode argmax --fields 4
"""
from __future__ import annotations

from paths import config_path, result_path, weights_path

import argparse
import json
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

from ipd_sim import payoff
from neural_policy import NeuralPolicy
from gru_policy import GRUPolicy
from adversarial_policy import AdversarialPolicy
from search_common import ParamPolicy
from train_multiobj import ParamResp, build_dist_field
from train_bayes_priors import ParamBayesPrior
from triple_hybrid import Triple

DISTS = ("original", "late", "grim30", "wide")


def load_cfgs():
    with open(result_path("train_multiobj_results.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    final_cfg = next(r["cfg"] for r in m["validation"] if r["kind"] == "final")
    resp_cfg = next(r["cfg"] for r in m["validation"] if r["kind"] == "resp")
    with open(result_path("train_bayes_priors_results.json"), encoding="utf-8") as fh:
        b = json.load(fh)
    bayes_cfg = max(b["validation"], key=lambda x: x["val_fit"])["cfg"]
    return final_cfg, resp_cfg, bayes_cfg


def make(kind, cfg):
    if kind == "final":
        return ParamPolicy(seed=20261024, **cfg)
    if kind == "resp":
        return ParamResp(seed=20261024, **cfg)
    if kind == "bayes":
        return ParamBayesPrior(cfg["bayes"])
    if kind == "triple":
        return Triple(cfg["final"], cfg["resp"], cfg["bayes"],
                      mode=cfg.get("mode", "majority"))
    if kind == "neural":
        return NeuralPolicy(cfg["weights"], seed=cfg.get("seed", 20261024),
                            mode=cfg.get("mode", "sample"),
                            name=cfg.get("name", "Neural"))
    if kind == "gru":
        return GRUPolicy(cfg["weights"], seed=cfg.get("seed", 20261024),
                         mode=cfg.get("mode", "sample"),
                         name=cfg.get("name", "GRU-Net"))
    if kind == "adversarial":
        return AdversarialPolicy.from_config(
            cfg["config"], seed=cfg.get("seed", 20261024),
            mode=cfg.get("mode", "sample"), name=cfg.get("name"),
        )
    raise ValueError(kind)


def eval_task(task):
    name, kind, cfg, dist, seeds, n = task
    vals = []
    for seed in seeds:
        field = build_dist_field(dist, seed, n)
        p = make(kind, cfg)
        total = 0
        for opp in field:
            p.reset()
            opp.reset()
            mine: list[int] = []
            theirs: list[int] = []
            s = 0
            for t in range(100):
                x = p.act(t, mine, theirs) if hasattr(p, "act") else p.move(t + 1, mine, theirs)
                y = opp.act(t, theirs, mine)
                mine.append(x)
                theirs.append(y)
                s += payoff(x, y)
            total += s
        vals.append(total / n)
    return name, dist, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp", choices=("mlp", "gru", "adversarial"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None,
                    help="adversarial config (default: adversarial_policy_config.json)")
    ap.add_argument("--fields", type=int, default=8)
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--neural-mode", default="sample", choices=("sample", "argmax"))
    ap.add_argument("--include-baselines", action="store_true", default=True)
    ap.add_argument("--no-baselines", dest="include_baselines", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.arch == "adversarial":
        if args.config is None:
            args.config = config_path("adversarial_policy_config.json")
        with open(args.config, encoding="utf-8") as fh:
            acfg = json.load(fh)
        label = f"Adv-{acfg.get('base_arch', 'mlp').upper()}"
        kind = "adversarial"
        if args.out is None:
            args.out = result_path("adversarial_benchmark.json")
    else:
        if args.weights is None:
            args.weights = (weights_path("gru_policy_weights.json") if args.arch == "gru"
                            else weights_path("neural_policy_weights.json"))
        if args.out is None:
            args.out = (result_path("gru_benchmark.json") if args.arch == "gru"
                        else result_path("neural_benchmark.json"))
        label = "GRU" if args.arch == "gru" else "Neural"
        kind = "gru" if args.arch == "gru" else "neural"

    final_cfg, resp_cfg, bayes_cfg = load_cfgs()
    cands = [
        ("Final-tuned", "final", final_cfg),
        ("Resp3-tuned", "resp", resp_cfg),
        ("Bayes-tuned", "bayes", {"bayes": bayes_cfg}),
        ("Triple-majority", "triple", {"final": final_cfg, "resp": resp_cfg,
                                       "bayes": bayes_cfg, "mode": "majority"}),
        (f"{label}-{args.neural_mode}", kind,
         ({"config": args.config, "mode": args.neural_mode}
          if kind == "adversarial"
          else {"weights": args.weights, "mode": args.neural_mode})),
    ]
    if not args.include_baselines:
        cands = cands[-1:]

    rng = random.Random(args.seed)
    seed_map = {d: [rng.randrange(1 << 30) for _ in range(args.fields)]
                for d in DISTS}
    tasks = [(name, kind, cfg, dist, seeds, args.n)
             for name, kind, cfg in cands
             for dist, seeds in seed_map.items()]

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, dist, vals in ex.map(eval_task, tasks, chunksize=1):
            results.setdefault(name, {})[dist] = {
                "mean": statistics.fmean(vals),
                "min": min(vals),
                "p05": sorted(vals)[max(0, int(0.05 * len(vals)))],
                "values": vals,
            }
            print(f"{name:16} {dist:9} mean={results[name][dist]['mean']:7.2f} "
                  f"min={results[name][dist]['min']:7.2f}", flush=True)

    print("\nSummary (mean score per opponent):")
    print(f"{'strategy':18}" + "".join(f"{d:>10}" for d in DISTS) +
          f"{'avg':>10}{'worst':>10}")
    summary = {}
    for name, _, _ in cands:
        means = [results[name][d]["mean"] for d in DISTS]
        row = {"dists": results[name],
               "avg": statistics.fmean(means),
               "worst": min(means),
               "min": min(results[name][d]["min"] for d in DISTS)}
        summary[name] = row
        print(f"{name:18}" + "".join(f"{m:10.2f}" for m in means) +
              f"{row['avg']:10.2f}{row['worst']:10.2f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"arch": args.arch, "seed": args.seed, "fields": args.fields,
                   "n": args.n, "weights": args.weights,
                   "config": getattr(args, "config", None),
                   "summary": summary},
                  fh, ensure_ascii=False, indent=2)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
