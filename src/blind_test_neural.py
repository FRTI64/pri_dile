#!/usr/bin/env python3
"""Blind ranking test that includes the exported neural policy.

Uses the same 300-player ``sample_wide`` fields as ``blind_test.py`` but reuses
the opponent-vs-opponent round robin across candidates, which makes it much
cheaper to compare several strategies on unseen fields.

Example::

    python3 src/blind_test_neural.py --fields 10 --workers 10
    python3 src/blind_test_neural.py --arch rl --fields 10 --workers 12
    python3 src/blind_test_neural.py --arch adv-rl \
        --config configs/adversarial_rl_policy_config_aggressive.json \
        --fields 10 --workers 12
"""
from __future__ import annotations

from paths import config_path, result_path, weights_path

import argparse
import copy
import json
import statistics
from concurrent.futures import ProcessPoolExecutor

from ipd_sim import payoff, play_match
from neural_policy import NeuralPolicy
from gru_policy import GRUPolicy
from adversarial_policy import AdversarialPolicy
from rl_policy import RLPolicy
from search_common import ParamPolicy
from train_multiobj import ParamResp
from train_bayes_priors import ParamBayesPrior
from train_wide import sample_wide
from triple_hybrid import Triple


def load_cfgs():
    with open(result_path("train_multiobj_results.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    final_cfg = next(r["cfg"] for r in m["validation"] if r["kind"] == "final")
    resp_cfg = next(r["cfg"] for r in m["validation"] if r["kind"] == "resp")
    with open(result_path("train_bayes_priors_results.json"), encoding="utf-8") as fh:
        b = json.load(fh)
    bayes_cfg = max(b["validation"], key=lambda x: x["val_fit"])["cfg"]
    return final_cfg, resp_cfg, bayes_cfg


def make_candidates(weights: str, neural_mode: str, arch: str = "mlp",
                    config: str | None = None):
    final_cfg, resp_cfg, bayes_cfg = load_cfgs()
    if arch == "gru":
        policy = GRUPolicy(weights, seed=20261024, mode=neural_mode)
        label = "GRU"
    elif arch in ("rl", "adv-rl"):
        with open(weights, encoding="utf-8") as fh:
            fmt = json.load(fh).get("format")
        cls = NeuralPolicy if fmt == "ipd-neural-policy-v1" else RLPolicy
        base = cls(weights, seed=20261024, mode=neural_mode)
        if arch == "rl":
            policy = base
            label = "RL-PPO"
        else:
            with open(config, encoding="utf-8") as fh:
                acfg = json.load(fh)
            policy = AdversarialPolicy(
                base, gap=acfg.get("gap"), min_c=acfg.get("min_c", 8),
                min_d=acfg.get("min_d", 3), late_day=acfg.get("late_day"),
                repair=acfg.get("repair", False), name="Adv-RL",
            )
            label = "Adv-RL"
    elif arch == "adversarial":
        policy = AdversarialPolicy.from_config(config, seed=20261024,
                                                mode=neural_mode)
        label = "Adv"
    else:
        policy = NeuralPolicy(weights, seed=20261024, mode=neural_mode)
        label = "Neural"
    return [
        ("Final-tuned", ParamPolicy(seed=20261024, **final_cfg)),
        ("Resp3-tuned", ParamResp(seed=20261024, **resp_cfg)),
        ("Bayes-tuned", ParamBayesPrior(bayes_cfg)),
        ("Triple-majority", Triple(final_cfg, resp_cfg, bayes_cfg,
                                   mode="majority")),
        (f"{label}-{neural_mode}", policy),
    ]


def field_task(task):
    seed, n, weights, neural_mode, arch, config = task
    base_field, _, _ = sample_wide(seed, n, 0.15)
    n_opp = len(base_field)

    # Opponent round robin is candidate-independent, so compute it once.  Each
    # candidate later gets a deep copy starting from the same RNG state.
    field = copy.deepcopy(base_field)
    opp_scores = [0] * n_opp
    for i in range(n_opp):
        for j in range(i + 1, n_opp):
            r = play_match(field[i], field[j])
            opp_scores[i] += r.score_a
            opp_scores[j] += r.score_b

    out = {}
    for name, cand in make_candidates(weights, neural_mode, arch, config):
        cand_score = 0
        boosted = list(opp_scores)
        field = copy.deepcopy(base_field)
        for i, opp in enumerate(field):
            cand.reset()
            opp.reset()
            mine: list[int] = []
            theirs: list[int] = []
            own = other = 0
            for t in range(100):
                x = cand.act(t, mine, theirs) if hasattr(cand, "act") else cand.move(t + 1, mine, theirs)
                y = opp.act(t, theirs, mine)
                mine.append(x)
                theirs.append(y)
                own += payoff(x, y)
                other += payoff(y, x)
            cand_score += own
            boosted[i] += other
        rank = 1 + sum(s > cand_score for s in boosted)
        out[name] = {"rank": rank, "score": cand_score}
    return seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=10)
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--arch", default="mlp",
                    choices=("mlp", "gru", "rl", "adv-rl", "adversarial"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--neural-mode", default="sample", choices=("sample", "argmax"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.arch in ("adversarial", "adv-rl"):
        if args.config is None:
            args.config = config_path(
                "adversarial_policy_config_aggressive.json"
                if args.arch == "adv-rl" else "adversarial_policy_config.json")
        if args.arch == "adv-rl" and args.weights is None:
            with open(args.config, encoding="utf-8") as fh:
                base = json.load(fh).get("base_weights",
                                         "rl_policy_checkpoint.json")
            args.weights = weights_path(base)
        if args.out is None:
            args.out = result_path(
                "blind_adversarial_rl_results.json"
                if args.arch == "adv-rl" else "blind_adversarial_results.json")
    else:
        if args.weights is None:
            if args.arch == "gru":
                args.weights = weights_path("gru_policy_weights.json")
            elif args.arch == "rl":
                args.weights = weights_path("rl_policy_checkpoint.json")
            else:
                args.weights = weights_path("neural_policy_weights.json")
        if args.out is None:
            if args.arch == "gru":
                args.out = result_path("blind_gru_results.json")
            elif args.arch == "rl":
                args.out = result_path("blind_rl_results.json")
            else:
                args.out = result_path("blind_neural_results.json")
    seeds = [args.seed + i * 7919 for i in range(args.fields)]
    tasks = [(seed, args.n, args.weights, args.neural_mode, args.arch,
              args.config) for seed in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (seed, out) in enumerate(ex.map(field_task, tasks, chunksize=1), 1):
            rows.append(out)
            summary = ", ".join(f"{name}:{v['rank']}" for name, v in out.items())
            print(f"field {i}/{args.fields} seed={seed} {summary}", flush=True)

    print(f"\nBlind test over {args.fields} unseen {args.n}-player fields:")
    print(f"{'strategy':20}{'score':>10}{'mean rank':>12}{'rank1':>8}{'top10':>8}{'worst':>8}")
    result = {}
    for name in rows[0]:
        scores = [r[name]["score"] / args.n for r in rows]
        ranks = [r[name]["rank"] for r in rows]
        result[name] = {
            "score": statistics.fmean(scores),
            "mean_rank": statistics.fmean(ranks),
            "rank1": sum(x == 1 for x in ranks),
            "top10": sum(x <= 10 for x in ranks),
            "worst": max(ranks),
            "scores": scores,
            "ranks": ranks,
        }
        print(f"{name:20}{result[name]['score']:10.2f}"
              f"{result[name]['mean_rank']:12.2f}"
              f"{result[name]['rank1']:5d}/{args.fields:<2d}"
              f"{result[name]['top10']:5d}/{args.fields:<2d}"
              f"{result[name]['worst']:8d}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"arch": args.arch, "seed": args.seed, "fields": args.fields,
                   "n": args.n, "neural_mode": args.neural_mode,
                   "config": args.config, "results": result},
                  fh, ensure_ascii=False, indent=2)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
