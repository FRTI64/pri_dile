#!/usr/bin/env python3
"""Fair-RNG held-out red screen for the three-way gate + MoE."""
from __future__ import annotations

import argparse
import json
import statistics

from ipd.core.simulator import play_match
from ipd.evaluation.policy_pool import load_policy
from ipd.policies.moe import DefectExpert, ThreeWayMoE
from ipd.policies.neural import NeuralPolicy
from ipd.training.opponent_model import ThreeWayGate, load_opponent_model

REDS = {
    "E4-red-BC": "weights/selfplay_improvement/E4_red_policy.json",
    "Red-RL-v5-heldout": "weights/red_pool/red_rl_v5_heldout.json",
    "Red-RL-v6-heldout": "weights/red_pool/red_rl_v6_heldout.json",
}
SEEDS = [20561026 + i * 7919 for i in range(20)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-model", default="weights/opponent_gate_threeway.json")
    ap.add_argument("--safe-threshold", type=float, default=0.6)
    ap.add_argument("--risky-threshold", type=float, default=0.6)
    ap.add_argument("--min-day", type=int, default=10)
    ap.add_argument("--response-model", default="")
    ap.add_argument("--max-response-probability", type=float, default=0.4)
    ap.add_argument("--out", default=(
        "results/p2_threeway_moe_20261026/moe_alld_red.json"))
    args = ap.parse_args()

    gate_model = load_opponent_model(args.gate_model)
    response_model = (load_opponent_model(args.response_model)
                      if args.response_model else None)
    out: dict[str, dict[str, float | int]] = {}
    for name, path in REDS.items():
        rows = {"base": [], "moe": []}
        diagnostics = []
        for seed in SEEDS:
            base = NeuralPolicy("weights/rl_policy_weights_repair.json",
                                seed=20261024, mode="sample")
            opponent = load_policy(path, seed=seed, mode="sample")
            result = play_match(base, opponent, rounds=100)
            rows["base"].append((result.score_a, result.score_b))

            fallback = NeuralPolicy(
                "weights/rl_policy_weights_repair.json",
                seed=20261024, mode="sample")
            gate = ThreeWayGate(
                gate_model, safe_threshold=args.safe_threshold,
                risky_threshold=args.risky_threshold, min_day=args.min_day,
                response_model=response_model,
                max_response_probability=(
                    args.max_response_probability if response_model else None))
            moe = ThreeWayMoE(fallback, gate, safe_expert=DefectExpert())
            opponent = load_policy(path, seed=seed, mode="sample")
            result = play_match(moe, opponent, rounds=100)
            rows["moe"].append((result.score_a, result.score_b))
            diagnostics.append(moe.diagnostics())
        base_mean = statistics.fmean(x[0] for x in rows["base"])
        moe_mean = statistics.fmean(x[0] for x in rows["moe"])
        rounds = sum(d["rounds"] for d in diagnostics)
        out[name] = {
            "base_blue": base_mean,
            "moe_blue": moe_mean,
            "blue_delta": moe_mean - base_mean,
            "base_diff": statistics.fmean(x[1] - x[0]
                                          for x in rows["base"]),
            "moe_diff": statistics.fmean(x[1] - x[0]
                                         for x in rows["moe"]),
            "safe_rate": sum(d["safe"] for d in diagnostics) / max(1, rounds),
            "risky_rate": (sum(d["risky"] for d in diagnostics)
                           / max(1, rounds)),
            "unknown_rate": (sum(d["unknown"] for d in diagnostics)
                             / max(1, rounds)),
        }
        print(name, out[name])
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
