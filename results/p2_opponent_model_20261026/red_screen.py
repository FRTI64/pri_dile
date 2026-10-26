#!/usr/bin/env python3
"""Fair-RNG held-out red screen for the gated belief planner."""
from __future__ import annotations

import json
import statistics

from ipd.core.simulator import play_match
from ipd.evaluation.policy_pool import load_policy
from ipd.policies.belief_planner import BeliefPlanner
from ipd.policies.neural import NeuralPolicy
from ipd.training.opponent_model import OpponentGate, load_opponent_model

REDS = {
    "E4-red-BC": "weights/selfplay_improvement/E4_red_policy.json",
    "Red-RL-v5-heldout": "weights/red_pool/red_rl_v5_heldout.json",
    "Red-RL-v6-heldout": "weights/red_pool/red_rl_v6_heldout.json",
}
SEEDS = [20561026 + i * 7919 for i in range(20)]


def main() -> None:
    out = {}
    for name, path in REDS.items():
        rows = {"base": [], "planner": []}
        diagnostics = []
        for seed in SEEDS:
            base = NeuralPolicy("weights/rl_policy_weights_repair.json",
                                seed=20261024, mode="sample")
            opponent = load_policy(path, seed=seed, mode="sample")
            result = play_match(base, opponent, rounds=100)
            rows["base"].append((result.score_a, result.score_b))

            base = NeuralPolicy("weights/rl_policy_weights_repair.json",
                                seed=20261024, mode="sample")
            gate = OpponentGate(
                load_opponent_model("weights/opponent_gate.json"),
                allowed=("allow",), threshold=0.7, min_day=5)
            planner = BeliefPlanner(
                base, horizon=4, top_k=6, max_hypotheses=48,
                fallback_confidence=0.9, margin=10, seed=20261024,
                gate=gate)
            opponent = load_policy(path, seed=seed, mode="sample")
            result = play_match(planner, opponent, rounds=100)
            rows["planner"].append((result.score_a, result.score_b))
            diagnostics.append(planner.diagnostics())
        base_mean = statistics.fmean(x[0] for x in rows["base"])
        planner_mean = statistics.fmean(x[0] for x in rows["planner"])
        base_diff = statistics.fmean(x[1] - x[0] for x in rows["base"])
        planner_diff = statistics.fmean(x[1] - x[0] for x in rows["planner"])
        out[name] = {
            "base_blue": base_mean,
            "planner_blue": planner_mean,
            "blue_delta": planner_mean - base_mean,
            "base_diff": base_diff,
            "planner_diff": planner_diff,
            "override_rate": (sum(d["overrides"] for d in diagnostics)
                              / max(1, sum(d["rounds"] for d in diagnostics))),
            "gate_fallback": sum(d.get("fallback_gate", 0)
                                 for d in diagnostics),
        }
        print(name, out[name])
    with open("results/p2_opponent_model_20261026/planner_gated_red.json",
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
