# P2 C1 — Bayesian belief planner offline screen（2026-10-26）

## 0. 结论

**C1 首版：No-Go（离线）。** 未注册候选、未改主权重；RL-Repair 仍是
base/fallback。详细设计、Gate C 评估和复现见
`docs/p2-c1-belief-planner.md`。

> 早期一次 screen 复用了同一个 base policy RNG，产生了 E4 +6.6 等伪影。
> 本目录已替换为 **fair-RNG** 结果：base 与 planner 使用独立、同 seed 的
> base 实例。

## 1. 结果摘要

| Screen | validation Δ | heldout_v2 Δ | min_norm | override | 判定 |
|---|---:|---:|---:|---:|---|
| aggressive（h6/top8/conf0.75） | -12.11 | -16.15 | 0.846 | 6.9–10.1% | No-Go |
| conservative（h4/top6/conf0.9/margin10） | -2.19 | +2.06 | 0.993 | 0.2–0.3% | No-Go |

保守版 heldout_v2 family：adaptive +8.75、temporal +8.70、extort 0、
noisy -3.00。Held-out red（fair RNG）：E4 +0.35、v5 0.00、v6 -0.10，
均无实质增益。

## 2. 诊断

- 模型库对 noisy / extort / non-responsive exploiter 的区分不足；
- 激进 planner 过度 defection，直接伤 extort/noisy；
- 提高 confidence/margin 后只在极少数状态覆盖 base（0.2–0.3%），
  只能带来局部小收益，无法形成稳定 family 改进；
- 下一步应做 C2 auxiliary opponent prediction 或 C3 classifier + MoE，
  用 learned responsiveness gating，而不是继续加深 planning horizon。

C3 的后续实验（learned gate）见 `results/p2_opponent_model_20261026/`。

## 3. 产物

```text
results/p2_belief_20261026/
  README.md
  planner_screen_h6.json
  planner_screen_conservative.json
  planner_red_screen.json
  checksums.txt
ipd/policies/belief_planner.py
ipd/evaluation/planner_eval.py
tests/test_belief_planner.py
docs/p2-c1-belief-planner.md
```

主权重 `weights/rl_policy_weights.json` 的 sha256 仍为
`bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。
