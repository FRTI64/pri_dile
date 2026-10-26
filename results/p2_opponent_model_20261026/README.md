# P2 C2/C3 — learned opponent model + gate（2026-10-26）

## 0. 结论

**C2/C3 离线实验：No-Go。** 学习到的 gate 有效（held-out type gate
accuracy 86.6%），但部署时过度保守：gated conservative 只把 heldout_v2 抬
+1.20，held-out red 完全不变；gated aggressive 则因 adaptive/noisy false
positives 把 extort 打掉 107.6。未注册候选、未改主权重。详细设计、指标和
下一步见 `docs/p2-c2c3-opponent-model.md`。

## 1. 模型指标

| Model | validation | heldout_v2 |
|---|---:|---:|
| family MLP（9 类） | 42.6% | 32.1% |
| gate MLP（allow/deny） | **86.6%** | **70.6%**（precision 70.2%，recall 38.7%） |
| next-action linear | 81.9% | - |

Heldout_v2 gate 的 false positives 全部来自 adaptive/noisy；random/late 的
precision 为 1.0，但 recall 低（75% / 24%）。

## 2. Planner screen（fair RNG）

| Planner | validation Δ | heldout_v2 Δ | min_norm | override |
|---|---:|---:|---:|---:|
| ungated aggressive | -12.11 | -16.15 | 0.846 | 6.9–10.1% |
| ungated conservative | -2.19 | +2.06 | 0.993 | 0.2–0.3% |
| **gated conservative（t0.7）** | **0.00** | **+1.20** | **0.999** | 0.1% |
| gated aggressive | -7.40 | -21.91 | 0.768 | 2.6% |

Gated conservative：adaptive +8.75、noisy -0.33，extort/random/temporal 0；
held-out red E4/v5/v6 均无变化。Gate C 的 +10 未达到。

## 3. 诊断

- 二分类 gate 能学，但只靠 85 维历史特征无法可靠区分 adaptive/noisy 与
  random/late；
- RL-Repair 很少主动 early defection，很多 family 在历史上不可区分，
  excitation 不足；
- 下一步：用 exploratory/counterfactual blue 轨迹训练 opponent model，
  或做三路 `safe/risky/unknown` gate + MoE；unknown 回退 RL-Repair。
- 不继续加深 planner horizon；不改主权重。

## 4. 产物

```text
ipd/training/opponent_model.py
tests/test_opponent_model.py
weights/opponent_classifier.json
weights/opponent_gate.json
weights/opponent_next_action.json
results/p2_opponent_model_20261026/
  README.md
  opponent_model_report.json
  planner_ungated_aggressive.json
  planner_ungated_conservative.json
  planner_gated_t07.json / planner_gated_t06.json
  planner_gated_aggressive.json
  planner_gated_red.json
  run_planner_screens.sh / red_screen.py
  checksums.txt
docs/p2-c2c3-opponent-model.md
```

主权重 `weights/rl_policy_weights.json` sha256 仍为
`bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。

后续 exploratory / counterfactual 数据实验见
`results/p2_excitation_20261026/` 和 `docs/p2-c2c3-excitation.md`：
训练 excitation 已解决，gate 改善，但 planner 仍 No-Go。
