# 研究记录

这里保存实验设计、协议、完整结果和 Go/No-Go。**实验文档不回写历史数字**；
当前推荐和 promotion 状态以 [../operations/handover.md](../operations/handover.md)
为准。

## 评估协议

- **当前主口径：Protocol v4**，见 [../reference/protocol-v4.md](../reference/protocol-v4.md)；
- legacy：`benchmark --fields 20` / `blind --fields 20` /
  `league --config configs/eval/e0_league.json`；
- `heldout_v2/v3/v4` 与 `official/blind` 只保留为历史记录；
- 指标定义见 [../reference/metrics.md](../reference/metrics.md)。

## 实验总表

| 阶段 | 文档 | 结论 |
|---|---|---|
| Protocol v4 | [../reference/protocol-v4.md](../reference/protocol-v4.md) | **已定义**；977 configs + level/complexity + Axelrod；baseline 未运行 |
| Plan 1 | [plan1-benchmark.md](plan1-benchmark.md) | 实现完成；test_v7 0.85h（2.97×），RNG parity 通过；0.5–0.6h 目标未达 |
| Plan 2 | [plan2-gpu-policies.md](plan2-gpu-policies.md) | **Gate 2 PASS**；test_v7 4.27×（714.97s）；argmax/float64 parity |
| Plan 3 | [plan3-fast-opponents.md](plan3-fast-opponents.md) | **Gate 3A C++**；规则实例 99.4% 覆盖；规则评估 7.55×、B=2048 训练 3.04×；小 batch/全 mix 回退 |
| E0 | [e0-league.md](e0-league.md) | **PASS**；建立 cross-play / Nash / exploitability / held-out red |
| E0 plan | [e0-league-plan.md](e0-league-plan.md) | implemented |
| E1 | [e1-robust-finetune.md](e1-robust-finetune.md) | E1a/E1b No-Go；**E1c Go candidate**（RL-Repair） |
| E2 | [e2-risk-ppo.md](e2-risk-ppo.md) | **No-Go**（CVaR/DRO 未过 tail Gate） |
| P3 | [p3-red-league.md](p3-red-league.md) | red pool Go；league-lite No-Go |
| P3 expansion | [p3-expansion.md](p3-expansion.md) | **No-Go**（Gate D FAIL） |
| PSRO | [psro.md](psro.md) | 实现完成；`psro` / `psro-br` 实验候选；**No-Go for promotion** |
| PSRO from scratch | [psro-scratch.md](psro-scratch.md) | 现有模型只做 red；`psro-scratch` red_win 0.0，但 dev 480.43；**No-Go** |
| P2 C1 | [p2-c1-belief-planner.md](p2-c1-belief-planner.md) | **No-Go** |
| P2 C2/C3 | [p2-c2c3-opponent-model.md](p2-c2c3-opponent-model.md) | **No-Go** |
| P2 C2/C3 数据 | [p2-c2c3-excitation.md](p2-c2c3-excitation.md) | 数据层 Go；planner No-Go |
| P2 C3 | [p2-c3-threeway-moe.md](p2-c3-threeway-moe.md) | **No-Go** |
| P2 C3 校准 | [p2-c3-repeated-calib.md](p2-c3-repeated-calib.md) | **No-Go** |
| SelfPlay | [selfplay-plan.md](selfplay-plan.md) / [selfplay-improvement.md](selfplay-improvement.md) | implemented；改进 **No-Go** |
| Roadmap | [long-term-plan.md](long-term-plan.md) | 长期路线与 Gate 状态 |

## 相关目录

- 策略方法：[../policies/](../policies/)
- 历史快照：[../history/](../history/)
- 架构决策：[../decisions/](../decisions/)
- 结构化结果与逐 family 表：`results/<experiment>/README.md`
