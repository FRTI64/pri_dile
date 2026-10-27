# 100 天囚徒困境策略项目

100 天迭代囚徒困境（IPD）策略工具包：规则状态机、贝叶斯推断、多数投票融合、
MLP / GRU、约束剥削包装、PPO 强化学习和 21 维自博弈策略。评估覆盖四分布、
blind、9 个测试集、158 类对手、冻结 league 和 held-out red。

> **当前状态（2026-10-27）**
>
> - 主权重：`weights/rl_policy_weights.json`（RL-PPO，从零 + hard）；
> - 最佳候选：`rl-repair`（RL-Repair），但**未 promotion**；
> - fallback：`rl-hard60`（A60）；
> - 测试：`python3 -m ipd test` → **183 tests OK**；注册候选 **20 个**；
> - anti-exploitation **未解决**：E4 red 仍接近 100% 击败当前蓝军。
>
> 完整结论、权重、风险和 promotion 条件见
> **[docs/operations/handover.md](docs/operations/handover.md)**。

## 快速开始

```bash
python3 -m pip install -e .
python3 -m ipd list-policies
python3 -m ipd test

# 四分布烟测 / 当前候选正式评估
python3 -m ipd benchmark --candidates core --fields 8 --workers 8
python3 -m ipd benchmark --candidates rl,rl-hard60,rl-repair \
  --fields 20 --workers 20
python3 -m ipd blind --candidates rl,rl-hard60,rl-repair \
  --fields 20 --workers 10
```

训练需要 PyTorch（`requirements-train.txt`）；推理和 C++ 部署不需要。
完整命令见 [docs/guides/](docs/guides/)，参考见 [docs/reference/](docs/reference/)。

## 当前候选

E0/E1c 冻结协议，20-field official/blind；旧 10-field 结果已归档。

| 候选 | 注册名 | official | blind 20 | heldout v2 | heldout v3 | 状态 |
|---|---|---:|---:|---:|---:|---|
| **RL-Repair** | `rl-repair` | **554.72** | **564.68** | **503.17** | **509.88** | 最佳候选，未 promotion |
| RL-PPO-Hard60 | `rl-hard60` | 554.27 | 563.87 | 500.07 | 505.97 | 候选 / fallback |
| RL-PPO | `rl` | 553.22 | 562.80 | 494.97 | 502.31 | 当前主权重 |
| Adv-RL | `adv-rl` | 554.05 | 562.97 | 502.39 | 504.84 | 仅分布匹配时考虑 |
| SelfPlay-GRU | `selfplay` | 550.97 | 561.13 | 491.86 | - | 21 维备选 |
| Triple-majority | `triple-majority` | 547.37 | 551.33 | 491.94 | - | 规则中最均衡 |
| Final-tuned | `final-tuned` | 546.35 | 547.75 | 490.04 | - | 纯规则基线 |

- 标准 CLI（独立 policy seed）：RL-PPO 553.33 / 562.83、A60 554.36 / 564.17、
  RL-Repair 554.76 / 564.61；与冻结协议略有差异，不能混表。
- `heldout_v3` 已被 E1c 使用；`heldout_v4` 未使用。
- 指标定义见 [docs/reference/metrics.md](docs/reference/metrics.md)。

## 目录

```text
ipd/          Python 包：core / policies / evaluation / training / tools
cpp/          C++17 推理与生成头文件
configs/      评估、对抗、league、self-play 配置
weights/      训练权重与 checkpoint
results/      基准、盲测、实验报告、训练曲线
tests/        分层测试
docs/         文档：架构 / 指南 / 参考 / 策略 / 研究 / 运维 / 历史 / ADR
```

分层和接口规范见 [docs/architecture/overview.md](docs/architecture/overview.md)。

## 当前实验结论

| 阶段 | 结论 |
|---|---|
| A60 / hard×2 | 候选；official 554.36 / blind 20-field 564.17 |
| E0 league | **PASS**；建立 cross-play / Nash / exploitability / held-out red |
| E1c repair | **Go candidate**（RL-Repair），未 promotion |
| E2 CVaR / DRO | No-Go |
| P3 red pool | red pool Go；league-lite / expansion Gate D **FAIL** |
| P2 C1 / C2 / C3 | 全部 offline No-Go |
| SelfPlay 改进 | No-Go；保留 21 维部署备选 |
| Axelrod | 可选对手库；默认口径不变 |

完整记录见 [docs/research/](docs/research/) 和
`results/sweep_20261026/README.md`（66 候选 × 158 类对手）。

## 文档

- 文档总入口：[docs/README.md](docs/README.md)
- 交接 / 下一 session：[docs/operations/handover.md](docs/operations/handover.md)
- 评估 / 训练 / 部署：[docs/guides/](docs/guides/)
- CLI / 候选 / 配置 / 指标：[docs/reference/](docs/reference/)
- 策略方法：[docs/policies/](docs/policies/)
- 实验与 Go/No-Go：[docs/research/](docs/research/)
- 历史快照：[docs/history/](docs/history/)
- 架构决策：[docs/decisions/](docs/decisions/)

## 参与开发

```bash
python3 -m ipd test
python3 -m ipd self-test --candidates all
```

开发约定、添加候选/评估任务、文档规则见 [CONTRIBUTING.md](CONTRIBUTING.md)；
版本变更见 [CHANGELOG.md](CHANGELOG.md)。

## 注意

- 所有候选都应在第 100 天选择背叛；
- GRU / SelfPlay-GRU 是有状态策略，每场必须 `reset()`；
- `ipd/core/machine.py` / `cpp/ipd_strategy.cpp` 默认参数不是 Final-tuned；
- 没有真实 300 人比赛数据，结论来自合成对手库和 held-out 评估；
- 未确认前不要替换主权重；promotion 规则见
  [docs/decisions/0002-no-promotion-without-reproduction.md](docs/decisions/0002-no-promotion-without-reproduction.md)。
