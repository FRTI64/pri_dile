# 100 天囚徒困境策略项目

100 天迭代囚徒困境（IPD）策略工具包：规则状态机、贝叶斯推断、多数投票融合、
MLP / GRU、约束剥削包装、PPO 强化学习和 21 维自博弈策略。评估以
**Protocol v2**（train / dev / test 泄漏审计）为主，另保留 legacy 四分布、
blind、9 个测试集、158 类对手、冻结 league 和 held-out red。

> **当前状态（2026-10-27）**
>
> - 主权重：`weights/rl_policy_weights.json`（RL-PPO，从零 + hard）；
> - 最佳候选：`rl-repair`（RL-Repair），但**未 promotion**；
> - fallback：`rl-hard60`（A60）；
> - Protocol v2 test_v5：**RL-Repair 477.05** > A60 469.91 > Adv-RL 467.95
>   （30 fields；top 3 预注册，已消费）；
> - 测试：`python3 -m ipd test` → **202 tests OK**；注册候选 **20 个**；
> - anti-exploitation **未解决**：Red-v5 / Red-v6 仍 100% 击败 top 3。
>
> 完整结论、权重、风险和 promotion 条件见
> **[docs/operations/handover.md](docs/operations/handover.md)**。

## 快速开始

```bash
python3 -m pip install -e .
python3 -m ipd list-policies
python3 -m ipd test

# Protocol v2：审计 split，跑 dev 选模
python3 -m ipd protocol audit
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family --fields 20 --workers 20 --baseline RL-PPO

# test_v5 只给预注册 top 3 跑一次
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v5 --fields 30 --acknowledge-once

# 旧口径（legacy，仅历史复现）
python3 -m ipd benchmark --candidates core --fields 8 --workers 8
python3 -m ipd blind --candidates rl,rl-hard60,rl-repair \
  --fields 20 --workers 10
```

训练需要 PyTorch（`requirements-train.txt`）；推理和 C++ 部署不需要。
完整命令见 [docs/guides/](docs/guides/)，参考见 [docs/reference/](docs/reference/)。

## 当前候选（Protocol v2）

dev = 20 fields × 299；test_v5 = 30 fields × 299，top 3 一次性消费。
完整 20 候选见 [results/protocol_v2/README.md](results/protocol_v2/README.md)。

| 候选 | 注册名 | dev_param | dev_family | test_v5 | 状态 |
|---|---|---:|---:|---:|---|
| **RL-Repair** | `rl-repair` | 520.09 | 582.71 | **477.05** | 最佳候选，未 promotion |
| RL-PPO-Hard60 | `rl-hard60` | 514.18 | 582.73 | 469.91 | 候选 / fallback |
| Adv-RL | `adv-rl` | **520.19** | 583.03 | 467.95 | dev 最高；red 弱 |
| Adv-GRU | `adv-gru` | 512.71 | 582.96 | - | dev only |
| Adv-MLP | `adv-mlp` | 512.18 | 582.98 | - | dev only |
| RL-PPO | `rl` | 510.55 | 582.53 | - | 当前主权重；baseline |
| SelfPlay-GRU | `selfplay` | 506.58 | 582.34 | - | 21 维备选 |
| Triple-majority | `triple-majority` | 505.63 | 582.88 | - | 规则中最均衡 |

- dev top 3 由 dev_param mean 预注册；Adv-RL 与 RL-Repair 的 dev 差 0.10，
  属于噪声；test_v5 上 RL-Repair +7.14 vs A60、+9.10 vs Adv-RL（paired
  bootstrap CI 均不跨 0）。
- 旧 `official` / `blind` / `heldout_v2/v3/v4` 已进入 legacy，不要与新表混用；
  历史数字见 [docs/operations/handover.md](docs/operations/handover.md)。
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
| Protocol v2 split | **启用**；dev/test 可审计，test_v5 已消费 |
| Protocol v2 baseline | RL-Repair test 第一；anti-exploitation 仍 100% 失败 |
| A60 / hard×2 | 候选 / fallback；Protocol v2 test 469.91 |
| E0 league | **PASS**（legacy）；建立 cross-play / Nash / exploitability |
| E1c repair | **Go candidate**（RL-Repair），未 promotion |
| E2 CVaR / DRO | No-Go |
| P3 red pool | red pool Go；league-lite / expansion Gate D **FAIL** |
| P2 C1 / C2 / C3 | 全部 offline No-Go |
| SelfPlay 改进 | No-Go；保留 21 维部署备选 |
| Axelrod | 可选对手库；不进入 Protocol v2 主口径 |

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
