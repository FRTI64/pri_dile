# 100 天囚徒困境策略项目

100 天迭代囚徒困境（IPD）策略工具包：规则状态机、贝叶斯推断、多数投票融合、
MLP / GRU、约束剥削包装、PPO 强化学习和 21 维自博弈策略。评估以
**Protocol v4**（747 configs / 新风格集 / Axelrod / 2048 人赛场）
为主，另保留 legacy 四分布、
blind、9 个测试集、158 类对手、冻结 league 和 held-out red。

> **当前状态（2026-10-27）**
>
> - 主权重：`weights/rl_policy_weights.json`（RL-PPO，从零 + hard）；
> - 最佳候选：`rl-repair`（RL-Repair），但**未 promotion**；
> - fallback：`rl-hard60`（A60）；
> - **Protocol v4**：747 configs；12 个 complex 风格 + 50 个 Axelrod 策略 +
>   15 个 AxelrodMix；
>   默认 2048 人（2000+）/ dev 30 fields / test 50 fields；
> - v4 baseline **尚未运行**（按本次要求不做测试）；v3 的 299/20/40 结果
>   只作 legacy；
> - 上次完整测试为 v3 的 204 tests；v4 改动后未运行测试；
> - anti-exploitation **未解决**：Red-v5 / Red-v6 仍 100% 击败 top 3。
>
> 完整结论、权重、风险和 promotion 条件见
> **[docs/operations/handover.md](docs/operations/handover.md)**。

## 快速开始

```bash
python3 -m pip install -e .
python3 -m ipd list-policies
python3 -m ipd test

# Protocol v4：审计 split，跑 dev 选模（默认 2048 人 × 30 fields）
python3 -m ipd protocol audit
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO

# test_v7 只给预注册 top 3 跑一次（默认 50 fields）
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v7 --fields 50 --n 2048 --acknowledge-once

# 旧口径（legacy，仅历史复现）
python3 -m ipd benchmark --candidates core --fields 8 --workers 8
python3 -m ipd blind --candidates rl,rl-hard60,rl-repair \
  --fields 20 --workers 10
```

训练需要 PyTorch（`requirements-train.txt`）；推理和 C++ 部署不需要。
完整命令见 [docs/guides/](docs/guides/)，参考见 [docs/reference/](docs/reference/)。

## 当前候选（Protocol v4；下表是 legacy v3 baseline）

下表是 **Protocol v3** 的 299/20/40 baseline，不是 v4 排名；v4 的
2048/30/50 baseline 尚未运行。完整 v3 表见
[results/protocol_v3/README.md](results/protocol_v3/README.md)。

| 候选 | 注册名 | dev_param | dev_family | dev_complex | dev_ood | test_v6 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| **RL-Repair** | `rl-repair` | **532.44** | 582.92 | **420.38** | **489.84** | **484.32** | 最佳候选，未 promotion |
| RL-PPO-Hard60 | `rl-hard60` | 527.97 | 582.94 | 410.09 | 485.12 | 476.54 | 候选 / fallback |
| Adv-RL | `adv-rl` | 526.58 | 583.20 | 415.19 | 487.22 | 471.44 | dev_complex 高；red 弱 |
| RL-PPO | `rl` | 524.37 | 582.72 | 397.90 | 478.61 | - | 当前主权重；baseline |
| Final-tuned | `final-tuned` | 523.39 | 583.13 | 413.64 | 484.57 | - | 纯规则基线 |
| Triple-majority | `triple-majority` | 517.07 | 583.14 | 410.02 | 480.14 | - | 规则中最均衡 |
| SelfPlay-GRU | `selfplay` | 513.85 | 582.45 | 390.57 | 471.19 | - | 21 维备选 |
| Adv-MLP | `adv-mlp` | 515.24 | 583.18 | 406.65 | 478.26 | - | dev only |

- 旧 v3 baseline：RL-Repair test 484.32，A60 476.54，Adv-RL 471.44。
- v4 增加 Axelrod 和更大规模后必须重跑 dev/test；不要直接沿用 v3 排名。
- `dev_complex` / `dev_axelrod` 是主要新增区分层。
- `official` / `blind` / `heldout_v2/v3/v4` / v2 / v3 均为 legacy。
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
| Protocol v4 split | **启用**；747 configs + 新风格/Axelrod/complex/policy；test_v7 未消费 |
| Protocol v4 baseline | **未运行**（本次不测试）；v3 baseline 仅作 legacy |
| A60 / hard×2 | 候选 / fallback；v3 test 476.54（legacy） |
| E0 league | **PASS**（legacy）；建立 cross-play / Nash / exploitability |
| E1c repair | **Go candidate**（RL-Repair），未 promotion |
| E2 CVaR / DRO | No-Go |
| P3 red pool | red pool Go；league-lite / expansion Gate D **FAIL** |
| P2 C1 / C2 / C3 | 全部 offline No-Go |
| SelfPlay 改进 | No-Go；保留 21 维部署备选 |
| Axelrod | **已接入 Protocol v4**：50 策略 + 15 组合；可选依赖 |

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
