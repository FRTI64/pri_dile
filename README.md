# 100 天囚徒困境策略项目

100 天迭代囚徒困境（IPD）策略工具包：规则状态机、贝叶斯推断、多数投票融合、
MLP / GRU、约束剥削包装、PPO 强化学习和 21 维自博弈策略。评估以
**Protocol v4**（977 configs / 新风格集 / Axelrod / 2048 人赛场）
为主，另保留 legacy 四分布、
blind、9 个测试集、158 类对手、冻结 league 和 held-out red。

> **当前状态（2026-10-29）**
>
> - 主权重：`weights/rl_policy_weights.json`（RL-PPO，从零 + hard）；
> - **Protocol v4 dev/test 最佳候选**：`rl-switch`（RL-Switch，窄 meta
>   RL-Wide160→RL-Repair），dev_param 548.58 / dev_complex 454.24 /
>   dev_axelrod 582.86；`test_v7` provisional **488.26**（seed 27）和
>   **487.51**（seed 29）；但 **未 promotion**，Python + C++ 已移植；
> - 前最佳候选 / fallback：`rl-wide160`（160/128）、`rl-wide`（128/128）、
>   `rl-pattern`（64/32）、`rl-repair`；A60 仍为 `rl-hard60`；
> - **Protocol v4**：977 configs；14 个 complex 风格 + 224 个 Axelrod 策略 +
>   15 个 AxelrodMix；
>   默认 2048 人（2000+）/ dev 30 fields / test 50 fields；
> - v4 全候选 baseline **尚未运行**；top 6 的 dev 已运行，v3 的
>   299/20/40 结果只作 legacy；
> - **330 tests OK**（本轮全量运行）；注册候选 31 个；
> - **PSRO / red 历史存档**：以下 PSRO 结果只用于复现，不再作为
>   当前评估或 promotion 门槛。
> - **PSRO 已引入**：完整 blue-vs-red PSRO + 2 轮 PPO oracle；`psro`
>   meta-policy 在 held-out Red-v5/v6 上 red-minus-blue -21.4%，但
>   dev_param 527.71 vs RL-Repair 543.35；`psro-br` 未进入 maximin support；
>   两者均为实验候选，**未 promotion**；
> - **PSRO v4 已训练**：`psro-v4` / `psro-v4-br`；3 轮 × 30 iter、
>   40% red、低 KL；BR_01 进入 maximin support（权重 1.0），
>   exploitability 290.00 → **0.00**；`psro-v4-br` 在 dev_red /
>   Red-v5/v6 上 **red_win 0.0**，但 dev_param 486.93 vs RL-Repair
>   546.47（-59.5）、min_norm 0.56；**anti-exploitation specialist，
>   未 promotion**；
> - **从零 PSRO 已加入**：`psro-scratch` / `psro-scratch-br`；blue 只含
>   random init + 从零 scratch base + 从零 BR，现有模型/规则只做 red；
>   **dev_param 480.43、min_norm 0.55**，但 dev_red / held-out Red-v5/v6
>   **red_win 0.0**（首次翻转 red 结果）；**未 promotion**；
> - **blue-red 对战已停用**（ADR 0011）：以后不再用 red /
>   exploitability 作为评估或 promotion 门槛；现有 red/PSRO 结果仅作
>   历史存档，不再新增 red 任务。
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

# test_v7 永久锁定为 provisional（不会消费；默认 50 fields）
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v7 --fields 50 --n 2048 --no-consume

# 只验证 pipeline，同样不会消费 test_v7
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v7 --fields 1 --n 256 --workers 4 \
  --no-consume --out /tmp/test_v7_smoke.json

# 按对手思考等级 / 策略复杂度
python3 -m ipd protocol eval --candidates all \
  --tiers dev_level3 --fields 30 --n 2048 \
  --out results/protocol_v4/protocol_v4_dev_level3.json
python3 -m ipd protocol eval --candidates all \
  --tiers dev_complexity4 --fields 30 --n 2048 \
  --out results/protocol_v4/protocol_v4_dev_complexity4.json

# PSRO（reference population；实验）
python3 -m ipd train psro --config configs/psro/psro_20261027.json \
  --workers 20

# PSRO v4（当前头部 blue + 更强 oracle；实验）
python3 -m ipd train psro --config configs/psro/psro_v4_20261029.json \
  --workers 20

# 从零 PSRO（blue 只有 random/scratch/BR；现有模型只做 red）
python3 -m ipd train psro --config configs/psro/psro_scratch_20261027.json \
  --workers 20

# 旧口径（legacy，仅历史复现）
python3 -m ipd benchmark --candidates core --fields 8 --workers 8
python3 -m ipd blind --candidates rl,rl-hard60,rl-repair \
  --fields 20 --workers 10
```

训练需要 PyTorch（`requirements-train.txt`）；推理和 C++ 部署不需要。
完整命令见 [docs/guides/](docs/guides/)，参考见 [docs/reference/](docs/reference/)。

## 当前候选（Protocol v4）

**Protocol v4 dev（30 fields × 2048，sample，2026-10-29）：**

| 候选 | 注册名 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| **RL-Switch** | `rl-switch` | **548.58** | 582.91 | 454.24 | **582.86** | **521.42** | v4 dev/test 最佳；Python + C++；未 promotion |
| RL-Wide160 | `rl-wide160` | 548.26 | **583.08** | **454.31** | 582.38 | 521.37 | 160/128；未 promotion |
| RL-Wide | `rl-wide` | 548.24 | **583.08** | 452.79 | 582.30 | 520.71 | 128/128 前最佳；未 promotion |
| RL-Pattern | `rl-pattern` | **548.45** | 583.07 | 447.64 | 582.22 | 519.46 | 64/32；未 promotion |
| RL-Repair | `rl-repair` | 547.96 | 582.76 | 434.28 | **582.46** | 515.19 | proven fallback |
| RL-PPO | `rl` | 540.35 | 582.53 | 440.31 | 574.64 | 513.48 | 当前主权重；baseline |

- `test_v7` 永久 locked；top provisional：RL-Switch **488.26**、
  RL-Repair 486.31、RL-Wide160 485.62、RL-Wide 485.50、RL-Pattern
  485.43、RL-PPO 477.26（`--no-consume`，不参与选模）。
- RL-Wide160 把第一层扩到 160：`pattern_breaker` 420.3 → 511.6，
  complex `min_norm` 0.845 → 0.997；dev_param 与 RL-Pattern 基本持平。
- 全候选 v4 baseline 尚未运行；完整报告见
  [docs/research/rl-pattern.md](docs/research/rl-pattern.md) 和
  [results/rl_v4_20261028/](results/rl_v4_20261028/)。

下表是 **Protocol v3** 的 299/20/40 baseline，仅作 legacy 对照；完整 v3
表见 [results/protocol_v3/README.md](results/protocol_v3/README.md)。

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
| Protocol v4 split | **启用**；977 configs + 新风格/Axelrod/complex/policy；test 消费永久锁定 |
| Protocol v4 baseline | **top 6 dev 已运行**；全候选 baseline 未运行；v3 baseline 仅作 legacy |
| **RL-Switch** | **窄 meta RL-Wide160→RL-Repair；test_v7 488.26/487.51；dev 基本不变**；未 promotion |
| **RL-Wide160** | **160/128；dev_complex 454.31、complex min_norm 0.997**；test provisional -0.69；未 promotion |
| **RL-Wide** | **128/128；dev_complex 452.79、complex min_norm 0.973**；test provisional -0.81；未 promotion |
| 输入特征 v2 | 85→97，三端 parity 通过；dev_complex +2～4 但 dev_param/test 回退；**No-Go** |
| 全量对手扫描 | 158 / 977 / 224 对手；薄弱点：extortion/ZD、periodic/Kamikaze、Gap/FSM、PB 尾部、noisy/Joss |
| Extort-hard | 新增 `extort_hard`；strong/gentle 均只小幅改善 extort，PB/complex 回退；**No-Go** |
| RL-Gate | 学习型上层 gate；CS 版 dev 548.62/454.61，但 test 486.98；**No-Go**（实验候选） |
| Periodic-hard | 新增 `periodic_hard`；test +0.77，但 complex/PB 回退；**No-Go** |
| Distill-Switch | RL-Switch 蒸馏单模型；BCE 0.19 但 dev_param 535.46；**No-Go** |
| **RL-Pattern** | **v4 dev 4/5 tier 超 RL-Repair**（pattern_breaker 236.8→420.3）；test provisional -0.88；未 promotion |
| A60 / hard×2 | 候选 / fallback；v3 test 476.54（legacy） |
| E0 league | **PASS**（legacy）；建立 cross-play / Nash / exploitability |
| E1c repair | **Go candidate**（RL-Repair），未 promotion |
| E2 CVaR / DRO | No-Go |
| P3 red pool | red pool Go；league-lite / expansion Gate D **FAIL** |
| **PSRO** | **实现 + 实验候选**；reference meta-policy dev -15.6；未 promotion。red 指标仅历史 |
| **PSRO from scratch** | blue=random/scratch/BR；dev 480.43 / min_norm 0.55；未 promotion。red 指标仅历史 |
| **PSRO v4** | 3 轮 × 30 iter 更强 oracle；dev_param 486.93 / min_norm 0.56；未 promotion。red 指标仅历史 |
| P2 C1 / C2 / C3 | 全部 offline No-Go |
| SelfPlay 改进 | No-Go；保留 21 维部署备选 |
| Axelrod | **已接入 Protocol v4**：224 策略 + 15 组合；可选依赖 |

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
