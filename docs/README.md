# 文档中心

本仓库采用接近 [Diátaxis](https://diataxis.fr/) 的分层：**指南**回答“怎么做”，
**参考**回答“是什么”，**架构**解释“为什么这样设计”，**研究**保存实验记录，
**历史**保存过期快照，**决策**记录不可逆或高成本的工程选择。

## 1. 从这里开始

| 目标 | 入口 |
|---|---|
| 了解项目和当前结论 | [根 README](../README.md) |
| 安装、跑测试、提交改动 | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| 评估当前候选 | [docs/guides/evaluation.md](guides/evaluation.md) |
| 新 split / 审计 / test 规则 | [docs/reference/protocol-v4.md](reference/protocol-v4.md) |
| 训练模型 / 复现实验 | [docs/guides/training.md](guides/training.md) |
| 部署 Python / C++ | [docs/guides/deployment.md](guides/deployment.md) |
| 查 CLI / 候选 / 配置 / 指标 | [docs/reference/](reference/) |
| 看目录与接口设计 | [docs/architecture/overview.md](architecture/overview.md) |
| 看实验与 Go/No-Go | [docs/research/](research/) |
| 交接 / 下一 session | [docs/operations/handover.md](operations/handover.md) |

## 2. 当前状态（2026-10-29）

| 项 | 值 |
|---|---|
| 主权重 | `weights/rl_policy_weights.json`（RL-PPO，从零 + hard） |
| v4 dev 综合最均衡（top 5） | `rl-wide160`（RL-Wide160，160/128，未 promotion） |
| 前最佳 / fallback | `rl-wide` / `rl-pattern` / `rl-repair`；A60 仍为 `rl-hard60` |
| 测试 | **320 tests OK**（本轮全量运行） |
| 注册候选 | 27 个（`python3 -m ipd list-policies`） |
| 默认评估 | **Protocol v4**：`dev_param` / `dev_family` / `dev_complex` / `dev_axelrod` / `dev_ood` 选模；test 永久 provisional |
| 规模 | 977 configs；2048 人 / dev 30 / test 50；14 complex + 224 Axelrod + 15 组合 |
| 分级诊断 | `dev_level0..4` / `dev_complexity0..4`（按思考等级/复杂度） |
| Protocol v4 baseline | **top 5 dev 已运行**；全候选未运行；v3 的 484.32/476.54/471.44 只作 legacy |
| held-out | test 消费永久锁定（provisional only）；v2/v3 已进入 legacy |
| PSRO | 完整 blue-vs-red PSRO + 2 轮 PPO oracle；`psro` / `psro-br` 实验候选，未 promotion |
| PSRO from scratch | blue 仅 random/scratch/BR，现有模型只做 red；`psro-scratch` / `psro-scratch-br`；red_win 0.0，但 dev 480.43；未 promotion |
| 已知未解决 | anti-exploitation；Red-v5/v6 仍 100% 击败 top 3；`psro-scratch` red_win 0.0，但 dev 480.43 / min_norm 0.55；PSRO-Meta dev 掉约 15.6 |

推荐、权重、风险和 promotion 条件以
[operations/handover.md](operations/handover.md) 为准。

## 3. 文档地图

### 架构（explanation）

| 文档 | 内容 |
|---|---|
| [architecture/overview.md](architecture/overview.md) | 分层、模块、接口、目录、测试布局 |
| [architecture/bayes-responsive-flow.md](architecture/bayes-responsive-flow.md) | BayesPlus / Responsive3 C++ 流程图与参数 |

### 指南（how-to）

| 文档 | 内容 |
|---|---|
| [guides/evaluation.md](guides/evaluation.md) | 四分布 / blind / suite / matchup / league / split |
| [guides/training.md](guides/training.md) | BC / PPO / self-play / opponent-model 训练 |
| [guides/deployment.md](guides/deployment.md) | Python 推理、C++ 编译、权重导出、单文件提交 |
| [guides/split-evaluation.md](guides/split-evaluation.md) | `eval-split` 配置与输出布局 |
| [guides/axelrod.md](guides/axelrod.md) | 可选 Axelrod 243/224/cheating 对手库 |

### 参考（information）

| 文档 | 内容 |
|---|---|
| [reference/cli.md](reference/cli.md) | `python3 -m ipd <command>` 命令与常用参数 |
| [reference/candidates.md](reference/candidates.md) | 27 个候选、别名、集合、权重、状态 |
| [reference/configuration.md](reference/configuration.md) | `configs/`、`weights/`、`results/`、路径与配置格式 |
| [reference/metrics.md](reference/metrics.md) | 四分布、blind、suite、league、CVaR、min_norm 的定义与口径 |
| [reference/protocol-v4.md](reference/protocol-v4.md) | Protocol v4 的 train/dev/test 划分、manifest、usage ledger |

### 策略方法

| 文档 | 内容 |
|---|---|
| [policies/rl.md](policies/rl.md) | PPO 的 MDP、训练、导出、风险 |
| [policies/neural.md](policies/neural.md) | MLP 行为克隆 + DAgger |
| [policies/gru.md](policies/gru.md) | GRU-Net 策略 |
| [policies/adversarial.md](policies/adversarial.md) | 约束剥削规则 |
| [policies/selfplay.md](policies/selfplay.md) | 21 维非对称自博弈 |
| [policies/psro.md](policies/psro.md) | PSRO meta-policy / best response 部署 |

### 研究记录

| 文档 | 内容 | 结论 |
|---|---|---|
| [research/README.md](research/README.md) | 研究索引、协议、Gate 总表 | - |
| [research/e0-league.md](research/e0-league.md) | E0 cross-play / Nash / exploitability | PASS |
| [research/e1-robust-finetune.md](research/e1-robust-finetune.md) | E1 domain-randomized / repair | E1c Go candidate |
| [research/e2-risk-ppo.md](research/e2-risk-ppo.md) | E2 CVaR / DRO | No-Go |
| [research/p3-red-league.md](research/p3-red-league.md) | P3 red pool + league-lite | red pool Go；league-lite No-Go |
| [research/p3-expansion.md](research/p3-expansion.md) | P3 multi-red + PSRO-lite | No-Go |
| [research/psro.md](research/psro.md) | 完整 PSRO：blue population + 红队 oracle | 实现完成；实验候选，No-Go for promotion |
| [research/psro-scratch.md](research/psro-scratch.md) | 从零 PSRO：现有模型只做 red | red_win 0.0，但 official 掉约 63；No-Go |
| [research/p2-c1-belief-planner.md](research/p2-c1-belief-planner.md) | C1 Bayesian planner | No-Go |
| [research/p2-c2c3-opponent-model.md](research/p2-c2c3-opponent-model.md) | C2/C3 opponent model + gate | No-Go |
| [research/p2-c2c3-excitation.md](research/p2-c2c3-excitation.md) | C2/C3 excitation / counterfactual | data Go；planner No-Go |
| [research/p2-c3-threeway-moe.md](research/p2-c3-threeway-moe.md) | C3 three-way gate + MoE | No-Go |
| [research/p2-c3-repeated-calib.md](research/p2-c3-repeated-calib.md) | C3 repeated calibration / EV probe | No-Go |
| [research/long-term-plan.md](research/long-term-plan.md) | 长期路线与各 Gate 状态 | roadmap |
| [research/selfplay-plan.md](research/selfplay-plan.md) | self-play 计划 | implemented |
| [research/selfplay-improvement.md](research/selfplay-improvement.md) | self-play 改进 | No-Go |

### 运维与交接

| 文档 | 内容 |
|---|---|
| [operations/handover.md](operations/handover.md) | 当前结论、评估协议、权重、风险、检查清单 |
| [operations/next-session.md](operations/next-session.md) | 下一 session 的起始 prompt 与任务优先级 |

### 历史归档

| 文档 | 内容 |
|---|---|
| [history/README.md](history/README.md) | 历史文档索引与替代链接 |
| [history/final-comparison.md](history/final-comparison.md) | 早期规则 + 神经/RL 主表 |
| [history/suite-results.md](history/suite-results.md) | 17 候选 9-suite |
| [history/opponent-performance.md](history/opponent-performance.md) | 早期 85 类对手表现 |
| [history/matchup-results.md](history/matchup-results.md) | 早期 85 类对阵 |
| [history/advanced-results.md](history/advanced-results.md) | P0/P1/P2 规则搜索过程 |
| [history/legacy-rules.md](history/legacy-rules.md) | 早期规则策略与手工实验 |

### 架构决策记录（ADR）

| 文档 | 决策 |
|---|---|
| [decisions/README.md](decisions/README.md) | ADR 索引与模板 |
| [decisions/0001-frozen-league-evaluation.md](decisions/0001-frozen-league-evaluation.md) | 用冻结 league + held-out 作为主评估 |
| [decisions/0002-no-promotion-without-reproduction.md](decisions/0002-no-promotion-without-reproduction.md) | 未独立复现不得替换主权重 |
| [decisions/0003-anti-exploitation-unsolved.md](decisions/0003-anti-exploitation-unsolved.md) | 不把 red 训练收益当成 anti-exploitation |
| [decisions/0004-archive-no-go-research.md](decisions/0004-archive-no-go-research.md) | No-Go 实验归档，不删除原始记录 |

## 4. 阅读路径

- **新贡献者**：根 README → CONTRIBUTING → architecture/overview → reference/cli
- **评估者**：guides/evaluation → reference/metrics → reference/candidates → handover
- **研究者**：research/README → 最近实验 → long-term-plan → decisions
- **部署者**：guides/deployment → reference/configuration → cpp/README
- **接手者**：operations/handover → operations/next-session → decisions

## 5. 文档约定

- 当前结论只写在根 README、handover、reference 和 architecture；实验文档不因
  新实验改写历史数字。
- 实验文档顶部必须有 Status、协议、日期和 Go/No-Go。
- 历史文档只加归档标记和替代链接，不删除原始表格。
- 任何 promotion 都要更新 checksum、推荐表、handover 和 `CHANGELOG.md`。
- 不以“跑分更高”替代协议说明；10-field 与 20-field、标准 CLI 与 league 不可混表。

## 6. 旧路径迁移

2026-10-27 文档重组后，旧扁平路径已移动。外部书签/脚本请按下表更新：

| 旧路径 | 新路径 |
|---|---|
| `HANDOVER.md` | `docs/operations/handover.md` |
| `docs/HANDOVER_PROMPT.md` | `docs/operations/next-session.md` |
| `docs/STRUCTURE.md` | `docs/architecture/overview.md` |
| `docs/BAYES_RESPONSIVE_FLOW.md` | `docs/architecture/bayes-responsive-flow.md` |
| `docs/AXELROD_OPPONENTS.md` | `docs/guides/axelrod.md` |
| `docs/SPLIT_EVALUATION.md` | `docs/guides/split-evaluation.md` |
| `docs/RL_POLICY.md` | `docs/policies/rl.md` |
| `docs/NEURAL_POLICY.md` | `docs/policies/neural.md` |
| `docs/GRU_POLICY.md` | `docs/policies/gru.md` |
| `docs/ADVERSARIAL_POLICY.md` | `docs/policies/adversarial.md` |
| `docs/SELFPLAY.md` | `docs/policies/selfplay.md` |
| `docs/E0_LEAGUE.md` | `docs/research/e0-league.md` |
| `docs/e0-league-plan.md` | `docs/research/e0-league-plan.md` |
| `docs/e1-robust-finetune.md` | `docs/research/e1-robust-finetune.md` |
| `docs/e2-risk-ppo.md` | `docs/research/e2-risk-ppo.md` |
| `docs/p3-red-league.md` | `docs/research/p3-red-league.md` |
| `docs/p3-expansion-plan.md` | `docs/research/p3-expansion.md` |
| `docs/p2-*.md` | `docs/research/p2-*.md` |
| `docs/ipd-long-term-improvement-plan.md` | `docs/research/long-term-plan.md` |
| `docs/ipd-selfplay-plan.md` | `docs/research/selfplay-plan.md` |
| `docs/ipd-selfplay-improvement-plan.md` | `docs/research/selfplay-improvement.md` |
| `docs/FINAL_COMPARISON.md` | `docs/history/final-comparison.md` |
| `docs/SUITE_RESULTS.md` | `docs/history/suite-results.md` |
| `docs/OPPONENT_PERFORMANCE.md` | `docs/history/opponent-performance.md` |
| `docs/MATCHUP_RESULTS.md` | `docs/history/matchup-results.md` |
| `docs/ADVANCED_RESULTS.md` | `docs/history/advanced-results.md` |
| `docs/IPD_README.md` | `docs/history/legacy-rules.md` |

历史 checksum 文件仍引用迁移前路径，这是有意保留的实验记录，不代表当前文件
位置。
