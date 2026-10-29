# 架构总览：分层、接口与目录

项目按“核心模拟 → 参赛策略 → 评估 → 训练”分层。所有 Python 代码位于
`ipd/` 包内；仓库根目录不再保留旧的平铺源码或兼容 shim。文档总入口见
[docs/README.md](../README.md)。

## 1. 目录结构

```text
ipd/
├── __init__.py            包入口（C/D、Strategy、Policy）
├── __main__.py            python3 -m ipd
├── cli.py                 统一命令行
├── paths.py               仓库路径解析
├── self_test.py           策略接口自检
├── core/                  与成绩无关的模拟内核
│   ├── primitives.py      C/D、payoff、Strategy 基类
│   ├── simulator.py       对局/锦标赛引擎
│   ├── opponents/         对手库（按类型拆分）
│   │   ├── basic.py       AllC/AllD/TFT/Grim/GTFT/...
│   │   ├── reciprocal.py  TFTProbe/Responsive/Resp3/ProbeBot/...
│   │   ├── defectors.py   LateD/EarlyD/Gradual/RandomP/...
│   │   ├── memory.py      MemoryOne/Reactive/RandomFSM/Periodic/...
│   │   ├── zero_determinant.py  ExtortZD/GenerousZD
│   │   ├── spec.py        TypeSpec 元数据
│   │   ├── bank.py        85 类基础对手银行 + 混合采样
│   │   ├── plus.py        额外 67 类细粒度/复杂对手
│   │   ├── axelrod_bank.py 可选 Axelrod 库适配器
│   │   └── legacy.py      旧固定权重赛场辅助函数
│   ├── machine.py         Final 状态机（默认参数不是 tuned）
│   ├── harvest.py         Harvest 变体
│   ├── early.py           EarlyProbe 变体
│   ├── bayes.py / bayes_plus.py / ema.py / ema_forgive.py
│   ├── features.py        85 维特征
│   └── opponents_extended.py / opponents_plus.py  兼容转发入口
├── policies/              参赛策略与统一接口
│   ├── base.py            Policy 协议、PolicyAdapter、ensure_policy
│   ├── candidates/        参赛/评估候选（一策略一文件、互不 import）
│   │   ├── __init__.py    kind -> 模块映射 + build_candidate
│   │   ├── final_base.py / final_tuned.py
│   │   ├── resp3_base.py / resp3_tuned.py
│   │   ├── bayes_base.py / bayes_tuned.py
│   │   ├── harvest45.py / harvest_tuned.py
│   │   ├── hybrid.py / triple_majority.py / gap_start.py
│   │   ├── neural.py / gru.py / rl.py
│   │   ├── rl_hard60.py / rl_repair.py / selfplay.py
│   │   ├── psro.py / psro_br.py
│   │   ├── psro_scratch.py / psro_scratch_br.py
│   │   └── adv_mlp.py / adv_gru.py / adv_rl.py
│   ├── rules/             训练/搜索用的共享规则构件
│   ├── opponents/         把自身策略作为对手的银行
│   ├── neural.py          MLP 推理（纯标准库）
│   ├── gru.py             GRU 推理（纯标准库）
│   ├── rl.py              PPO actor-critic 推理
│   ├── selfplay_gru.py    21 维 SelfPlay-GRU 推理
│   ├── selfplay_gru_obs.py 21 维观测（纯标准库）
│   ├── psro.py            PSRO meta-policy（每场采样一个成员）
│   ├── adversarial.py     约束剥削包装
│   ├── moe.py             C3 safe/risky/unknown 三路 MoE
│   ├── belief_planner.py  C1 Bayesian belief planner
│   ├── tuned.py           从 results/ 读取 tuned 参数
│   └── registry.py        候选名称、CandidateSpec、make_policy
├── evaluation/            评估代码（不实现策略）
│   ├── distributions.py   original/late/grim30/wide、9 suites、Axelrod
│   ├── common.py          对局、field 评估、逐对手统计、并行任务
│   ├── benchmark.py       四分布统一基准
│   ├── blind.py           未见赛场盲测
│   ├── split.py           按候选/对手拆分的评估驱动
│   ├── split_config.py    每个候选独立参数配置
│   ├── protocol.py        Protocol v4 train/dev/test catalog + audit
│   ├── protocol_cli.py    `python -m ipd protocol` 入口
│   ├── suite.py           9 个家族加权测试集
│   ├── matchup.py         逐对手类型矩阵 + 9 测试集加权
│   ├── opponent_suite.py  冻结 validation / heldout_v2 / heldout_v3
│   ├── metagame.py        cross-play、Nash、Elo、exploitability
│   ├── league.py          E0 联赛 harness
│   ├── planner_eval.py    C1 planner 离线评估
│   ├── moe_eval.py        C3 gate + MoE 离线评估
│   ├── policy_pool.py     P3 frozen blue/red policy pool
│   ├── robust.py          E1 robust / repair 训练分布
│   └── hard2.py           hard2 训练分布
├── training/              训练/搜索代码
│   ├── cli.py             python3 -m ipd train <task>
│   ├── common.py          统一参数、seed、JSON 产物
│   ├── neural.py          MLP BC + DAgger
│   ├── gru.py             GRU BC + DAgger
│   ├── rl.py              PPO
│   ├── adversarial.py     约束剥削规则搜索
│   ├── risk.py            E2 CVaR / Dirichlet / DRO
│   ├── opponent_model.py  C2/C3 对手模型 / gate
│   ├── league.py          P3 maximin / opponent manifest
│   ├── psro.py            完整 PSRO：元博弈 / blue-red maximin / oracle
│   ├── selfplay/          21 维非对称自博弈训练器
│   └── search/            multiobj / bayes / joint / wide / triple / p1 / p2
└── tools/                 C++ 权重导出、matchup 透视

tests/                     分层测试（core/policies/evaluation/training）
configs/                   对抗/评估/self-play 配置
weights/                   MLP/GRU/RL/SelfPlay 权重与 checkpoint
results/                   基准、盲测、实验报告、训练曲线、搜索记录
cpp/                       C++17 推理实现和生成的权重头文件
docs/                      方法、实验报告、结构规范、文档导航
```

## 1.1 候选策略独立文件

`ipd/policies/candidates/` 中每个参赛/评估候选一个文件。约束：

- 候选文件只从 `ipd.core` 或对应模型实现（`ipd.policies.neural/gru/rl`、
  `ipd.policies.selfplay_gru`）取依赖；
- 候选文件之间不互相 import；
- tuned 参数直接写在候选文件里，便于逐文件翻译；
- `ipd/policies/rules/` 供训练/搜索复用，不作为候选注册入口；
- `tests/test_candidates.py` 检查“无候选间 import”和与历史实现的动作一致性。

注册新候选：

1. 在 `candidates/` 新建文件，实现 `KIND`、`DISPLAY` 和 `build(...)`；
2. 在 `candidates/__init__.py` 的 `MODULES` 中登记；
3. 在 `policies/registry.py` 中补充 `DISPLAY`、`ALIASES`、集合别名；
4. 运行 `python3 -m ipd self-test --candidates <name>` 和测试。

候选文件统一入口：

```python
def build(*, seed=20261024, weights=None, config=None, mode="sample",
          name=None, tuned=None, **kwargs):
    policy = MyPolicy(...)
    policy.name = name or "MyPolicy"
    return policy
```

`registry.make_policy` 会调用 `build`，再统一套上 `PolicyAdapter`。

## 1.2 候选集合

| 名称 | 数量 | 内容 |
|---|---:|---|
| `core` | 4 | Final-tuned、Resp3-tuned、Bayes-tuned、Triple-majority |
| `base` / `legacy` | 7 | Final/Resp3/Bayes（base+tuned）+ Hybrid |
| `classic` | 10 | base + Harvest45 + Harvest-tuned + GapStart |
| `suite` | 9 | base + Harvest45 + Triple-majority |
| `learned` | 13 | Neural、GRU、RL、Hard60、Repair、PSRO-Meta、PSRO-BR、PSRO-Scratch、PSRO-Scratch-BR、Adv-MLP、Adv-GRU、Adv-RL、SelfPlay |
| `all` | 24 | 全部注册候选 |

`python3 -m ipd list-policies` 是最终口径；新增候选后同步更新本表。

## 2. 依赖方向

```text
core  ←  policies  ←  evaluation
                  ←  training
tools 只依赖 public API
```

必须遵守：

- `core` 不导入 `policies`、`evaluation`、`training`；
- `policies` 不导入 `training`；候选文件内嵌 tuned 参数，
  `policies.tuned` 仍供训练/搜索读取 `results/*.json`；
- `evaluation` 只通过 `policies.registry` 构造候选，通过
  `policies.base.ensure_policy` 调用策略；
- `training` 不定义参赛策略；它复用 `policies.rules` 与
  `evaluation.distributions`。

## 3. 策略接口

规范接口是 0-based 的 `act`：

```python
from ipd import C, D

class MyPolicy:
    name = "MyPolicy"

    def reset(self) -> None:
        ...

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        # t = 0..99；mine/theirs 是当前决策之前的动作
        return C if t < 99 else D
```

历史策略使用 1-based `move(day, mine, theirs)`。评估代码统一经过：

```python
from ipd.policies.base import ensure_policy

policy = ensure_policy(policy)
policy.reset()
action = policy.act(t, mine, theirs)
```

适配器会拒绝非 `0/1` 的动作，并自动适配 `move`。GRU / SelfPlay-GRU 是有状态
策略：每场必须 `reset()`，同一时间步只调用一次 `act()`。

## 4. 评估接口

```bash
# Protocol v4（当前主口径）
python3 -m ipd protocol audit
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_ood \
  --fields 20 --baseline RL-PPO

# 四分布（legacy；默认 8 fields；正式比较建议 20）
python3 -m ipd benchmark --candidates core --fields 20

# 盲测（默认 10 fields；当前正式比较建议 20）
python3 -m ipd blind --candidates core --fields 20

# 9 个测试集
python3 -m ipd suite --fields 12

# 逐对手矩阵 + 9 测试集加权
python3 -m ipd matchup --candidates all --opponent-set all --reps 16

# 冻结 league（E0/E1c 口径）
python3 -m ipd league --config configs/eval/e0_league.json

# 拆分评估
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json

# 离线错误分析 / 复现实验
python3 -m ipd planner-eval
python3 -m ipd moe-eval
```

`--arch` 是 benchmark/blind 的旧兼容捷径；新命令统一用 `--candidates`。
评测协议和数字比较规则见 [docs/README.md](../README.md#6-评估口径速查)。

新增评估脚本时，优先调用 `evaluation.common.eval_task` 或
`evaluate_field`，不要自行实现匹配循环。

## 5. 训练接口

```bash
python3 -m ipd train --list
python3 -m ipd train neural --device auto
python3 -m ipd train gru --device auto
python3 -m ipd train rl --device auto --iterations 100
python3 -m ipd train psro --config configs/psro/psro_20261027.json
python3 -m ipd train psro --config configs/psro/psro_scratch_20261027.json
python3 -m ipd train adversarial --arch mlp
python3 -m ipd train selfplay --config configs/selfplay/full.json --device auto
python3 -m ipd train selfplay-bc --device auto --episodes 4000 --epochs 10
python3 -m ipd train opponent-model --data-mode mixed
python3 -m ipd train multiobj --workers 24
```

训练文件也可以直接执行：

```bash
python3 ipd/training/neural.py --device auto
python3 ipd/training/rl.py --device auto --iterations 100
python3 ipd/training/search/multiobj.py --workers 24
```

所有训练任务：

- 接受 `--seed`、`--workers`（适用时）、`--n`（适用时）、`--out`；
- 学习类任务额外接受 `--device`、`--history-out`；
- 暴露 `main(argv=None)`，由 `ipd.training.cli` 调度；
- 产物写 JSON，路径由 `ipd.paths` 解析。

## 6. C++ 文件

```text
cpp/
├── policies/                 每个策略/模型一个头文件
│   ├── final_policy.hpp      FinalParams + FinalPolicy
│   ├── responsive3_policy.hpp Responsive3Params + Responsive3
│   ├── bayes_plus_policy.hpp BayesPlusParams + BayesPlus
│   ├── neural_policy.hpp     NeuralParams + NeuralPolicy
│   ├── gru_policy.hpp        GRUParams + GRUPolicy
│   └── adversarial_rl_policy.hpp AdvRLParams + AdversarialRLPolicy
├── opponents/                每个对手一个头文件
│   ├── all_c.hpp / all_d.hpp / tft.hpp / grim.hpp / late_d.hpp
│   ├── random_p.hpp
│   ├── opponent.hpp          统一 Opponent 接口 + play_against
│   └── demo.hpp              演示用对手组合
├── ipd_features.hpp          MLP/GRU 共用的 85 维特征
├── neural_policy_weights.hpp / gru_policy_weights.hpp
├── rl_policy_weights.hpp / rl_policy_weights_bc.hpp
└── *.cpp                     保持原有编译入口的演示程序
```

所有策略构造函数都接受参数结构体：

```cpp
ipd::FinalPolicy final_policy(20261024u, ipd::FinalParams::tuned());
ipd::Responsive3 resp(20261024u, ipd::Responsive3Params::tuned());
ipd::BayesPlus bayes(ipd::BayesPlusParams::tuned());
ipd::GRUPolicy gru(20261024u, ipd::GRUParams{true, true, true});
ipd::AdversarialRLPolicy adv(20261024u, true,
    ipd::AdvRLParams{0.2, 12, 2, 40, true});
```

顶层的 `cpp/neural_policy.hpp` / `cpp/gru_policy.hpp` 保留为兼容入口。
**没有 C++ 版 Triple-majority**；Adv-MLP / Adv-GRU 只有 Python 版，Adv-RL
有 C++ 版。完整编译示例见 [`cpp/README.md`](../../cpp/README.md)。

## 7. 测试布局

```text
tests/test_layout.py             目录、路径、必需产物
tests/test_core.py               模拟器、对手库、分布
tests/test_policies.py           PolicyAdapter、registry、边界、RL 导出一致性
tests/test_candidates.py         候选文件独立性、参数/历史等价性
tests/test_evaluation.py         run_match、并行任务、评估 CLI
tests/test_metagame.py           Nash averaging、Elo、exploitability
tests/test_league.py             E0 cross-play/suite/red harness
tests/test_league_meta.py        P3 maximin / opponent manifest
tests/test_robust.py             E1 domain-randomized 训练分布
tests/test_risk.py               E2 CVaR episode weights / Dirichlet
tests/test_policy_pool.py        P3 frozen blue/red policy pool
tests/test_opponent_model.py     C2/C3 对手模型 / gate
tests/test_belief_planner.py     C1 belief planner
tests/test_moe.py                C3 three-way gate + MoE
tests/test_selfplay.py           21 维观测、GRU、PPO、BC、resume
tests/test_psro.py               PSRO 元博弈、population、oracle 命令、端到端
tests/test_psro_policy.py        PSRO meta-policy 采样/重置
tests/test_axelrod_opponents.py  Axelrod 243/224/1、适配器、CLI
tests/test_training.py           训练任务可导入、统一参数、JSON 工具
tests/test_protocol.py           Protocol v4 catalog / audit / manifest / usage
```

运行：

```bash
python3 -m ipd test
# 等价：python3 -m unittest discover -s tests -t . -v
```

最近完整测试：**312 tests OK**。新模块仍应先写测试，再训练/评估。
