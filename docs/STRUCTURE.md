# 项目结构与接口规范

本项目按“核心模拟 → 参赛策略 → 评估 → 训练”分层，所有代码都位于
`ipd/` 包内，仓库根目录不再保留旧的平铺源码或兼容 shim。

## 1. 目录结构

```text
ipd/
├── __init__.py            包入口（C/D、Strategy、Policy）
├── __main__.py            python3 -m ipd
├── cli.py                 统一命令行：benchmark / blind / eval-split / suite / matchup / league / train / export
├── paths.py               仓库路径解析
├── self_test.py           统一策略接口自检
├── core/                  与比赛成绩无关的模拟内核
│   ├── primitives.py      C/D、payoff、Strategy 基类
│   ├── simulator.py       对局/锦标赛引擎，并兼容重导出基础对手
│   ├── opponents/         对手库（按类型拆分）
│   │   ├── basic.py       AllC/AllD/TFT/Grim/GTFT/... 基础对手
│   │   ├── reciprocal.py  TFTProbe/Responsive/Resp3/ProbeBot/...
│   │   ├── defectors.py   LateD/EarlyD/Gradual/RandomP/...
│   │   ├── memory.py      MemoryOne/Reactive/RandomFSM/Periodic/...
│   │   ├── zero_determinant.py  ExtortZD/GenerousZD
│   │   ├── spec.py        TypeSpec 元数据
│   │   ├── bank.py        85 类对手银行与混合采样
│   │   ├── plus.py        额外参数化对手（PLUS）
│   │   └── legacy.py      旧固定权重赛场辅助函数
│   ├── machine.py         Final 状态机（Machine）
│   ├── harvest.py         Harvest 变体
│   ├── early.py           EarlyProbe 变体
│   ├── bayes.py / bayes_plus.py / ema.py / ema_forgive.py
│   ├── features.py        P2 规则特征
│   └── opponents_extended.py / opponents_plus.py  兼容入口（转发到 opponents/）
├── policies/              参赛策略（提交文件）与统一接口
│   ├── base.py            Policy 协议、PolicyAdapter、ensure_policy
│   ├── candidates/        参赛/评估候选，一策略一文件、互不 import
│   │   ├── __init__.py    kind -> 模块 映射 + build_candidate
│   │   ├── final_base.py / final_tuned.py
│   │   ├── resp3_base.py / resp3_tuned.py
│   │   ├── bayes_base.py / bayes_tuned.py
│   │   ├── harvest45.py / harvest_tuned.py
│   │   ├── hybrid.py / triple_majority.py / gap_start.py
│   │   ├── neural.py / gru.py / rl.py
│   │   └── adv_mlp.py / adv_gru.py / adv_rl.py
│   ├── rules/             训练/搜索用的共享规则构件（不再作为候选入口）
│   │   ├── params.py      各策略的参数 dataclass
│   │   ├── final.py       Final / ParamPolicy
│   │   ├── harvest.py     Harvest / ParamHarvest
│   │   ├── responsive3.py ParamResp
│   │   ├── bayes.py       ParamBayes / ParamBayesPrior
│   │   ├── hybrid.py      Final+Resp3
│   │   ├── triple.py      Triple-majority
│   │   ├── gap_start.py   GapStart
│   │   └── early_probe.py EarlyProbe
│   ├── opponents/         把自身策略作为对手（按用途拆分）
│   │   ├── smart.py       SMART 银行
│   │   ├── tuned_smart.py TUNED_SMART 银行
│   │   └── bank.py        PLUS / ALL_SPECS
│   ├── neural.py          MLP 推理（纯标准库）
│   ├── gru.py             GRU-Net 推理（纯标准库）
│   ├── rl.py              PPO actor-critic 推理
│   ├── adversarial.py     约束剥削包装
│   ├── tuned.py           从 results/ 读取 tuned 参数，带内置回退值
│   └── registry.py        候选名称、CandidateSpec、make_policy
├── evaluation/            测试/评估代码（不实现策略）
│   ├── distributions.py   original/late/grim30/wide/hard、9 个 suite
│   ├── common.py          run_match、evaluate_field、逐对手统计、并行任务
│   ├── benchmark.py       四分布统一基准
│   ├── blind.py           未见赛场盲测
│   ├── split.py           按候选/对手拆分的评估驱动
│   ├── split_config.py    每个候选独立参数配置
│   ├── suite.py           9 个族权重测试集
│   ├── matchup.py         逐对手类型矩阵 + 9 个测试集家族加权
│   ├── opponent_suite.py  冻结评估 suite（含 heldout_v2/v3 新家族/参数）
│   ├── metagame.py        cross-play 矩阵、Nash averaging、Elo、exploitability
│   ├── league.py          E0 联赛 harness（cross-play / suite / red）
│   ├── policy_pool.py     P3 frozen blue/red policy pool
│   └── robust.py          E1 robust / repair domain-randomized 训练对手分布
├── training/              训练/搜索代码（只通过 registry 和 distributions 取策略/对手）
│   ├── common.py          统一参数、seed、JSON 产物
│   ├── cli.py             python3 -m ipd train <task>
│   ├── neural.py          MLP BC + DAgger
│   ├── gru.py             GRU-Net BC + DAgger
│   ├── rl.py              PPO
│   ├── adversarial.py     约束剥削规则搜索
│   └── search/            multiobj / bayes / joint / wide / triple / p1 / p2...
└── tools/                 部署/结果工具（C++ 权重导出、matchup 透视）

tests/                     分层测试
configs/eval/              按候选拆分参数的评估配置
configs/ weights/ results/ cpp/ docs/
```

## 1.1 按策略/模型/对手拆分

### 候选策略独立文件

`ipd/policies/candidates/` 中每个参赛/评估候选一个文件：

| 候选 | 文件 |
|---|---|
| Final-base / Final-tuned | `final_base.py` / `final_tuned.py` |
| Resp3-base / Resp3-tuned | `resp3_base.py` / `resp3_tuned.py` |
| Bayes-base / Bayes-tuned | `bayes_base.py` / `bayes_tuned.py` |
| Harvest45 / Harvest-tuned | `harvest45.py` / `harvest_tuned.py` |
| Final+Resp3 | `hybrid.py` |
| Triple-majority | `triple_majority.py` |
| GapStart | `gap_start.py` |
| Neural / GRU / RL-PPO | `neural.py` / `gru.py` / `rl.py` |
| Adv-MLP / Adv-GRU / Adv-RL | `adv_mlp.py` / `adv_gru.py` / `adv_rl.py` |

约束：

- 每个文件只从 `ipd.core` 或对应的模型实现（`ipd.policies.neural/gru/rl`）取依赖；
- 候选文件之间不互相 import；
- tuned 参数直接写在该候选文件里，便于逐文件翻译；
- `ipd/policies/rules/` 仍供训练/搜索复用，但不再作为候选注册入口；
- `tests/test_candidates.py` 会检查“无候选间 import”和与历史实现的动作一致性。

### 评估结果

`python3 -m ipd eval-split` 读取 `configs/eval/*.json`，为每个候选单独设置
`fields`/`n`/`seed`/`workers`，并把结果拆成：

```text
results/<run>/
  config.json
  summary.json
  by_candidate/<candidate>/benchmark.json
  by_candidate/<candidate>/blind.json
  by_candidate/<candidate>/opponents.json
  by_opponent/<opponent>.json
```

- `by_candidate/<candidate>/` 是该策略/模型的独立文件；
- `by_opponent/<opponent>.json` 汇总所有候选对某一对手类型的成绩；
- `benchmark` 与 `blind` 可以给不同候选不同参数；盲测中参数完全相同的候选
  仍会复用同一赛场的对手循环赛。

```bash
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json
python3 -m ipd eval-split --config configs/eval/retrain_20261025_fast.json
```

### C++ 文件

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
└── *.cpp                     保持原有编译入口的演示程序
```

所有策略构造函数都接受参数结构体，例如：

```cpp
ipd::FinalPolicy final_policy(20261024u, ipd::FinalParams::tuned());
ipd::Responsive3 resp(20261024u, ipd::Responsive3Params::tuned());
ipd::BayesPlus bayes(ipd::BayesPlusParams::tuned());
ipd::GRUPolicy gru(20261024u, ipd::GRUParams{true, true, true});
```

顶层的 `cpp/neural_policy.hpp`、`cpp/gru_policy.hpp` 等仍保留为兼容入口。
更完整的编译/参数示例见 [`cpp/README.md`](../cpp/README.md)。

---

## 2. 依赖方向

```text
core  ←  policies  ←  evaluation
                  ←  training
tools 只依赖 public API
```

必须遵守：

- `core` 不导入 `policies`、`evaluation`、`training`。
- `policies` 不导入 `training`；候选文件把 tuned 参数直接写在自己文件里，
  `policies.tuned` 仍供训练/搜索读取 `results/*.json`。
- `evaluation` 只通过 `policies.registry` 构造候选，通过
  `policies.base.ensure_policy` 调用策略。
- `training` 不定义参赛策略；它复用 `policies.rules` 与
  `evaluation.distributions`。

## 3. 参赛策略接口

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

历史策略使用 1-based `move(day, mine, theirs)`。评估代码不检查
`hasattr`，而是统一经过：

```python
from ipd.policies.base import ensure_policy

policy = ensure_policy(policy)
policy.reset()
action = policy.act(t, mine, theirs)
```

适配器会拒绝非 `0/1` 的动作，并自动适配 `move`。新增策略只需在
`ipd/policies/candidates/` 新建一个独立文件（实现 `build(...)`），在
`candidates/__init__.py` 的 `MODULES` 中登记，再在 `registry.py` 中补充
`DISPLAY` 和集合别名即可；候选文件之间不要互相 import。

候选文件的统一入口：

```python
def build(*, seed=20261024, weights=None, config=None, mode="sample",
          name=None, tuned=None, **kwargs):
    policy = MyPolicy(...)
    policy.name = name or "MyPolicy"
    return policy
```

`registry.make_policy` 会调用这个函数，再统一套上 `PolicyAdapter`。

## 4. 统一评估接口

```bash
python3 -m ipd benchmark --candidates core            # 四分布
python3 -m ipd benchmark --candidates all --fields 20 # 全部候选
python3 -m ipd blind --candidates core --fields 10    # 盲测排名
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json
python3 -m ipd suite --fields 12                      # 9 个测试集
python3 -m ipd matchup --candidates all --opponent-set all --reps 16
# 逐类型矩阵 + 9 个测试集加权，输出 CSV/JSON
# 结果文件：matchup_all.csv / matchup_all.json /
#           matchup_all_suite_weighted.csv
python3 ipd/tools/pivot_matchup.py
# matchup_all_wide.csv：候选 × 对手，单元格为 我方分数|对手分数
# matchup_all_scores.csv：长表 score=我方分数|对手分数
python3 -m ipd self-test --candidates all             # 快速接口自检
```

候选集合：

| 名称 | 内容 |
|---|---|
| `core` | Final-tuned、Resp3-tuned、Bayes-tuned、Triple-majority |
| `base` / `legacy` | 7 个规则基线（Final/Resp3/Bayes/base+tuned、Hybrid） |
| `classic` | base + Harvest45 + Harvest-tuned + GapStart |
| `suite` | 9 个测试集使用的 8 个规则候选（含 Harvest45） |
| `learned` | Neural、GRU、RL、Adv-MLP、Adv-GRU、Adv-RL |
| `all` | 全部注册候选 |

新增测试脚本时，只调用 `evaluation.common.eval_task` 或
`evaluate_field`，不要自行实现匹配循环。

## 5. 统一训练接口

```bash
python3 -m ipd train --list
python3 -m ipd train neural --device auto --epochs 40
python3 -m ipd train gru --device auto
python3 -m ipd train rl --device auto --iterations 100
python3 -m ipd train adversarial --arch mlp
python3 -m ipd train multiobj --workers 24
python3 -m ipd train wide --workers 24
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
- 暴露 `main(argv=None)`，因此可以被 `ipd.training.cli` 调度；
- 产物写 JSON，路径仍由 `ipd.paths` 解析。

## 6. 测试布局

```text
tests/test_layout.py       目录、路径、必需产物
tests/test_core.py         模拟器、对手库、分布
tests/test_policies.py     PolicyAdapter、registry、边界、RL 导出一致性
tests/test_candidates.py   候选文件独立性、参数与历史实现等价性
tests/test_evaluation.py   run_match、并行任务、三个评估 CLI
tests/test_metagame.py     Nash averaging、Elo、exploitability
tests/test_league.py       E0 cross-play/suite/red harness、heldout_v2/v3
tests/test_robust.py       E1 domain-randomized 训练分布
tests/test_risk.py         E2 CVaR episode weights / Dirichlet mixture
tests/test_policy_pool.py  P3 frozen blue/red policy pool
tests/test_training.py     训练任务可导入、统一参数、JSON 工具
```

运行：

```bash
python3 -m ipd test
# 等价：python3 -m unittest discover -s tests -v
```
