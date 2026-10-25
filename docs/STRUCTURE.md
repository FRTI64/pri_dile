# 项目结构与接口规范

本项目按“核心模拟 → 参赛策略 → 评估 → 训练”分层，所有代码都位于
`ipd/` 包内，仓库根目录不再保留旧的平铺源码或兼容 shim。

## 1. 目录结构

```text
ipd/
├── __init__.py            包入口（C/D、Strategy、Policy）
├── __main__.py            python3 -m ipd
├── cli.py                 统一命令行：benchmark / blind / suite / matchup / train / export
├── paths.py               仓库路径解析
├── self_test.py           统一策略接口自检
├── core/                  与比赛成绩无关的模拟内核
│   ├── simulator.py       收益、Strategy、play_match、基础对手
│   ├── opponents.py       38 个手写对手 + 字段采样
│   ├── opponents_extended.py  85 类对手银行与场景混合
│   ├── opponents_plus.py  额外参数化/ZD/FSM 对手
│   ├── machine.py         Final 状态机（Machine）
│   ├── harvest.py         Harvest 变体
│   ├── early.py           EarlyProbe 变体
│   ├── bayes.py / bayes_plus.py / ema.py / ema_forgive.py
│   └── features.py        P2 规则特征
├── policies/              参赛策略（提交文件）与统一接口
│   ├── base.py            Policy 协议、PolicyAdapter、ensure_policy
│   ├── rules.py           Final/Resp3/Bayes/Triple/Hybrid/GapStart
│   ├── neural.py          MLP 推理（纯标准库）
│   ├── gru.py             GRU-Net 推理（纯标准库）
│   ├── rl.py              PPO actor-critic 推理
│   ├── adversarial.py     约束剥削包装
│   ├── opponents.py       把自身策略作为对手的 smart 变体
│   ├── tuned.py           从 results/ 读取 tuned 参数，带内置回退值
│   └── registry.py        候选名称、CandidateSpec、make_policy
├── evaluation/            测试/评估代码（不实现策略）
│   ├── distributions.py   original/late/grim30/wide/hard、9 个 suite
│   ├── common.py          run_match、evaluate_field、聚合指标、并行任务
│   ├── benchmark.py       四分布统一基准
│   ├── blind.py           未见赛场盲测
│   ├── suite.py           9 个族权重测试集
│   └── matchup.py         逐对手类型对阵矩阵
├── training/              训练/搜索代码（只通过 registry 和 distributions 取策略/对手）
│   ├── common.py          统一参数、seed、JSON 产物
│   ├── cli.py             python3 -m ipd train <task>
│   ├── neural.py          MLP BC + DAgger
│   ├── gru.py             GRU-Net BC + DAgger
│   ├── rl.py              PPO
│   ├── adversarial.py     约束剥削规则搜索
│   └── search/            multiobj / bayes / joint / wide / triple / p1 / p2...
└── tools/                 部署工具（C++ 权重导出）

tests/                     分层测试
configs/ weights/ results/ cpp/ docs/
```

## 2. 依赖方向

```text
core  ←  policies  ←  evaluation
                  ←  training
tools 只依赖 public API
```

必须遵守：

- `core` 不导入 `policies`、`evaluation`、`training`。
- `policies` 不导入 `training`；tuned 参数由 `policies.tuned` 从
  `results/*.json` 读取，缺失时使用内置值。
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
`ipd/policies/registry.py` 中注册一个 `kind`，所有测试脚本即可通过名字使用。

## 4. 统一评估接口

```bash
python3 -m ipd benchmark --candidates core            # 四分布
python3 -m ipd benchmark --candidates all --fields 20 # 全部候选
python3 -m ipd blind --candidates core --fields 10    # 盲测排名
python3 -m ipd suite --fields 12                      # 9 个测试集
python3 -m ipd matchup --candidates base --reps 16    # 逐类型矩阵
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
tests/test_evaluation.py   run_match、并行任务、三个评估 CLI
tests/test_training.py     训练任务可导入、统一参数、JSON 工具
```

运行：

```bash
python3 -m ipd test
# 等价：python3 -m unittest discover -s tests -v
```
