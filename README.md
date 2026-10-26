# 100 天囚徒困境策略项目

本项目实现并比较了多类 100 天迭代囚徒困境（IPD）策略：规则策略、贝叶斯策略、
多数投票融合策略，可导出到 C++ 的 MLP / GRU-Net 神经策略、约束对抗剥削版本，
以及直接用比赛收益训练的完整 PPO 强化学习策略。最终从零训练的 RL 是最高盲测，
加上约束剥削规则（Adv-RL）后是最高四分布。

当前推荐版本和完整实验结论见 **[HANDOVER.md](HANDOVER.md)**。

## 目录结构

```text
.
├── ipd/                  规范 Python 包
│   ├── core/             模拟内核、对手库、规则状态机、特征
│   ├── policies/         参赛策略（每候选独立文件）+ 统一接口 + 注册表
│   ├── evaluation/       四分布/盲测/拆分评估/9 测试集/逐类型对阵
│   ├── training/         MLP/GRU/RL/对抗训练与规则参数搜索
│   ├── tools/            C++ 权重导出
│   └── cli.py            `python3 -m ipd ...` 统一入口
├── cpp/                  C++17 推理实现、共用特征头文件和生成的权重头文件
├── configs/              对抗策略配置（需与 weights/ 中的权重配对使用）
│   └── eval/             按候选拆分参数的评估配置
├── weights/              训练好的 MLP / GRU / RL 权重（JSON）
├── results/              基准测试、盲测、训练曲线和搜索记录（JSON / CSV）
├── docs/                 方法说明、结果表、流程图、结构规范
├── tests/                分层测试（core/policies/evaluation/training）
├── HANDOVER.md           交接文档：推荐方案、参数、风险、检查清单
└── requirements-train.txt
```

项目分层和接口规范见 **[docs/STRUCTURE.md](docs/STRUCTURE.md)**。核心规则：

- `core` 不依赖策略、评估或训练；
- `policies` 是唯一放参赛策略的目录，统一实现 `reset()` + `act(t, mine, theirs)`；
- `evaluation` 只通过候选注册表取策略，不再直接 import 具体策略类；
- `training` 不定义参赛策略，训练入口统一为 `python3 -m ipd train <task>`。

请在仓库根目录运行；路径由 `ipd.paths` 统一解析。如需从其他目录运行，先执行 `pip install -e .`：

```bash
python3 -m ipd benchmark --candidates core --fields 8
python3 -m ipd blind --candidates core --fields 10 --seed 20261025
```

## 统一接口

参赛策略规范接口（0-based）：

```python
from ipd import C, D

class MyPolicy:
    name = "MyPolicy"

    def reset(self) -> None:
        ...

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        return C if t < 99 else D
```

旧式 1-based `move(day, mine, theirs)` 会被 `ipd.policies.base.ensure_policy`
自动适配；非法动作会被拒绝。新策略注册到
`ipd/policies/registry.py` 后，所有评估脚本都可以通过名字使用：

```bash
python3 -m ipd benchmark --candidates final-tuned,my-policy
```

## 快速开始

### 测试

```bash
python3 -m ipd test
python3 -m ipd self-test --candidates all
# 等价命令
python3 -m unittest discover -s tests -v
```

### 规则策略与评估

```bash
python3 -m ipd list-policies
python3 -m ipd benchmark --candidates core --fields 20 --workers 24
python3 -m ipd blind --candidates core --fields 10 --workers 12
python3 -m ipd suite --fields 12 --workers 24
python3 -m ipd matchup --candidates base --reps 16
python3 -m ipd matchup --candidates all --opponent-set all --reps 16
# 全策略 × 158 对手 + 9 个测试集加权
python3 -m ipd matchup \
  --spec-config configs/eval/sweep_all_weights_20261026.json \
  --opponent-set all --reps 16 --workers 24 \
  --out results/sweep_20261026/matchup.csv
# 66 个策略/模型/权重 × 158 对手 × 10 家族；结果见 results/sweep_20261026/
```

### 神经策略推理（不需要 PyTorch）

```bash
python3 -m ipd self-test --candidates neural \
  --weights weights/neural_policy_weights.json
python3 -m ipd self-test --candidates gru \
  --weights weights/gru_policy_weights.json
python3 -m ipd self-test --candidates adv-mlp \
  --config configs/adversarial_policy_config.json
```

### 训练（需要 PyTorch）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt

python3 -m ipd train neural --device auto
python3 -m ipd train gru --device auto
python3 -m ipd train adversarial --arch mlp --seed 20261026
python3 -m ipd train rl --device auto --iterations 100 \
  --dist-weights 1,1,1,2 --bc-kl-coef 0.1
# 非对称自博弈（蓝/红 GRU，三阶段课程）
python3 -m ipd train selfplay --config configs/selfplay/full.json \
  --device auto --final-test
# 可选：将现有 RL-PPO 蒸馏为 21 维 GRU，再做 self-play 微调
python3 -m ipd train selfplay-bc --device auto --episodes 4000 --epochs 10
```

训练文件也支持直接执行，例如：

```bash
python3 ipd/training/neural.py --device auto
python3 ipd/training/search/multiobj.py --workers 24
```

### 学习策略的四分布与盲测

```bash
python3 -m ipd benchmark --candidates core,neural --fields 20 --workers 20
python3 -m ipd benchmark --candidates core,gru --fields 20 --workers 20
python3 -m ipd benchmark --candidates core,adv-mlp \
  --config configs/adversarial_policy_config_aggressive.json --fields 20 --workers 20
python3 -m ipd benchmark --candidates core,rl --fields 20 --workers 20
python3 -m ipd benchmark --candidates core,adv-rl \
  --config configs/adversarial_rl_policy_config_aggressive.json --fields 20 --workers 20

python3 -m ipd blind --candidates core,neural --fields 10 --workers 12
python3 -m ipd blind --candidates core,adv-mlp \
  --config configs/adversarial_policy_config_aggressive.json --fields 10 --workers 12
python3 -m ipd blind --candidates core,rl --fields 10 --workers 12
python3 -m ipd blind --candidates core,adv-rl \
  --config configs/adversarial_rl_policy_config_aggressive.json --fields 10 --workers 12
python3 -m ipd benchmark --candidates selfplay \
  --weights weights/selfplay_gru_policy.json --fields 20 --workers 20
python3 -m ipd blind --candidates selfplay \
  --weights weights/selfplay_gru_policy.json --fields 10 --workers 12
```

按策略/模型拆分参数并输出独立结果文件：

```bash
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json
python3 -m ipd eval-split --config configs/eval/retrain_20261025_fast.json
```

每次运行会生成 `results/split_<name>/`，其中每个候选有独立的
`benchmark.json`、`blind.json`、`opponents.json`，另有一个
`by_opponent/<opponent>.json` 汇总所有候选对单个对手类型的成绩。配置格式见
[docs/SPLIT_EVALUATION.md](docs/SPLIT_EVALUATION.md)。

旧的平铺入口已移除；所有评估统一使用上面的
`python3 -m ipd benchmark|blind|eval-split|suite|matchup` 命令。拆分评估的
配置格式和输出布局见 [docs/SPLIT_EVALUATION.md](docs/SPLIT_EVALUATION.md)。

### C++ 推理

```bash
# MLP / GRU（权重头文件已生成在 cpp/）
g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo
g++ -std=c++17 -O2 -DGRU_POLICY_DEMO cpp/gru_policy.cpp -o gru_demo
./neural_demo
./gru_demo

# BayesPlus / Responsive3
g++ -std=c++17 -O2 -DRESP_BAYES_DEMO cpp/bayesplus_responsive3.cpp -o demo
./demo

# 完整 RL 策略（权重头文件已生成）
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
  -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o rl_demo
./rl_demo

# Adv-RL（RL + 两条约束剥削规则；最终 RL base）
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
  -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_demo
./adv_rl_demo

# Adv-RL（BC 初始化 base，对应 554.29 / 562.91）
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_BC_WEIGHTS \
  -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_bc_demo
./adv_rl_bc_demo
```

重新训练权重后，刷新 C++ 头文件：

```bash
python3 -m ipd export --arch mlp   # weights/neural_policy_weights.json -> cpp/neural_policy_weights.hpp
python3 -m ipd export --arch gru   # weights/gru_policy_weights.json    -> cpp/gru_policy_weights.hpp
python3 -m ipd export --weights weights/rl_policy_weights.json \
  --out cpp/rl_policy_weights.hpp  # 最终 RL -> cpp/rl_policy_weights.hpp
python3 -m ipd export --weights weights/rl_policy_weights_bc.json \
  --out cpp/rl_policy_weights_bc.hpp  # Adv-RL base
```

策略实现已按模型拆到 `cpp/policies/`，对手拆到 `cpp/opponents/`。每个策略的
构造函数都接受参数结构体，例如 `ipd::FinalParams::tuned()`、
`ipd::Responsive3Params::tuned()`、`ipd::BayesPlusParams::tuned()`；顶层的
`cpp/neural_policy.hpp` / `cpp/gru_policy.hpp` 仍保留为兼容入口。详见
[docs/STRUCTURE.md](docs/STRUCTURE.md#11-按策略模型对手拆分)。

## 结果摘要

统一四分布测试（20 个赛场 × 299 人）：

| 策略 | 四分布均值 | 盲测均分 / 平均排名 |
|---|---:|---:|
| **RL-PPO（从零 + hard，最终）** | **553.33** | **563.23 / 8.00** |
| **RL-PPO-Hard60（hard×2 fine-tune，候选）** | **554.36** | **564.17 / 5.75（20 fields）** |
| **Adv-RL 激进（RL + 约束剥削）** | **554.29** | **562.91 / 8.00** |
| **SelfPlay-GRU（21 维，BC 蒸馏）** | **550.76** | **560.43 / 8.10** |
| RL-PPO（从零 base） | 551.87 | 559.27 / 8.30 |
| RL-PPO（BC 初始化 + hard） | 550.50 | 558.60 / 8.30 |
| Adv-MLP 激进 | 552.24 | 559.51 / 8.10 |
| Adv-GRU 激进 | 551.69 | 559.05 / 8.10 |
| Adv-MLP 保守 | 551.62 | 557.44 / 8.30 |
| GRU-Net | 547.81 | 555.22 / 8.30 |
| Neural MLP | 548.15 | 555.10 / 8.30 |
| Triple-majority | 547.51 | 552.52 / 9.20 |
| Final-tuned | 546.35 | 549.39 / 10.40 |

新增的 21 维 SelfPlay-GRU（BC 蒸馏 + self-play 训练器）在 20-field 四分布上
550.76、blind 560.43 / 8.10；纯从零 15k self-play 为 524.77（2 fields）/
504.57 final test。2026-10-26 的改进 A/B（BC+red fine-tune、red best
response）为 **No-Go**：E1a 仅 +0.43 mean / +0.016 min_norm，E1b 崩坏，
red best response 100% 击败 BC/RL/Adv；详见
`results/selfplay_improvement/final_decision.md`。最终 sanity check（2026-10-26）：
benchmark/blind 完全复现，权重 checksum 与冻结建议见
`results/selfplay_improvement/sanity/README.md`；主提交建议 RL-PPO，21 维备选
SelfPlay-GRU，Adv-RL 仅在官方四分布构成匹配时使用。

新候选 **RL-PPO-Hard60**（`rl-hard60`，实验名 A60）：
官方 20-field 554.36、blind 20-field 564.17 / 5.75、held-out final 509.76
（RL-PPO 506.49），且 held-out family 无回退。权重在
`weights/rl_policy_weights_hard60.json`；报告见
`results/hard2_20261026/README.md`。当前未替换主权重，作为下一提交候选。

当前重新生成的统一结果见 `results/benchmark.json` 和 `results/blind_results.json`；
按候选/对手拆分的结果见 `results/split_retrain_20261025/`；全策略 × 全对手
逐类型结果和 9 个测试集加权结果见 `results/matchup_all_README.md`，透视表见
`results/matchup_all_wide.csv`。历史快照与 tuned 参数来源仍保留在 `results/`。
详细说明见 `HANDOVER.md`。

## 文档索引

| 文档 | 内容 |
|---|---|
| [HANDOVER.md](HANDOVER.md) | 交接总览、推荐方案、参数、风险、检查清单 |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | 目录分层、Policy/评估/训练统一接口 |
| [docs/SPLIT_EVALUATION.md](docs/SPLIT_EVALUATION.md) | 按策略/模型/对手拆分评估 |
| [docs/SELFPLAY.md](docs/SELFPLAY.md) | 非对称自博弈：21 维观测、对手池、red league、验证拆分 |
| [docs/IPD_README.md](docs/IPD_README.md) | 模块与结果索引 |
| [docs/NEURAL_POLICY.md](docs/NEURAL_POLICY.md) | MLP 策略：训练、接口、结果、风险 |
| [docs/GRU_POLICY.md](docs/GRU_POLICY.md) | GRU-Net 策略 |
| [docs/RL_POLICY.md](docs/RL_POLICY.md) | 完整 PPO 强化学习策略：MDP、训练、结果、风险 |
| [docs/ADVERSARIAL_POLICY.md](docs/ADVERSARIAL_POLICY.md) | 约束对抗剥削规则与调参 |
| [docs/FINAL_COMPARISON.md](docs/FINAL_COMPARISON.md) | 各测试集主表 |
| [docs/SUITE_RESULTS.md](docs/SUITE_RESULTS.md) | 9 个测试集结果 |
| [docs/OPPONENT_PERFORMANCE.md](docs/OPPONENT_PERFORMANCE.md) | 85 类对手逐类型表现 |
| [docs/ADVANCED_RESULTS.md](docs/ADVANCED_RESULTS.md) | P0/P1/P2 完整过程 |
| [docs/MATCHUP_RESULTS.md](docs/MATCHUP_RESULTS.md) | 早期对阵汇总 |
| [docs/BAYES_RESPONSIVE_FLOW.md](docs/BAYES_RESPONSIVE_FLOW.md) | BayesPlus / Responsive3 C++ 流程图 |

## 注意事项

- 参赛/评估候选现在位于 `ipd/policies/candidates/`，每个候选一个独立文件；
  候选文件之间不要互相 import，`ipd/policies/rules/` 仅供训练/搜索复用。
- `ipd/core/machine.py` / `cpp/ipd_strategy.cpp` 的默认参数是早期版本，不是 Final-tuned。
- GRU 是有状态策略：每场必须 `reset()`，同一时间步只调用一次 `act()`。
- 所有候选策略都应在第 100 天选择背叛。
- Adv-MLP / Adv-GRU 目前只有 Python 版；Adv-RL 另有 `cpp/adversarial_rl.cpp`。
