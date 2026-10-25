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
│   ├── policies/         参赛策略 + 统一 Policy 接口 + 候选注册表
│   ├── evaluation/       四分布/盲测/9 测试集/逐类型对阵
│   ├── training/         MLP/GRU/RL/对抗训练与规则参数搜索
│   ├── tools/            C++ 权重导出
│   ├── experiments/      历史研究脚本（run_*）
│   └── cli.py            `python3 -m ipd ...` 统一入口
├── src/                  旧路径兼容层（仅转发到 ipd，保留旧命令）
├── cpp/                  C++17 推理实现、共用特征头文件和生成的权重头文件
├── configs/              对抗策略配置（需与 weights/ 中的权重配对使用）
├── weights/              训练好的 MLP / GRU / RL 权重（JSON）
├── results/              基准测试、盲测、训练曲线和搜索记录（JSON / CSV）
├── docs/                 方法说明、结果表、流程图、结构规范
├── tests/                分层测试（core/policies/evaluation/training/compat）
├── HANDOVER.md           交接文档：推荐方案、参数、风险、检查清单
└── requirements-train.txt
```

项目分层和接口规范见 **[docs/STRUCTURE.md](docs/STRUCTURE.md)**。核心规则：

- `core` 不依赖策略、评估或训练；
- `policies` 是唯一放参赛策略的目录，统一实现 `reset()` + `act(t, mine, theirs)`；
- `evaluation` 只通过候选注册表取策略，不再直接 import 具体策略类；
- `training` 不定义参赛策略，训练入口统一为 `python3 -m ipd train <task>`；
- `src/*.py` 只是兼容 shim，新代码请使用 `ipd.*`。

路径由 `ipd.paths` 解析，因此既可以在仓库根目录运行，也可以从其他目录运行：

```bash
python3 -m ipd benchmark --candidates core --fields 8
python3 src/blind_test.py --fields 10 --seed 20261025   # 旧命令仍可用
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
# 等价旧命令
python3 -m unittest discover -s tests -v
```

### 规则策略与评估

```bash
python3 -m ipd list-policies
python3 -m ipd benchmark --candidates core --fields 20 --workers 24
python3 -m ipd blind --candidates core --fields 10 --workers 12
python3 -m ipd suite --fields 12 --workers 24
python3 -m ipd matchup --candidates base --reps 16
```

### 神经策略推理（不需要 PyTorch）

```bash
python3 -m ipd self-test --candidates neural,gru,rl,adv-mlp,adv-gru,adv-rl
python3 src/neural_policy.py weights/neural_policy_weights.json
python3 src/gru_policy.py weights/gru_policy_weights.json
python3 src/adversarial_policy.py configs/adversarial_policy_config.json
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
```

旧命令（`python3 src/benchmark_neural_policy.py --arch rl ...` 等）仍然可用，
并会转发到同一套评估实现。

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

旧的 `src/export_neural_cpp.py` / `src/export_gru_cpp.py` 仍可用。

## 结果摘要

统一四分布测试（20 个赛场 × 299 人）：

| 策略 | 四分布均值 | 盲测均分 / 平均排名 |
|---|---:|---:|
| **RL-PPO（从零 + hard，最终）** | **553.33** | **563.23 / 8.00** |
| **Adv-RL 激进（RL + 约束剥削）** | **554.29** | **562.91 / 8.00** |
| RL-PPO（从零 base） | 551.87 | 559.27 / 8.30 |
| RL-PPO（BC 初始化 + hard） | 550.50 | 558.60 / 8.30 |
| Adv-MLP 激进 | 552.24 | 559.51 / 8.10 |
| Adv-GRU 激进 | 551.69 | 559.05 / 8.10 |
| Adv-MLP 保守 | 551.62 | 557.44 / 8.30 |
| GRU-Net | 547.81 | 555.22 / 8.30 |
| Neural MLP | 548.15 | 555.10 / 8.30 |
| Triple-majority | 547.51 | 552.52 / 9.20 |
| Final-tuned | 546.35 | 549.39 / 10.40 |

不同盲测脚本的绝对分数不可直接混用；详细说明见 `HANDOVER.md`。

## 文档索引

| 文档 | 内容 |
|---|---|
| [HANDOVER.md](HANDOVER.md) | 交接总览、推荐方案、参数、风险、检查清单 |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | 目录分层、Policy/评估/训练统一接口、兼容层 |
| [docs/IPD_README.md](docs/IPD_README.md) | 全部模块和实验脚本清单 |
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

- `src/ipd_strategy.py` / `cpp/ipd_strategy.cpp` 的默认参数是早期版本，不是 Final-tuned。
- GRU 是有状态策略：每场必须 `reset()`，同一时间步只调用一次 `act()`。
- 所有候选策略都应在第 100 天选择背叛。
- 对抗策略目前只有 Python 版；C++ 版需要在外层补充两条剥削规则。
