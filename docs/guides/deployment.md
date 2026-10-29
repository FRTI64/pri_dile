# 部署指南

本项目有两种部署路径：

1. **Python**：纯标准库推理，不需要 PyTorch；
2. **C++17**：MLP / GRU / RL / Adv-RL / Final / Resp3 / BayesPlus。

训练需要 PyTorch，但部署不需要。

## 1. Python 推理

```python
from ipd.policies.rl import RLPolicy

p = RLPolicy("weights/rl_policy_weights.json", seed=20261024)
p.reset()
action = p.act(t, mine, theirs)
```

动作约定：

```text
C = 0, D = 1
t = 0..99（0-based）
mine/theirs 是当前决策之前的动作
每场开始调用 reset()
对手从未 D 时保持 C；第 100 天 D
```

运行自检：

```bash
python3 -m ipd self-test --candidates rl
python3 -m ipd self-test --candidates rl-repair
python3 -m ipd self-test --candidates neural --weights weights/neural_policy_weights.json
python3 -m ipd self-test --candidates gru --weights weights/gru_policy_weights.json
python3 -m ipd self-test --candidates selfplay --weights weights/selfplay_gru_policy.json
```

## 2. 依赖与可提交产物

| 策略 | 必需文件 | 备注 |
|---|---|---|
| Final / Resp3 / Bayes / Triple | Python 源码 | 无外部权重 |
| Neural | `weights/neural_policy_weights.json` | 约 243 KB |
| GRU | `weights/gru_policy_weights.json` | 约 395 KB；有状态 |
| RL / A60 / Repair | `weights/rl_policy_weights*.json` | 85 维 MLP |
| RL-Switch | `rl_policy_weights_wide160.json` + `rl_policy_weights_repair.json` | Python + C++ 双 MLP；C++ 已通过 action parity |
| SelfPlay-GRU | `weights/selfplay_gru_policy.json` | 21 维 GRU；有状态 |
| Adv-MLP / Adv-GRU | 对应 base 权重 + `configs/adversarial_*_config*.json` | 只有 Python |
| Adv-RL | `weights/rl_policy_weights_bc.json` + aggressive config | 有 C++ |
| PSRO-BR | `weights/psro/psro_policy.json` | 85 维 MLP；有 C++ |
| PSRO-Meta | `weights/psro/meta_strategy.json` + 全部 blue 成员 | Python 采样器；C++ 需自行实现 |
| PSRO-Scratch-BR | `weights/psro_scratch/psro_policy.json` | 85 维 MLP；有 C++ |
| PSRO-Scratch | `weights/psro_scratch/meta_strategy.json` + 从零 blue 成员 | Python 采样器 |

单文件提交时，需要把 JSON 权重内嵌到 Python/C++ 源文件；仓库不提供自动
单文件打包脚本。

## 3. C++ 编译

```bash
# MLP（BC）
g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo

# GRU
g++ -std=c++17 -O2 -DGRU_POLICY_DEMO cpp/gru_policy.cpp -o gru_demo

# RL-PPO（主权重）
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
  -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o rl_demo

# Adv-RL（BC base）
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_BC_WEIGHTS \
  -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_bc_demo

# BayesPlus / Responsive3
g++ -std=c++17 -O2 -DRESP_BAYES_DEMO cpp/bayesplus_responsive3.cpp -o demo

# RL-Switch（先导出双权重头文件）
python3 -m ipd.tools.export_switch_cpp \
  --base weights/rl_policy_weights_wide160.json \
  --alt weights/rl_policy_weights_repair.json \
  --out cpp/rl_switch_weights.hpp
g++ -std=c++17 -O2 -Icpp cpp/rl_switch_demo.cpp -o rl_switch_demo
./rl_switch_demo
```

头文件与参数说明见 [../../cpp/README.md](../../cpp/README.md)。

## 4. 重新导出权重头文件

```bash
python3 -m ipd export --arch mlp
python3 -m ipd export --arch gru
python3 -m ipd export --weights weights/rl_policy_weights.json \
  --out cpp/rl_policy_weights.hpp
python3 -m ipd export --weights weights/rl_policy_weights_bc.json \
  --out cpp/rl_policy_weights_bc.hpp
python3 -m ipd.tools.export_switch_cpp \
  --base weights/rl_policy_weights_wide160.json \
  --alt weights/rl_policy_weights_repair.json \
  --out cpp/rl_switch_weights.hpp
```

导出后重新编译，并跑一次 `python3 -m ipd test` 和 C++ demo。

## 5. 当前 C++ 覆盖

| 策略 | C++ | 文件 |
|---|---|---|
| Final | 是 | `cpp/policies/final_policy.hpp` |
| Responsive3 | 是 | `cpp/policies/responsive3_policy.hpp` |
| BayesPlus | 是 | `cpp/policies/bayes_plus_policy.hpp` |
| Neural MLP | 是 | `cpp/policies/neural_policy.hpp` |
| GRU | 是 | `cpp/policies/gru_policy.hpp` |
| RL-PPO | 是 | `cpp/policies/neural_policy.hpp` + `cpp/rl_policy_weights.hpp` |
| RL-Switch | 是 | `cpp/policies/rl_switch_policy.hpp` + `cpp/rl_switch_weights.hpp` |
| Adv-RL | 是 | `cpp/policies/adversarial_rl_policy.hpp` |
| Adv-MLP / Adv-GRU | 否 | 需在 Neural/GRU 外自行加两条规则 |
| PSRO-BR | 是（标准 MLP） | `cpp/policies/neural_policy.hpp` + 导出的 PSRO 权重 |
| PSRO-Meta | 否 | 需要保留 population 和采样器；不能只导出单个 MLP |
| PSRO-Scratch-BR | 是（标准 MLP） | 同 PSRO-BR，权重换为 scratch 版 |
| PSRO-Scratch | 否 | 需要保留从零 blue population 和采样器 |
| Triple-majority | 否 | 需移植 Final + Resp3 + Bayes 并投票 |

## 6. 部署检查清单

- [ ] 固定 seed；
- [ ] 每场 `reset()`；GRU / SelfPlay 同一时间步只调用一次 `act()`；
- [ ] 第 1 天 C、对手未 D 时 C、第 100 天 D；
- [ ] 权重 checksum 与报告一致；
- [ ] 单文件限制下确认权重已内嵌；
- [ ] C++ 重新导出后与 Python 动作一致；
- [ ] 在目标设备上跑一次 `self-test` 和 1-field benchmark。
