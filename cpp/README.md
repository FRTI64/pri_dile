# C++ 提交/推理文件

策略和对手已按文件拆分，顶层 `*.cpp` 保留为可编译演示入口。

## 策略/模型

| 文件 | 类 | 参数 |
|---|---|---|
| `policies/final_policy.hpp` | `ipd::FinalPolicy`（兼容别名 `IPDStrategy`） | `ipd::FinalParams` / `FinalParams::tuned()` |
| `policies/responsive3_policy.hpp` | `ipd::Responsive3` | `ipd::Responsive3Params` / `::tuned()` |
| `policies/bayes_plus_policy.hpp` | `ipd::BayesPlus` | `ipd::BayesPlusParams` / `::tuned()` |
| `policies/neural_policy.hpp` | `ipd::NeuralPolicy` | `ipd::NeuralParams` |
| `policies/gru_policy.hpp` | `ipd::GRUPolicy` | `ipd::GRUParams` |
| `policies/adversarial_rl_policy.hpp` | `ipd::AdversarialRLPolicy` | `ipd::AdvRLParams` |

MLP/RL 权重仍由编译宏选择：

```bash
# 行为克隆 MLP
g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo

# 最终 RL
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
  -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o rl_demo

# Adv-RL 的 BC base
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_BC_WEIGHTS \
  -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_bc_demo
```

## 对手

`opponents/` 下每个对手一个头文件：

- `all_c.hpp` / `all_d.hpp` / `tft.hpp` / `grim.hpp`
- `late_d.hpp`（可传 `k`）
- `random_p.hpp`（可传 `p` 和 seed）
- `opponent.hpp`：统一 `ipd::Opponent` 和 `ipd::play_against`
- `demo.hpp`：演示用对手组合

## 参数示例

```cpp
ipd::FinalPolicy final_policy(20261024u, ipd::FinalParams::tuned());
ipd::Responsive3 resp(20261024u, ipd::Responsive3Params::tuned());
ipd::BayesPlus bayes(ipd::BayesPlusParams::tuned());
ipd::GRUPolicy gru(20261024u, ipd::GRUParams{true, true, true});
ipd::NeuralPolicy mlp(20261024u, ipd::NeuralParams{true, true, true});
ipd::AdversarialRLPolicy adv(20261024u, true,
    ipd::AdvRLParams{0.2, 12, 2, 40, true});
```

`FinalParams::tuned()`、`Responsive3Params::tuned()`、`BayesPlusParams::tuned()`
的值与 Python 的 `DEFAULT_FINAL_TUNED` / `DEFAULT_RESP_TUNED` /
`DEFAULT_BAYES_TUNED` 一致。
