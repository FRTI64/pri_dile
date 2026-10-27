# PSRO 策略（`psro` / `psro-br`）

## 1. 两个部署形态

| 候选 | 类型 | 行为 | 权重 |
|---|---|---|---|
| `psro` | PSRO-Meta | 每场 `reset()` 时按 blue maximin 权重采样一个成员，整场使用该成员 | `weights/psro/meta_strategy.json` |
| `psro-br` | PSRO-BR | 最后一轮 PPO best response，单策略 | `weights/psro/psro_policy.json` |
| `psro-scratch` | PSRO-Scratch | 从零 blue population（random/scratch base/BR）的 meta-policy | `weights/psro_scratch/meta_strategy.json` |
| `psro-scratch-br` | PSRO-Scratch-BR | 从零 PSRO 的最终 best response | `weights/psro_scratch/psro_policy.json` |

`psro` 是 PSRO 的完整输出；`psro-br` 是对照/研究 artifact。两者都**未
promotion**。

## 2. 推理接口

```python
from ipd.policies.registry import make_policy

p = make_policy("psro")          # PSRO-Meta
p.reset()                        # 采样一个成员并 reset
action = p.act(t, mine, theirs)  # 0-based
```

`PSROMetaPolicy` 的关键语义：

- 采样发生在 `reset()`，不是每天；一个成员打完一整场；
- 成员按 `blue_meta.weights_by_name` 采样，零权重成员不会被选中；
- 成员实例按名字缓存并 `reset()`，RL / GRU 的状态不会跨场泄漏；
- 采样 RNG 由 candidate 的 `seed` 固定，同一 seed 的采样序列可复现；
- 成员可以是规则候选（`kind`）、JSON 策略（`path`）或 `spec`。

`psro-br` 是标准 85 维 MLP，可直接用 `NeuralPolicy` / C++ 部署；`psro`
meta-policy 需要 Python 侧采样器，C++ 部署需要自行实现同样的采样。

## 3. 文件与校验

```text
weights/psro/meta_strategy.json   population + blue/red meta weights
weights/psro/psro_policy.json     最后一轮 best response（psro-br）
results/psro_20261027/            report / population / oracle / diagnostics
```

当前 checksum：

```text
deedc25c7c384e86c0223ec5395ba6f3dfd21b5358de57c8d903ce056d92ed9b  weights/psro/psro_policy.json
7d1c886b0c96d28dbe14c713d30e7bc8c646c0c37dc2e74ed426987a9cd94cd9  weights/psro/meta_strategy.json
e70148b7d3d5fd31037ff7e3f31fc059833e862c3c93f064293471a61fe4aa4e  weights/psro_scratch/psro_policy.json
ec3e73aab9cc45ba1c330046347b6174a3bf78d35bedbbc322698fe598e7bf0f  weights/psro_scratch/meta_strategy.json
```

## 4. 从零版本（`psro-scratch` / `psro-scratch-br`）

从零版本遵守“已有模型/策略只能做对手参考”：

- blue 只有 `Scratch-Init`（随机 MLP）、`Scratch-Base`（从零官方 PPO）
  和 `PSRO-BR-*`（从 scratch base 初始化的 PPO BR）；
- existing RL / rule / selfplay 全部在 red 池，不进入 blue，也不作为
  oracle 初始化；
- 随机 init 只继承 feature mean/std，不继承 policy weights。

诊断结果（4×512，10 red seeds）：

| 模型 | dev_param | min_norm | dev_red red win | Red-v5/v6 red win |
|---|---:|---:|---:|---:|
| `psro-scratch` | 480.43 | 0.5501 | **0.0** | **0.0** |
| `psro-scratch-br` | 481.11 | 0.5437 | **0.0** | **0.0** |
| RL-Repair | 543.35 | 1.0000 | 1.0 | 1.0 |

这是项目里第一次在 held-out Red-v5/v6 上把 red win 打到 0；代价是
official `dev_param` 掉约 63，`noisy` family 只剩 249。它是
anti-exploitation specialist，不是主权重替代。完整实验见
[../research/psro-scratch.md](../research/psro-scratch.md)。

## 5. 评估结论

- Blue maximin：Final-tuned 0.725 + Triple-majority 0.275；
- 2 轮 PPO oracle 未进入 support（权重 0），meta exploitability 284.80 不变；
- dev_param（4×512 诊断）：527.71 vs RL-Repair 543.35；
- held-out Red-v5/v6（10 seeds）：373.70 vs RL-Repair 348.35；
  red-minus-blue 均值 378.5 → 297.5（**-21.4%**，v5 0%、v6 -37.3%）；
- 所有候选 red win 仍 100%；Gate D 仍因 official/family 约束失败。

因此 `psro` 是一个 anti-exploitation / official 的 trade-off 实验候选，
不是主权重替代。方法和完整结果见
[../research/psro.md](../research/psro.md)。
