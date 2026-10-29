# PSRO 策略（`psro` / `psro-br` / `psro-v4`）

## 1. 部署形态

| 候选 | 类型 | 行为 | 权重 |
|---|---|---|---|
| `psro` | PSRO-Meta | 每场 `reset()` 时按 blue maximin 权重采样一个成员，整场使用该成员 | `weights/psro/meta_strategy.json` |
| `psro-br` | PSRO-BR | 最后一轮 PPO best response，单策略 | `weights/psro/psro_policy.json` |
| `psro-scratch` | PSRO-Scratch | 从零 blue population（random/scratch base/BR）的 meta-policy | `weights/psro_scratch/meta_strategy.json` |
| `psro-scratch-br` | PSRO-Scratch-BR | 从零 PSRO 的最终 best response | `weights/psro_scratch/psro_policy.json` |
| `psro-v4` | PSRO-V4-Meta | 2026-10-29 更强 oracle 的 blue maximin meta-policy | `weights/psro_v4/meta_strategy.json` |
| `psro-v4-br` | PSRO-V4-BR | v4 进入 maximin support 的 BR_01（单策略） | `weights/psro_v4/psro_policy.json` |

`psro` 是 PSRO 的完整输出；`psro-br` 是对照/研究 artifact。`psro-v4` /
`psro-v4-br` 是 3 轮、30 iter/轮、40% red、低 KL 的更强 oracle 复现。
全部**未 promotion**。

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
e97e3cff866b4cb6c366b9dc5849b5429c7854829486eeb3936b6c3d377e6fac  weights/psro_v4/psro_policy.json
0d2856b63a231c12602f6ff651c1c54fde1965d03bf98576ecafd1dc7b0c82bd  weights/psro_v4/meta_strategy.json
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

## 6. PSRO v4（`psro-v4` / `psro-v4-br`）

配置 `configs/psro/psro_v4_20261029.json`：3 轮 × 30 iter，40% red，
`bc_kl_coef=0.02`，blue population 加入 RL-Switch / RL-Wide160 /
RL-Pattern 等当前头部候选。

- 初始 exploitability 290.00；**第 1 轮 BR_01 直接进入 blue maximin
  support（权重 1.0），exploitability 降到 0.00**；BR_02/03 未进入
  support（权重 0）。
- 部署的是 support 内的 BR_01，不是最后一轮 BR_03。
- 诊断：`dev_param`（10×1024）**486.93 vs RL-Repair 546.47**，min_norm
  0.5599，worst family `noisy` 260.0；`dev_complex` 436.43（RL-Repair
  434.28）；`dev_heldout_red` / Red-v5/v6 **red win 0.0**（blue 530.55
  vs RL-Repair 348.35），`dev_red` red win 0.0。
- 判定：**No-Go for promotion**；这是当前最强的 anti-exploitation
  specialist，但 official 代价约 -59.5，不能替代 RL-Repair / 主权重。
- 完整结果：`results/psro_v4_20261029/`；方法与下一步见
  [../research/psro.md](../research/psro.md) 第 8 节。
