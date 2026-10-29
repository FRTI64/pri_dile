# GRU-DAgger：把 RL-Switch 蒸馏成单模型

> **状态（2026-10-29）：实验候选 `rl-dagger-gru` / `GRU-DAgger`，未
> promotion。** 单模型 GRU（64 hidden、1 层）在 Protocol v4 dev 上
> 与 RL-Switch 持平，complex / OOD 略高；两个官方 seed 都复现。
> `test_v7` provisional 低 0.37 / 0.49，在噪声范围内，但 test 仍锁定，
> 不参与选模。主权重保持不变。

## 1. 动机

RL-Switch 是当前最佳候选，但它需要两份权重和 match-level 开关逻辑：

- 前 6 回合使用 RL-Wide160；
- 若双方前 6 回合各 D≥2 且对手最近一步 C，则余下回合使用
  RL-Repair。

之前的单模型蒸馏（`results/distill_20261042/`）只得到 dev_param
535.46，判定 No-Go。复查发现它有一个关键实现错误：

- 训练从 `rl_policy_checkpoint_wide160.json` 初始化，checkpoint 的
  `feature_mean` / `feature_std` 不是 0/1；
- 蒸馏脚本却用 `mean=0, std=1` 直接喂原始特征，并把这些 0/1
  写进导出文件；
- 用同一 wide160 权重做对照，在 dev_param（5×1024）上，正确
  normalization 为 **544.74**，identity normalization 只有
  **498.61**，旧蒸馏为 **531.75**。

因此旧蒸馏的 535.46 主要是 normalization bug，而不是“蒸馏不可行”。
本实验用正确的 normalization 重做 MLP，并进一步比较 DAgger 与
GRU 状态模型。

## 2. 方法

### 2.1 教师

教师是确定性的 RL-Switch 规则。对每个状态：

```text
if len(theirs) >= 6
   and theirs[:6].count(D) >= 2
   and mine[:6].count(D) >= 2
   and C in theirs[6:]:
       expert = RL-Repair
else:
       expert = RL-Wide160
target = expert.P(defect)
```

标签使用所选 expert 的软概率，而不是采样动作，以减少教师随机性。
学生只在这些状态上学习：`t>0`、对手至少 D 过一次、`t<99`；第 1 天
强制 C、第 100 天强制 D，由推理层统一处理。

### 2.2 训练分布

六个来源，每源若干 field（每 field 48 个对手）：

| 来源 | 作用 |
|---|---|
| `train`（Protocol v4 train_v4） | 主分布 |
| `train_complex` | complex 风格 |
| `train_pattern_breaker` | 修复 pattern_breaker |
| `extort_hard` | extortion / ZD |
| `periodic_hard` | periodic / Kamikaze |
| `hard` | 尾风险混合 |

BC 使用教师轨迹；DAgger 使用学生自己的轨迹（epsilon=0.05
exploration），再用教师在同一 history 上打标签。每轮把新数据
加入聚合集，重新训练。

### 2.3 学生

| 学生 | 结构 | 初始化 |
|---|---|---|
| MLP | 160/128 tanh，无状态 | RL-Wide160 checkpoint |
| GRU | 85→GRU(64)→64→tanh→1，1 层 | 从零 |

GRU 的 hidden state 可以显式保留“本场是否已经切换”，这是无状态
MLP 做不到的。GRU 导出为标准 `ipd-gru-policy-v1`，推理层
`GRUPolicy` 和 GPU `BatchGRU` 都直接支持。

超参数：

```text
BC fields: 12/source (GRU final), n=48
DAgger fields: 6/source/round, rounds=2, explore=0.05
GRU: hidden=64, layers=1, epochs=30, lr=1e-3, batch=64 matches
MLP: epochs=10, lr=1e-4, batch=8192 states
```

## 3. 结果

### 3.1 MLP：normalization 修复后仍受 pattern_breaker 限制

修正 normalization 后，MLP BC 在官方 30×2048 dev 上：

| 集 | RL-Switch | MLP-BC | diff |
|---|---:|---:|---:|
| dev_param | 548.58 | **548.84** | +0.26 |
| dev_family | 582.91 | **582.97** | +0.06 |
| dev_complex | **454.24** | 446.59 | -7.65 |
| dev_axelrod | 582.86 | **582.92** | +0.06 |
| dev_ood | **521.42** | 518.95 | -2.47 |

dev_complex 的差距几乎全部来自 `pattern_breaker`：

```text
RL-Switch 512.94
MLP-BC     383.19   (-129.75)
```

加入 `train_pattern_breaker`、冻结 trunk、DAgger 后，5×1024 诊断
仍只有约 447–451 complex（同口径 RL-Switch 为 453.47）。无状态
MLP 无法同时保留 base 行为和切换后的行为。

### 3.2 GRU：BC 已经接近，DAgger 补齐

5×1024 诊断（dev_param / dev_complex）：

| 模型 | dev_param | dev_complex | pattern_breaker |
|---|---:|---:|---:|
| RL-Switch | 545.10 | 453.47 | 511.88 |
| GRU BC（12 fields/source） | 544.91 | 453.59 | 507.31 |
| GRU BC + 2 轮 DAgger | **545.50** | **455.38** | **516.16** |

DAgger 让验证集 BCE 从 0.1322 降到 0.1304，switch 子集准确率从
98.9% 升到 99.4%。

### 3.3 官方 dev（30 fields × 2048）

seed 20261027（官方主 seed）：

| 集 | RL-Switch | GRU-DAgger | diff |
|---|---:|---:|---:|
| dev_param | 548.58 | **548.58** | +0.00 |
| dev_family | **582.91** | 582.90 | -0.01 |
| dev_complex | 454.24 | **455.13** | +0.89 |
| dev_axelrod | **582.86** | 582.49 | -0.37 |
| dev_ood | 521.42 | **521.54** | +0.12 |

seed 20261029（独立复现）：

| 集 | RL-Switch | GRU-DAgger | diff |
|---|---:|---:|---:|
| dev_param | 546.80 | **546.97** | +0.16 |
| dev_family | **582.62** | 582.59 | -0.03 |
| dev_complex | 454.27 | **454.39** | +0.12 |
| dev_axelrod | **582.13** | 581.93 | -0.20 |
| dev_ood | 522.84 | **523.05** | +0.21 |

`min_norm`（相对 RL-Wide160 的 family 归一化）在两个 seed 上都
≥0.9919；dev_param 的 0.9955/0.9957 与 RL-Switch 相同，complex 的
0.9935 也相同。最弱 family 是 `conditional_extort`，与 RL-Switch
一致。

### 3.4 dev_complex family 对比（seed 27）

| family | RL-Switch | GRU-DAgger | diff |
|---|---:|---:|---:|
| pattern_breaker | 512.94 | **519.28** | +6.33 |
| lookahead | 377.35 | **384.05** | +6.70 |
| delayed_extort | 357.27 | **358.05** | +0.78 |
| anchor | 488.16 | **488.50** | +0.35 |
| adaptive_mix | 296.47 | **296.80** | +0.33 |
| meta_mix | 497.11 | **497.32** | +0.22 |
| phase_switch | 491.06 | **491.16** | +0.11 |
| portfolio | 470.27 | **470.35** | +0.08 |
| endgame_hunter | 397.93 | **397.96** | +0.03 |
| contrition_cycle | 595.00 | 595.00 | 0.00 |
| memory_two | 595.00 | 595.00 | 0.00 |
| noise_adaptive | 595.00 | 595.00 | 0.00 |
| reputation_grim | 595.00 | 595.00 | 0.00 |
| probe_exploit | 271.54 | 271.37 | -0.16 |
| conditional_extort | 289.48 | 289.22 | -0.27 |

### 3.5 level / complexity 诊断（seed 27）

| tier | RL-Switch | GRU-DAgger | diff |
|---|---:|---:|---:|
| dev_level0 | 625.19 | **625.20** | +0.01 |
| dev_level1 | **550.78** | 550.74 | -0.03 |
| dev_level2 | **551.29** | 551.09 | -0.21 |
| dev_level3 | 451.94 | **452.45** | +0.51 |
| dev_level4 | 448.67 | **449.88** | +1.20 |
| dev_complexity0 | 625.19 | **625.20** | +0.01 |
| dev_complexity1 | **543.84** | 543.77 | -0.07 |
| dev_complexity2 | 535.69 | **535.70** | +0.01 |
| dev_complexity3 | **523.69** | 523.59 | -0.10 |
| dev_complexity4 | 402.65 | **403.91** | +1.26 |

全部在 ±1.3 内；level3/4 和 complexity4 略有提升。

### 3.6 test_v7 provisional（仅诊断，不选模）

| seed | RL-Switch | GRU-DAgger | diff |
|---|---:|---:|---:|
| 20261027 | **488.26** | 487.89 | -0.37 |
| 20261029 | **487.51** | 487.02 | -0.49 |

test 消费永久锁定；这里只运行一次用于诊断，结果
`selection_eligible=false`，不写 usage ledger，也不影响推荐。
两次都在噪声范围内，但不能宣称 test 超越 RL-Switch。

### 3.7 安全规则与行为检查

- 第 1 天 C，第 100 天 D；
- 对手未 D 时保持 C；
- 对 AllC / TFT / Grim 得分 604，对 AllD 得分 99，与 RL-Switch
  完全一致；
- Python `GRUPolicy` 与 GPU `BatchGRU` 固定序列 logit 差 <1e-6，
  argmax 100/100 一致；
- C++ `RLDAggerGRUPolicy` 与 Python 固定序列 action parity 100/100。

## 4. 为什么 GRU 有效

1. **状态可保留**：switch 是一个“曾经满足条件”的 latch；
   GRU hidden state 能直接记住，无状态 MLP 只能从 85 维摘要里
   间接推断。
2. **正确 normalization**：从 wide160 checkpoint 初始化时必须使用
   其 `feature_mean/std`，否则等价于把输入尺度整体改掉。
3. **覆盖 pattern_breaker**：12 fields/source 的 BC 数据让
   pattern_breaker 从 424（6 fields）恢复到 507（5×1024），DAgger
   再提升到 516。
4. **DAgger 修正分布偏移**：学生自己的轨迹被教师重新标注，减少
   了 BC 在 switch 边界附近的误差累积。

## 5. 部署

- Python：`make_policy("rl-dagger-gru")` 或
  `GRUPolicy("weights/rl_dagger_gru_weights.json")`；
- GPU：`BatchGRU` 直接读取同一 `ipd-gru-policy-v1` 文件；
- C++：`cpp/policies/rl_dagger_gru_policy.hpp` +
  `cpp/rl_dagger_gru_weights.hpp`，`cpp/rl_dagger_gru_demo.cpp`；
  重新导出 header 后要把 namespace 改成 `rl_dagger_gru_weights`
  （见 demo 顶部注释）；
- 每场必须 `reset()`；同一时间步只调用一次 `act()`；
- 单文件 1,063,718 bytes，sha256
  `7b7bbe677db157dd7101a35c018b89ca6bcc9f4357091c58d5e7b14c80cca196`。

## 6. 风险与判定

**不是 promotion。** 当前结论是“单模型部署候选”：

- dev 两个 seed 与 RL-Switch 持平，complex/OOD 略高；
- test provisional 低 0.4 左右，不能当作独立 final；
- 没有外部比赛结果；
- GRU 有状态，部署时必须每场 reset；如果比赛接口不能保证，
  应继续用 RL-Switch 或 RL-Wide160；
- 单文件 1.1MB，比 RL-Pattern（约 249KB）大，但比“双权重 +
  开关”更简单；
- 不替换主权重 `weights/rl_policy_weights.json`。

下一步若要 promotion：外部比赛/独立种子复现 + 用户确认；若要
继续研究，可尝试 hidden=96、更多 pattern_breaker 数据，或把
GRU 作为 gate 而不是完整策略。

## 7. 复现

```bash
# 训练（最终配置）
python3 results/distill_20261043/dagger_switch.py \
  --student gru --rounds 2 --bc-fields 12 --dagger-fields 6 \
  --val-fields 3 --n 48 --hidden 64 --layers 1 \
  --out results/distill_20261043/gru_dagger_ckpt.json

# 官方 dev（seed 27）
python3 results/distill_20261043/eval_tier_gpu_seed.py \
  weights/rl_dagger_gru_weights.json dev_param GRU-DAgger \
  30 2048 rl-dagger-gru 20261027

# C++ parity
python3 -m unittest tests.test_dagger_gru_cpp -v
```

数据/结果：`results/distill_20261043/`；汇总：
`results/distill_20261043/summary.json`。
