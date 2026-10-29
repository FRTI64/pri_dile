# RL-Switch 单模型蒸馏（旧结果 No-Go；已由 GRU-DAgger 接替）

> **状态（2026-10-29）：旧 MLP 结果 No-Go，但主要原因是
> normalization bug。** 本页保留原始实验；修正 normalization 后
> MLP BC 官方 dev_param 548.84，但 dev_complex 仍只有 446.59
> （pattern_breaker 383.19）。后续的 GRU + DAgger 单模型
> `rl-dagger-gru` 已与 RL-Switch 持平，见
> [dagger-gru.md](dagger-gru.md)。**本页的 535.46 不应再作为
> “蒸馏不可行”的证据。**

## 1. 动机

RL-Switch 是一个 hard switch：

- 前 6 回合由 RL-Wide160 执行；
- 若双方各 D≥2 且对手最新 C，则剩余回合切 RL-Repair。

它需要两份权重和开关逻辑。一个自然的想法是把它蒸馏成一个单模型，
既简化部署，也可能学到更平滑的切换策略。

## 2. 方法

- 教师：`RL-Switch`（sample 模式）；
- 学生：160/128 MLP，从 RL-Wide160 checkpoint 初始化；
- 训练数据：train-tier / train_complex / extort_hard / periodic_hard /
  hard，共 96,196 个非强制状态；
- 目标：教师当前使用的 expert 的 `P(D)`（软标签）；
- 学生只训练 policy trunk/head，BCE，Adam 1e-4，4 epochs；
- 数据来自教师轨迹（behavior cloning），没有 DAgger。

## 3. 结果

训练：

```text
states=96196 switch_share=0.611
epoch 1 bce=0.42882
epoch 4 bce=0.18989
```

dev：

| 集 | RL-Wide160 | RL-Switch | Distill |
|---|---:|---:|---:|
| dev_param | 548.26 | **548.58** | 535.46 |
| dev_family | **583.08** | 582.91 | 582.28 |
| dev_complex | **454.31** | 454.24 | 442.43 |
| dev_axelrod | 582.38 | **582.86** | 571.99 |
| dev_ood | 521.37 | **521.42** | 511.24 |

## 4. 为什么失败

0. **normalization bug（主因）**：从 wide160 checkpoint 初始化，
   却用 `mean=0, std=1` 训练并导出；同一权重在 dev_param
   （5×1024）上正确归一化 544.74，identity 只有 498.61。旧蒸馏
   531.75 与这个错误一致。修正后 MLP BC 到 548.84；
1. **match-level 状态不可见**：教师每场只决定一次（前 6 回合后），
   学生是无状态 MLP，只能从当前 history 推断“本场是否已经切换”；
2. **教师轨迹分布偏移**：BC 数据只覆盖教师访问的状态，学生一 drift
   就进入没见过的状态；
3. **软标签平均化**：教师 target 是当前 expert 的概率，学生在
   switch 边界附近会学到中间概率，导致行为不够果断；
4. **数据覆盖**：switch_share 0.611，训练目标偏向 Repair 区域，
   complex/Axelrod 的 false positive 代价大。

## 5. 判定与下一步

**旧 MLP 方案 No-Go。** 不注册 `rl-distill`，不替换 RL-Switch。
后续已按下面方向完成 **GRU-DAgger**：

1. 用 **DAgger**（学生轨迹 + 教师查询）而不是纯 BC；
2. 用 **GRU / 有状态模型**，显式保留“本场是否已切换”；
3. 结果：`rl-dagger-gru` dev 与 RL-Switch 持平，见
   [dagger-gru.md](dagger-gru.md)。

当前最佳仍是双权重 RL-Switch；单模型蒸馏不是优先方向。

## 6. 复现

```bash
python3 results/distill_20261042/distill_switch.py \
  --fields 6 --n 48 --epochs 4 \
  --out results/distill_20261042/distill_switch_ckpt.json
```

数据/结果：`results/distill_20261042/`。
