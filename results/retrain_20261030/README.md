# 主模型重训实验（2026-10-30）

> **结论：No-Go。** 按 `docs/guides/training.md` 的主权重配方从头重训
> RL-PPO（scratch base → hard），两种 hard 阶段配置均在 Protocol v5
> dev 上比现有 `weights/rl_policy_weights.json` 低约 2.5–3.1 分。
> 未替换主权重，未注册候选。

## 1. 目的

在当前代码和 Protocol v5 split 下重跑主权重配方，确认：

1. 现有主权重是否可复现；
2. 重训模型是否能在 v5 dev 上超过现有 RL-PPO。

## 2. 命令

```bash
# Stage 1: from scratch, 30 iter
python3 -m ipd train rl --device cuda --from-scratch \
  --iterations 30 --fields-per-dist 5 --n 80 \
  --dist-weights 1,1,1,2 --ent-coef 0.001 --lr 3e-4 \
  --center-advantages --select-metric avg \
  --out results/retrain_20261030/scratch_base_weights.json \
  --checkpoint-out results/retrain_20261030/scratch_base_checkpoint.json \
  --history-out results/retrain_20261030/scratch_base_history.json

# Stage 2a: hard fine-tune, 文档原命令（默认 6 fields/dist）
python3 -m ipd train rl --device cuda --iterations 40 \
  --init-weights results/retrain_20261030/scratch_base_checkpoint.json \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 --ent-coef 0.0001 --lr 1e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --select-metric blind \
  --out results/retrain_20261030/rl_policy_weights_retrained.json \
  --checkpoint-out results/retrain_20261030/rl_policy_checkpoint_retrained.json \
  --history-out results/retrain_20261030/rl_training_history_retrained.json

# Stage 2b: 历史主权重口径（5 fields/dist + 3 val fields × 100）
python3 -m ipd train rl --device cuda --iterations 40 \
  --fields-per-dist 5 --val-fields-per-dist 3 --val-n 100 \
  --init-weights results/retrain_20261030/scratch_base_checkpoint.json \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 --ent-coef 0.0001 --lr 1e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --select-metric blind \
  --out results/retrain_20261030/rl_policy_weights_retrained_f5.json \
  --checkpoint-out results/retrain_20261030/rl_policy_checkpoint_retrained_f5.json \
  --history-out results/retrain_20261030/rl_training_history_retrained_f5.json
```

训练耗时约 2–3 分钟（30 iter）+ 约 6 分钟（40 iter），RTX 5090
Laptop，CUDA。两次 hard 阶段的最佳 checkpoint：

| 运行 | fields/dist | best iter | best val avg | best selection |
|---|---:|---:|---:|---:|
| 历史主权重（`rl_training_history.json`） | 5 | 30 | 556.55 | 553.71 |
| 重训（默认） | 6 | 35 | 556.23 | 553.54 |
| 重训（f5） | 5 | 25 | 556.33 | 553.37 |

训练内验证与历史主权重基本持平（差 0.17–0.34），但 v5 dev 外部评估
出现一致回退。

## 3. Protocol v5 dev（30 fields × 2048）

### 3.1 主 seed 20261027

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-PPO（现有主权重）** | **525.99** | **582.55** | **457.26** | **563.43** | **513.30** |
| 重训（6 fields/dist） | 523.49 | 582.29 | 454.32 | 560.89 | 511.01 |
| 重训（5 fields/dist） | 522.91 | 582.27 | 454.29 | 561.63 | 510.96 |

### 3.2 独立 seed 20261028

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-PPO（现有主权重）** | **525.09** | **582.15** | **458.73** | **563.09** | **511.72** |
| 重训（6 fields/dist） | 522.24 | 581.86 | 455.74 | 560.45 | 509.27 |

两个 seed 方向一致：重训模型 `dev_param` 低 2.85–3.08，
`dev_complex` 低 2.94–2.99，`dev_ood` 低 2.29–2.45。
`min_norm` 也略差（dev_param 0.969–0.972，dev_complex 0.839–0.880，
dev_ood 0.835–0.887；现有 RL-PPO 按定义 1.000）。

## 4. 判定

- **No-Go：** 重训配方不能复现现有主权重的 v5 dev 水平，也没有超过它；
- 训练内 val 与历史基本一致，说明差距不是训练崩溃，而是当前代码 /
  split 下模型行为的真实差异；
- 不替换 `weights/rl_policy_weights.json`，不注册新候选，不消费
  `test_v8`；
- 若继续提升主模型，应使用 Protocol v5 的 `train_v5` profile 做定向
  训练（`docs/guides/training.md` §3.4），而不是重复同一主权重配方。

## 5. 产物

- `scratch_base_weights.json` / `scratch_base_checkpoint.json`
- `rl_policy_weights_retrained.json` / `_checkpoint_`
- `rl_policy_weights_retrained_f5.json` / `_checkpoint_`
- `*_history.json`
- `protocol_v5_dev_retrained.json`、`protocol_v5_dev_retrained_seed2.json`、
  `protocol_v5_dev_retrained_f5.json`
- `*_train.log`、`eval_v5_dev*.log`
- 评估 wrapper：`results/protocol_v5/eval_weights.py`

权重 sha256：

```text
239414164745538fed05a1c0a3d50ca2130475de3d03c49a1332b1aea6e0dab7  rl_policy_weights_retrained.json
5065f8865f6e4d69b75eb1dcd964f6e44df3c865b48f9c27db9dda9d276f63da  rl_policy_weights_retrained_f5.json
```
