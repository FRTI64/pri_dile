# RL-Wide：把 64/32 扩到 128/128

> **状态（2026-10-29）：实验候选，未 promotion。**
> `rl-wide` 在 Protocol v4 dev 上 dev_param 548.24、dev_complex 452.79、
> dev_ood 520.71；`min_norm` 从 RL-Pattern 的 0.845/0.842 提到
> **0.973/0.960**。test_v7 provisional 485.50，与 RL-Pattern 基本持平。

## 1. 动机

RL-Pattern 已经修掉 `pattern_breaker`，但 64/32 的容量仍然限制了
complex / OOD 的上限。直接加宽无法加载旧权重，因此先用 net2wider 把
64/32 扩到 128/128，再做轻量 fine-tune，避免从零重训。

目标不是追求最大 dev_complex，而是同时满足：

- dev_param 不下降；
- dev_complex / dev_ood 继续提升；
- `min_norm` 不出现新的最差 family；
- test_v7 provisional 不出现明显退化。

## 2. net2wider 扩展

新增 `ipd/training/expand.py`：

```bash
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_pattern.json \
  --hidden 128 64 \
  --out results/rl_v4_20261029/wide_ac.json \
  --weights-out results/rl_v4_20261029/wide_weights.json
```

规则：

- 复制隐藏单元，并把下一层对应列除以复制次数，保持初始函数接近原模型；
- 被复制的单元加入 1% 相对噪声，打破对称性；
- policy / value 两条 trunk 同步扩展；
- 支持 `ipd-rl-policy-v1` 和 `ipd-neural-policy-v1` 两种格式。

64/32 → 128/64 的实际 logit 差异：200 个随机历史上 max 0.065、
mean 0.018，因此 PPO 从接近 RL-Pattern 的策略开始。

## 3. 训练路线

### 3.1 128/64 探索

先扩到 128/64，然后用两类配方：

| 配方 | 训练分布 | 结果 |
|---|---|---|
| PB-heavy | legacy + `train_v4` + `train_pattern_breaker`（4/11） | dev_complex 457.9，但 dev_param 547.03 |
| gentle | legacy + `train_v4`（无额外 PB） | dev_param 548.55，但 `min_norm` 0.78 |

PB-heavy 的 complex 高但 dev_param 掉；gentle 的 dev_param 好但
`pattern_breaker` 仍是最差 family。两者都不够均衡。

### 3.2 128/128 平衡版（最终）

从 128/64 gentle 模型再扩到 128/128，然后做 4 iter 小步 fine-tune：

```bash
python3 -m ipd train expand \
  --input results/rl_v4_20261029/wide_gentle_ckpts/iter_008.json \
  --hidden 128 128 \
  --out results/rl_v4_20261029/wide128_ac.json

python3 -m ipd train rl --device cuda --hidden 128 128 \
  --iterations 4 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261036 \
  --init-weights results/rl_v4_20261029/wide128_ckpts/iter_004.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261029/wide_balance_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261029/wide_balance_weights_best.json \
  --checkpoint-out results/rl_v4_20261029/wide_balance_ac_best.json \
  --history-out results/rl_v4_20261029/wide_balance_history.json
```

最终取 `wide_balance_ckpts/iter_004.json`，不是 `--out` 的 best。
PB 权重只给 1/11，避免 128/128 的大容量再次把 PB 学成唯一目标。

## 4. 结果

### 4.1 Protocol v4 dev（30 fields × 2048，sample，v5 GPU）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| RL-PPO | 540.35 | 582.53 | 440.31 | 574.64 | 513.48 |
| RL-Repair | 547.96 | 582.76 | 434.28 | 582.46 | 515.19 |
| RL-Pattern | **548.45** | 583.07 | 447.64 | 582.22 | 519.46 |
| **RL-Wide** | 548.24 | **583.08** | **452.79** | **582.30** | **520.71** |

`min_norm`（vs RL-PPO）：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| RL-Pattern | 0.9970 | 1.0000 | 0.8448 | 1.0072 | 0.8424 |
| **RL-Wide** | 0.9968 | 1.0000 | **0.9726** | **1.0072** | **0.9596** |

- 相对 RL-Pattern：dev_param -0.21、dev_family +0.01、
  dev_complex **+5.15**、dev_axelrod +0.08、dev_ood **+1.25**；
- `pattern_breaker` 420.3 → **483.8**，`lookahead` 355.2 → **379.6**；
- 代价：`extort` -2.5、`temporal` -2.6、`probe_exploit` -3.3；
- `min_norm` 大幅改善，说明 worst-family 从 PB 转移到了更均匀的分布。

### 4.2 test_v7（50 fields × 2048，sample，provisional）

| 候选 | test_v7 | min field | p05 | worst family |
|---|---:|---:|---:|---:|
| RL-PPO | 477.26 | 470.50 | 471.20 | 138.7 |
| RL-Repair | **486.31** | **480.55** | **480.79** | 121.1 |
| RL-Pattern | 485.43 | 478.69 | 479.54 | 129.2 |
| **RL-Wide** | 485.50 | 479.38 | 480.11 | 128.4 |

- 相对 RL-Repair -0.81；相对 RL-Pattern +0.07；相对 RL-PPO +8.24；
- `pattern_breaker` 407.7 → 442.1（test 参数），`lookahead` 693 → 717；
- test-only 仍有回退：`hyper_extort`、`axelrod_extra` 未恢复到
  RL-Repair 水平；因此仍不能 promotion；
- legacy 20-field 兼容性：四分布 avg **554.96**、blind **564.72**
  （RL-Pattern 554.93 / 564.91）。

## 5. 消融：宽度与 PB 权重

| 变体 | 宽度 | PB 权重 | dev_param | dev_complex | complex min_norm | test_v7 |
|---|---|---:|---:|---:|---:|---:|
| RL-Pattern | 64/32 | 4/11 | 548.45 | 447.64 | 0.845 | 485.43 |
| Wide PB-heavy | 128/64 | 4/11 | 547.03 | 457.90 | 0.926 | 485.18 |
| Wide gentle | 128/64 | 0 | 548.55 | 447.13 | 0.781 | 486.25 |
| Wide 128/128 gentle | 128/128 | 0 | 548.89 | 445.90 | 0.730 | **486.38** |
| **Wide balanced** | 128/128 | 1/11 | 548.24 | 452.79 | **0.973** | 485.50 |

结论：

- 宽度本身主要提升 dev_complex / OOD 的上限，但会放大 PB 的波动；
- 完全不训 PB 时，`pattern_breaker` 重新成为最差 family（min_norm 0.73–0.78）；
- 小权重 PB（1/11）能在不牺牲 dev_param 的情况下把 min_norm 拉到 0.97；
- 继续加大 PB 或宽度不是下一步首选，应先做独立 seed 复现和 red 检查。

## 6. 判定与下一步

**判定：注册 `rl-wide` 为实验候选，不 promotion。**

- 它是 Protocol v4 dev 上综合最均衡的候选：dev_param 与 RL-Pattern
  基本持平，complex/OOD 和 min_norm 更好；
- test_v7 provisional 与 RL-Pattern 持平，仍低于 RL-Repair 0.81；
- 主权重不变，RL-Repair / RL-Pattern / A60 / RL-PPO 仍为 fallback。

下一步：

1. 独立 seed 复现 dev_param / dev_complex / min_norm；
2. 在 `dev_level0..4` / `dev_complexity0..4` 上确认没有新的分级回退；
3. 跑 `dev_red` 与外部比赛；如果 red 不比 RL-Repair 差，再讨论
   promotion；
4. 若继续加宽，优先做 192/128 或 128/128 + 特征扩展，而不是继续
   加 PB 权重。

## 7. 复现

```bash
python3 -m ipd list-policies | grep rl-wide
python3 -m ipd self-test --candidates rl-wide

python3 -m ipd protocol eval --candidates RL-PPO,RL-Repair,RL-Pattern,RL-Wide \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-PPO \
  --out results/rl_v4_20261029/protocol_v4_dev_rl_wide.json

python3 -m ipd protocol eval --candidates RL-PPO,RL-Repair,RL-Pattern,RL-Wide \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 --candidate-chunk 1 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 2 --seed 20261027 --no-consume \
  --out results/rl_v4_20261029/protocol_v4_test_v7_rl_wide_provisional.json
```

权重：

- `weights/rl_policy_weights_wide.json`
  sha256 `176787bdb6048d7a5cb0e5adcda9afef12fb77ed1d676b4f2c6ce67b1669d00b`
- `weights/rl_policy_checkpoint_wide.json`
  sha256 `3a2ba29e34e07aded66b531cc564f26ee444990ac7bd073fd76c2e23d5a7fde4`
