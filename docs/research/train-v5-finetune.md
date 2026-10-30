# RL-V5-FT：`train_v5` 定向 fine-tune

> **状态（2026-10-30）：实验候选 `rl-v5-ft`，未 promotion。**
> 从 RL-Wide160 出发，仅用 Protocol v5 的 `train_v5` profile 做短程
> PPO fine-tune。两个 dev seed 和 `test_v8` provisional 均优于
> RL-Wide160，但 `probe_exploit` 仍是瓶颈，`min_norm` 只有 0.927。

## 1. 动机

RL-Wide160 使用 v4 训练分布 + `train_pattern_breaker`。Protocol v5
重新划分 train/dev/test 后，需要确认：

1. 纯 `train_v5`（标准 + complex + policy + Axelrod + anchors）能否
   提升 `dev_complex` / `dev_axelrod` / `dev_ood`；
2. 增益能否延续到 `test_v8`；
3. 是否能在不破坏安全规则的前提下改善 probe。

## 2. 配方

- 起点：`weights/rl_policy_checkpoint_wide160.json`（160/128）；
- 训练分布：`train_v5`，不混入 probe/PB 专项 profile；
- 12 iter，3 fields/dist × 160，lr `1e-4`，bc-kl 0.1，target-kl 0.02；
- 每 4 iter 在 5 个 dev tier × 3 fields × 160 验证；
- 选择 iter8，导出 `weights/rl_policy_weights_v5_ft.json`。

先试过 lr `3e-5` 的 16 iter 版本，只有 +0.3～0.5 的 `dev_param`
增益；提高到 `1e-4` 后 complex/OOD 收益明显变大。

## 3. 结果

Protocol v5 dev（30 fields × 2048，sample）：

| seed | 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---|---:|---:|---:|---:|---:|
| 20261027 | **RL-V5-FT** | **532.18** | **583.14** | **482.64** | **568.74** | **523.94** |
| 20261027 | RL-Wide160 | 531.66 | 583.08 | 477.77 | 567.77 | 522.24 |
| 20261028 | **RL-V5-FT** | **531.08** | **582.68** | **483.81** | **568.59** | **522.07** |
| 20261028 | RL-Wide160 | 530.59 | 582.62 | 479.01 | 567.56 | 520.49 |

相对 RL-Wide160：

| seed | dev_param | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|
| 20261027 | +0.52 | +4.87 | +0.97 | +1.70 |
| 20261028 | +0.49 | +4.80 | +1.03 | +1.58 |

关键 family（seed 27）：`pattern_breaker` 634 → **683**，
`adaptive_mix` 308 → 317；`probe_exploit` 115.7 → 114.8，
因此 `min_norm` 从 0.934 降到 **0.927**。

`test_v8` provisional（50 fields × 2048，`--no-consume`）：

| 候选 | test_v8 |
|---|---:|
| **RL-V5-FT** | **496.37** |
| RL-Wide160 | 495.71 |
| RL-PPO | 487.83 |

相对 RL-Wide160 +0.66，相对 RL-PPO +8.54。

## 4. 判定

- **实验候选，不 promotion。** 收益可复现，但 `probe_exploit` 仍约
  115（RL-PPO 124），`min_norm` 0.927，低于旧 0.99 门槛；
- 不替换主权重；`test_v8` 未消费；
- 若继续，应在保持 `train_v5` 的前提下做 probe 定向修复，或把该
  checkpoint 作为 v5 新基线，而不是继续加 LR / 加轮数。

## 5. 复现

完整命令、逐 family 表、日志与 checksum 见
[`results/protocol_v5/train_v5_finetune_lr1e4_20261030/README.md`](../../results/protocol_v5/train_v5_finetune_lr1e4_20261030/README.md)。

```bash
python3 -m ipd self-test --candidates rl-v5-ft
python3 -m ipd protocol eval --candidates rl-v5-ft \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v5/protocol_v5_rl_v5_ft_dev.json
```
