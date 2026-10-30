# train_v5 定向 fine-tune（2026-10-30）

> **结论：可复现的小幅正收益，但不 promotion。**
> 从 `RL-Wide160` 出发，只用 Protocol v5 的 `train_v5` profile 做短程
> PPO fine-tune。两个独立 dev seed 与 `test_v8` provisional 都优于
> RL-Wide160：`dev_param +0.49～0.52`、`dev_complex +4.8～4.9`、
> `dev_axelrod +1.0`、`dev_ood +1.6～1.7`、`test_v8 +0.66`。
> 但 `probe_exploit` 仍只有约 115（RL-PPO 124），`min_norm` 0.927
> （低于旧 0.99 门槛），因此只作为实验候选 `rl-v5-ft`，不替换主权重、
> 不消费 `test_v8`。

## 1. 动机

现有 `RL-Wide160` 是用 v4 训练分布 + `train_pattern_breaker` 得到的。
Protocol v5 重新划分 train/dev/test 后，先用纯 `train_v5`（标准 +
complex + policy + Axelrod + anchors）做一次短程 fine-tune，检查：

1. v5 train 是否能提升 `dev_complex` / `dev_axelrod` / `dev_ood`；
2. 增益是否能延续到 `test_v8`；
3. 是否能在不破坏安全规则的前提下改善 `probe_exploit`。

## 2. 训练配方

起点：`weights/rl_policy_checkpoint_wide160.json`（160/128）。
训练分布：`train_v5`，不加入 `train_probe_pattern` 或其他定向 profile。

```bash
python3 -m ipd train rl --device cuda \
  --iterations 12 --fields-per-dist 3 --n 160 --workers 20 \
  --seed 20261061 \
  --hidden 160 128 \
  --init-weights weights/rl_policy_checkpoint_wide160.json \
  --train-dists train_v5 \
  --dists dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --val-fields-per-dist 3 --val-n 160 --val-every 4 \
  --select-metric balanced \
  --save-all-checkpoints results/protocol_v5/train_v5_finetune_lr1e4_20261030/ckpts \
  --bc-kl-coef 0.1 --lr 1e-4 --ent-coef 0.01 --target-kl 0.02 \
  --out results/protocol_v5/train_v5_finetune_lr1e4_20261030/train_v5_ft_lr1e4_weights.json \
  --checkpoint-out results/protocol_v5/train_v5_finetune_lr1e4_20261030/train_v5_ft_lr1e4_checkpoint.json \
  --history-out results/protocol_v5/train_v5_finetune_lr1e4_20261030/train_v5_ft_lr1e4_history.json
```

关键设置：

- `lr=1e-4`，`bc-kl=0.1`，`target-kl=0.02`，安全规则保持开启；
- 每 4 iter 用 5 个 dev tier × 3 fields × 160 选模；
- 训练内最佳 checkpoint 为 **iter 8**（`ckpts/iter_008.json`）；
- `--out` 的 deployable weights 与 iter8 checkpoint 逐权重一致
  （max abs diff = 0）。

先跑过一版 `lr=3e-5` 的 16 iter fine-tune（`train_v5_finetune_20261030/`），
两个 dev seed 只有 +0.3～0.5 的 `dev_param` 增益；提高到 `1e-4` 后
complex/OOD 增益明显变大，因此保留本目录结果。

## 3. Protocol v5 dev（30 fields × 2048）

### 3.1 seed 20261027

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **TrainV5-lr1e4-iter8** | **532.18** | **583.14** | **482.64** | **568.74** | **523.94** |
| RL-Wide160 | 531.66 | 583.08 | 477.77 | 567.77 | 522.24 |
| RL-PPO | 525.99 | 582.55 | 457.26 | 563.43 | 513.30 |

### 3.2 seed 20261028

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **TrainV5-lr1e4-iter8** | **531.08** | **582.68** | **483.81** | **568.59** | **522.07** |
| RL-Wide160 | 530.59 | 582.62 | 479.01 | 567.56 | 520.49 |
| RL-PPO | 525.09 | 582.15 | 458.73 | 563.09 | 511.72 |

相对 RL-Wide160：

| seed | dev_param | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|
| 20261027 | +0.52 | +4.87 | +0.97 | +1.70 |
| 20261028 | +0.49 | +4.80 | +1.03 | +1.58 |

### 3.3 关键 family（seed 27，dev_complex）

| family | TrainV5-lr1e4-iter8 | RL-Wide160 |
|---|---:|---:|
| `pattern_breaker` | **683.15** | 634.02 |
| `adaptive_mix` | **317.41** | 307.79 |
| `conditional_extort` | **326.21** | 324.79 |
| `probe_exploit` | 114.82 | 115.67 |

`pattern_breaker` +49.1、`adaptive_mix` +9.6；`probe_exploit` 反而 -0.85，
因此 `dev_complex` / `dev_ood` 的 `min_norm` 从 0.934 降到 **0.927**
（瓶颈仍是 probe，不是 PB）。

## 4. test_v8 provisional（50 fields × 2048）

`--no-consume`，不写 usage ledger、不参与选模。

| 候选 | test_v8 | min_norm | worst family | p05 |
|---|---:|---:|---:|---:|
| **TrainV5-lr1e4-iter8** | **496.37** | 0.865 | 128.91 | 488.59 |
| RL-Wide160 | 495.71 | 0.888 | 129.92 | 488.40 |
| RL-PPO | 487.83 | 1.000 | 138.77 | 480.34 |

相对 RL-Wide160 **+0.66**，相对 RL-PPO **+8.54**。
test_v8 上 `probe_exploit` 267.39 vs 264.56（+2.8），
但 `pattern_breaker` 392.87 vs 403.60（-10.7），
说明 dev 上的 PB 增益没有完全延续；总体均值仍小幅领先。

## 5. 判定

- **实验候选，不 promotion。** 两个 dev seed 和 test provisional 方向
  一致，收益不是单次噪声；
- 均值、complex、Axelrod、OOD 均有正收益，但 `probe_exploit` 仍是
  硬瓶颈：`min_norm ~0.927`，远低于旧 `0.99` 门槛；
- 已注册为实验候选 `rl-v5-ft`（部署权重
  `weights/rl_policy_weights_v5_ft.json`）；不替换
  `weights/rl_policy_weights.json`，不消费 `test_v8`；
- 若继续，优先在保持 `train_v5` 的前提下做 probe 定向修复，或把本
  checkpoint 作为新的 v5 基线，而不是继续加 LR / 加训练轮数。

## 6. 产物

- `train_v5_ft_lr1e4_weights.json`：iter8 的 deployable
  `ipd-neural-policy-v1` 权重，sha256
  `7d7a5665a0004212ab00201bf9ef7169454c002d060c98eda36298b50850402d`
- 已复制到注册权重 `weights/rl_policy_weights_v5_ft.json`；
  checkpoint 复制到 `weights/rl_policy_checkpoint_v5_ft.json`
- `train_v5_ft_lr1e4_checkpoint.json` / `ckpts/iter_008.json`：
  iter8 actor-critic checkpoint，sha256
  `17d122e27a8784c99be50ad80e522db5b40656434033f189ef1a97b856d3580a`
- `ckpts/iter_004.json`、`iter_012.json`：其他验证点
- `screen_dev.json`、`screen_dev_seed2.json`、`test_v8_provisional.json`
- `train.log`、`screen_dev*.log`、`test_v8_provisional.log`
- 评估 wrapper：`results/protocol_v5/eval_weights.py`

## 7. 复现命令

```bash
# dev seed 27 / 28
python3 results/protocol_v5/eval_weights.py \
  --specs results/protocol_v5/train_v5_finetune_lr1e4_20261030/screen_specs.json \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --seed 20261027 --baseline RL-PPO \
  --out results/protocol_v5/train_v5_finetune_lr1e4_20261030/screen_dev.json

# test_v8 provisional（不消费）
python3 results/protocol_v5/eval_weights.py \
  --specs results/protocol_v5/train_v5_finetune_lr1e4_20261030/test_provisional_specs.json \
  --tiers test_v8 --fields 50 --n 2048 --workers 20 --seed 20261027 \
  --baseline RL-PPO --no-consume \
  --out results/protocol_v5/train_v5_finetune_lr1e4_20261030/test_v8_provisional.json
```
