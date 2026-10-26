# Hard-opponent RL-PPO fine-tune — 2026-10-26

## 0. 结论

**推荐新主提交：A60（现有 RL-PPO + hard 权重 2，60 iterations，KL 0.1，lr 1e-4）。**

A60 相对当前 RL-PPO：

| 指标 | RL-PPO | A60 | Δ |
|---|---:|---:|---:|
| 官方四分布（20 fields × 299） | 553.33 | **554.36** | +1.03 |
| 官方 blind（20 fields） | 562.83 | **564.17** | +1.34 |
| 官方 blind 平均排名 | 5.80 | **5.75** | 略好 |
| held-out final（86 × 20） | 506.49 | **509.76** | +3.27 |
| held-out confirm（新 seeds） | 508.79 | **511.33** | +2.54 |
| held-out min_norm | 1.000 | **1.000** | 0 |
| E4 red blue mean | 306.95 | 305.95 | -1.00 |

第二 seed（20261031）也一致：

| 指标 | RL-PPO | A60 seed2 | Δ |
|---|---:|---:|---:|
| 官方四分布 | 553.33 | **554.12** | +0.79 |
| 官方 blind（20 fields） | 562.83 | **564.46** | +1.63 |
| held-out final | 506.49 | **508.95** | +2.46 |

A60 没有 family 崩塌；E4 red 基本持平（仍 100% 击败，anti-exploitation 没有解决）。
如果只看“未知对手、一次固定提交”的稳健性，A60 是比原 RL-PPO 更好的默认选择。

## 1. 为什么是 A60

A60 使用现有 `hard` 池（不是新的 `hard2`），只是：

```text
train-dists: original,late,grim30,wide,hard
dist-weights: 1,1,1,2,2   # 原来 hard 权重 1，现在 2
iterations: 60
bc-kl-coef: 0.1
lr: 1e-4
ent-coef: 0.0001
target-kl: 0.02
init: weights/rl_policy_checkpoint.json
select: avg
```

A、B、C 三组对比（seed 20261030）：

| 实验 | 训练池 | KL | lr | 官方 | blind20 | held-out final |
|---|---|---:|---:|---:|---:|---:|
| RL-PPO | - | - | - | 553.33 | 562.83 | 506.49 |
| A30 | hard×2 | 0.10 | 1e-4 | 553.79 | 563.88 | 510.11 |
| A60 | hard×2 | 0.10 | 1e-4 | **554.36** | **564.17** | **509.76** |
| B30 | hard2×2 | 0.10 | 1e-4 | 553.87 | 563.21 | 511.82 |
| C30 | hard2×2 | 0.05 | 5e-5 | 553.60 | 562.22 | 513.10 |

B/C 对 held-out 的均值更高，但 blind 略低；A60 是官方、blind、held-out 三者最均衡的。
在“未知对手、一次固定提交”的前提下，A60 比 B/C 更稳。

## 2. Held-out per-family

A60 vs RL-PPO（final 86×20）：

| family | RL-PPO | A60 | Δ |
|---|---:|---:|---:|
| good | 604.0 | 604.0 | 0 |
| top | 595.0 | 595.0 | 0 |
| noisy | 503.5 | 506.1 | +2.6 |
| extort | 359.0 | 379.3 | **+20.3** |
| random | 434.6 | 437.1 | +2.5 |
| special | 398.8 | 399.4 | +0.6 |
| heldout | 599.5 | 599.5 | 0 |

confirm 新 seeds 上同样没有回退：

| family | RL-PPO | A60 | Δ |
|---|---:|---:|---:|
| extort | 364.3 | 377.7 | +13.4 |
| noisy | 513.7 | 515.6 | +2.0 |
| random | 419.3 | 423.0 | +3.7 |
| special | 402.5 | 403.9 | +1.5 |
| good/top/heldout | 不变 | 不变 | 0 |

## 3. 官方四分布细节

A60：

```text
original 556.87 / late 549.01 / grim30 552.36 / wide 559.19
avg 554.36
```

RL-PPO：

```text
original 555.88 / late 548.41 / grim30 552.09 / wide 556.95
avg 553.33
```

四个分布全部上涨。

## 4. Backtest 文件

```text
weights/rl_policy_weights_hard2.json       # A60 可部署策略
weights/rl_policy_checkpoint_hard2.json    # A60 checkpoint
sha256:
  42c64c8a7524115f5ec17511a482dc56a2259cfcb6740ab5b223a193797ff60e  rl_policy_weights_hard2.json
  13276c70302d0d79e3eea3e034139262eeb896b8dbe5bea396f5722a7ed72ab5  rl_policy_checkpoint_hard2.json
```

原始实验产物：

```text
weights/hard2_20261026/A60_policy.json
weights/hard2_20261026/A60_checkpoint.json
weights/hard2_20261026/seed2/A60_policy.json
results/hard2_20261026/A60_benchmark_20fields.json
results/hard2_20261026/blind20/A60.json
results/hard2_20261026/compare_final_A60.json
results/hard2_20261026/compare_final_confirm_A60.json
results/hard2_20261026/compare_validation_A60.json
results/hard2_20261026/E4_red_vs_base_A60.json
results/hard2_20261026/seed2/A60_benchmark_20fields.json
results/hard2_20261026/seed2/A60_blind20.json
results/hard2_20261026/seed2/compare_final_A60.json
```

## 5. 复现命令

```bash
.venv/bin/python -m ipd train rl --device auto --seed 20261030 \
  --iterations 60 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,hard --dist-weights 1,1,1,2,2 \
  --init-weights weights/rl_policy_checkpoint.json \
  --bc-kl-coef 0.1 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --out weights/rl_policy_weights_hard2.json \
  --checkpoint-out weights/rl_policy_checkpoint_hard2.json \
  --history-out results/hard2_20261026/A60_history.json
```

评估：

```bash
.venv/bin/python -m ipd benchmark --candidates rl \
  --weights weights/rl_policy_weights_hard2.json --fields 20 --workers 20
.venv/bin/python -m ipd blind --candidates rl \
  --weights weights/rl_policy_weights_hard2.json --fields 20 --workers 10
```

## 6. 注意

- A60 没有解决针对性 red best response 的 anti-exploitation；E4 red 仍然 100% 击败。
  它只是在官方/blind/held-out 上更稳一点。
- 这次 final suite 已经用于 A/B/C 比较；严格来说它现在更像 validation。
  confirm 使用了新的 seed base（20761026），但仍是同一批 opponent 参数。
  如果要继续选模，必须再留一套完全新的 family/参数做最终 test。
- 是否把 `weights/rl_policy_weights.json` 替换成 A60，需要最终确认。
  当前已保留原 RL-PPO 权重，并新增 `weights/rl_policy_weights_hard2.json`。
