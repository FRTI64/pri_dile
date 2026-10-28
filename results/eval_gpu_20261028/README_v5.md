# GPU sample v5 验证（2026-10-28）

> v5 是 opt-in per-match RNG；不是 v4 的 bitwise 等价，不能与 v4 sample 表混排名。

- 用时：1365.63 秒；sample_rng=per-match；gpu=True

## v5 GPU sample 排名

| # | 候选 | mean | 相对于 v4 Δ | 路径 |
|---:|---|---:|---:|---|
| 1 | `RL-Repair` | **486.31** | +0.16 | GPU v5 |
| 2 | `Resp3-base` | **480.07** | -0.08 | CPU v4 |
| 3 | `RL-PPO-Hard60` | **479.92** | -0.03 | GPU v5 |
| 4 | `Final-base` | **477.57** | +0.03 | CPU v4 |
| 5 | `RL-PPO` | **477.26** | +0.28 | GPU v5 |
| 6 | `Resp3-tuned` | **476.82** | +0.18 | CPU v4 |
| 7 | `Adv-RL` | **476.11** | -0.04 | CPU v4 |
| 8 | `Bayes-base` | **472.44** | +0.01 | CPU v4 |
| 9 | `Final-tuned` | **472.36** | -0.02 | CPU v4 |
| 10 | `PSRO-Meta` | **472.29** | -0.10 | CPU v4 |
| 11 | `Triple-majority` | **471.54** | -0.14 | CPU v4 |
| 12 | `Neural` | **470.95** | -0.25 | GPU v5 |
| 13 | `GRU` | **470.74** | -0.07 | GPU v5 |
| 14 | `Final+Resp3` | **470.68** | -0.08 | CPU v4 |
| 15 | `PSRO-BR` | **470.41** | -0.07 | GPU v5 |
| 16 | `Adv-MLP` | **470.32** | -0.19 | CPU v4 |
| 17 | `Bayes-tuned` | **470.06** | -0.00 | CPU v4 |
| 18 | `Adv-GRU` | **469.16** | -0.11 | CPU v4 |
| 19 | `SelfPlay-GRU` | **465.92** | -0.13 | GPU v5 |
| 20 | `Harvest-tuned` | **462.45** | -0.02 | CPU v4 |
| 21 | `Harvest45` | **449.77** | -0.17 | CPU v4 |
| 22 | `PSRO-Scratch` | **448.51** | +0.01 | CPU v4 |
| 23 | `PSRO-Scratch-BR` | **448.36** | -0.18 | GPU v5 |
| 24 | `GapStart` | **383.86** | +0.01 | CPU v4 |

## 仅 GPU v5 支持的候选：v4 sample vs v5 sample vs argmax

| 候选 | v4 sample | v5 sample | v5−v4 | argmax | argmax−v4 |
|---|---:|---:|---:|---:|---:|
| `GRU` | 470.82 | 470.74 | -0.07 | 457.42 | -13.40 |
| `Neural` | 471.20 | 470.95 | -0.25 | 463.19 | -8.01 |
| `PSRO-BR` | 470.48 | 470.41 | -0.07 | 467.42 | -3.06 |
| `PSRO-Scratch-BR` | 448.53 | 448.36 | -0.18 | 448.34 | -0.19 |
| `RL-PPO` | 476.98 | 477.26 | +0.28 | 466.66 | -10.31 |
| `RL-PPO-Hard60` | 479.95 | 479.92 | -0.03 | 463.29 | -16.66 |
| `RL-Repair` | 486.16 | 486.31 | +0.16 | 484.68 | -1.47 |
| `SelfPlay-GRU` | 466.05 | 465.92 | -0.13 | 466.03 | -0.02 |

## 结论

- v5 sample 与 v4 sample 的差异在 0–2 分量级；argmax 的差异是 10–17 分。
- v5 恢复了随机合作/修复，不再把 sample 训练的策略压成确定性策略。
- `Adv-*` 与 PSRO meta 仍走 CPU v4；规则/Axelrod 不变。
- v5 需要单独基线；test_v7 仍为 provisional。
