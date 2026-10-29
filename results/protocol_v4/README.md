# Protocol v4 test_v7（provisional）— 2026-10-27

> **不是 v4 baseline，也不是 final。** `test_v7` 永久锁定；本次只做 provisional 诊断，不写 usage ledger，`selection_eligible=false`。

- 候选：24（20 个注册候选 + 4 个 PSRO，分批运行后合并）
- 规模：50 fields × 2048，100 天
- seed：20261027
- 结果：`protocol_v4_test_v7_all24_provisional.json`
- v4 top 5 dev 已运行（`RL-PPO` / `RL-Repair` / `RL-Pattern` /
  `RL-Wide` / `RL-Wide160`），见 `results/rl_v4_20261028/`、
  `results/rl_v4_20261029/` 和 `results/rl_v4_20261032/`；全候选
  dev baseline 仍未运行

## 排名

| # | 候选 | mean | min field | p05 | worst family | min_norm vs RL-PPO |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `RL-Repair` | **486.16** | 480.0 | 480.41 | 121.15 | 0.5523 |
| 2 | `Resp3-base` | **480.14** | 475.1 | 475.54 | 129.71 | 0.7167 |
| 3 | `RL-PPO-Hard60` | **479.95** | 473.3 | 473.85 | 136.26 | 0.7654 |
| 4 | `Final-base` | **477.55** | 470.9 | 472.20 | 130.11 | 0.7267 |
| 5 | `RL-PPO` | **476.98** | 469.1 | 471.49 | 138.73 | 1.0000 |
| 6 | `Resp3-tuned` | **476.64** | 470.8 | 471.69 | 130.12 | 0.7834 |
| 7 | `Adv-RL` | **476.15** | 467.4 | 470.68 | 131.71 | 0.8977 |
| 8 | `Bayes-base` | **472.44** | 463.1 | 465.42 | 128.36 | 0.7580 |
| 9 | `PSRO-Meta` | **472.39** | 465.7 | 465.79 | 130.55 | 0.7802 |
| 10 | `Final-tuned` | **472.39** | 464.9 | 466.58 | 130.14 | 0.7685 |
| 11 | `Triple-majority` | **471.68** | 464.6 | 465.42 | 131.71 | 0.7994 |
| 12 | `Neural` | **471.20** | 463.0 | 465.06 | 131.01 | 0.8133 |
| 13 | `GRU` | **470.82** | 462.7 | 464.03 | 131.54 | 0.7984 |
| 14 | `Final+Resp3` | **470.76** | 464.4 | 465.57 | 130.47 | 0.7922 |
| 15 | `Adv-MLP` | **470.51** | 464.4 | 464.62 | 131.08 | 0.8276 |
| 16 | `PSRO-BR` | **470.48** | 461.3 | 464.27 | 138.93 | 0.5294 |
| 17 | `Bayes-tuned` | **470.07** | 460.4 | 462.90 | 127.65 | 0.7252 |
| 18 | `Adv-GRU` | **469.27** | 462.9 | 464.02 | 131.63 | 0.7919 |
| 19 | `SelfPlay-GRU` | **466.05** | 457.8 | 459.14 | 137.38 | 0.3866 |
| 20 | `Harvest-tuned` | **462.47** | 454.8 | 456.70 | 130.30 | 0.7381 |
| 21 | `Harvest45` | **449.94** | 444.0 | 444.56 | 130.12 | 0.5546 |
| 22 | `PSRO-Scratch-BR` | **448.53** | 439.2 | 441.36 | 116.77 | 0.3726 |
| 23 | `PSRO-Scratch` | **448.50** | 438.8 | 441.30 | 116.46 | 0.3729 |
| 24 | `GapStart` | **383.85** | 372.9 | 375.00 | 41.98 | 0.0866 |

## 解释

- `mean` 是 50 个 field 的均分；`worst family` 是 test_v7 内最差 family 的原始分。
- `min_norm` 是该候选每个 family / RL-PPO 同 family 的最小值；跨 family 原始分不可比，故用归一化。
- 所有头部候选都在 `early_harvest` 崩到 121–139；RL-Repair 均值第一但 min_norm 只有 0.5523，不能只用均值选模。
- test_v7 结果不参与排名或调参；选模仍应使用 `dev_param` / `dev_family` / `dev_complex` / `dev_axelrod` / `dev_ood`。

## GPU sample v5（2026-10-28）

`protocol_v4_test_v7_all24_gpu_sample_v5.json` 是 opt-in
`--sample-rng per-match` 的全量结果（22:45）。它与上表 v4 sample 的
GPU 支持候选差异 <0.3 分；GPU argmax 差异 10–17 分。
详见 `results/eval_gpu_20261028/README_v5.md` 与
`docs/research/plan5-gpu-sample.md`。默认 v4 表不变。

## Plan 6：v5 sample 调度与 Python 瓶颈（2026-10-28）

Plan 6 的性能优化保持**分数不变**：24/24 候选与 Plan 5 v5 baseline
逐位一致。`Adv-*` 尝试过 GPU sample，但因会切换到 v5 per-match RNG、
改变分数（+0.10/+0.23/+0.25）而回退到 CPU。

| 配置 | wall |
|---|---:|
| Plan 5 v5 baseline（8 GPU / 12 CPU，fb2） | 1365.63s |
| 仅 policy 对手上 GPU（8/12，fb2） | 1310.57s |
| Plan 6 exact，native 前（4 GPU / 16 CPU，fb4） | 1057.82s |
| **Plan 6 exact + native（4 GPU / 16 CPU，fb4）** | **272.77s** |

- `Adv-*`、PSRO meta、规则/Axelrod 仍走 CPU；policy 对手 GPU 化只影响
  原本就在 GPU 的候选，且分数逐位一致。
- 6.6 的严格同序 C++ policy 前向把 wall 进一步降到 272.77s（5.01×），
  24/24 候选仍逐位一致；621.94s 属于被回退的 `Adv-*` 改 RNG 实验，
  不作为结果。
- dev 五层 argmax 是本轮迁移的一次性诊断（历史记录）：695.57s →
  **475.38s**，24 候选与 argmax baseline 逐位一致；不能与 sample 表
  混用，也不作为长期 gate
  （`plan6_migration_diagnostic_dev_argmax_w8fb2.json`）。
- 明细与复现见 `docs/research/plan6-python-bottleneck.md`。
