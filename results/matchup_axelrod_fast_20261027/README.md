# Axelrod-fast（224 strategies）对战（2026-10-27）

- 候选：24 个（含 PSRO）
- 对手：224 个 Axelrod fast strategies（剔除 19 个 long-run Meta）
- 重复：16 次；每场 100 天
- 种子：20261024
- 用时：228.36 秒

## 排名

| # | 候选 | 等权均值 |
|---:|---|---:|
| 1 | `RL-Repair` | **575.39** |
| 2 | `RL-PPO-Hard60` | **574.95** |
| 3 | `Adv-RL` | **572.79** |
| 4 | `Final-tuned` | **571.16** |
| 5 | `Final+Resp3` | **571.12** |
| 6 | `RL-PPO` | **571.07** |
| 7 | `Triple-majority` | **569.37** |
| 8 | `GRU` | **568.69** |
| 9 | `Adv-GRU` | **568.58** |
| 10 | `Neural` | **568.47** |
| 11 | `Adv-MLP` | **567.42** |
| 12 | `Final-base` | **565.21** |
| 13 | `PSRO-BR` | **565.07** |
| 14 | `Resp3-tuned` | **564.56** |
| 15 | `SelfPlay-GRU` | **564.27** |
| 16 | `Resp3-base` | **561.54** |
| 17 | `PSRO-Meta` | **556.76** |
| 18 | `Harvest-tuned` | **553.47** |
| 19 | `Bayes-tuned` | **549.10** |
| 20 | `Bayes-base` | **548.42** |
| 21 | `Harvest45` | **532.40** |
| 22 | `PSRO-Scratch-BR` | **521.57** |
| 23 | `PSRO-Scratch` | **520.37** |
| 24 | `GapStart` | **442.97** |

## 说明

- 完整 Axelrod 243 含 19 个 long-run Meta；单场 >10 分钟，未纳入本次全候选测试。
- `family=axelrod`，因此 9 suite 加权与类型等权相同。
- 文件：`matchup.csv`、`matchup_wide.csv`、`matchup_scores.csv`、`matchup_families.csv`、`matchup_suite_weighted.csv`、`matchup.json`。
