# Protocol v4 全 catalog 对战（2026-10-27）

- 候选：24 个（含 PSRO 4 个）
- 对手：977 个 catalog config（38 family）
- 每 config 重复：16 次；每场 100 天
- 随机种子：20261024
- 用时：997.55 秒
- 对局总数：24 × 977 × 16

## 总排名（每个 config 等权）

| # | 候选 | config 等权 | family 等权 | train | dev | test | anchor |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `RL-Repair` | **505.76** | 479.95 | 505.35 | 535.76 | 497.71 | 486.21 |
| 2 | `RL-PPO-Hard60` | **501.87** | 478.85 | 493.36 | 530.39 | 491.72 | 485.81 |
| 3 | `Adv-RL` | **500.79** | 476.75 | 493.52 | 530.36 | 491.12 | 488.84 |
| 4 | `Triple-majority` | **499.08** | 477.16 | 483.30 | 527.62 | 478.90 | 488.44 |
| 5 | `Final-tuned` | **498.09** | 475.77 | 485.62 | 518.76 | 479.27 | 488.10 |
| 6 | `Final+Resp3` | **498.05** | 475.62 | 484.17 | 520.14 | 479.92 | 488.01 |
| 7 | `Neural` | **497.31** | 473.69 | 482.50 | 520.06 | 485.75 | 488.36 |
| 8 | `Resp3-tuned` | **496.84** | 480.27 | 480.87 | 529.51 | 479.81 | 487.96 |
| 9 | `Adv-GRU` | **496.78** | 472.80 | 483.48 | 526.49 | 487.08 | 488.45 |
| 10 | `GRU` | **496.31** | 472.93 | 481.80 | 523.50 | 485.53 | 488.45 |
| 11 | `Adv-MLP` | **495.45** | 473.10 | 480.91 | 524.31 | 488.44 | 488.36 |
| 12 | `RL-PPO` | **494.50** | 469.46 | 485.38 | 518.80 | 487.52 | 485.31 |
| 13 | `Resp3-base` | **494.44** | 474.55 | 486.80 | 524.62 | 489.42 | 485.41 |
| 14 | `Final-base` | **493.84** | 474.99 | 482.73 | 519.22 | 485.45 | 487.43 |
| 15 | `PSRO-BR` | **491.14** | 465.71 | 480.43 | 514.25 | 489.50 | 488.94 |
| 16 | `PSRO-Meta` | **488.11** | 465.36 | 476.19 | 511.28 | 483.52 | 486.80 |
| 17 | `SelfPlay-GRU` | **487.98** | 460.10 | 477.07 | 508.07 | 486.85 | 484.99 |
| 18 | `Bayes-base` | **487.95** | 472.46 | 479.12 | 525.38 | 488.86 | 491.04 |
| 19 | `Bayes-tuned` | **486.41** | 469.79 | 476.91 | 525.05 | 486.35 | 491.21 |
| 20 | `Harvest-tuned` | **481.76** | 462.73 | 470.41 | 509.55 | 476.24 | 481.05 |
| 21 | `Harvest45` | **477.40** | 451.89 | 479.12 | 522.60 | 483.43 | 473.43 |
| 22 | `PSRO-Scratch` | **448.69** | 443.91 | 413.75 | 475.29 | 446.30 | 491.11 |
| 23 | `PSRO-Scratch-BR` | **448.43** | 443.55 | 414.01 | 470.58 | 446.34 | 491.41 |
| 24 | `GapStart` | **419.20** | 384.64 | 449.56 | 519.58 | 465.01 | 453.93 |

## 家族冠军

| family | 最佳候选 | 分数 |
|---|---|---:|
| adaptive | `Final+Resp3` | 593.12 |
| adaptive_mix | `Final+Resp3` | 314.76 |
| adaptive_probe | `Bayes-base` | 665.67 |
| anchor | `PSRO-Scratch-BR` | 491.41 |
| axelrod | `RL-Repair` | 576.24 |
| axelrod_extra | `RL-PPO-Hard60` | 575.85 |
| axelrod_mix | `Final-base` | 558.10 |
| conditional_extort | `Adv-MLP` | 343.77 |
| contrition_cycle | `GapStart` | 596.00 |
| defect | `RL-Repair` | 744.99 |
| delayed_extort | `GapStart` | 496.49 |
| delayed_grim | `Adv-GRU` | 604.00 |
| early_harvest | `GapStart` | 179.00 |
| early_probe | `Final-base` | 588.00 |
| endgame_hunter | `Adv-RL` | 399.00 |
| extort | `RL-Repair` | 417.64 |
| forgiving | `GapStart` | 616.95 |
| frequent_kamikaze | `Harvest-tuned` | 504.00 |
| hyper_extort | `RL-Repair` | 207.55 |
| late | `GapStart` | 554.20 |
| lookahead | `GapStart` | 583.78 |
| memory_two | `Adv-GRU` | 595.00 |
| meta_mix | `RL-Repair` | 553.99 |
| mirror_adaptive | `Adv-GRU` | 595.00 |
| nice | `PSRO-Meta` | 602.29 |
| noise_adaptive | `Adv-GRU` | 595.00 |
| noisy | `RL-Repair` | 466.49 |
| pattern_breaker | `PSRO-Scratch` | 830.89 |
| phase_switch | `GapStart` | 445.96 |
| policy | `Adv-GRU` | 594.98 |
| portfolio | `PSRO-BR` | 517.37 |
| probe_exploit | `GapStart` | 585.00 |
| probe_harvest | `GapStart` | 464.52 |
| punitive | `Adv-GRU` | 601.00 |
| random | `PSRO-Scratch-BR` | 531.26 |
| reputation_grim | `Harvest-tuned` | 695.00 |
| temporal | `Harvest45` | 491.40 |
| wild | `GapStart` | 495.87 |

## 头部候选的最弱 family

| 候选 | 最弱 4 个 family |
|---|---|
| `RL-Repair` | early_harvest 126.3, frequent_kamikaze 170.7, hyper_extort 207.6, probe_exploit 219.9 |
| `RL-PPO-Hard60` | early_harvest 135.0, hyper_extort 152.1, frequent_kamikaze 170.0, probe_exploit 205.8 |
| `Adv-RL` | early_harvest 132.7, hyper_extort 143.9, probe_exploit 191.7, frequent_kamikaze 203.7 |
| `Triple-majority` | early_harvest 130.2, hyper_extort 133.4, probe_exploit 162.6, frequent_kamikaze 262.8 |
| `Final-tuned` | early_harvest 130.3, hyper_extort 131.8, probe_exploit 174.1, frequent_kamikaze 264.7 |
| `Final+Resp3` | early_harvest 130.3, hyper_extort 133.3, probe_exploit 171.2, frequent_kamikaze 264.7 |
| `Neural` | hyper_extort 126.6, early_harvest 131.3, probe_exploit 168.5, frequent_kamikaze 204.5 |
| `Resp3-tuned` | early_harvest 130.0, hyper_extort 132.1, probe_exploit 161.6, adaptive_mix 300.2 |
| `Adv-GRU` | hyper_extort 125.7, early_harvest 133.3, probe_exploit 167.9, frequent_kamikaze 226.3 |
| `GRU` | hyper_extort 125.7, early_harvest 133.3, probe_exploit 167.9, frequent_kamikaze 265.5 |

## 说明

- 这是逐 config 等权覆盖，不等于 Protocol v4 的 field 抽样权重；分数不能与 158 类 legacy / 9 suite 直接混表。
- catalog 包含 train/dev/test/anchor/policy/Axelrod；test-only family（early_harvest、hyper_extort、frequent_kamikaze 等）对所有头部候选都很苛刻。
- 文件：`matchup.csv`、`matchup_wide.csv`、`matchup_scores.csv`、`matchup_families.csv`、`matchup_suite_weighted.csv`、`matchup.json`。
