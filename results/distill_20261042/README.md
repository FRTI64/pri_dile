# RL-Switch distillation（2026-10-29）

> 把 RL-Switch 蒸馏成单个 160/128 MLP。**No-Go**：dev_param 535.46
> vs RL-Switch 548.58，complex 442.43 vs 454.24。

## 结果

| 集 | RL-Wide160 | RL-Switch | Distill |
|---|---:|---:|---:|
| dev_param | 548.26 | **548.58** | 535.46 |
| dev_family | **583.08** | 582.91 | 582.28 |
| dev_complex | **454.31** | 454.24 | 442.43 |
| dev_axelrod | 582.38 | **582.86** | 571.99 |
| dev_ood | 521.37 | **521.42** | 511.24 |

训练：96,196 个非强制教师状态，BCE 0.19（4 epochs）。
失败原因是 match-level switch 无法被无状态 MLP 稳定表示，且 BC
存在分布偏移。

## 文件

- `distill_switch.py`：教师轨迹 + 学生训练脚本；
- `distill_switch_ckpt.json`：学生 actor-critic checkpoint；
- `distill_switch_dev_*.json`：逐 tier 结果。

完整分析见 `docs/research/distill-switch.md`。
