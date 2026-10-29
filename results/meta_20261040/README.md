# RL-Gate（2026-10-29）

> 学习型 match-level meta gate：前 10 回合后选择 RL-Wide160 或
> RL-Repair。**No-Go**：阈值 0.5 过度切换、complex -18.8；
> 阈值 0.9 退化为 RL-Wide160，test 只 +0.22。

## 结果

| 集 | RL-Wide160 | RL-Switch | RL-Gate t=.5 | RL-Gate t=.9 |
|---|---:|---:|---:|---:|
| dev_param | 548.26 | **548.58** | 548.76 | 548.30 |
| dev_complex | **454.31** | 454.24 | 435.55 | 454.63 |
| dev_ood | 521.37 | **521.42** | 516.07 | 521.34 |
| test_v7 | 485.62 | **488.26** | 485.67 | 485.84 |

训练：1052 个反事实样本，25 维 gate 特征，logistic regression；
train_acc 0.810、val_acc 0.816（阈值 0.9 重训后 0.816）。

## 文件

| 文件 | 内容 |
|---|---|
| `build_gate.py` | 数据生成 + gate 训练脚本 |
| `gate_dataset.json` | 1052 个反事实样本 |
| `protocol_v4_dev_rl_gate_final.json` | 阈值 0.9 官方 dev |
| `protocol_v4_test_v7_rl_gate_final_provisional.json` | 阈值 0.9 test provisional |
| `protocol_v4_dev_rl_gate.json` / `_t90.json` | 阈值 0.5 / 0.9 中间结果 |
| `protocol_v4_test_v7_rl_gate_provisional.json` / `_t90_...` | 中间 test |

Gate 权重：`weights/meta_gate_wide160_repair.json`（25 个系数）。
完整分析与复现见 `docs/research/rl-gate.md`。
