# RL-Gate（2026-10-29）

> 学习型 match-level meta gate：前 10 回合后选择 RL-Wide160 或
> RL-Repair。分类器版 **No-Go**；cost-sensitive 版 dev 接近手写
> RL-Switch，但 test 仍低 1.28，**不推荐 promotion**。

## 结果

| 集 | RL-Wide160 | RL-Switch | RL-Gate t=.5 | RL-Gate t=.9 | RL-Gate CS |
|---|---:|---:|---:|---:|---:|
| dev_param | 548.26 | 548.58 | 548.76 | 548.30 | **548.62** |
| dev_complex | 454.31 | 454.24 | 435.55 | 454.63 | **454.61** |
| dev_ood | 521.37 | **521.42** | 516.07 | 521.34 | 521.40 |
| test_v7 | 485.62 | **488.26** | 485.67 | 485.84 | 486.98 |

训练：1052 个反事实样本，25 维 gate 特征，logistic regression；
分类器 val_acc 0.816；cost-sensitive 版 val utility 518.96
（alwaysA 515.24，alwaysB 518.25），complex 样本权重 2.5、
margin 5。

## 文件

| 文件 | 内容 |
|---|---|
| `build_gate.py` | 数据生成 + gate 训练脚本 |
| `gate_dataset.json` | 1052 个反事实样本 |
| `protocol_v4_dev_rl_gate_cs.json` | **cost-sensitive 最终 dev** |
| `protocol_v4_test_v7_rl_gate_cs_provisional.json` | **cost-sensitive 最终 test** |
| `train_cost_sensitive_gate.py` | CS gate 训练脚本 |
| `meta_gate_classifier_t90.json` | 分类器版 gate（历史） |
| `protocol_v4_dev_rl_gate_final.json` / `..._t90...` | 分类器版中间结果 |

Gate 权重：

- `weights/meta_gate_cs_wide160_repair.json`（最终 `rl-gate` 使用）
- `weights/meta_gate_wide160_repair.json`（分类器版，历史）
完整分析与复现见 `docs/research/rl-gate.md`。
