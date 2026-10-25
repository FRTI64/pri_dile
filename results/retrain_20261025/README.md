# 2026-10-25 全策略重训与对抗训练

本目录保存一次从现有规则策略出发的完整重训产物：MLP/GRU 行为克隆 +
DAgger、三类约束对抗调参、PPO（BC + hard、从零、从零 + hard）。所有新权重
与配置单独存放，未覆盖仓库中原有已提交版本。

## 环境

- 24 核 CPU，31 GiB 内存
- Python 3.14 + PyTorch 2.11.0+cu128
- 本机 CUDA 不可用，所有训练在 CPU 上完成
- 训练时缺少 NumPy，仅触发 PyTorch 警告，不影响训练/推理

## 1. 现有规则策略作为教师

教师为 `Triple-majority`：Final-tuned + Responsive3-tuned + BayesPlus-tuned
多数投票。训练配置与仓库原实验一致：4 个官方分布 × 12 场/分布，
60 对手/场，DAgger 1 轮（6 场/分布），40 epochs。

```bash
.venv/bin/python -m ipd train neural --device cpu --workers 20 \
  --seed 20261025 \
  --out weights/retrain_20261025/neural_policy_weights.json \
  --history-out results/retrain_20261025/neural_training_history.json

.venv/bin/python -m ipd train gru --device cpu --workers 20 \
  --seed 20261025 \
  --out weights/retrain_20261025/gru_policy_weights.json \
  --history-out results/retrain_20261025/gru_training_history.json
```

| 学生 | 参数量 | 验证 BCE | 动作一致率 | P(D) | R(D) | 温度 |
|---|---:|---:|---:|---:|---:|---:|
| MLP 64-32 | 7,617 | 0.021970 | 99.216% | 0.9676 | 0.9819 | 0.95 |
| GRU(32)-32 | 12,513 | 0.078435 | 97.336% | 0.9686 | 0.9833 | 1.025 |

## 2. 约束对抗训练

对固定 base 做黑盒搜索：48 个候选，4 个搜索场/分布，前 6 名在
12 场/分布 × 299 对手上验证；只接受四个分布均不下降的配置。

```bash
# MLP / GRU / RL BC base
.venv/bin/python -m ipd train adversarial --arch mlp \
  --weights weights/retrain_20261025/neural_policy_weights.json \
  --seed 20261026 \
  --out configs/retrain_20261025/adversarial_policy_aggressive.json \
  --results results/retrain_20261025/adversarial_search_results.json

.venv/bin/python -m ipd train adversarial --arch gru \
  --weights weights/retrain_20261025/gru_policy_weights.json \
  --seed 20261026 \
  --out configs/retrain_20261025/adversarial_gru_policy_aggressive.json \
  --results results/retrain_20261025/adversarial_gru_search_results.json

.venv/bin/python -m ipd train adversarial --arch rl \
  --weights weights/retrain_20261025/rl_policy_weights_bc.json \
  --seed 20261026 \
  --out configs/retrain_20261025/adversarial_rl_policy_aggressive.json \
  --results results/retrain_20261025/adversarial_rl_search_results.json
```

三个 base 都选中了同一组参数：

```text
gap=0.05, min_c=12, min_d=2, late_day=50, repair=false
```

验证集四分布均值相对 base 的提升：

| Base | base 均值 | 对抗后均值 | 最差分布 Δ |
|---|---:|---:|---:|
| MLP | 551.01 | 555.39 | +3.20 |
| GRU | 550.95 | 555.12 | +3.20 |
| RL BC | 553.60 | 557.08 | +2.47 |

## 3. PPO 强化学习

阶段 1（BC 初始化 + wide 加权 + KL）沿用 22:57 已完成的
`weights/rl_policy_*_stage1.json`，并复制到本目录。后续阶段从它继续：

```bash
# 阶段 2：BC + hard（Adv-RL base）
.venv/bin/python -m ipd train rl --device cpu --iterations 60 \
  --fields-per-dist 5 --n 80 \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 --ent-coef 0.0001 --lr 1e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --val-fields-per-dist 4 --val-n 100 --workers 16 \
  --init-weights weights/retrain_20261025/rl_policy_checkpoint_stage1.json \
  --select-metric blind \
  --out weights/retrain_20261025/rl_policy_weights_bc.json \
  --checkpoint-out weights/retrain_20261025/rl_policy_checkpoint_bc.json \
  --history-out results/retrain_20261025/rl_training_history_bc.json

# 阶段 3：从零 base
.venv/bin/python -m ipd train rl --device cpu --from-scratch \
  --iterations 30 --fields-per-dist 5 --n 80 \
  --dist-weights 1,1,1,2 --ent-coef 0.001 --lr 3e-4 \
  --center-advantages --select-metric avg \
  --val-fields-per-dist 3 --val-n 100 --workers 20 \
  --init-weights weights/retrain_20261025/neural_policy_weights.json \
  --out weights/retrain_20261025/rl_policy_weights_scratch_base.json \
  --checkpoint-out weights/retrain_20261025/rl_policy_checkpoint_scratch_base.json \
  --history-out results/retrain_20261025/rl_training_history_scratch_base.json

# 阶段 4：从零 + hard（最终单独 RL）
.venv/bin/python -m ipd train rl --device cpu --iterations 40 \
  --fields-per-dist 5 --n 80 \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 --ent-coef 0.0001 --lr 1e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --select-metric blind --val-fields-per-dist 3 --val-n 100 --workers 20 \
  --init-weights weights/retrain_20261025/rl_policy_checkpoint_scratch_base.json \
  --out weights/retrain_20261025/rl_policy_weights.json \
  --checkpoint-out weights/retrain_20261025/rl_policy_checkpoint.json \
  --history-out results/retrain_20261025/rl_training_history.json
```

| 阶段 | best iter | 验证四分布均值 | 最差分布 | 备注 |
|---|---:|---:|---:|---|
| stage1（复用） | 95 | 551.77 | 544.12 | BC + wide + KL |
| stage2 BC + hard | 15 | 552.38 | 546.12 | Adv-RL base |
| stage3 从零 base | 25 | 554.09 | 542.52 | 30 iter |
| stage4 从零 + hard | 20 | 556.47 | 545.11 | 最终单独 RL |

## 4. 统一评估

```bash
.venv/bin/python results/retrain_20261025/evaluate.py \
  --benchmark-fields 20 --blind-fields 10 --n 299
```

该脚本用同一批 `CandidateSpec` 跑四分布和盲测；盲测的对手循环赛每个赛场只
计算一次，由所有候选共享。结果写入 `benchmark_all.json` 和 `blind_all.json`。

### 4.1 四分布（20 场 × 299 对手，种子 20261026）

| 候选 | 原始混合 | 末期收割 | 高 Grim | 宽+聪明 | 均值 | 旧版本均值 |
|---|---:|---:|---:|---:|---:|---:|
| RL-final-retrain | 555.24 | 547.71 | 551.78 | 556.82 | 552.89 | 553.33 |
| RL-BC-retrain | 554.15 | 544.54 | 550.44 | 553.50 | 550.66 | 550.50 |
| RL-scratch-base-retrain | 554.02 | 547.51 | 551.92 | 554.60 | 552.01 | 551.87 |
| RL-stage1-retrain | 551.66 | 543.81 | 550.28 | 552.44 | 549.55 | 549.55 |
| **Adv-RL-retrain** | **556.64** | **549.25** | **553.41** | **557.30** | **554.15** | **554.29** |
| Adv-RL-conservative-retrain | 555.88 | 549.10 | 553.41 | 555.17 | 553.39 | 553.73 |
| Adv-RL-final-retrain | 555.34 | 548.41 | 552.47 | 556.33 | 553.14 | — |
| Adv-MLP-retrain | 552.68 | 548.03 | 553.33 | 554.03 | 552.02 | 552.24 |
| Adv-GRU-retrain | 551.99 | 547.85 | 553.29 | 553.76 | 551.72 | 551.69 |
| Neural-retrain | 549.45 | 542.74 | 550.19 | 549.34 | 547.93 | 548.15 |
| GRU-retrain | 549.24 | 542.72 | 550.23 | 549.22 | 547.85 | 547.81 |

### 4.2 盲测（10 个未见赛场 × 299 对手，种子 20261025）

| 候选 | 均分 | 平均排名 | 第 1 名 | 前 10 | 最差 | 旧版本均分 |
|---|---:|---:|---:|---:|---:|---:|
| **RL-final-retrain** | **562.22** | **8.10** | 7/10 | 7/10 | 25 | 563.23 |
| **Adv-RL-retrain** | **561.46** | **8.10** | 7/10 | 7/10 | 25 | 562.91 |
| Adv-RL-conservative-retrain | 560.56 | 8.20 | 7/10 | 7/10 | 26 | 561.05 |
| Adv-RL-final-retrain | 560.67 | 8.10 | 7/10 | 7/10 | 25 | — |
| Adv-GRU-retrain | 559.18 | 8.10 | 7/10 | 7/10 | 25 | 559.05 |
| RL-scratch-base-retrain | 559.17 | 8.30 | 7/10 | 7/10 | 27 | 559.27 |
| RL-BC-retrain | 558.93 | 8.30 | 7/10 | 7/10 | 27 | 558.60 |
| Adv-MLP-retrain | 558.57 | 8.20 | 7/10 | 7/10 | 26 | 559.51 |
| RL-stage1-retrain | 557.95 | 8.30 | 7/10 | 7/10 | 27 | 557.95 |
| GRU-retrain | 555.19 | 8.30 | 7/10 | 7/10 | 27 | 555.22 |
| Neural-retrain | 554.00 | 8.30 | 7/10 | 7/10 | 27 | 555.10 |

## 5. 结论

本轮完整重训成功复现了原有训练流程，但没有超过已有最好版本：

- 四分布最高仍是 **Adv-RL**（新 554.15，旧 554.29，差 0.14）；
- 盲测最高仍是 **最终单独 RL**（新 562.22，旧 563.23，差 1.01）；
- MLP/GRU/Adv-MLP/Adv-GRU 与旧版本均在 ±0.5 分内；
- stage1 与旧版完全一致，说明复用和数据处理流程可复现；
- 三套对抗搜索都选择了同一组参数：
  `gap=0.05, min_c=12, min_d=2, late_day=50, repair=false`。

因此没有覆盖官方 `weights/` 和 `configs/`；新产物只保留在本目录，供复现和后续
对比。若要提交，仍建议使用仓库中原有的 RL-PPO / Adv-RL 权重和配置。

