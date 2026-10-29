# RL-Gate：用机器学习学习上层 meta 开关（No-Go）

> **状态（2026-10-29）：基础设施 + 实验完成，效果 No-Go。**
> 用监督学习训练一个 match-level gate，在前 10 回合后选择
> RL-Wide160 或 RL-Repair。阈值 0.5 时会过度切换、dev_complex
> 掉 18.8；阈值校准到 0.9 后基本退化为 RL-Wide160，test 只
> +0.2，远不如手写 RL-Switch 的 +2.6。
> **不推荐 promotion；保留为实验候选。**

## 1. 动机

手写 RL-Switch 只用了“前 6 回合 mutual D + 对手最新 C”这一条规则。
一个自然的想法是让上层 meta 自己学：给定前 K 回合的对手统计，
预测“切到 RL-Repair 是否会在本场剩余回合得分更高”。

## 2. 方法

- 前缀：RL-Wide160 打 K=10 回合；
- 反事实：从同一前缀状态分别用 RL-Wide160 / RL-Repair 打完 100 回合；
- 标签：`score(Repair) - score(Wide) > 2`；
- 特征：25 维紧凑向量（D 率、最近 4 手、p_c/p_d、switch rate、
  period-2/3/4、run length、score diff、t）；
- 模型：logistic regression（25→1），BCE，85/15 划分；
- 训练对手：`train` / `train_complex` / `train_axelrod` /
  `extort_hard` / `hard`，共 1052 个反事实样本；
- 切换阈值：0.5（原始）和 0.9（dev 校准）。

训练脚本：`results/meta_20261040/build_gate.py`；
gate 权重：`weights/meta_gate_wide160_repair.json`。

## 3. 训练结果

```text
dataset rows=1052 skipped=4 switch_share=0.200
train_acc=0.810 val_acc=0.816
```

分类准确率约 82%，但“分类准确”不等于“得分最优”：false positive
会把复杂对手错切到 RL-Repair。

## 4. 评估结果

### 阈值 0.5

| 集 | RL-Wide160 | RL-Switch | RL-Gate |
|---|---:|---:|---:|
| dev_param | 548.26 | **548.58** | 548.76 |
| dev_complex | **454.31** | 454.24 | **435.55** |
| dev_ood | **521.37** | 521.42 | 516.07 |
| test_v7 | 485.62 | **488.26** | 485.67 |

过度切换：complex 掉 18.8、ood 掉 5.3，test 也没有收益。

### 阈值 0.9（dev 校准）

| 集 | RL-Wide160 | RL-Switch | RL-Gate |
|---|---:|---:|---:|
| dev_param | 548.26 | **548.58** | 548.30 |
| dev_family | **583.08** | 582.91 | 583.07 |
| dev_complex | 454.31 | 454.24 | **454.63** |
| dev_axelrod | 582.38 | **582.86** | 582.49 |
| dev_ood | **521.37** | **521.42** | 521.34 |
| test_v7 | 485.62 | **488.26** | 485.84 |

阈值 0.9 基本退化为 RL-Wide160；只比 wide 多 +0.04 dev_param、
+0.22 test，远低于手写 RL-Switch 的 +0.32/+2.64。

## 5. 为什么失败

1. **标签与目标不一致**：标签只问“Repair 是否在本场更高”，没有
   考虑 false positive 对 complex/ood 的代价；
2. **样本偏差**：训练对手以 train_v4/extort_hard 为主，复杂对手
   的反事实收益小，但误切代价大；
3. **K=10 信息不足**：前 10 回合还不足以区分 extortion、Gap、
   Random、Joss 和普通合作对手；
4. **单阈值全局切换**：一个概率阈值无法同时满足“多切 extortion”
   和“少切 complex”。

## 6. 判定与下一步

**No-Go for promotion / recommendation。** 保留 `rl-gate` 作为实验
候选（gate 权重很小，便于后续对照），但不进入提交推荐。

若继续 meta 学习，建议：

- 用 **cost-sensitive / constrained RL** 训练 gate，把 complex/ood
  的 false-positive 代价写进 reward；
- 用 97 维 v2 特征或更长的 K，并做 per-round gate；
- 或者保留手写 RL-Switch，把精力转向 periodic/Kamikaze 训练，
  而不是继续调 gate 阈值。

## 7. 复现

```bash
python3 results/meta_20261040/build_gate.py \
  --out weights/meta_gate_wide160_repair.json \
  --dataset-out results/meta_20261040/gate_dataset.json \
  --threshold 0.9

python3 -m ipd protocol eval \
  --candidates RL-Wide160,RL-Switch,RL-Gate \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 1 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-Wide160 \
  --out results/meta_20261040/protocol_v4_dev_rl_gate_final.json

python3 -m ipd protocol eval \
  --candidates RL-Wide160,RL-Switch,RL-Gate \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 \
  --candidate-chunk 1 --mode sample --sample-rng per-match \
  --gpu on --gpu-workers 4 --gpu-field-batch 2 --seed 20261027 \
  --no-consume \
  --out results/meta_20261040/protocol_v4_test_v7_rl_gate_final_provisional.json
```
