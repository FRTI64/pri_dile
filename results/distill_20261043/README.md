# GRU-DAgger / MLP-DAgger distillation（2026-10-29）

> 从 RL-Switch 教师蒸馏单模型。**最终候选 `rl-dagger-gru` /
> `GRU-DAgger`**：1 层 GRU（64 hidden），dev 两个 seed 与 RL-Switch
> 持平，complex/OOD 略高；未 promotion。旧 MLP 蒸馏的主要问题是
> feature normalization bug。

## 最终结果

官方 dev（30 fields × 2048，seed 20261027）：

| 集 | RL-Switch | GRU-DAgger | diff |
|---|---:|---:|---:|
| dev_param | 548.58 | **548.58** | +0.00 |
| dev_family | **582.91** | 582.90 | -0.01 |
| dev_complex | 454.24 | **455.13** | +0.89 |
| dev_axelrod | **582.86** | 582.49 | -0.37 |
| dev_ood | 521.42 | **521.54** | +0.12 |

独立 seed 20261029：546.97 / 582.59 / 454.39 / 581.93 / 523.05，
与 RL-Switch 差 +0.16 / -0.03 / +0.12 / -0.20 / +0.21。

`test_v7` provisional：487.89（seed 27）、487.02（seed 29），
分别比 RL-Switch 低 0.37 / 0.49；test 锁定，不参与选模。

## 关键发现

1. 旧 MLP 蒸馏从 wide160 checkpoint 初始化，却用
   `mean=0, std=1` 训练和导出；同一权重在 dev_param（5×1024）上
   正确 normalization 为 544.74，identity 只有 498.61，旧蒸馏
   531.75。修正后 MLP BC 可到 dev_param 548.84，但 complex
   446.59（pattern_breaker 383.19）。
2. 加入 `train_pattern_breaker`、DAgger、GRU 状态模型后，
   pattern_breaker 恢复到 519.28（官方 seed 27），complex 反超
   RL-Switch +0.89。
3. GRU + 2 轮 DAgger 的验证 BCE 从 0.1322 降到 0.1304，switch
   子集准确率 99.4%。

## 文件

| 文件 | 内容 |
|---|---|
| `dagger_switch.py` | BC + DAgger 训练脚本（`--student mlp|gru`） |
| `eval_tier_gpu_seed.py` | 可指定 seed 的 Protocol v4 tier 评估 |
| `gru_dagger_ckpt.json` | **最终 GRU checkpoint**（= `weights/rl_dagger_gru_weights.json`） |
| `gru_dagger_ckpt_history.json` | 最终训练历史 |
| `gru_bc_ckpt.json` / `gru_bc12_ckpt.json` | GRU BC 消融 |
| `mlp_bc_ckpt.json` | 修正 normalization 的 MLP BC（官方 dev） |
| `mlp_bc2_*` / `mlp_bc12_*` / `mlp_dagger_frozen_*` | MLP 数据/冻结/DAgger 消融 |
| `eval_mlp_bc/` | MLP BC 官方 30×2048 结果 |
| `eval_gru_dagger/` | GRU-DAgger seed 27 官方 dev |
| `eval_gru_dagger_seed29/` | GRU-DAgger seed 29 复现 |
| `eval_gru_dagger_levels/` | level / complexity 诊断 |
| `eval_gru_dagger_test/` | test_v7 provisional |
| `summary.json` | 上述结果的汇总、checksum、min_norm |

最终 checkpoint：

```text
format      ipd-gru-policy-v1
n_features  85
hidden      64
num_layers  1
bytes       1,063,718
sha256      7b7bbe677db157dd7101a35c018b89ca6bcc9f4357091c58d5e7b14c80cca196
```

## 复现

```bash
# 训练
python3 results/distill_20261043/dagger_switch.py \
  --student gru --rounds 2 --bc-fields 12 --dagger-fields 6 \
  --val-fields 3 --n 48 --hidden 64 --layers 1 \
  --out results/distill_20261043/gru_dagger_ckpt.json

# 官方 dev（seed 27）
python3 results/distill_20261043/eval_tier_gpu_seed.py \
  weights/rl_dagger_gru_weights.json dev_param GRU-DAgger \
  30 2048 rl-dagger-gru 20261027
```

完整分析见 `docs/research/dagger-gru.md`。
