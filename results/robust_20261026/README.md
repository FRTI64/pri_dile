# E1 — domain-randomized A60 fine-tune（2026-10-26）

## 0. 结论

**E1a: No-Go；E1b: borderline No-Go；E1c: Go candidate。**

| 实验 | heldout_v2 Δ vs A60 | min_norm Δ | CVaR10 Δ | red blue Δ | heldout_v3 final | 判定 |
|---|---:|---:|---:|---:|---:|---|
| E1a（KL 0.1, lr 1e-4） | +1.36 | +0.0036 | -2.15 | +12.50 | 未运行 | **No-Go** |
| E1b（KL 0.05, lr 2e-4） | +1.94（selection +3.05） | -0.0036 | -2.32 | +11.70 | 未运行 | **No-Go** |
| E1c（repair, 20 iter） | **+3.10**（selection +4.90） | +0.0002 | -0.74 | +4.25 | **+3.91** | **Go candidate** |

决策：

- **E1c `RL-Repair` 注册为候选 `rl-repair`**，但不覆盖
  `weights/rl_policy_weights.json`；RL-PPO/A60 仍是 fallback。
- E1a/E1b 的 `weights/rl_policy_weights_robust*.json` 保留为研究 artifact。
- **heldout_v3 已被 E1c 一次性使用**；后续新候选必须新建 heldout_v4。
- E1c 的收益主要来自 extort +23~28 和 random 修复；red blue 只比 A60 高
  4.25，E4 red 仍 100% 击败，anti-exploitation 仍未解决。

## 1. 训练分布与协议

从 A60 checkpoint 初始化，新增 `robust` domain-randomized 训练分布
（`ipd/evaluation/robust.py`）：

```text
train-dists: original,late,grim30,wide,hard,robust
dist-weights: 1,1,1,2,1,2
iterations: 40
init: weights/rl_policy_checkpoint_hard60.json
select: official validation avg
```

随机化范围：Extort chi [1.5,3.8]、Random p [0.01,0.95]、
LateD k [20,95]、NoisyTFT/Joss p [0.01,0.50]、DynamicWindow W [3,10]、
Kamikaze period [8,30]、Harvest start [20,90]；另有 TFT/Grim/AllD/Bully/AllC
anchor。robust 只用于训练，不用于 reporting。

## 2. E1a（KL 0.1, lr 1e-4）

官方 validation：

```text
initial (A60)  avg=558.87  worst=553.45
best iter 40   avg=559.07  worst=554.09
```

E0 风格评估：

| Candidate | official | blind | validation | heldout_v2 | min_norm | CVaR10 | red blue |
|---|---:|---:|---:|---:|---:|---:|---:|
| RL-PPO | 553.22 | 562.80 | 504.45 | 494.97 | 1.0000 | 387.26 | 310.55 |
| RL-PPO-Hard60 | 554.27 | 563.87 | 507.85 | 500.07 | 0.9928 | 391.13 | 299.20 |
| **RL-Robust (E1a)** | **554.71** | **564.63** | **508.47** | **501.43** | **0.9964** | 388.98 | **311.70** |

相对 A60：overall +0.54、official +0.44、blind +0.76、validation +0.62、
heldout_v2 +1.36、min_norm +0.0036、CVaR -2.15、red blue +12.50。

Held-out v2 per-family（E1a - A60）：

| Family | A60 | E1a | Δ |
|---|---:|---:|---:|
| reciprocal_v2 | 599.2 | 599.2 | 0 |
| punitive_v2 | 600.6 | 600.6 | 0 |
| noisy_v2 | 425.6 | 429.4 | +3.8 |
| extort_v2 | 441.9 | 458.1 | **+16.3** |
| random_v2 | 420.6 | 414.5 | **-6.1** |
| temporal_v2 | 450.9 | 450.9 | 0 |
| adaptive_v2 | 582.8 | 581.1 | -1.7 |
| **mean** | 500.07 | 501.43 | +1.36 |

**Gate：FAIL。** held-out +1.36 < +3，min_norm +0.0036 < +0.02，
CVaR 下降。E4 red 仍 100% 击败；blue mean 的提升不等于 anti-exploitation 解决。

## 3. E1b（KL 0.05, lr 2e-4）

E1a 的 policy 变化很小（KL 约 0.004），因此 E1b 降低 KL、提高 lr，
并保存 iter 5..40 的 8 个 checkpoint。

官方 validation：

```text
initial (A60)  avg=559.01  worst=550.63
best iter 40   avg=559.80  worst=550.51
```

Checkpoint selection（heldout_v2，固定 policy seed）：

| Candidate | heldout_v2 mean | min_norm |
|---|---:|---:|
| RL-PPO-Hard60 | 499.50 | 1.0000 |
| E1b-040 | **502.54** | 0.9967 |

E1b-040 相对 A60 **+3.05**，刚好通过 selection screen；增益仍主要来自
extort +25.3、random -1.4。`e1b_selection.md` 的 Gate 显示 INCOMPLETE 是
因为该次只跑了 heldout_v2（`--skip-cross-play --skip-red`），不是模型结论。

完整 E1b 评估（index-based policy seeds，和 E1a/E0 同 suite/seeds）：

| Candidate | official | blind | validation | heldout_v2 | min_norm | CVaR10 | red blue |
|---|---:|---:|---:|---:|---:|---:|---:|
| RL-PPO | 553.22 | 562.80 | 504.45 | 494.97 | 1.0000 | 387.26 | 310.55 |
| RL-PPO-Hard60 | 554.27 | 563.87 | 507.85 | 500.07 | 0.9928 | 391.13 | 299.20 |
| **RL-Robust-B (E1b)** | **554.66** | **564.67** | **509.99** | **502.00** | 0.9892 | 388.81 | **310.90** |

相对 A60：official +0.38、blind +0.79、validation +2.14、heldout_v2 +1.94、
min_norm -0.0036、CVaR -2.32、red blue +11.70。官方四分布没有下降
（original +0.40、late +0.29、grim30 -0.16、wide +1.00）。

Held-out v2 per-family（E1b - A60）：

| Family | A60 | E1b | Δ |
|---|---:|---:|---:|
| reciprocal_v2 | 599.2 | 599.2 | 0 |
| punitive_v2 | 600.6 | 600.6 | 0 |
| noisy_v2 | 425.6 | 430.1 | +4.4 |
| extort_v2 | 441.9 | 462.4 | **+20.5** |
| random_v2 | 420.6 | 414.2 | **-6.3** |
| temporal_v2 | 450.9 | 451.2 | +0.3 |
| adaptive_v2 | 582.8 | 580.9 | -2.0 |
| **mean** | 500.07 | 502.00 | +1.94 |

**判定：No-Go。** 两次 validation 不一致（selection +3.05，full +1.94），
full run 未达到 +3；min_norm/CVaR 均下降，random 仍回退 6.3。
red blue 改善 +11.7 是正向信号，但 E4 red 仍 100% 击败，不能据此 promotion。
不运行 heldout_v3。

## 4. E1c repair（Go candidate）

从 E1b checkpoint 初始化，用新的 `repair` 分布修复 random/adaptive 尾部：

```text
Random 5.0 (p 收窄到 [0.08, 0.92])
NoisyTFT 3.0, Joss 2.0, DynamicWindow 2.0, LateD 2.0
Extort 1.5, Kamikaze 1.0, Harvest 1.0
TFT 3.0, GTFT 2.0, Pavlov 2.0, AllC 2.0
Grim 1.0, AllD 1.0, Bully 1.0
```

训练 20 iter，lr 1e-4，KL 0.05。官方 validation 554.39→554.88；
训练脚本按 official 选 iter 15，但 heldout_v2 选模选 **iter 20**，并导出为
`weights/rl_policy_weights_repair.json`。

Checkpoint selection（heldout_v2，固定 policy seed）：

| Candidate | heldout_v2 mean | min_norm |
|---|---:|---:|
| RL-PPO-Hard60 | 499.50 | 1.0000 |
| RL-Robust-B (E1b) | 502.54 | 0.9967 |
| **E1c-020** | **504.40** | **0.9972** |

E1c-020 相对 A60 **+4.90**；相对 E1b **+1.86**。per-family 相对 A60：
extort +28.3、random **+5.3**（回退修复）、temporal +5.2、noisy -1.2、
adaptive -0.1。

完整 E1c 评估（同 suite/seeds）：

| Candidate | official | blind | validation | heldout_v2 | min_norm | CVaR10 | red blue |
|---|---:|---:|---:|---:|---:|---:|---:|
| RL-PPO | 553.22 | 562.80 | 504.45 | 494.97 | 1.0000 | 387.26 | 310.55 |
| RL-PPO-Hard60 | 554.27 | 563.87 | 507.85 | 500.07 | 0.9928 | 391.13 | 299.20 |
| **RL-Repair (E1c)** | **554.72** | **564.68** | **509.93** | **503.17** | **0.9930** | 390.39 | **303.45** |
| Adv-RL | 554.05 | 562.97 | 501.87 | 502.39 | 0.9815 | 387.14 | 325.55 |

相对 A60：official +0.45、blind +0.80、validation +2.08、heldout_v2 +3.10、
min_norm +0.0002、CVaR -0.74、red blue +4.25。官方四分布
original +0.78 / late +0.43 / grim30 -0.06 / wide +0.62，均未下降超过 2。

标准 CLI 复现（独立 policy seed）：

```text
benchmark 20 fields: 557.26 / 549.39 / 552.28 / 560.12，avg 554.76
blind 20 fields:     564.61 / 5.75（16/20 第一，16/20 前 10，worst 27）
```

Held-out v2 per-family（E1c - A60）：

| Family | A60 | E1c | Δ |
|---|---:|---:|---:|
| reciprocal_v2 | 599.2 | 599.2 | 0 |
| punitive_v2 | 600.6 | 600.6 | 0 |
| noisy_v2 | 425.6 | 429.7 | +4.0 |
| extort_v2 | 441.9 | 467.5 | **+25.6** |
| random_v2 | 420.6 | 421.2 | **+0.6** |
| temporal_v2 | 450.9 | 450.8 | -0.1 |
| adaptive_v2 | 582.8 | 581.0 | -1.9 |
| **mean** | 500.07 | 503.17 | **+3.10** |

**Heldout_v3 final（只跑一次）：**

| Candidate | heldout_v3 mean | min_norm | CVaR10 | red blue |
|---|---:|---:|---:|---:|
| RL-PPO | 502.31 | 1.0000 | 401.43 | 295.60 |
| RL-PPO-Hard60 | 505.97 | 0.9999 | 405.65 | 310.00 |
| **RL-Repair** | **509.88** | **0.9996** | 405.58 | 306.35 |
| Adv-RL | 504.84 | 0.9807 | **407.62** | **328.30** |

E1c vs A60 on v3：**+3.91**；per-family extort +23.4、random +4.7、
noisy +2.5、temporal +2.2、adaptive -0.2、anchor -0.1；CVaR 基本持平
（-0.07），min_norm 基本持平（-0.0003）。red blue 306.35 vs 310.00
（-3.65），仍 100% 被 E4 red 击败。

**判定：Go candidate。** 两次 validation（+4.90 / +3.10）和一次 final
（+3.91）方向一致，官方/blind 不降，无 family 崩塌。注册 `rl-repair`，
但不覆盖主权重；anti-exploitation 仍未解决。

## 5. Gate E1 汇总

| 条件 | E1a | E1b | E1c |
|---|---|---|---|
| heldout_v2 mean +3 或 min_norm +0.02 | FAIL（+1.36/+0.0036） | borderline FAIL（+1.94/-0.0036） | **PASS（+3.10/+0.0002）** |
| 官方不降超过 2 | PASS（+0.44） | PASS（+0.38） | PASS（+0.45） |
| 无 family 崩塌 | PASS（random -6.1） | PASS（random -6.3） | PASS（adaptive -1.9） |
| red blue 不比 A60 差 10 | PASS（+12.5） | PASS（+11.7） | PASS（+4.25） |
| heldout_v3 方向 | 未运行 | 未运行 | **PASS（+3.91）** |
| exploitability 不 materially 增加 | 无法区分 | 无法区分 | 无法区分 |

## 6. 诊断与下一步

E1c 证明了两点：

1. domain randomization 的 extort 收益是真实的（heldout_v2 +25.6，
   heldout_v3 +23.4）；
2. E1a/E1b 的 random 回退可以通过 repair 分布修复（random +5.3/+4.7），
   不必牺牲 extort。E1c 的 final v3 +3.91，两次 validation +4.90/+3.10，
   方向一致，因此判 Go candidate。

仍未解决：

- E4 red 仍 100% 击败 RL-Repair；red blue 只比 A60 高 4.25（v2）/
  -3.65（v3），anti-exploitation 没有解决。
- CVaR 基本持平（v3 -0.07），不是尾部风险的质变。
- `heldout_v3` 已被 E1c 使用；后续候选必须新建 heldout_v4。

下一步建议：

1. **E2**：CVaR/worst-family 辅助目标，直接优化尾部而不是 mixture 权重。
2. **P3**：加入 red/exploiter checkpoint 池；当前 cross-play 仍退化。
3. 若要把 RL-Repair 提升为主权重，先做一次独立复现（新 seeds）和
   用户确认；当前只注册为候选，`weights/rl_policy_weights.json` 不变。

## 7. 复现

E1a：

```bash
.venv/bin/python -m ipd train rl --device auto --seed 20261030 \
  --iterations 40 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,hard,robust \
  --dist-weights 1,1,1,2,1,2 \
  --init-weights weights/rl_policy_checkpoint_hard60.json \
  --bc-kl-coef 0.1 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --out weights/rl_policy_weights_robust.json \
  --checkpoint-out weights/rl_policy_checkpoint_robust.json \
  --history-out results/robust_20261026/E1_history.json

.venv/bin/python -m ipd league --config configs/eval/e1_robust_league.json \
  --workers 20 --out results/robust_20261026/e1_report.json \
  --md-out results/robust_20261026/e1_report.md
```

E1b：

```bash
.venv/bin/python -m ipd train rl --device auto --seed 20261031 \
  --iterations 40 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,hard,robust \
  --dist-weights 1,1,1,2,1,2 \
  --init-weights weights/rl_policy_checkpoint_hard60.json \
  --bc-kl-coef 0.05 --lr 2e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --save-all-checkpoints results/robust_20261026/E1b_checkpoints \
  --out weights/rl_policy_weights_robust_b.json \
  --checkpoint-out weights/rl_policy_checkpoint_robust_b.json \
  --history-out results/robust_20261026/E1b_history.json

.venv/bin/python -m ipd league \
  --config results/robust_20261026/e1b_selection_config.json \
  --skip-cross-play --skip-red --workers 20 \
  --out results/robust_20261026/e1b_selection.json

.venv/bin/python -m ipd league --config configs/eval/e1b_robust_league.json \
  --workers 20 --out results/robust_20261026/e1b_report.json \
  --md-out results/robust_20261026/e1b_report.md
```

E1c：

```bash
.venv/bin/python -m ipd train rl --device auto --seed 20261032 \
  --iterations 20 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,repair \
  --dist-weights 1,1,1,2,2 \
  --init-weights weights/rl_policy_checkpoint_robust_b.json \
  --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --save-all-checkpoints results/robust_20261026/E1c_checkpoints \
  --out weights/rl_policy_weights_repair.json \
  --checkpoint-out weights/rl_policy_checkpoint_repair.json \
  --history-out results/robust_20261026/E1c_history.json

.venv/bin/python -m ipd league \
  --config results/robust_20261026/e1c_selection_config.json \
  --skip-cross-play --skip-red --workers 20 \
  --out results/robust_20261026/e1c_selection.json

.venv/bin/python -m ipd league --config configs/eval/e1c_repair_league.json \
  --workers 20 --out results/robust_20261026/e1c_report.json \
  --md-out results/robust_20261026/e1c_report.md

# one-shot final; do not rerun for selection
.venv/bin/python -m ipd league --config configs/eval/e1c_final_league.json \
  --workers 20 --out results/robust_20261026/e1c_final_v3.json \
  --md-out results/robust_20261026/e1c_final_v3.md
```

## 8. 产物

```text
results/robust_20261026/
  README.md
  E1_history.json / E1b_history.json / E1c_history.json
  e1_report.json / e1_report.md          # E1a full
  e1b_selection.json / e1b_selection.md  # E1b checkpoint selection
  e1b_report.json / e1b_report.md        # E1b full
  e1c_selection.json / e1c_selection.md  # E1c checkpoint selection
  e1c_report.json / e1c_report.md        # E1c full
  e1c_final_v3.json / e1c_final_v3.md    # one-shot final
  train.log / eval.log / e1b_train.log / e1b_eval.log
  e1c_train.log / e1c_eval.log / e1c_selection.log / e1c_final.log
  checksums.txt
weights/
  rl_policy_weights_robust.json / rl_policy_checkpoint_robust.json
  rl_policy_weights_robust_b.json / rl_policy_checkpoint_robust_b.json
  rl_policy_weights_repair.json / rl_policy_checkpoint_repair.json
```

主权重 `weights/rl_policy_weights.json` 的 sha256 仍为
`bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`；
A60 权重仍为
`42c64c8a7524115f5ec17511a482dc56a2259cfcb6740ab5b223a193797ff60e`。
E1b 的 8 个中间 checkpoint 未提交（selected iter 40 与
`weights/rl_policy_checkpoint_robust_b.json` 相同）；E1c 的中间 checkpoint
也未提交，selected iter 20 已导出为
`weights/rl_policy_weights_repair.json` / `weights/rl_policy_checkpoint_repair.json`。
