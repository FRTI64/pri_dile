# E1 — domain-randomized A60 fine-tune（2026-10-26）

## 0. 结论

**E1a: No-Go；E1b: borderline No-Go。两者都不 promotion。**

| 实验 | heldout_v2 Δ vs A60 | min_norm Δ | CVaR10 Δ | red blue Δ | 判定 |
|---|---:|---:|---:|---:|---|
| E1a（KL 0.1, lr 1e-4） | +1.36 | +0.0036 | -2.15 | +12.50 | **No-Go** |
| E1b（KL 0.05, lr 2e-4） | +1.94（selection +3.05） | -0.0036 | -2.32 | +11.70 | **No-Go** |

因此：

- 不替换 `weights/rl_policy_weights.json`；
- 不注册 `rl-robust` / `rl-robust-b` 为正式候选；
- `weights/rl_policy_weights_robust*.json` 和 checkpoint 保留为研究 artifact；
- **heldout_v3 没有运行**：E1a 未过 validation gate；E1b 的两次 validation
  结果不一致（selection +3.05，full +1.94），按保守规则不消耗一次性 final test。
  `heldout_v3` 留给下一个稳定通过 validation 的候选。

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

## 4. Gate E1 汇总

| 条件 | E1a | E1b |
|---|---|---|
| heldout_v2 mean +3 或 min_norm +0.02 | FAIL（+1.36/+0.0036） | **borderline FAIL**（+1.94/-0.0036） |
| 官方不降超过 2 | PASS（+0.44） | PASS（+0.38） |
| 无 family 崩塌 | PASS（random -6.1） | PASS（random -6.3） |
| red blue 不比 A60 差 10 | PASS（+12.5） | PASS（+11.7） |
| exploitability 不 materially 增加 | 无法区分 | 无法区分 |

## 5. 诊断与下一步

domain randomization 有真实信号：extort 稳定 +16~20，red blue +12 左右；
但当前 `robust` mixture 同时把 random 拉低 6 分，导致 mean 增益不稳定、
CVaR 下降。下一步建议：

1. **E1c**：降低 `Random` 权重/收窄 random 参数区间，增加 cooperative
   anchors；或把 random 高背叛区间拆成 curriculum。仍用 heldout_v2 做
   validation。
2. **E2**：在 PPO 目标里加入 CVaR/worst-family 辅助项，直接约束 random/
   adaptive 尾部，而不是只靠 mixture 权重。
3. 只有连续两次 validation 都过 Gate（+3 mean 或 +0.02 min_norm）后，
   才运行一次 heldout_v3 final test。

## 6. 复现

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

## 7. 产物

```text
results/robust_20261026/
  README.md
  E1_history.json / E1b_history.json
  e1_report.json / e1_report.md          # E1a full
  e1b_selection.json / e1b_selection.md  # E1b checkpoint selection
  e1b_report.json / e1b_report.md        # E1b full
  train.log / eval.log / e1b_train.log / e1b_eval.log
  checksums.txt
weights/
  rl_policy_weights_robust.json / rl_policy_checkpoint_robust.json
  rl_policy_weights_robust_b.json / rl_policy_checkpoint_robust_b.json
```

主权重 `weights/rl_policy_weights.json` 的 sha256 仍为
`bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`；
A60 权重仍为
`42c64c8a7524115f5ec17511a482dc56a2259cfcb6740ab5b223a193797ff60e`。
E1b 的 8 个中间 checkpoint 未提交（selected iter 40 与
`weights/rl_policy_checkpoint_robust_b.json` 相同）。
