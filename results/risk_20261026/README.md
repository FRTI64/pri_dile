# E2 — CVaR / risk-sensitive PPO（2026-10-26）

## 0. 结论（E2a + E2b + E2c）

**E2a/E2b/E2c: No-Go。** 从 RL-Repair 出发，CVaR episode weighting、
更强 CVaR、以及 worst-family DRO 都没有达到尾部 Gate。

| 实验 | risk | best heldout_v2 mean | Δ mean | Δ CVaR item | Δ CVaR family | Δ min_norm |
|---|---|---:|---:|---:|---:|---:|
| E2a | cvar 1.0 / q 0.2 | 505.30 | +0.90 | +2.22（+1.2%） | +4.04 | +0.0001 |
| E2b | cvar 3.0 / q 0.1 | 505.78 | +1.39 | +0.88（+0.5%） | +4.70 | +0.0027 |
| E2c | DRO eta 0.5 / q 0.5 | 505.33 | +0.94 | -0.89（-0.5%） | +1.39 | +0.0025 |

E2a 最好 E2-015；E2b 最好 E2b-005；E2c 最好 E2c-015。三者 CVaR 改善都
远低于 +10%，min_norm 也远低于 +0.02。official validation 也没有超过
起始 RL-Repair（E2a best iter 0；E2b best iter 5，+0.11；E2c best iter 15，
+0.29）。

因此：

- 不注册 `rl-risk`，不替换任何权重；
- RL-Repair 仍是当前 Go candidate；
- **heldout_v4 没有运行**，保留给下一个通过 validation 的候选；
- episode-level CVaR surrogate 不足以改变 held-out family 尾部，下一步应改
  worst-family/DRO 重加权，而不是继续调 coef。

## 1. 实现

- `ipd/training/risk.py`：`cvar_episode_weights`（worst quantile 加权，
  mean weight = 1）和 `dirichlet_weights`（每轮 mixture 采样）。
- `ipd/training/rl.py` 新 flags：
  - `--risk-objective none|cvar`
  - `--risk-coef`
  - `--risk-quantile`
  - `--dirichlet-concentration`
- history row 新增 `cvar_return`、`risk_weight_max/min`、`mixture`。

注意：episode-level CVaR 加权是 tail-risk surrogate，不是精确 CVaR
policy gradient；它的目标是让最差 episode 的 advantage 权重更高。

## 2. E2a 训练

```bash
.venv/bin/python -m ipd train rl --device auto --seed 20261033 \
  --iterations 20 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,repair \
  --dist-weights 1,1,1,2,2 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --risk-objective cvar --risk-coef 1.0 --risk-quantile 0.2 \
  --dirichlet-concentration 20 \
  --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --save-all-checkpoints results/risk_20261026/E2_checkpoints \
  --out weights/rl_policy_weights_risk.json \
  --checkpoint-out weights/rl_policy_checkpoint_risk.json \
  --history-out results/risk_20261026/E2_history.json
```

Checkpoint selection（heldout_v2，固定 policy seed）：

```bash
.venv/bin/python -m ipd league \
  --config results/risk_20261026/e2_selection_config.json \
  --skip-cross-play --skip-red --workers 20 \
  --out results/risk_20261026/e2_selection.json
```

## 3. Gate

| 条件 | E2a | E2b | E2c |
|---|---|---|---|
| heldout_v2 mean 不比 RL-Repair 低超过 1 | PASS（+0.90） | PASS（+1.39） | PASS（+0.94） |
| CVaR10 +10% 或 min_norm +0.02 | FAIL（+1.2% / +0.0001） | FAIL（+0.5% / +0.0027） | FAIL（-0.5% / +0.0025） |
| official 不降超过 2 | PASS（best iter 0） | PASS（best iter 5，+0.11） | PASS（best iter 15，+0.29） |
| red 不比 RL-Repair 差 10 | 未跑（未过 CVaR Gate） | 未跑（未过 CVaR Gate） | 未跑（未过 CVaR Gate） |

**No-Go。** 风险目标没有改变 policy 的尾部结构；episode-level
weighting 可能被 cooperative episodes 的高 advantage 抵消，DRO 重加权
也没有把 held-out family 尾部抬起。

E2b（coef=3.0, quantile=0.1）和 E2c（DRO eta=0.5, q=0.5）的
heldout_v2 checkpoint selection：

| Candidate | mean | min_norm | CVaR family | CVaR item |
|---|---:|---:|---:|---:|
| RL-Repair | 504.40 | 0.9972 | 424.19 | 189.63 |
| E2b-005 | **505.78** | **0.9999** | **428.89** | **190.51** |
| E2c-015 | 505.33 | 0.9997 | 425.59 | 188.74 |

E2b 更强风险权重只把 mean 提高 +1.39、CVaR item +0.5%；E2c DRO
mean +0.94，但 CVaR item **-0.5%**。因此 E2a/E2b/E2c 都不 promotion；
`heldout_v4` 未运行。

## 4. 诊断

- 训练 reward 约 519–549，和 E1c 接近；policy KL 只有约 0.005，变化很小。
- CVaR family 由 worst family（random）主导，episode-level 加权没有直接
  针对 family 尾部。
- E2c DRO 确实把训练 mixture 推向 repair（约 0.45–0.48），但 held-out
  CVaR 没有改善；说明训练分布的平均 reward 不是 held-out family 尾部的
  好代理。
- 下一步若继续 risk 方向，应直接用 **held-out family-level CVaR 约束**
  或 meta-game exploitability 作为目标；否则优先转 P3 league/red pool。

## 5. 测试

```bash
python3 -m unittest tests.test_risk tests.test_training -v
```

## 6. 产物

```text
results/risk_20261026/
  E2_history.json / E2b_history.json / E2c_history.json
  e2_train.log / e2b_train.log / e2c_train.log
  e2_selection.json / e2_selection.md
  e2b_selection.json / e2b_selection.md
  e2c_selection.json / e2c_selection.md
  e2_selection_config.json / e2b_selection_config.json /
  e2c_selection_config.json
  README.md
  checksums.txt
weights/
  rl_policy_weights_risk.json / rl_policy_checkpoint_risk.json      # E2a (未注册)
  rl_policy_weights_risk_b.json / rl_policy_checkpoint_risk_b.json  # E2b (未注册)
  rl_policy_weights_risk_c.json / rl_policy_checkpoint_risk_c.json  # E2c (未注册)
```

主权重 `weights/rl_policy_weights.json` 与 RL-Repair 权重均未改动。
E2/E2b/E2c 的中间 checkpoint 未提交；selection config 中的
`E2_checkpoints/`、`E2b_checkpoints/`、`E2c_checkpoints/` 路径需要重训
才会存在。
