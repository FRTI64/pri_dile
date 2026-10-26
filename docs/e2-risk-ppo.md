# E2 — CVaR / risk-sensitive PPO

- **Status:** complete — **No-Go**（E2a/E2b both failed tail gate）
- **Parent:** `RL-Repair` / `weights/rl_policy_checkpoint_repair.json`
- **New code:** `ipd/training/risk.py`, PPO flags in `ipd/training/rl.py`
- **Goal:** improve the held-out tail (CVaR/min family) without giving back the
  E1c mean gain.

## 1. Risk objective

`cvar_episode_weights` upweights the worst `risk_quantile` fraction of
episodes. Weights have mean 1.0, so the PPO loss scale is unchanged. This is a
practical tail-risk surrogate, not an exact CVaR policy-gradient estimator.

`dirichlet_weights` samples the per-iteration distribution mixture around the
base `--dist-weights` prior. This prevents overfitting one fixed mixture.

## 2. Training command

```bash
python3 -m ipd train rl --device auto --seed 20261033 \
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

Checkpoint selection uses heldout_v2 validation with a fixed policy seed.
`heldout_v3` is already consumed by E1c; E2 final test would need a new
`heldout_v4`.

## 3. Gate E2

Go only if all hold:

1. heldout_v2 mean is not lower than E1c by more than 1;
2. heldout_v2 CVaR10 improves by at least 10% relative **or** min_norm
   improves by at least +0.02;
3. official four-distribution mean is not lower than E1c by more than 2;
4. no held-out family collapses; E4 red blue is not worse than E1c by more
   than 10.

If the gate fails, keep RL-Repair as the candidate and record No-Go.

## 4. Results (No-Go)

| 实验 | risk | best heldout_v2 mean | Δ mean | Δ CVaR item | Δ min_norm |
|---|---|---:|---:|---:|---:|
| E2a | cvar 1.0 / q 0.2 | 505.30 | +0.90 | +1.2% | +0.0001 |
| E2b | cvar 3.0 / q 0.1 | 505.78 | +1.39 | +0.5% | +0.0027 |
| E2c | DRO eta 0.5 / q 0.5 | 505.33 | +0.94 | -0.5% | +0.0025 |

E2a/E2b/E2c 都没有达到 +10% CVaR 或 +0.02 min_norm；official validation
也没有超过起始 RL-Repair。**No-Go**：保留 RL-Repair，不注册 `rl-risk`
/`rl-dro`，`heldout_v4` 未运行。

诊断：episode-level CVaR 加权没有直接约束 held-out family 尾部；DRO
把训练 mixture 推向 repair，但训练分布平均 reward 不是 held-out family
尾部的可靠代理。下一步应直接用 family-level CVaR/exploitability 目标，
或转 P3 red/exploiter league。

## 5. E2c DRO（worst-family reweighting）

E2a/E2b 的 episode-level CVaR 没有直接约束 family 尾部。E2c 改为
exponentiated-gradient / worst-family DRO：每轮根据训练分布的平均 reward
更新 mixture，低分分布获得更高权重，并用 `dro-floor` 防止塌缩。

```bash
python3 -m ipd train rl --device auto --seed 20261035 \
  --iterations 20 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,repair \
  --dist-weights 1,1,1,2,2 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --risk-objective dro --dro-eta 0.5 --dro-quantile 0.5 \
  --dro-blend 0.5 --dro-floor 0.1 \
  --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --save-all-checkpoints results/risk_20261026/E2c_checkpoints \
  --out weights/rl_policy_weights_risk_c.json \
  --checkpoint-out weights/rl_policy_checkpoint_risk_c.json \
  --history-out results/risk_20261026/E2c_history.json
```

Gate 与 E2a/E2b 相同：mean 不降超过 1，CVaR +10% 或 min_norm +0.02，
official 不降超过 2，red 不差 10。通过才使用 heldout_v4 final。

实现注意：DRO 返回归一化权重（sum=1），训练时按初始 `dist_weights` 总和
重新缩放，保持总 field 预算不变；否则 rollout 会从 35 fields 缩到 5。

## 6. Tests

```bash
python3 -m unittest tests.test_risk tests.test_training -v
```

Plus a one-iteration CPU smoke with `--risk-objective cvar
--dirichlet-concentration 20`, checking history contains `cvar_return`,
risk weights and the sampled `mixture`.
