# PSRO — Policy-Space Response Oracles（2026-10-27）

- **Status:** complete — **No-Go for promotion**; experimental candidates
  `psro` / `psro-br` are registered.
- **v4 (2026-10-29):** stronger oracle; BR_01 enters the maximin support
  with weight 1.0 and exploitability drops from **290.00 to 0.00**.  It also
  flips `dev_red` / Red-v5/v6 to red win **0.0**, but `dev_param` falls to
  **486.93 vs 546.47** for RL-Repair.  **No-Go for promotion**; `psro-v4` /
  `psro-v4-br` are experimental anti-exploitation specialists.  See §8.
- **Code:** `ipd/training/psro.py`, `ipd/policies/psro.py`,
  `ipd/policies/candidates/psro.py`, `ipd/policies/candidates/psro_br.py`
- **Config:** `configs/psro/psro_20261027.json`
- **Result:** `results/psro_20261027/`
- **From-scratch companion:** [psro-scratch.md](psro-scratch.md)（blue 只有
  random/scratch/BR，现有模型只做 red）
- **Parent:** `docs/research/p3-expansion.md`（maximin-only PSRO-lite）

## 1. 为什么做完整 PSRO

P3 expansion 训练了多个 red，并用 `meta_weights_from_report` 求 red 的
maximin 混合，再训练一个 blue response。它只优化单侧（red 权重），没有
blue population，且 blue response 只选一个；Gate D 失败（-5.9%）。

完整 PSRO 把两边都建模：

1. blue population 和 red population 分开；
2. 每次 cross-play 估计矩形收益矩阵
   `M[i][j] = blue_i score - red_j score`；
3. blue 求 maximin 混合 `p`，red 求对偶 maximin 混合 `q`；
4. 以 `q`（加 uniform 正则）为 opponent pool，训练新的 blue best response；
5. 把新 blue 加入 population，重新求解；重复。

与 P3-lite 的区别：

- P3-lite 只对 red 求 maximin，blue 始终是一条策略；
- PSRO 的 blue 是一个 population，部署时可以是 maximin 混合
  （`psro` meta-policy）或最后一轮 BR（`psro-br`）；
- PSRO 的 evaluation exploitability 用纯 blue/red maximin，不用 uniform
  训练混合，避免把多样性正则误当成 meta-strategy。

## 2. Population

`configs/psro/psro_20261027.json` 的 blue population：

| Blue | 来源 | 备注 |
|---|---|---|
| RL-Repair | `weights/rl_policy_weights_repair.json` | 当前最佳候选 |
| RL-PPO-Hard60 | `weights/rl_policy_weights_hard60.json` | A60 |
| RL-PPO | `weights/rl_policy_weights.json` | 主权重 |
| Adv-RL | aggressive config | 规则包装 |
| Final-tuned | 规则 | 无权重 |
| Triple-majority | 规则 | 无权重 |
| SelfPlay-GRU | `weights/selfplay_gru_policy.json` | 21 维 |
| Blue-Meta-v2 | `weights/red_pool/blue_meta_v2.json` | P3 blue response |

Red population（训练用，不是 held-out）：`Red-RL-v1`、`Red-RL-v3`、
`Red-RL-v4`。**Red-v5 / Red-v6 不进入 population**，只用于最终诊断。

Cross-play：8 seeds，双方各打一次；100 回合；20 workers。

## 3. Meta-solver

`solve_blue_red` 返回：

```text
blue_meta:  maximin weights over blues, expected payoff, worst_case
red_meta:   maximin weights over reds, training_weights (uniform mix 0.15),
            expected payoff
exploitability = max(0, -blue_worst_case)
```

`uniform_mix` 只作用于 `red_meta.training_weights`，让每个 red 至少保留
`0.15 / n_red` 的 oracle 采样概率；evaluation 仍用纯 maximin。

## 4. Oracle

默认 oracle 复用现有 PPO：

```text
python3 -m ipd train rl \
  --opponent-pool <round manifest> --opponent-fraction 0.6 \
  --opponent-penalty 1.0 --no-validation \
  --iterations 12 --fields-per-dist 4 --n 64 --workers 20 \
  --train-dists original,late,grim30,wide,repair \
  --dist-weights 1,1,1,2,2 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02
```

- 60% 训练 field 来自 red 训练混合，40% 来自 official/repair，防止纯 red
  过拟合；
- `--opponent-penalty 1.0` 使 PPO reward 为 `own - opponent`；
- `--no-validation`：PSRO 需要 best response，不用 validation 选模；
- `--init-weights` 固定为 RL-Repair checkpoint，便于两轮对比。

## 5. 结果

### 5.1 元博弈

初始 blue maximin：

| Blue | Weight | Expected vs red |
|---|---:|---:|
| Final-tuned | 0.7251 | -284.78 |
| Triple-majority | 0.2749 | -284.81 |
| RL-Repair | 0.0000 | -395.08 |
| PSRO-BR-01 | 0.0000 | -372.91 |
| PSRO-BR-02 | 0.0000 | -375.89 |

Red maximin：Red-v4 0.6606、Red-v1 0.3393、Red-v3 0.0001。
Blue worst-case **-284.797**；exploitability **284.797**。

两轮 oracle 后的 exploitability：284.799 → 284.799 → 284.804。
两个 RL best response 都没有进入 blue maximin support（权重 0），
说明它们相对 Final/Triple 仍然更容易被 Red-v1/v4 剥削。

### 5.2 诊断评估

**不是完整 Protocol v4 baseline**；4 dev fields × 512，10 held-out-red seeds。
完整 JSON：`results/psro_20261027/eval_dev_param_smoke.json`、
`eval_heldout_red.json`。

| Candidate | dev_param | min_norm | held-out red blue mean |
|---|---:|---:|---:|
| RL-Repair | **543.35** | 1.0000 | 348.35 |
| Triple-majority | 530.61 | 0.9375 | 373.05 |
| **PSRO-Meta (`psro`)** | 527.71 | 0.9106 | **373.70** |
| PSRO-BR (`psro-br`) | 525.58 | 0.7066 | 354.45 |
| Final-tuned | 525.73 | 0.9159 | 375.70 |

- PSRO-Meta 在 held-out Red-v5/v6 上比 RL-Repair **+25.4 blue points**；
  red-minus-blue 均值 **378.5 → 297.5（-21.4%）**，其中 Red-v5 0%、
  Red-v6 -37.3%；
- 但 dev_param **-15.6**，`min_norm` 0.9106，`late` family 只有 308.04；
- PSRO-BR 在 red 上只比 RL-Repair +6.1，dev 也更低；
- 所有候选对 Red-v5/v6 仍是 red win 100%。

因此 **Gate D 仍不通过**：exploitability-reduction 目标在 Red-v5/v6 上
达到（-21.4%），但 official/dev 和 family guardrail 失败（dev -15.6，
`late` -4.7%），而且这是 10-seed / 4-field 诊断，不是正式 Gate 运行。
PSRO-Meta 的价值是展示了“规则混合更抗 red、但 official 更弱”的 trade-off，
不是 anti-exploitation 解决。

## 6. 决策

- 不 promotion，不覆盖 `weights/rl_policy_weights.json`；
- `psro` / `psro-br` 注册为实验候选，便于后续对比；
- 需要新的 held-out red 才能继续 P3/PSRO；不要用 Red-v5/v6 选模；
- 若继续 PSRO，优先尝试：
  1. 更强 blue oracle：更多迭代、从零 / 规则初始化、降低 KL；
  2. red oracle：训练 red best response 对 blue maximin，形成
     double oracle；
  3. 更大 population / 更多 cross-play seeds，降低 maximin 噪声；
- 不要重复“同样的 2 轮 PPO oracle + 同样 3 个 red”配置。

## 7. 复现

```bash
python3 -m ipd train psro --config configs/psro/psro_20261027.json \
  --workers 20

# 仅元博弈，不训练
python3 -m ipd train psro --config configs/psro/psro_20261027.json \
  --dry-run --out /tmp/psro_dry

# 诊断评估
bash results/psro_20261027/run_eval.sh
```

权重 checksum：

```text
deedc25c7c384e86c0223ec5395ba6f3dfd21b5358de57c8d903ce056d92ed9b  weights/psro/psro_policy.json
7d1c886b0c96d28dbe14c713d30e7bc8c646c0c37dc2e74ed426987a9cd94cd9  weights/psro/meta_strategy.json
```

完整结果与 caveats 见 `results/psro_20261027/README.md`。

## 8. PSRO v4 — stronger oracle（2026-10-29）

上一轮 2 轮 / 12 iter / 60% red / `bc_kl=0.05` 的 oracle 没有进入 blue
maximin support。v4 按 §6 的建议只改 oracle 和 population，没有改
meta-solver：

- Config：`configs/psro/psro_v4_20261029.json`（3 轮）；
- Blue：RL-Switch、RL-Wide160、RL-Pattern、RL-Repair、RL-PPO、Final-tuned、
  Triple-majority、Adv-RL；
- Red：Red-RL-v1 / v3 / v4（Red-v5/v6 仍只做 held-out）；
- Cross-play：16 seeds × 100 rounds；
- Oracle：RL-Repair 初始化，30 iter/轮，40% red，`bc_kl=0.02`，
  `opponent_penalty=1.0`，`lr=1e-4`，`fields_per_dist=5`，`n=96`。

### 元博弈

初始 blue maximin 仍是 Final-tuned 0.998 + Triple-majority 0.002，
exploitability **290.00**。第 1 轮 BR_01 的 cross-play 对 Red-v1/v3/v4
取得 **+33.1 / -265.9 / -203.8** 的 score difference；加入后 blue
support 直接变成 **BR_01 权重 1.0**，exploitability **0.00**。第 2/3 轮
BR 的 expected 只有 -73.3 / -10.0，均未进入 support。

这说明“更强 oracle + 更低 KL”可以训练出真正入 support 的 learned
best response；此前 v1 的 learned BR 全是权重 0。

### 诊断

| 指标 | PSRO-V4-BR | RL-Repair | 差 |
|---|---:|---:|---:|
| dev_param（10×1024） | **486.93** | 546.47 | **-59.54** |
| dev_param min_norm | 0.5599 | 1.0000 | -0.4401 |
| worst family | noisy 260.0 | late 305.98 | -45.9 |
| dev_complex（3×512） | **436.43** | 434.28 | +2.15 |
| dev_ood（3×512） | 477.72 | 515.19 | -37.47 |
| dev_axelrod（3×512） | 498.11 | 582.46 | -84.35 |
| dev_red red win（10 seeds） | **0.0** | 1.0 | — |
| Red-v5/v6 red win（10 seeds） | **0.0** | 1.0 | — |
| Red-v5/v6 blue mean | **530.55** | 348.35 | **+182.20** |

`dev_param` 的损失集中在 `noisy`（260.0 vs 464.4）、`adaptive`
（397.1 vs 585.8）和 `extort`（366.2 vs 496.8）；`defect` 680.0 仍是
强项。`dev_complex` 基本持平，说明 v4 不是简单的“全面变差”。

### 决策

- **No-Go for promotion**：dev_param -59.5、min_norm 0.56，远低于
  RL-Repair；主权重不变。
- 但它是目前最强的 anti-exploitation specialist：dev_red 和 held-out
  Red-v5/v6 都到 **red win 0.0**，且 dev_param 比 `psro-scratch` 高约
  6.5；`psro-v4-br` 是单策略可部署版本。
- `psro-v4` meta 只含 BR_01 一个成员，但每场重新采样 member RNG，
  因此仍有约 20–30% red win；对 red 稳定性以 `psro-v4-br` 为准。
- 下一步：约束 / 多目标 oracle（official family floor + red objective），
  或 red oracle / double oracle；不要重复同一 blue-only 3 轮配置。
- 完整结果：`results/psro_v4_20261029/`；checksum 见
  `results/psro_v4_20261029/checksums.txt`。
