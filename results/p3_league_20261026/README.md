# P3 expansion — multi-red pool + PSRO-lite blue response（2026-10-26）

## 0. 结论

**Red pool 扩展：Go（研究 artifact）；PSRO-lite blue response：No-Go。**

- 新训练 5 个 85 维 red：v2（Adv-RL/SelfPlay/规则池）、v3（RL 家族 +
  上一版 blue response）、v4（Adv/SelfPlay/RL-Repair/blue response），以及
  严格 held-out 的 v5/v6。
- red pool eval 后，learned-blue maximin 选择 **Red-RL-v4** 为主（0.70），
  v1/v2/v3 各 0.10；`python -m ipd meta` 生成 weighted opponent manifest。
- 两个 blue response（50% / 60% red）和第三个 v3-heavy 对照训练完成。
  B1（50% red）是唯一同时保持 official/blind/heldout_v2 的候选；B2 的
  extort family 掉 7.1%，B3 的 official validation 直接回退到初始
  RL-Repair。
- **Gate D：FAIL。** B1 的 held-out exploitability 从 390.33 降到 367.33
  （**-5.9%**，要求 -20%）；official +0.24 通过，但 `validation:extort`
  掉 **-3.21%**，超过 2% family 约束。E4 -10.5%、Red-v5 -1.7%、
  Red-v6 -4.7%；三个 held-out red 仍然 100% 击败 blue。
- **不 promotion。** RL-Repair 仍是当前最佳候选；不注册 `rl-meta`；
  `heldout_v4` 仍未使用。P3 方向判定为**没有达到 Gate D 的信号**，下一步
  转向 P2 C1（Bayesian belief planner 离线评估）。

## 1. Red pool 结果

训练 pool 与 seed 见 `docs/p3-expansion-plan.md`。下表是 20 seeds 下
red − blue 的均值（learned blues = RL-PPO/A60/RL-Repair/Adv-RL/
SelfPlay/Blue-vs-Red-Comp-v1；越高表示 red 越强）：

| Red | seed | learned-blue mean | worst learned blue | vs RL-Repair |
|---|---:|---:|---:|---:|
| Red-RL-v1（已有） | 20261036 | 377.2 | 339.0 | 399.5 |
| Red-RL-v2 | 20261039 | 319.8 | 306.5 | 321.5 |
| Red-RL-v3 | 20261040 | 442.2 | 321.5 | 486.0 |
| Red-RL-v4 | 20261043 | 383.6 | 352.5 | 396.5 |
| Red-RL-v5-heldout | 20261041 | 330.5 | 323.5 | 331.0 |
| Red-RL-v6-heldout | 20261042 | 406.9 | 337.0 | 429.0 |

关键观察：

- v3 在 RL 家族上最锋利（RL-Repair 486），但对 Adv-RL 只有 321.5；
- v4 是最稳的强 red（worst learned blue 352.5），因此 maximin 给它
  0.70 权重；
- v5 意外地“均衡但弱”（331 vs RL-Repair），v6 更强（429 vs RL-Repair）；
- 六个 red 对所有 blue 的 red win rate 都是 100%，再次说明 anti-exploitation
  未解决。

完整 per-pair 数据：`red_eval.json`、`red_eval.md`。

## 2. Meta 权重

`python -m ipd meta` 使用 learned-blue columns，排除 Final/Triple 规则
baseline（否则弱 column 会把 maximin 推向 v2 这种“均衡但弱”的 red）：

| Red | maximin | final weight（0.6 maximin + 0.4 uniform） |
|---|---:|---:|
| Red-RL-v1 | 0.000 | 0.100 |
| Red-RL-v2 | 0.000 | 0.100 |
| Red-RL-v3 | 0.00005 | 0.100 |
| Red-RL-v4 | 0.99995 | 0.700 |

worst-case red−blue = 345.75 vs Adv-RL。权重文件：
`meta_weights.json`；训练 manifest：`blue_meta_pool.json`。

## 3. Blue response 训练与选模

| 候选 | red 比例 | 训练 pool | official val best | 结果 |
|---|---:|---|---:|---|
| Blue-Meta-b1 | 50% | maximin v4-heavy | 557.92（iter 5） | **选中** |
| Blue-Meta-b2 | 60% | maximin v4-heavy | 557.35（iter 10） | extort -7.1%，拒绝 |
| Blue-Meta-b3 | 60% | v3-heavy（70% v3） | 563.40（iter 0） | 无改进，拒绝 |

B1 的后续 checkpoint（iter 15/20）虽然进一步降低训练 red exploitability，
但 blind `defect` family 崩塌（min_norm 0.867/0.816），不可用。最终只使用
B1 iter 5；B2/B3 保留为研究 artifact。

B1 的最终指标（`blue_vs_heldout_red.json`）：

```text
official        554.999
blind           564.345
validation      508.982
heldout_v2      502.938
min_norm        0.9679
CVaR10 family   393.90
CVaR10 item     241.21
```

相对 RL-Repair（official 554.764、blind 564.605、heldout_v2 504.395）：
official +0.24、blind -0.26、heldout_v2 -1.46、CVaR family +2.08、
CVaR item -2.15。标准 CLI 独立 seed 复现（legacy `--arch rl` 会把自定义
权重显示为 RL-PPO）：benchmark 20 fields **554.79**、blind 20 fields
**564.28 / 5.75**。

## 4. Gate D

| 条件 | 结果 | 判定 |
|---|---|---|
| held-out exploitability 降 ≥20% | 390.33 → 367.33（**-5.9%**） | **FAIL** |
| official 不降 >2 | +0.24 | PASS |
| 无 family 掉 >2% | `validation:extort` **-3.21%** | **FAIL** |
| 单个 held-out red 不变差 >5% | E4 -10.5%、v5 -1.7%、v6 -4.7% | PASS |

Per-red：

| Held-out red | RL-Repair diff | Blue-Meta-v2 diff | Δ | red win |
|---|---:|---:|---:|---:|
| E4-red-BC | 408.5 | 365.5 | -10.5% | 1.00 |
| Red-RL-v5-heldout | 330.5 | 325.0 | -1.7% | 1.00 |
| Red-RL-v6-heldout | 432.0 | 411.5 | -4.7% | 1.00 |
| **mean** | **390.33** | **367.33** | **-5.9%** | 1.00 |

**Gate D: FAIL。** 与上一版单 red league-lite（E4 -12.8%）相比，多 red +
maximin 并没有提高 E4 降幅，只把 v6 的降幅做到 -4.7%。official 约束下
可用的 B1 变化太小；更激进的 B2 则以 extort family 崩塌为代价。

## 5. 决策与下一步

- 不 promotion，不注册新候选；`weights/rl_policy_weights.json` 与
  RL-Repair 权重均未改动。
- P3 结论：**当前 reward（own − opponent）+ 85 维 MLP + official 选模
  无法在保持 family 的同时达到 -20% exploitability**。继续加 red 数量或
  调 meta 权重的边际收益低。
- 按计划转 **P2 C1**：Bayesian opponent belief + finite-horizon planning，
  先做离线评估和 fallback，不改主策略。P3 的 red/held-out 评估 harness
  保留复用。
- `heldout_v4` 仍未使用。

## 6. 复现

```bash
# 训练 red（v2-v4；v5/v6 用 train_heldout_reds.sh）
bash results/p3_league_20261026/train_reds.sh
bash results/p3_league_20261026/train_heldout_reds.sh

# red pool eval + maximin manifest
bash results/p3_league_20261026/evaluate_red_pool.sh

# blue responses
bash results/p3_league_20261026/train_blue.sh
bash results/p3_league_20261026/train_blue_b3.sh

# 选模（official/blind/heldout_v2 + training reds v3/v4）
python3 -m ipd league \
  --config configs/eval/p3_blue_checkpoint_selection.json \
  --skip-cross-play --workers 20 \
  --out results/p3_league_20261026/blue_checkpoint_selection.json

# 选中 B1 iter5 后做一次 held-out red final test + Gate D
cp weights/red_pool/blue_meta_b1.json weights/red_pool/blue_meta_v2.json
cp /tmp/p3_league/blue_b1_checkpoint.json \
   weights/red_pool/blue_meta_v2_checkpoint.json
bash results/p3_league_20261026/finalize_blue.sh
```

## 7. 产物

```text
configs/league/
  p3_red_v2_blues.json ... p3_red_v6_heldout_blues.json
configs/eval/
  p3_red_pool_20261026.json
  p3_blue_selection_trainreds.json
  p3_blue_checkpoint_selection.json
  p3_blue_vs_heldout_red_20261026.json
weights/red_pool/
  red_rl_v2.json ... red_rl_v6_heldout.json
  blue_meta_b1.json / blue_meta_b2.json / blue_meta_b3.json
  blue_meta_v2.json / blue_meta_v2_checkpoint.json
results/p3_league_20261026/
  README.md
  train_reds.sh / train_heldout_reds.sh
  evaluate_red_pool.sh / train_blue.sh / train_blue_b3.sh / finalize_blue.sh
  red_v*_train.log / red_v*_train.json
  red_eval.json / red_eval.md
  meta_weights.json / meta_weights_prelim.json
  meta_weights_v3heavy.json
  blue_meta_b*_train.log / blue_meta_b*_train.json
  blue_selection_trainreds.json / .md
  blue_checkpoint_selection.json / .md
  blue_vs_heldout_red.json / .md
  gate_d.json
  benchmark/blue_meta_v2.json / blind/blue_meta_v2.json
  checksums.txt
```

主权重 `weights/rl_policy_weights.json` sha256 仍为
`bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。
