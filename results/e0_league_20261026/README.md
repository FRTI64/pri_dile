# E0 — cross-play / Nash / exploitability（2026-10-26）

## 0. 结论

**E0 Gate: PASS。** 冻结的 9 个候选都在同一 suite/seeds 上完成评估，
cross-play、Nash、CVaR、held-out v2 和 E4 red 数据完整；主权重未改动。

关键结论：

1. **RL-PPO-Hard60（A60）仍是当前最强候选**：overall 552.11、held-out v2
   500.07、CVaR10 391.13，分别比 RL-PPO 高 **+1.31 / +5.10 / +3.87**。
   但它不是严格占优：blind `wild` family 归一化 0.9928（-0.72%），E4 red blue
   299.20（RL-PPO 310.55，低 11.35）；anti-exploitation 仍未解决。
2. **Adv-RL 的 held-out v2 最高（502.39）**，但 min_norm 0.9815、red blue
   325.55，且官方/blind 与 A60 接近；只在分布构成匹配时考虑。
3. **SelfPlay-GRU/Bc 确认 No-Go**：min_norm 0.949、held-out v2 491.8，
   低于 RL-PPO/A60；保留为 21 维/部署备选。
4. **cross-play 目前是退化的**：所有候选互相合作，payoff 594.3–595.3，
   Elo 约 1000，Nash 约均匀且 exploitability 0。说明蓝军池内部没有
   rock-paper-scissors 结构，**不能只靠 cross-play 选模**；必须加入
   red/exploiter pool（P3）才能测出真实被剥削风险。
5. E4 red 对所有候选 red win rate 95–100%。这再次确认：当前没有
   anti-exploitation 策略；“对训练 red 有效”不能当作解决。

**决策：不替换 `weights/rl_policy_weights.json`；A60 继续作为候选/fallback。
E0 只建立评估，不做权重 promotion。**

## 1. 冻结候选与 suite

- 候选：Final-tuned、Resp3-tuned、Triple-majority、RL-PPO、RL-PPO-Hard60、
  RL-BC、Adv-RL、SelfPlay-GRU、SelfPlay-BC。
- 官方四分布：20 fields × 299，seed 20261026。
- blind：20 fields × 299，seed 20261025。
- validation：51 固定对手 × 20 seeds。
- **heldout_v2**：新 74 对手、7 个新 family/参数，20 seeds，seed base
  20961026；本报告后视为 validation，不再用于反复选模。
- E4 red：20 seeds，seed base 20561026。
- cross-play：所有候选两两双向，20 seeds。

## 2. 总表

| Candidate | official | blind | validation | heldout_v2 | min_norm | CVaR10 fam | red blue |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final-tuned | 546.35 | 547.75 | 489.23 | 490.04 | 0.8540 | 360.92 | 334.05 |
| Resp3-tuned | 545.98 | 546.38 | 489.55 | 492.31 | 0.8698 | 358.40 | 326.20 |
| Triple-majority | 547.37 | 551.33 | 491.21 | 491.94 | 0.8739 | 357.34 | 337.55 |
| RL-PPO | 553.22 | 562.80 | 504.45 | 494.97 | 1.0000 | 387.26 | 310.55 |
| **RL-PPO-Hard60** | **554.27** | **563.87** | **507.85** | **500.07** | 0.9928 | **391.13** | 299.20 |
| RL-BC | 550.65 | 558.52 | 499.86 | 499.43 | 0.9589 | 381.36 | 333.95 |
| Adv-RL | 554.05 | 562.97 | 501.87 | **502.39** | 0.9815 | 387.14 | 325.55 |
| SelfPlay-GRU | 550.97 | 561.13 | 502.01 | 491.86 | 0.9494 | 377.55 | 301.65 |
| SelfPlay-BC | 550.92 | 560.48 | 502.50 | 491.79 | 0.9489 | 378.11 | 303.60 |

`min_norm` 是相对 RL-PPO 的逐 family 最小归一化分数；raw min_family
不可跨 family 比较。完整 per-family 见 `report.json` 和 `auto_report.md`。

## 3. Held-out v2 per-family（关键候选）

| Family | RL-PPO | A60 | Adv-RL | SelfPlay-GRU |
|---|---:|---:|---:|---:|
| reciprocal_v2 | 599.2 | 599.2 | 599.2 | 599.2 |
| punitive_v2 | 600.6 | 600.6 | 600.6 | 600.6 |
| noisy_v2 | 421.5 | 425.6 | 418.3 | 419.1 |
| extort_v2 | 419.1 | **441.9** | 421.8 | 411.6 |
| random_v2 | 409.1 | 420.6 | **442.3** | 398.8 |
| temporal_v2 | 447.6 | 450.9 | **462.7** | 444.8 |
| adaptive_v2 | 582.8 | 582.8 | 581.3 | 581.4 |

A60 的增益主要来自 extort/random/noisy；Adv-RL 的强项是 random/temporal。
这两个候选的强项不同，后续 multi-teacher/gating 有信号可做。

## 4. Cross-play / meta-game

- 所有 pair 的 payoff 在 594.3–595.3，win-rate 基本 0.5；Elo 998–1002。
- Nash 权重接近均匀（0.140–0.146，Resp3/RL-PPO 为 0），
  meta exploitability = 0，pool exploitability ≤ 0.0017。
- 这不是“所有策略都很稳”，而是**蓝军池互相合作，cross-play 区分度不足**。
  要测真正的 exploitability，需要 P3 的 red/exploiter checkpoint 池。

## 5. Held-out red

| Blue | Blue mean | Red mean | Diff | Red win |
|---|---:|---:|---:|---:|
| Final-tuned | 334.05 | 607.55 | +273.50 | 1.00 |
| Triple-majority | 337.55 | 606.55 | +269.00 | 1.00 |
| RL-PPO | 310.55 | 654.05 | +343.50 | 1.00 |
| RL-PPO-Hard60 | 299.20 | 651.20 | +352.00 | 1.00 |
| RL-BC | 333.95 | 614.45 | +280.50 | 1.00 |
| Adv-RL | 325.55 | 608.55 | +283.00 | 0.95 |
| SelfPlay-GRU | 301.65 | 663.65 | +362.00 | 1.00 |

E4 red 只针对 frozen BC blue 训练；对非 BC 蓝军是 held-out 压力测试，
但不代表真实比赛一定 100% 被击败。**anti-exploitation 未解决。**

## 6. 复现

```bash
# 全量（约 11 分钟；SelfPlay 纯 Python 推理较慢）
python3 -m ipd league --config configs/eval/e0_league.json --workers 20 \
  --out results/e0_league_20261026/report.json \
  --md-out results/e0_league_20261026/README.md

# 快速 cross-play 复现（无 suite）
python3 -m ipd league --config configs/eval/e0_league.json \
  --skip-suites --skip-red --workers 20 \
  --out /tmp/e0_cross.json

# 测试
python3 -m unittest tests.test_metagame tests.test_league -v
```

报告 fingerprint：`9e6b9598cdfbd7203b0d3c7ba27da4bb3e0521fb03c57643019ff92cee72761f`。
cross-play 重复运行两次 fingerprint 均为
`8719c95310f98b032ca81a21417027e70143a2f0bf33bb0d48dca5da1a3f99f5`，
矩阵和 matches 完全一致（Gate E0 可复现性通过）。
自动生成的原始 Markdown 保留在 `auto_report.md`；结构化数据在 `report.json`。

## 7. 下一步

1. **E1**：从 A60 做 domain-randomized fine-tune（`robust` 训练分布已实现并有测试）。
   选模只看官方 validation；heldout_v2 作为 validation；**新建 heldout_v3**
   做一次性 final test。E1a/E1b 均未通过 validation Gate，因此 heldout_v3
   截至本次仍未运行（保留给下一个候选）。
2. E1 若 Go，再考虑 E2 CVaR/robust PPO 或 E3 multi-teacher。
3. P3 league 不能只做蓝军 cross-play；必须加入 red/exploiter 池，否则
   当前均衡退化为“互相合作”。

## 8. 产物

```text
results/e0_league_20261026/
  report.json        # 结构化报告（含 per-family / cross-play / red / gate）
  README.md          # 本文件
  auto_report.md     # CLI 自动生成版本
  run.log            # 运行日志
  checksums.txt      # 权重/config/report checksum
```

主权重 `weights/rl_policy_weights.json` 的 sha256 为
`bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`，
与 E0 前一致；完整 checksum 见 `checksums.txt`。
