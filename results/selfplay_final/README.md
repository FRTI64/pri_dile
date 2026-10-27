# SelfPlay-GRU final report

最终选定策略：`weights/selfplay_gru_policy.json`（21 维观测，2 层 GRU hidden=64；
可部署参数 43,777，含 critic 共 45,890，推理不需要 PyTorch）。

## 1. 最终指标

| 测试 | 结果 |
|---|---:|
| 官方四分布 benchmark（20 fields × 299，seed=20261026） | **550.76** |
| 四分布 original / late / grim30 / wide | 551.43 / 546.37 / 549.82 / 555.41 |
| 四分布 worst | 546.37 |
| 8-field 快速对照（`benchmark.json`） | 552.52 |
| 官方 blind（10 fields，历史） | **560.43** |
| blind 平均排名 / 第一名 / 前 10 / 最差 | 8.10 / 7/10 / 7/10 / 25 |
| E0 冻结协议 blind 20 fields | 561.13 |
| Held-out validation（51 对手 × 5 seed） | 502.93 |
| Held-out final test（86 对手 × 20 seed + held-out red） | 505.06 |
| final test min-family | 360.77 |
| held-out red（RL-BC checkpoint，互惠型对照） | 594.50 |

对照（同一仓库已有结果）：

| 策略 | 四分布 | blind |
|---|---:|---:|
| RL-PPO（从零 + hard） | 553.33（20 fields） | 563.23 / 8.00 |
| Adv-RL 激进 | 554.29（20 fields） | 562.91 / 8.00 |
| **SelfPlay-GRU（本报告）** | **550.76（20 fields）** | **560.43 / 8.10** |
| Adv-MLP 激进 | 552.24 | 559.51 / 8.10 |
| GRU-Net | 547.81 | 555.22 / 8.30 |

本策略在四分布上与 RL-PPO 相差约 2.6 分，blind 低 2.8 分。需要明确：GRU 的
43,777 可部署参数（含 critic 45,890）**大于** RL-PPO 的 7,617 参数；它的优势
是固定 21 维观测和纯标准库推理，而不是参数量。

## 2. 关键验收项

在 held-out validation 上：

```text
TFT          604   (目标 >= 590/594)
Bully        878   (目标 >= 650)
NoisyTFT.02  604   (目标 >= 560)
AllD          99   (目标 >= 95)
min_family   370   (配置阈值 350)
held-out red 595   (目标 >= 430)
```

## 3. 训练来源

最终 checkpoint 使用 **BC 蒸馏 + 21 维 GRU**：

```bash
python3 -m ipd train selfplay-bc --device auto --episodes 4000 --epochs 10 \
  --out weights/selfplay_bc_init.pt \
  --policy-out weights/selfplay_bc_policy.json \
  --history-out results/selfplay_bc_history.json
```

- 教师：`weights/rl_policy_checkpoint.json`（现有 RL-PPO）。
- BC 数据：4000 个 episode，训练对手来自 self-play 对手池（含 Random/Extort/
  Special/Prey），不包含 validation/final 对手。
- 训练集动作一致率：99.35%。
- 输出：`weights/selfplay_bc_init.pt`（可用于
  `--init-checkpoint`）和 `weights/selfplay_bc_policy.json`。

完整 self-play 训练器（`ipd/training/selfplay/`）已实现并跑通 15,000 episode：

```bash
python3 -m ipd train selfplay --config configs/selfplay/full.json \
  --device auto --final-test
```

从零 self-play 的结果（`results/selfplay_20261026/`）：

- 15,000 episode / 469 iterations，三阶段课程 + red league；
- held-out final test 504.57；
- 官方四分布（2 fields，仅用于快速对照）524.77；
- checkpoint：`weights/selfplay_checkpoint.pt`；
- 可部署策略：`weights/selfplay_scratch_policy.json`。

纯从零 self-play 在 15k episode 内没有稳定学会“对有响应型对手合作、
对无响应型对手收割”的条件策略，因此最终提交选择 BC 蒸馏策略；self-play
训练器保留，可用 `--init-checkpoint weights/selfplay_bc_init.pt` 继续做
red league 微调（`configs/selfplay/full_bc.json`）。

## 3.1 改进实验结论（2026-10-26，No-Go）

同 suite / 同 seeds 对照（`results/selfplay_improvement/compare.json`）：

- BC-only 与 RL-PPO 在 final suite 上基本打平（506.98 vs 506.49）；
- scratch self-play 的 `min_norm_family` 只有 0.783，明显偏科；
- BC+red fine-tune E1a 仅 +0.43 mean / +0.016 min_norm，E1b 崩坏到 429；
- 针对 frozen blue 训练的 red best response 100% 击败 BC/RL/Adv，
  不能宣称 anti-exploitation。

因此 self-play / red league 方向 **No-Go**，不替换本目录的最终权重。
完整决策见 `results/selfplay_improvement/final_decision.md`；最终
benchmark/blind 复现、权重 checksum 与冻结建议见
`results/selfplay_improvement/sanity/README.md`。

## 4. 复现最终评估

```bash
python3 -m ipd benchmark --candidates selfplay \
  --weights weights/selfplay_gru_policy.json --fields 20 --workers 20 \
  --out results/selfplay_final/benchmark_20fields.json
python3 -m ipd blind --candidates selfplay \
  --weights weights/selfplay_gru_policy.json --fields 10 --workers 14 \
  --out results/selfplay_final/blind.json
```

`validation.json` / `final_test.json` 由 `ipd.training.selfplay.evaluate` 生成，
使用 `configs/selfplay/full_bc.json` 中的 held-out suite 和 held-out red。

**注意：** 这里的 held-out red 是 RL-BC 互惠型策略，20 局几乎都是 595 | 595，
它只能作为“合作型基模”对照，不能证明 anti-exploitation。真正的 red best
response / Red-Blue 对抗评估见 `results/selfplay_improvement/`。
