# SelfPlay 改进实验：Go / No-Go 结论

- **Date:** 2026-10-26
- **Plan:** `docs/research/selfplay-improvement.md`
- **结论:** **No-Go（停止 self-play 方向）**。不替换 `weights/selfplay_gru_policy.json`，
  不覆盖 `results/selfplay_final/`。
- **最终策略建议：**
  - 无约束、只看分数：**RL-PPO**（553.33 / 563.23）或 **Adv-RL**（554.29 / 562.91）；
  - 强制 21 维 / GRU / 无 PyTorch：保留 **SelfPlay-GRU（BC-only）**
    （550.76 / 560.43 / 8.10），但必须标注它是 BC 蒸馏，不是 self-play 增益。

---

## 1. P0：公平对照（同 suite / 同 seeds）

`results/selfplay_improvement/compare.json` 汇总了：

- `compare_E0_final.json`：final suite 86 对手 × 20 seeds；
- `compare_E0_validation.json`：validation suite 51 对手 × 5 seeds；
- `compare_E1_validation.json`：BC / RL / E1a / E1b 同 seeds 对照。

### 1.1 Validation（51 对手 × 5 seeds）

| 策略 | mean | min_raw | min_norm vs RL | deployable params |
|---|---:|---:|---:|---:|
| BC-only | 501.66 | 369.7 | 0.953 | 43,777 |
| RL-PPO | 505.33 | 369.4 | 1.000 | 7,617 |
| Adv-RL | 505.86 | 367.1 | 0.994 | 7,617 |
| Scratch self-play 15k | 483.87 | 339.4 | 0.764 | 43,777 |

### 1.2 Final test（86 对手 × 20 seeds）

| 策略 | mean | min_raw | min_norm vs RL | 说明 |
|---|---:|---:|---:|---|
| BC-only | 506.98 | 365.6 | 0.991 | 与 RL 基本打平 |
| RL-PPO | 506.49 | 359.0 | 1.000 | baseline |
| Adv-RL | 504.08 | 347.5 | 0.968 | extort 较低 |
| Scratch self-play 15k | 505.79 | 340.2 | 0.783 | random 340，明显偏科 |

结论：BC-only 在 final suite 上不比 RL-PPO 差；scratch self-play 的
`min_norm_family` 只有 0.783，是明确的负向结果。

### 1.3 参数量更正

```text
SelfPlay-GRU: 可部署 43,777；含 critic 45,890
RL-PPO:       可部署 7,617；部署权重本身无 critic
```

GRU **不是更小**。它的优势是固定 21 维观测和纯标准库推理。

---

## 2. P1：BC + red league fine-tune A/B

所有实验都从 `weights/selfplay_bc_init.pt` 初始化，未覆盖已有最终结果。

| ID | episodes | bc_kl | blue_lr | held-out mean | min_norm | 官方 4-field | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| BC-only | - | - | - | 501.66 | 0.953 | 546.84 | baseline |
| E1a | 15,000 | 0.2 | 2e-4 | 502.09 (+0.43) | 0.969 (+0.0156) | 546.00 (-0.84) | 未达 Go |
| E1b | 15,000 | 0.1 | 1e-4 | 429.21 | 0.610 | 未测 | 灾难性退化 |

### Go / No-Go 判定

计划要求同时满足：

1. held-out mean +10 **或** min_norm +0.03；
2. 官方四分布下降不超过 2；
3. TFT/Bully/NoisyTFT.02/AllD 红线全过；
4. 对 red best response 不差于 BC；
5. 无新 family 崩塌。

实际：

- E1a：mean +0.43（<10），min_norm +0.0156（<0.03）；官方 -0.84（通过）；
  红线通过；但核心提升不足。
- E1b：mean -72.45，min_norm -0.343，直接崩坏；训练期没有 checkpoint
  通过红线，回退到 iter 32 的早期 checkpoint。
- **因此 E1a/E1b 都不满足 Go；E1c（30k）不启动。**

E1a 的 `extort` family 有改善（417→436），但 `noisy`（514→504）和
`random`（433→430）略降，整体 mean 几乎不变；这说明当前 red league 不能带来跨 family
的鲁棒性增益，只是局部调参。

---

## 3. E4：真正的 red best response

按计划新增 `--freeze-blue`，用 E4 训练一个针对 frozen BC blue 的 red
best response（6,000 episodes，red fraction 30–40%，blue 不更新）。

`results/selfplay_improvement/E4_red_eval.json` / `E4_red_eval_baselines.json`：

| Blue | Blue mean | Red mean | Diff | Red win rate |
|---|---:|---:|---:|---:|
| BC-only | 304.60 | 636.10 | -331.50 | 100% |
| Scratch self-play | 274.40 | 518.40 | -244.00 | 100% |
| E1a | 304.25 | 623.75 | -319.50 | 100% |
| E1b | 293.55 | 667.55 | -374.00 | 100% |
| RL-PPO | 306.95 | 648.45 | -341.50 | 100% |
| Adv-RL | 328.00 | 621.00 | -293.00 | 100% |

结论：

- 针对 frozen blue 训练的 Red-Blue best response 能 100% 击败所有当前
  blue（包括 RL-PPO / Adv-RL），因此“对针对性 best response 的
  anti-exploitation”目前没有被任何策略解决；
- BC-only 对 red 的表现（304.6）与 RL-PPO（306.95）基本打平，略差于
  Adv-RL（328.0）；不能据此说 BC 特别脆弱，但也不能宣称 anti-exploitation；
- 旧 `final_test.json` 中的 held-out red（RL-BC 595 | 595）只是互惠型
  对照，**不能作为 anti-exploitation 证据**，已在 `results/selfplay_final/README.md`
  和 `docs/policies/selfplay.md` 中标注。

---

## 4. 最终决策

```text
E0: BC-only ≈ RL-PPO > Adv-RL > scratch self-play
        │
E1a/E1b: mean +0.43 / min_norm +0.016；E1b 崩坏
        │
        └─ No-Go
              ├─ 不继续 E1c / E3 hard mining
              ├─ 不替换最终 weights/results
              └─ 提交建议：
                   score-only 默认: RL-PPO
                   条件选项: Adv-RL（仅官方四分布构成匹配时）
                   21 维/GRU/无 PyTorch: SelfPlay-GRU (BC-only)
```

### 4.1 最终 sanity check（2026-10-26）

- benchmark（20 fields × 299）复现：RL-PPO 553.33、Adv-RL 554.29、
  SelfPlay-GRU 550.76，与记录值完全一致；
- blind（10 fields）复现：RL-PPO 563.23 / 8.00、Adv-RL 562.91 / 8.00、
  SelfPlay-GRU 560.43 / 8.10，与记录值完全一致；
- 权重 checksum、路径与冻结建议：
  `results/selfplay_improvement/sanity/README.md` 和
  `results/selfplay_improvement/sanity/checksums.txt`；
- 主提交建议 RL-PPO；21 维备选 SelfPlay-GRU；Adv-RL 仅在官方四分布构成
  匹配时使用；不替换 `weights/selfplay_gru_policy.json`。

### 不要做的事

- 不要把 `weights/selfplay_gru_policy.json` 说成 self-play 训练的最终策略；
  它是 RL-PPO 的 21 维 BC 蒸馏。
- 不要用 `final_test.json` 的 RL-BC 595|595 宣称 anti-exploitation。
- 不要用 raw min_family 360 跨 family 比较；看 `min_norm_family` 和
  per-opponent 明细。
- 不要宣称 GRU 参数更小。
- 不要在 No-Go 后继续堆 self-play 功能；保留代码作为研究 artifact。

---

## 5. 产物索引

```text
results/selfplay_improvement/
  compare.json                     # E0/E1 同 seeds 对照汇总
  compare_E0_final.json            # final 86 × 20
  compare_E0_validation.json       # validation 51 × 5
  compare_E1_validation.json       # BC/RL/E1a/E1b
  E1a/                             # 15k 训练日志、eval_full_*.json、history
  E1b/
  E4_red/                          # frozen blue + red best response 训练
  E4_red_eval.json                 # BC/scratch vs E4 red
  E4_red_eval_E1.json              # E1a/E1b vs E4 red
  E4_red_eval_baselines.json       # RL/Adv vs E4 red
  E1a_benchmark_4fields.json       # E1a 官方四分布 4-field
  BC_benchmark_4fields.json        # BC 官方四分布 4-field
  final_decision.md                # 本文件
weights/selfplay_improvement/
  E1a_policy.json / E1a_checkpoint.pt
  E1b_policy.json / E1b_checkpoint.pt
  E4_red_policy.json / E4_red_checkpoint.pt
  E4_blue_frozen.json
```

复现命令：

```bash
python3 -m ipd train selfplay-compare \
  --candidates bc,rl-ppo,adv-rl,scratch --suite final --seeds 20 \
  --out results/selfplay_improvement/compare_E0_final.json

python3 -m ipd train selfplay \
  --config configs/selfplay/E1a.json --device auto \
  --init-checkpoint weights/selfplay_bc_init.pt

python3 -m ipd train selfplay \
  --config configs/selfplay/E1b.json --device auto \
  --init-checkpoint weights/selfplay_bc_init.pt

python3 -m ipd train selfplay \
  --config configs/selfplay/E4_red.json --device auto \
  --init-checkpoint weights/selfplay_bc_init.pt \
  --freeze-blue --red-out weights/selfplay_improvement/E4_red_policy.json

python3 -m ipd train selfplay-red-eval \
  --red weights/selfplay_improvement/E4_red_policy.json \
  --blue bc=weights/selfplay_bc_policy.json \
  --blue scratch=weights/selfplay_scratch_policy.json --seeds 20
```
