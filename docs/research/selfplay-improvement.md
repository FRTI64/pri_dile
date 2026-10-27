# SelfPlay-GRU 改进计划

- **Status:** executed 2026-10-26; **No-Go**; see `results/selfplay_improvement/final_decision.md`
- **Repo:** `/home/frti64/pri_dile`
- **Based on:** `results/selfplay_final/`、`docs/policies/selfplay.md`、`weights/selfplay_gru_policy.json`
- **目标:** 判断 self-play + red league 是否真的能提升 held-out 鲁棒性；如果不能，明确停止条件并选择最佳提交策略。

---

> 执行结果（2026-10-26）：E1a +0.43 mean / +0.016 min_norm，E1b 崩坏；
> red best response 100% 击败 BC/RL/Adv。按计划停止 self-play 方向，
> 不替换现有最终权重。详细对照见 `results/selfplay_improvement/compare.json`。

## 0. 当前结论

### 0.1 最终策略是什么

`weights/selfplay_gru_policy.json` 的 metadata：

```text
algorithm = bc-distillation
teacher   = weights/rl_policy_checkpoint.json
episodes  = 4000
```

也就是说，**最终提交是 RL-PPO 的 21 维 BC 蒸馏，不是 self-play 训练出来的策略**。

### 0.2 关键结果

| 指标 | SelfPlay-GRU | RL-PPO | 说明 |
|---|---:|---:|---|
| 官方四分布 | 550.76 | 553.33 | 低 2.57 |
| 官方 blind（10 fields，历史） | 560.43 / 8.10 | 563.23 / 8.00 | 低 2.80 |
| 同 held-out suite / 同 5 seeds | **509.73** | **509.33** | 基本打平 |
| held-out final test（20 seeds） | 505.06 | — | suite 本身更难 |
| 参数量 | **45,890** | **7,617** | GRU 反而更大 |

结论：

- 21 维 GRU 忠实压缩了 RL-PPO，但没有比 RL-PPO 更鲁棒；
- 官方 -2.6/-2.8 是压缩成本；
- held-out 505 vs 官方 550 不能直接全部归因于 overfitting，因为 held-out suite 包含更多高敲诈/低概率随机/晚变脸/高噪声对手，raw 上限更低；
- 真正的问题是：**self-play 的增量还没有被验证**。

### 0.3 主要弱点（final test）

明显偏弱：

- `Extort 2.5 / 3.2 / 3.4 / 3.6`：约 170–228
- `Random 0.02 / 0.08 / 0.17 / 0.27`：约 119–344
- `LateD 68 / 72 / 88 / 94`：约 126–257
- `NoisyTFT 0.28–0.35`：约 392–413
- `Joss 0.5`：约 221
- `Kamikaze8`：约 194
- `DynamicW2`：约 217

明显强：

- TFT / Grim / Pavlov：604
- Final-tuned / Triple-majority / RL-final：595
- MirrorAdaptive / DelayedGrim：595–604
- Bully：878
- Random 0.92：824

这是典型的“合作基本盘强、对高敲诈/低概率随机/晚期变脸/高噪声弱”的 profile，和 RL-PPO 的已知尾部风险一致。

### 0.4 held-out red 测试不够强

`final_test.json` 的 held-out red 是 `weights/rl_policy_checkpoint_bc.json`，20 局几乎全是 `595 | 595` 互惠。  
它没有针对最终策略训练，也没有施加剥削压力，因此**不能证明 anti-exploitation**。

### 0.5 min_family 指标需要归一化

`min_family 360.8` 是 extort family；RL-PPO 在同一 suite 的 extort 约 `361.4`。  
raw score 跨 family 不可比：Extort/AllD 的上限本来就低。  
不能只用 `360 > 350` 证明鲁棒性，也不能只用 raw min_family 判断崩坏。

---

## 1. 改进目标

### P0：建立公平对照

- 同一 suite、同一 seeds 上比较：BC-only、RL-PPO、Adv-RL、scratch self-play、BC+red fine-tune。
- 加 per-family normalized 指标，而不是只看 raw mean / raw min_family。
- 用真正的 held-out red best response 测 anti-exploitation。

### P1：跑缺失的核心实验

- `BC init + red league fine-tune`（`configs/selfplay/full_bc.json`）还没有作为最终结果跑完/评估。
- 这是判断 self-play 是否有用的核心 A/B，优先于任何新功能。

### P2：如果 P1 失败，停止 self-play

- 不要继续用官方四分布调参。
- 直接选最佳提交策略，保留 self-play 代码作为研究/复现 artifact。

### P3：如果 P1 成功，再做定向改进

- 观测特征 `pD-pC` / 响应差；
- hard-example mining；
- 更强的 red best response / league；
- 21 维 MLP vs GRU 对比。

---

## 2. 非目标

- 不追求“官方四分布再高 0.5 分”而忽略 held-out。
- 不在 final test 上反复选模。
- 不把“统计 TFT / 赤字触发 / 硬切 AllD / 动态抢跑”直接写死成策略逻辑。
- 不宣称 GRU 比 RL-PPO 参数更小（事实相反）。
- 不把 RL-BC 当作“强对抗 red”来验收。

---

## 3. P0：评估对照工具（最高优先级）

### 3.1 新增 compare 脚本

建议新增：

```text
ipd/training/selfplay/compare.py
```

或 CLI：

```bash
python3 -m ipd train selfplay-compare \
  --candidates bc,rl-ppo,adv-rl,scratch,fine-tune \
  --suite final --seeds 20 --out results/selfplay_improvement/compare.json
```

输入：

- candidate 列表（每个 candidate 有 build 函数 / weights / config）；
- suite：`validation` 或 `final`；
- seeds；
- baseline 名称（默认 RL-PPO）。

输出：

- 每个 candidate 的 raw mean；
- per-family mean；
- **normalized family score**：`score / baseline_family` 或 `score / max_achievable_family`；
- `min_norm_family`；
- per-opponent 明细；
- `top_family_contribution`、`max-min family gap`；
- vs FT/TM；
- red 对照：score difference、win rate、blue cooperation score。

### 3.2 修正 min_family 指标

不要只报 raw `min_family`。至少同时报：

```text
min_family_raw
min_family_norm_vs_baseline
min_family_norm_vs_ceiling
max_family - min_family
top_family contribution to mean
```

对 Extort/AllD 这类低上限 family，raw score 必须和 baseline/上限一起看。

### 3.3 真正的 held-out red

建议同时保留三类：

| Red 类型 | 来源 | 用途 |
|---|---|---|
| Red-Base | 冻结 Final-tuned / Triple-majority / RL-final | 模拟“研究基模的强队” |
| Red-Blue | 对最终 blue 训练 N 轮 best response，然后冻结 | 模拟“现场 scout 你的对手” |
| Red-Strong | 现有强策略 / 绝对分场 | 模拟“不专门针对你但很强” |

评估指标：

- red 对 blue 的 score difference；
- red 胜率；
- blue 的绝对分；
- blue 的 TFT/Grim 合作分是否被 red 压崩。

**不要把 RL-BC 互惠 595 当作 anti-exploitation 验收。**

### 3.4 冻结 final test

- final test seeds / suite 只跑一次；
- 所有调参、early stop、模型选择都在 validation 上；
- compare 脚本对 final test 只做最终报告。

---

## 4. P1：BC init + red league fine-tune A/B

### 4.1 实验配置

以 `configs/selfplay/full_bc.json` 为起点：

```text
init_checkpoint = weights/selfplay_bc_init.pt
bc_kl_coef      = 0.2
blue_lr         = 2e-4
center_advantages = true
entropy 1a/1b/2 = 0.01, 3 = 0.005
red fraction Phase 2 = 0.15-0.25, Phase 3 = 0.20-0.30
```

建议做三组：

| ID | Episodes | bc_kl_coef | blue_lr | 目的 |
|---|---:|---:|---:|---|
| E1a | 15,000 | 0.2 | 2e-4 | 默认 fine-tune |
| E1b | 15,000 | 0.1 | 1e-4 | 更保守，减少偏离 BC |
| E1c | 30,000 | 最佳 | 最佳 | 看预算是否是瓶颈 |

先跑 2 个 seed 做 screening；只对最佳配置跑 full seed。

### 4.2 命令

```bash
python3 -m ipd train selfplay \
  --config configs/selfplay/full_bc.json \
  --init-checkpoint weights/selfplay_bc_init.pt \
  --device auto --final-test
```

注意：

- 不要覆盖 `weights/selfplay_gru_policy.json` 和 `results/selfplay_final/`；
- 新增 `results/selfplay_improvement/E1a/` 等目录；
- 记录 KL、action agreement with BC、red constraint violation、league quarantine 率。

### 4.3 Go / No-Go 标准

只有同时满足才继续：

1. held-out mean 比 BC-only 提升 **≥ 10 分**，或 `min_norm_family` 提升 **≥ 0.03**；
2. 官方四分布不比 BC-only 低超过 **2 分**；
3. TFT / Bully / NoisyTFT.02 / AllD 红线全部通过；
4. 对 held-out red best response 的分差/胜率不差于 BC-only；
5. 没有 family 出现新的崩塌。

如果 E1a/E1b/E1c 都不满足：**停止 self-play 方向**，进入 P2。

---

## 5. P2：如果 P1 失败（当前更可能）

### 5.1 提交策略

> 本节是 self-play 改进计划执行前的历史建议；当前最佳候选已变为
> RL-Repair，见 [README.md](../../README.md) / [docs/operations/handover.md](../operations/handover.md)。

| 场景 | 建议提交 |
|---|---|
| 只看分数，允许现有 RL-Repair | **RL-Repair（554.76 / 564.61）** |
| 保守 / 不替换主权重 | RL-PPO（553.33 / 563.23，10 fields） |
| 允许 Adv-RL | Adv-RL（554.29 / 562.91，10 fields） |
| 强制 21 维 / GRU / 无 PyTorch | SelfPlay-GRU（550.76 / 560.43），定位为可部署压缩版 |
| 需要更小模型 | 先试 21 维 MLP，不要默认 2 层 GRU |

### 5.2 报告要改的地方

- 参数量：45,890 vs 7,617，GRU 更大；
- 加同一 held-out suite 的 RL-PPO 对照；
- 说明 held-out red 是 RL-BC 互惠，不是强对抗；
- 解释 raw min_family 不可跨 family 比较；
- 明确最终策略是 BC-only，不是 self-play fine-tune。

---

## 6. P3：如果 P1 成功，再做定向改进

### 6.1 观测特征

- v2：第 17 维换成响应差 `pD - pC`；
- v3：同时保留 v1 和 v2（总维度需重新定义）；
- 加 windowed conditional defection rates，而不是只看累计；
- 测试时从 BC init 跑 5k–10k，不要从零。

### 6.2 Hard-example mining

重点提升：

- `Extort` chi 2.5–3.6；
- `Random` p 0.02–0.3；
- `LateD` T 55–94；
- `NoisyTFT` 0.28–0.4；
- `Joss` 0.5；
- `Kamikaze` / `DynamicW2`。

方法：

- 按 validation deficit 做 prioritized opponent sampling；
- 每个 family 保留 held-out 参数 slice，避免过拟合；
- 不要把所有弱 family 同时拉满，分阶段加。

### 6.3 Red 强化

- Red-Blue 用旧 blue 快照，不直接用当前 blue；
- 训练一个对 final blue 的 best response，冻结后作为 held-out red；
- 保持 red league，检查 quarantine 率；
- per-family anchor，不要只看全局平均。

### 6.4 模型对比

- 21 维 MLP（例如 21→64→32→1）vs 2 层 GRU 64；
- 比较参数量、推理速度、held-out mean、min_norm_family；
- 如果 MLP 在 2 分以内，优先 MLP：更小、更简单、更容易导出。

---

## 7. 实验矩阵

| ID | 目的 | 初始化 | 预算 | 主要指标 | 通过条件 |
|---|---|---|---:|---|---|
| E0 | 公平对照 | BC / RL-PPO / Adv-RL / scratch | eval only | 同 suite/seeds | 建立 baseline |
| E1a | BC+red 默认 | BC | 15k | held-out, official | +10 mean 或 +0.03 min_norm |
| E1b | BC+red 保守 | BC | 15k | 同上 | 同上 |
| E1c | BC+red 长预算 | BC | 30k | 同上 | 同上 |
| E2 | 特征 v2/v3 | BC | 5–10k | extort/random/lateD | 弱 family 提升 |
| E3 | hard-example mining | BC | 10k | 弱 family | 弱 family 提升且无新崩坏 |
| E4 | red best response | final blue | eval only | difference/win rate | 不差于 baseline red |
| E5 | MLP vs GRU | BC | 4k BC + eval | params/speed/score | MLP 在 2 分内则选 MLP |

---

## 8. 计算与时间预算

- E0：只做评估，约 1 小时内。
- E1a/E1b：15k episodes；先 1 个 seed 筛选，再 2 个 seed 确认。
- E1c：30k episodes；只在 E1a/E1b 有正向信号时跑。
- E2/E3：5–10k episodes，从 BC init 开始。
- E4：只训练 red best response，不训练 blue；用于评估。
- E5：BC 4k + 评估，成本低。

**原则：先用 held-out 小预算筛选，不要一上来跑 full 15k/30k。**

---

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| 在 validation 上反复调参导致过拟合 | 保留 final test，不参与选择 |
| red best response 太强，blue 过度防御 | 监控 TFT/Grim 合作分，设 red 强度上限 |
| 特征 v2 震荡 | KL to BC、低 LR、短预算 screening |
| hard mining 过拟合弱 family | 每个 family 留 held-out 参数 slice |
| 计算时间不够 | 只跑 E1a/E1b，E1c 只在正向时跑 |
| 报告继续误读 raw min_family | 强制输出 normalized 指标 |
| 参数/模型声明错误 | 报告前自动统计参数量 |

---

## 10. 交付物

- `results/selfplay_improvement/compare.json`：同 suite 多候选对照；
- `results/selfplay_improvement/E1*/`：fine-tune 结果；
- `results/selfplay_improvement/final_decision.md`：go/no-go 与提交建议；
- 如需替换：新的 `weights/selfplay_*` 与 `results/selfplay_final/README.md` 更新；
- 修正 `docs/policies/selfplay.md` 的参数量、held-out red、min_family 说明。

---

## 11. 最终决策树

```text
E0 baseline
  │
  ├─ E1a/E1b (15k BC+red)
  │     │
  │     ├─ held-out +10 / min_norm +0.03，官方不降 >2
  │     │     └─ E1c 30k → 若稳定 → 替换最终策略
  │     │
  │     └─ 没有提升
  │           ├─ 如果弱 family 明显且特征 v2 有信号 → E2/E3
  │           └─ 否则停止 self-play → 提交 RL-PPO / Adv-RL
  │
  └─ E4 red best response
        └─ 若 blue 被打穿 → 当前策略不能宣称 anti-exploitation
```

---

## 12. 不要做的事

- 不要继续只用官方四分布调参。
- 不要在 final test 上反复选 checkpoint。
- 不要把 RL-BC 互惠 595 当作“通过 anti-exploitation”。
- 不要用 raw min_family 360 证明鲁棒性。
- 不要宣称 GRU 参数更小。
- 不要在 P1 没有正向信号时继续堆 self-play 功能。
- 不要把 hard-coded 启发式直接写进策略逻辑。
- 不要覆盖已有的 `weights/selfplay_gru_policy.json` 和 `results/selfplay_final/`，新实验单独目录。
