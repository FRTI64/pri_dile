# 100 天囚徒困境项目交接文档

> **交接状态（2026-10-27）**
>
> - 主权重未替换：`weights/rl_policy_weights.json`，sha256
>   `bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。
> - **RL-Repair（`rl-repair`）是当前最佳候选**，但只注册、不 promotion：
>   `weights/rl_policy_weights_repair.json`，sha256
>   `a759a7b6e7eb5773aaf0884f9d6c6ca22587f82e1f4fc4c0cc16e983e3c1bd2b`。
> - `python3 -m ipd test`：**183 tests OK**；`python3 -m ipd list-policies`：
>   20 个注册候选。
> - `heldout_v3` 已被 E1c 一次性使用；**`heldout_v4` 未使用**，只留给下一个
>   通过 validation 的候选做 final test。
> - anti-exploitation **未解决**：E4 red best response 对当前蓝军仍接近
>   100% 胜率。
> - 文档重组前最新提交：`efa9fbb Finalize handover after Axelrod and P2 C3`。

本文只保留当前结论和可执行信息。实验细节、逐 family 表和复现脚本在
`results/*/README.md` 与 `docs/**/*.md`；文档导航见 `docs/README.md`。

---

## 1. 当前建议

| 场景 | 建议 | 依据 |
|---|---|---|
| 现在提交，允许新候选 | **RL-Repair** | 官方 / blind / heldout v2/v3 均优于 A60 和主权重 |
| 保守提交，不替换主权重 | **RL-PPO（从零 + hard）** | 当前主权重，尾部 min_norm 最好 |
| 需要第二候选 / fallback | **RL-PPO-Hard60（A60）** | 官方与 blind 高，heldout 稳定，独立 seed 有复现 |
| 限制 21 维观测、无 PyTorch 推理 | **SelfPlay-GRU** | 固定 21 维、纯标准库；但四分布低约 2.6 分 |
| 需要 C++ | 主权重 RL-PPO / Adv-RL / Neural / GRU / Final / Resp3 / Bayes | Triple 没有 C++ 版 |
| 纯规则、无权重 | **Final-tuned** 或 **Triple-majority** | Triple 更均衡，Final 最易移植 |

**Promotion 条件：** 先用独立 seeds 复现 RL-Repair 的 official / blind /
heldout v2，再确认 C++ 导出、第 1/100 天行为、权重 checksum；得到确认后才把
`weights/rl_policy_weights.json` 替换为 RL-Repair。当前文档不假设已经替换。

---

## 2. 评估口径（先看这一节）

项目里同时存在几种协议，数字略有差异是正常的，**不能混表**。

### 2.1 标准 CLI

```bash
python3 -m ipd benchmark --candidates <name> --fields 20 --workers 20
python3 -m ipd blind --candidates <name> --fields 20 --workers 10
```

- `benchmark`：四分布 `original / late / grim30 / wide`，20 fields × 299；
- `blind`：20 个未见 seed 的 wide/smart 赛场；默认 CLI 是 10 fields，正式
  比较建议显式 `--fields 20`；
- 学习策略默认 `sample`；`argmax` 只作对照；
- 标准 CLI 的 policy seed 来自默认值，和 E0/E1c 冻结的 index-based policy
  seed 不同，因此同一候选会差 0.0x～1 分。

### 2.2 E0 / E1c 冻结 league

```bash
python3 -m ipd league --config configs/eval/e0_league.json \
  --out results/e0_league_20261026/report.json
```

- 固定候选 spec、权重路径、suite、opponent seeds 和 policy seeds；
- official / blind 均为 20 fields；
- 额外输出 validation、heldout_v2、min_norm、CVaR10、E4 red、cross-play、
  Nash / Elo；
- E1c 的 heldout_v3 final 用 `configs/eval/e1c_final_league.json` 单独跑一次。

### 2.3 结果表

E0/E1c 冻结协议（推荐用于候选比较）：

| 候选 | 注册名 | official | blind 20 | heldout_v2 | heldout_v3 | min_norm | CVaR10 | E4 red blue |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **RL-Repair** | `rl-repair` | **554.72** | **564.68** | **503.17** | **509.88** | 0.9930 | 390.39 | 303.45 |
| RL-PPO-Hard60 | `rl-hard60` | 554.27 | 563.87 | 500.07 | 505.97 | 0.9928 | **391.13** | 299.20 |
| RL-PPO | `rl` | 553.22 | 562.80 | 494.97 | 502.31 | **1.0000** | 387.26 | **310.55** |
| Adv-RL 激进 | `adv-rl` | 554.05 | 562.97 | 502.39 | 504.84 | 0.9815 | 387.14 | **325.55** |
| SelfPlay-GRU | `selfplay` | 550.97 | 561.13 | 491.86 | - | 0.9494 | 377.55 | 301.65 |
| Triple-majority | `triple-majority` | 547.37 | 551.33 | 491.94 | - | 0.8739 | 357.34 | 337.55 |
| Final-tuned | `final-tuned` | 546.35 | 547.75 | 490.04 | - | 0.8540 | 360.92 | 334.05 |

标准 CLI、独立 policy seed 的头部结果：

| 候选 | official 20-field | blind 20-field | 备注 |
|---|---:|---:|---|
| RL-PPO | 553.33 | 562.83 / 5.80 | 主权重 |
| RL-PPO-Hard60 | 554.36 | 564.17 / 5.75 | A60，独立 seed2 也一致 |
| RL-Repair | 554.76 | 564.61 / 5.75 | 16/20 第一、16/20 前 10、worst 27 |

**不要混用：** 10-field blind 的旧数字（RL-PPO 563.23 / 8.00、Adv-RL
562.91 / 8.00、Adv-MLP 559.51 / 8.10、SelfPlay 560.43 / 8.10）只适用于
对应实验表；与 20-field 表不可直接比较。`min_norm` 是相对 RL-PPO 的逐
family 最小归一化分数，raw `min_family` 不可跨 family 比较。`E4 red blue`
是蓝军对 red 的均分，越高越好，但当前所有候选仍被 red 接近 100% 击败。

---

## 3. 当前候选与权重

### 3.1 RL 系列（85 维 MLP）

| 文件 | 内容 |
|---|---|
| `weights/rl_policy_weights.json` | 主权重：从零 + hard，run6 |
| `weights/rl_policy_checkpoint.json` | 主权重对应的 actor-critic checkpoint |
| `weights/rl_policy_weights_hard60.json` | A60 部署权重 |
| `weights/rl_policy_checkpoint_hard60.json` | A60 checkpoint |
| `weights/rl_policy_weights_repair.json` | E1c RL-Repair 部署权重 |
| `weights/rl_policy_checkpoint_repair.json` | E1c checkpoint |
| `weights/rl_policy_weights_bc.json` | BC 初始化 + hard（Adv-RL base） |
| `weights/rl_policy_weights_scratch_base.json` | 纯从零 base |
| `weights/rl_policy_weights_stage1.json` | run3（wide + KL） |

架构与训练：

- 85 维历史特征 → 64 → 32 → 1，策略头 7,617 参数；
- PPO：clip 0.2、GAE(gamma=1.0, lambda=0.95)、优势按场中心化；
- `bc_kl_coef=0.1` 把策略拉向初始 BC 策略；`--dist-weights 1,1,1,2`
  提高 wide 权重；`--train-dists ...,hard` 加入 hard 对手；
- 默认 `safe=true`：第 1 天 C、对手从未 D 时 C、第 100 天 D；
- `--allow-probe` 会去掉前两条，但在 grim30 上风险很大，未证明有收益。

训练路径：

```text
run3  BC + wide + KL          549.55 / 557.95
run4  BC + hard               550.50 / 558.60  -> Adv-RL base
run5  从零 base，30 iter      551.87 / 559.27
run6  从零 + hard，40 iter    553.33 / 563.23  -> 主权重（10-field blind）
A60   hard×2，60 iter         554.36 / 564.17  -> rl-hard60
E1c   repair，20 iter         554.76 / 564.61  -> rl-repair
```

### 3.2 规则策略

| 候选 | 实现 | 备注 |
|---|---|---|
| Final-tuned | `ipd/policies/candidates/final_tuned.py` | 参数直接内嵌；`ipd/core/machine.py` 默认参数不是 tuned |
| Resp3-tuned | `ipd/policies/candidates/resp3_tuned.py` | Responsive3 状态机 |
| Bayes-tuned | `ipd/policies/candidates/bayes_tuned.py` | 对 Grim/punitive 强，对 smart/wide 弱 |
| Triple-majority | `ipd/policies/candidates/triple_majority.py` | Final + Resp3 + Bayes 多数票 |
| Final+Resp3 | `ipd/policies/candidates/hybrid.py` | Final-tuned + 默认 Resp3，不一致时 D |
| Harvest45 / Harvest-tuned / GapStart | `candidates/` | 特化/搜索候选，不推荐单独提交 |

Final-tuned 参数（与候选文件一致）：

```text
P_UNPROVOKED = 0.2395699546614404
P_MUTUAL     = 0.6399155368157192
P_RESPONSIVE = 0.9544467968114831
MIN_C = 5   MIN_D = 2
RESP_GAP = 0.3871088255996472
DEFECT_RATE = 0.1170465108152342
PEACE_TESTS = 1   PEACE_COOLDOWN = 9
```

Responsive3-tuned：

```text
g_unprov = 0.31711381663242244
g_mutual = 0.26173130084075913
g_resp   = 0.9804446329081643
min_c = 5   min_d = 2
resp = 0.46788262082576104
rate = 0.197064176521839
max_tests = 3   cooldown = 14
```

BayesPlus-tuned：

```text
coop_scale = 1.8369823500713045
punitive_scale = 2.4705958305576154
late_scale = 0.5216017158776368
noise_scale = 0.5817914534720606
probe_scale = 3.315461421931959
NONRESP_THRESHOLD = 0.27974873408266854
PUNITIVE_THRESHOLD = 0.7531347429866336
```

Triple 规则：三个子策略各自维护状态，按多数票出招（2:1 或 3:0）；
第 100 天三家都会 D。Final+Resp3：动作一致时采用，不一致时 D（p=0）。

### 3.3 神经 / 自博弈

| 候选 | 权重 | 结构 | 备注 |
|---|---|---|---|
| Neural | `weights/neural_policy_weights.json` | 85→64→32→1 MLP，7,617 参数 | Triple 蒸馏；10-field blind 555.10 |
| GRU | `weights/gru_policy_weights.json` | 85→GRU(32)→32→1，12,513 参数 | 有状态，必须 reset；10-field blind 555.22 |
| SelfPlay-GRU | `weights/selfplay_gru_policy.json` | 21 维观测，2 层 GRU(hidden=64)，可部署 43,777 参数 | BC 蒸馏；E0 20-field blind 561.13 |

SelfPlay 结论：纯从零 15k self-play 不稳定；最终策略用 BC 蒸馏，四分布
550.76（20 fields）、10-field blind 560.43。BC+red fine-tune E1a 仅
+0.43 mean / +0.016 min_norm，E1b 崩坏；red best response 100% 击败
BC/RL/Adv。**改进实验 No-Go，不替换主权重。**

### 3.4 Adv 系列

| 候选 | base | 四分布 | 10-field blind | 备注 |
|---|---|---:|---:|---|
| Adv-MLP 激进 | MLP | 552.24 | 559.51 / 8.10 | 只有 Python / C++ 需自行加规则 |
| Adv-MLP 保守 | MLP | 551.62 | 557.44 / 8.30 | 逐类型风险更小 |
| Adv-GRU 激进 | GRU | 551.69 | 559.05 / 8.10 | 只有 Python |
| Adv-GRU 保守 | GRU | 551.43 | 557.00 / 8.30 | 只有 Python |
| Adv-RL 激进 | RL-BC | 554.29 | 562.91 / 8.00 | 有 C++；E0 20-field 554.05 / 562.97 |
| Adv-RL 保守 | RL-BC | 553.73 | 561.05 / 8.20 | 规则更保守 |

两条事后规则（都要求对手已经先 D）：

1. 首次 D 在 `late_day` 之后且下一回合继续 D → 永久 D；
2. `n_c>=min_c, n_d>=min_d` 且 `pD-pC<=gap` → 永久 D。

激进配置：`gap=0.2, min_c=12, min_d=2, late_day=40, repair=true`。
约束是“四个分布均不下降”，不是“每个对手类型都不下降”；`MemOne-rand`、
`Extort3`、`NoisyTFT.10/20` 等 wild 类型可能有小损失。

---

## 4. 实验结论账本

### E0 — league / cross-play / exploitability（PASS）

- 入口：`python3 -m ipd league`，报告
  `results/e0_league_20261026/README.md`；
- 9 个候选在同一 suite/seeds 上完成评估；
- A60 overall 552.11（RL-PPO 550.80），heldout_v2 500.07（494.97），
  CVaR10 391.13（387.26）；
- cross-play 退化：所有蓝军互相合作（594–595），Elo/Nash 不可区分；
  必须加入 red/exploiter 池才能测真实 exploitability；
- E4 red 对所有候选 red win 95–100%；anti-exploitation 未解决；
- **不改主权重**。

### E1 — domain-randomized / repair fine-tune（E1c Go candidate）

| 实验 | heldout_v2 Δ vs A60 | min_norm Δ | CVaR10 Δ | heldout_v3 | 判定 |
|---|---:|---:|---:|---:|---|
| E1a | +1.36 | +0.0036 | -2.15 | 未运行 | No-Go |
| E1b | +1.94（selection +3.05） | -0.0036 | -2.32 | 未运行 | No-Go |
| E1c | **+3.10**（selection +4.90） | +0.0002 | -0.74 | **+3.91** | **Go candidate** |

- E1c 收益主要来自 extort +23～28 和 random 修复；
- `heldout_v3 final`：RL-Repair 509.88 vs A60 505.97；
- E4 red 仍 100% 击败；red blue 仅 +4.25（v2）/ -3.65（v3）；
- 注册 `rl-repair`，**不覆盖主权重**；
- 细节：`results/robust_20261026/README.md`、`docs/research/e1-robust-finetune.md`。

### E2 — CVaR / risk PPO（No-Go）

- 新增 `ipd/training/risk.py`：CVaR episode weights、Dirichlet mixture、
  worst-family DRO；
- E2a/E2b/E2c 在 heldout_v2 上最好 mean 505.78，CVaR item 仅
  -0.5%～+1.2%，min_norm +0.003，未达 +10% / +0.02 Gate；
- official validation 也未超过 RL-Repair；
- **No-Go**；不注册 `rl-risk` / `rl-dro`，不消耗 heldout_v4；
- 细节：`results/risk_20261026/README.md`、`docs/research/e2-risk-ppo.md`。

### P3 — red / exploiter pool（red pool Go；league-lite / expansion No-Go）

- **Red pool 第一里程碑 Go：** 85 维 `Red-RL-v1` 在全部 held-out blue 上强于
  E4 red（对 RL-Repair red 727.6 vs 703.8）；
- **League-lite No-Go：** competitive blue response 把 E4 exploitability
  407.0 → 355.0（-12.8%），未达 -20% Gate；official 554.74、blind 563.80；
- **P3 expansion No-Go：** v1–v6 red + maximin 权重，选中 Red-v4 0.70；
  Blue-Meta-v2 在 held-out red 上 exploitability 390.33 → 367.33
  （-5.9%），`validation:extort` -3.21%，Gate D FAIL；
- 不注册 `rl-meta`；E4 red 仍 100% 击败；`heldout_v4` 未用；
- 细节：`results/red_pool_20261026/README.md`、
  `results/p3_league_20261026/README.md`、`docs/research/p3-red-league.md`、
  `docs/research/p3-expansion.md`。

### P2 C1 / C2 / C3（offline No-Go）

- **C1 belief planner：** fair-RNG 后激进版 validation -12.11 /
  heldout_v2 -16.15；保守版 validation -2.19 / heldout_v2 +2.06；
  held-out red E4 +0.35、v5 0.00、v6 -0.10；未达 Gate C +10；
- **C2/C3 opponent model + gate：** held-out type 86.6%，heldout_v2
  70.6%，false positives 集中在 adaptive/noisy；gated conservative
  heldout_v2 +1.20、red 不变；gated aggressive extort -107.6；
- **excitation / counterfactual 数据：** `excited` 率 25.8% → 59.3%，
  cooperative/forgiving 0% → 56.9%/48.6%；数据层 Go，但部署轨迹 excitation
  仍不足，planner 3-seed heldout_v2 仅 +0.88；
- **three-way gate + MoE：** safe→AllD 5-seed heldout_v2 +4.04、validation
  +0.53，best-validation +0.90/+1.41，未达 +10；safe→planner -1.28；
- **repeated counterfactual calibration：** safe precision/recall
  74.7%/62.1%；response veto 0.2 后 validation -0.31、heldout_v2 +2.93、
  safe rate 2.5%；held-out red 0.00；
- 结论：**C1/C2/C3 全部 No-Go**，不改主权重。不要继续扫 C3 threshold；
  若继续，需要显式 value-of-information probe 或更好的后验/特征；
- 细节：`docs/research/p2-*.md`、`results/p2_*_20261026/`。

### SelfPlay（改进 No-Go）

- 同 suite/seeds：BC-only 与 RL-PPO 在 final suite 基本打平
  （506.98 vs 506.49）；scratch self-play min_norm 仅 0.783；
- E1a +0.43 mean / +0.016 min_norm；E1b 崩坏到 429；
- red best response 100% 击败 BC / RL / Adv；
- 保留 `selfplay` 作为 21 维 / 无 PyTorch 部署备选，不 promotion；
- 细节：`results/selfplay_improvement/final_decision.md`、
  `results/selfplay_final/README.md`、`docs/policies/selfplay.md`。

### Axelrod 对手库（可选功能）

- 新增 `ipd/core/opponents/axelrod_bank.py`，延迟导入；
- `axelrod` 243 普通 / `axelrod-fast` 224（剔除 19 个 long-run）/
  `axelrod-cheating` 1（4.14.0 只有 `Darwin`）；
- 默认四分布、85/158 对手集合和主权重均未改变；
- 224 fast × 100 回合：Triple 560.95、Resp3 555.37、Final 555.27、
  Bayes 546.25；**不能与官方四分布 554.x 直接比较**；
- 完整 243 很慢；日常用 `axelrod-fast`，完整集只做定向分析或
  `matchup --rounds 10` smoke；
- 细节：`docs/guides/axelrod.md`、`results/axelrod_20261027/`。

### 全量逐对手扫描（2026-10-26）

- `results/sweep_20261026/README.md`：66 个候选/权重 × 158 类对手 × 16 次；
- RL-Hard60 与 Adv-RL-BC-agg 的 9 测试集加权并列第一（556.79），
  RL-Hard60 类型等权 551.05、Adv-RL-BC-agg 553.19；
- 对抗规则只对 BC / stage1 基座正收益；套到 hard2 / RL-main 上会掉分；
- 这是加权对阵指标，不等同于 official / blind 排名。

---

## 5. 复现命令

### 5.1 基础

```bash
cd /home/frti64/pri_dile
python3 -m ipd test
python3 -m ipd list-policies
python3 -m ipd self-test --candidates all
```

### 5.2 当前头部候选

```bash
# 主权重
python3 -m ipd benchmark --candidates rl --fields 20 --workers 20
python3 -m ipd blind --candidates rl --fields 20 --workers 10

# A60
python3 -m ipd benchmark --candidates rl-hard60 --fields 20 --workers 20
python3 -m ipd blind --candidates rl-hard60 --fields 20 --workers 10

# RL-Repair（当前最佳候选）
python3 -m ipd self-test --candidates rl-repair
python3 -m ipd benchmark --candidates rl-repair --fields 20 --workers 20
python3 -m ipd blind --candidates rl-repair --fields 20 --workers 10
```

### 5.3 E0 / E1c 冻结评估

```bash
python3 -m ipd league --config configs/eval/e0_league.json --workers 20 \
  --out results/e0_league_20261026/report.json \
  --md-out results/e0_league_20261026/README.md

# heldout_v3 已被 E1c 使用；E1c final 复现见 5.4 末尾，不要用于继续选模。
```

### 5.4 训练

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt

# MLP / GRU BC + DAgger
python3 -m ipd train neural --device auto
python3 -m ipd train gru --device auto

# RL 主权重（从零 + hard，精简版；完整参数见 docs/policies/rl.md）
python3 -m ipd train rl --device auto --iterations 40 \
  --init-weights weights/rl_policy_checkpoint_scratch_base.json \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 --bc-kl-coef 0.1 --lr 1e-4 \
  --select-metric blind \
  --out weights/rl_policy_weights.json \
  --checkpoint-out weights/rl_policy_checkpoint.json

# A60（hard×2）
python3 -m ipd train rl --device auto --seed 20261030 --iterations 60 \
  --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,hard --dist-weights 1,1,1,2,2 \
  --init-weights weights/rl_policy_checkpoint.json \
  --bc-kl-coef 0.1 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 --select-metric avg \
  --out weights/rl_policy_weights_hard60.json \
  --checkpoint-out weights/rl_policy_checkpoint_hard60.json

# E1c repair（完整协议见 results/robust_20261026/README.md）
python3 -m ipd train rl --device auto --seed 20261032 \
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

# heldout_v3 已一次性使用；以下只用于复现历史结论
python3 -m ipd league --config configs/eval/e1c_final_league.json \
  --workers 20 --out results/robust_20261026/e1c_final_v3.json \
  --md-out results/robust_20261026/e1c_final_v3.md
```

训练任务与参数以 `python3 -m ipd train --list` 和对应 `--help` 为准；
实验当时的部分命令保存在 `results/*/run.sh` 或训练日志中（例如 E1c 的
`results/robust_20261026/e1c_train.log`）。

### 5.5 C++ 编译

```bash
g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo
g++ -std=c++17 -O2 -DGRU_POLICY_DEMO cpp/gru_policy.cpp -o gru_demo
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
  -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o rl_demo
g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_BC_WEIGHTS \
  -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_bc_demo
g++ -std=c++17 -O2 -DRESP_BAYES_DEMO cpp/bayesplus_responsive3.cpp -o demo
```

### 5.6 Axelrod（可选）

```bash
python3 -m pip install -r requirements-axelrod.txt
python3 -m ipd benchmark --candidates core --dists axelrod-fast \
  --fields 1 --n 224 --workers 5
python3 -m ipd matchup --candidates final-tuned --opponent-set axelrod-fast \
  --reps 4 --workers 12
```

---

## 6. C++ / 部署现状

- `cpp/policies/final_policy.hpp`：Final，`FinalParams::tuned()`；
- `cpp/policies/responsive3_policy.hpp` / `bayes_plus_policy.hpp`：
  Responsive3 / BayesPlus；
- `cpp/policies/neural_policy.hpp` / `gru_policy.hpp`：
  MLP / GRU 推理；
- `cpp/policies/adversarial_rl_policy.hpp` / `cpp/adversarial_rl.cpp`：
  Adv-RL；
- `cpp/opponents/*.hpp`：演示/测试用对手；
- **没有 C++ 版 Triple-majority**；如需 C++ Triple，需要移植 Final-tuned +
  Resp3 + BayesPlus 并做多数投票；
- **Adv-MLP / Adv-GRU 只有 Python 版**；Adv-RL 有 C++；
- C++ 与 Python 的特征/概率对齐已有测试（MLP 200 组、GRU 60 组，最大差异
  约 5e-10）；
- 重新训练后先 `python3 -m ipd export ...` 刷新 `cpp/*_weights.hpp` 再编译。

Python 与 C++ 的动作约定：

```text
C = 0, D = 1
day = 1..100
mine/theirs = 到此日之前的动作
每场开始调用 reset()
```

---

## 7. 风险与注意事项

1. **没有真实 300 人数据**：所有结果是合成对手库 + 分布内/held-out 评估；
   盲测只是未见 seed，不等于真实比赛。
2. **分布依赖强**：Harvest 在 late 最强、Bayes 在 punitive 最强、Triple/Final
   更均衡；不要用单一分布外推。
3. **Anti-exploitation 未解决**：E4 red 对当前蓝军仍接近 100% 胜率；
   P3 的 red pool / league-lite / expansion 都没有过 Gate D。
4. **RL 有尾部风险**：最终从零 RL 在 `Rand3->TFT` 上约 347（BC 约 554），
   `Extort3`、`Random.50/70` 也可能回退；`ProbeBot` 已修复到约 603。
5. **10 vs 20 fields 不可混用**：旧表多为 10-field blind；当前主表用 20-field
   或 E0 冻结协议。
6. **`heldout_v2` 是 validation**：它已被多次用于选模；`heldout_v3` 已被
   E1c 使用；`heldout_v4` 仍未使用，只做一次性 final test。
7. **SelfPlay 不是 anti-exploitation 证据**：final test 中的 held-out red
   是 RL-BC 互惠型对照，不能证明能防针对性 red。
8. **GRU / SelfPlay-GRU 有状态**：每场必须 `reset()`；同一时间步不要重复
   调用 `act()`。
9. **规则默认参数不是 tuned**：`ipd/core/machine.py`、`cpp/ipd_strategy.cpp`
   的默认参数是早期版本；提交时使用候选或 `FinalParams::tuned()`。
10. **权重文件必需**：Neural / GRU / RL / SelfPlay 都要带 JSON 权重；单文件
    提交时需要内嵌。
11. **Axelrod 是可选重量级依赖**：固定 `4.14.0`；完整 243 很慢；Darwin 的
    genome 是类变量，结果与顺序相关，不能当稳定强度。
12. **所有候选都应在第 100 天 D**；移植时漏掉会损失最后一天收益。
13. **NPU 路径未实机验证**：代码支持 `--device npu`，本轮只在 CUDA/CPU
    上完成训练和测试。

---

## 8. 交接检查清单

- [ ] 确认比赛接口：`act` / `move` 签名、历史格式、是否允许 `reset()`。
- [ ] 确认是否允许每天运行三个子策略（Triple）。
- [ ] 确认是否允许提交权重文件；单文件限制下决定是否内嵌。
- [ ] 确定提交版本：RL-Repair / A60 / RL-PPO / Adv-RL / SelfPlay / 规则。
- [ ] 若提交 RL-Repair，先做独立 seed 复现，再决定是否替换主权重。
- [ ] 运行 `python3 -m ipd test`（183 tests）和
      `python3 -m ipd self-test --candidates all`。
- [ ] 对学习策略跑一次 `benchmark --fields 20` 和 `blind --fields 20`。
- [ ] 检查第 1 天、第 100 天和“对手未 D 时 C”的安全规则。
- [ ] 确认 GRU / SelfPlay 每场 `reset()`，同一时间步只调用一次。
- [ ] 若提交 C++，刷新权重头文件并重新编译；Triple 需要自行移植。
- [ ] 若使用 Axelrod，安装 `requirements-axelrod.txt`，日常只用
      `axelrod-fast`，完整 243 不进常规 CI。
- [ ] 任何 promotion 都更新权重 checksum、`README.md`、`docs/operations/handover.md` 和报告。
- [ ] 保留最终提交文件、配置 JSON、运行日志和评估 seed。

---

## 9. 关键文件

| 文件 | 内容 |
|---|---|
| `ipd/policies/registry.py` | 候选名、别名、集合、构造入口 |
| `ipd/policies/candidates/` | 每个候选一个独立文件 |
| `ipd/evaluation/league.py` | E0 冻结评估 harness |
| `ipd/evaluation/opponent_suite.py` | validation / heldout_v2 / v3 |
| `ipd/evaluation/policy_pool.py` | P3 frozen blue/red pool |
| `ipd/training/rl.py` | PPO 训练、rollout、验证、导出 |
| `ipd/training/risk.py` | E2 CVaR / Dirichlet / DRO |
| `ipd/training/opponent_model.py` | C2/C3 对手模型 / gate |
| `ipd/training/selfplay/` | 21 维自博弈训练器 |
| `ipd/policies/moe.py` | C3 three-way gate + MoE |
| `ipd/policies/belief_planner.py` | C1 belief planner |
| `docs/README.md` | 文档导航与状态 |
| `docs/architecture/overview.md` | 分层与接口 |
| `results/*/README.md` | 各实验的完整报告 |
