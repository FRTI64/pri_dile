# 100 天囚徒困境项目交接文档

> **交接状态（2026-10-30）**
>
> - 主权重未替换：`weights/rl_policy_weights.json`，sha256
>   `bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。
> - **Protocol v5 已启用**：977 configs 重新划分 train/dev/test；
>   `test_v8` 为新的 held-out final（初始锁定），v4 `test_v7` 及全部
>   v4 结果进入 legacy（ADR 0013）。
> - **RL-Switch（`rl-switch`）是 v4 legacy dev/test 最佳（top 7
>   对比）**，但只注册、不 promotion；Python + C++ 双权重实现已通过 action parity：
>   默认 RL-Wide160；第 6 回合前双方各 D≥2 且对手最新 C 时切换
>   RL-Repair。dev_param 548.58 / dev_complex 454.24 / dev_axelrod
>   582.86；v4 `test_v7` provisional 488.26（seed 27）/ 487.51（seed 29）。
>   v5 需重新跑 baseline。
> - **GRU-DAgger（`rl-dagger-gru`）是单模型替代**：1 层 GRU（64
>   hidden）用 2 轮 DAgger 蒸馏 RL-Switch；dev_param 548.58 /
>   dev_complex **455.13**；两个 seed 与 RL-Switch 持平；v4 `test_v7`
>   provisional 487.89（seed 27）/ 487.02（seed 29）；Python + C++
>   已通过 action parity；未 promotion。详见
>   `docs/research/dagger-gru.md`。
> - **前最佳 / fallback：RL-Wide160（`rl-wide160`）、RL-Wide
>   （`rl-wide`）、RL-Pattern（`rl-pattern`）与 RL-Repair
>   （`rl-repair`）**；均未 promotion。
> - **最佳纯规则（v4 legacy）：Triple-v4（`triple-v4`）**：
>   Triple-majority + Final 噪声互惠分支；dev_param 538.66 /
>   dev_complex 447.08 / dev_ood 514.41；独立 seed 20261029 复现；
>   v4 `test_v7` provisional 473.68；无权重、无 probe；仍低于
>   RL-Repair / RL-PPO，**未 promotion**；v5 需重跑。
> - 最近完整测试：**346 tests OK**（本轮全量运行）。
>   `python3 -m ipd list-policies`：**33 个注册候选（all 集合）**。
> - `rl-gate` 是学习型上层 gate 实验：阈值 0.9 基本退化为
>   RL-Wide160（test 485.84），阈值 0.5 会过度切换；**No-Go**，
>   仅保留为实验候选。
> - **PSRO / red 历史存档**：`ipd/training/psro.py`、`psro` /
>   `psro-br` / `psro-scratch` / `psro-scratch-br` / `psro-v4` /
>   `psro-v4-br` 仍注册为历史实验候选；dev/red 结果只用于复现，
>   **不再作为当前评估或 promotion 门槛**。
> - **Protocol v5 已定义**：977 configs；14 complex 风格 + 224 Axelrod + 15 组合；
>   2048 人 / dev 30 fields / test 50 fields；`dev_param` / `dev_family` /
>   `dev_complex` / `dev_axelrod` / `dev_ood` 选模；`SPLIT_SALT=protocol_v5`
>   重新分配 train/dev/test。
> - **v5 dev baseline 已运行**（33 candidates，5 tiers × 30 fields ×
>   2048，1662.15s）：dev_param 第一是 RL-Gate 532.82，但 `dev_ood`
>   min_norm 0.899；RL-Switch / GRU-DAgger / RL-Wide / RL-Wide160
>   约 532 / 477 / 522，共同弱项 `probe_exploit`（min_norm ~0.934）。
>   旧 `min_norm ≥ 0.99` 门槛下无 top candidate 通过，**不 promotion**。
> - **v5 分级诊断**（`protocol_v5_dev_levels.json`，1605.51s）：
>   `probe_exploit` 是共同瓶颈（头部学习策略约 116，RL-PPO 124）；
>   `pattern_breaker` 对 RL-Wide160 / RL-Switch / GRU-DAgger 已高于
>   RL-PPO（632–639 vs 449–452），RL-Repair 的 PB 才偏低（357）。
>   已新增 `train_probe_pattern`，以 RL-Wide160 做定向 fine-tune。
> - **定向 fine-tune 结果**：ProbePattern-v2-iter4 把 `dev_complex`
>   477.77 → **481.81**、`dev_ood` 522.24 → **523.14**、
>   `pattern_breaker` 632 → **762**；但 `dev_param` -0.55、
>   `probe_exploit` 仍只有 **118.1**（RL-PPO 124.0）、min_norm 约
>   0.951。结论：PB 可训，probe_exploit 不是简单加权重能解决；需要
>   新特征 / 显式反 probe 规则 / specialist + gate。未注册、未 promotion。
> - **probe 专项按 3→1→2 试过**：probe-only specialist 与 v2 特征都
>   没有增益；精确规则（首次 D=5/7 切 AllD）能把 probe 拉到
>   123–129，但 `dev_param` -1.9～-4.5、`dev_axelrod` -1.1～-2.7，
>   净收益为负。probe 专项暂停：**不采用 unsafe specialist /
>   `--allow-probe` / 预判 gate**，也不为 probe 修改 protocol 或
>   safety 规则；v5 保持 977 configs。
> - v4 的 top 7 dev、`test_v7`、v3 的 RL-Repair 484.32 / A60 476.54 /
>   Adv-RL 471.44 只作 legacy。`test_v8` / `test_red` 初始锁定；
>   final 前需显式解禁（ADR 0013）。
> - **blue-red 对战已停用**（ADR 0011）：以后不再用 red /
>   exploitability 作为评估或 promotion 门槛；不新增 red 任务。
> - 最近提交：`git log -1`（GRU-DAgger 单模型蒸馏 + C++）。

> **2026-10-28 更新**
>
> - per-match sample RNG 已 opt-in（`--sample-rng per-match`）；
>   Plan 6 把 sample policy 对手搬到 GPU、Axelrod adapter 改为增量历史，
>   并加入严格同序 C++ CPU policy 前向。v4 `test_v7` 的 exact 运行降到
>   **272.77s**；`Adv-*` / PSRO meta 保持 CPU，确保 **24/24 候选分数与
>   CPU baseline 逐位一致**。dev argmax 只在本轮迁移中做过一次性加速
>   诊断，不进入长期 gate。
> - 最近完整测试：**307 tests OK**。详见
>   `docs/research/plan6-python-bottleneck.md`。

本文只保留当前结论和可执行信息。实验细节、逐 family 表和复现脚本在
`results/*/README.md` 与 `docs/**/*.md`；文档导航见 `docs/README.md`。

---

## 0. Protocol v5（2026-10-30 起的当前主口径）

v5 保留 977-config catalog，但用 `SPLIT_SALT=protocol_v5` 重新划分
train/dev/test；`test_v8` 是新的 held-out final。v4 的 `test_v7`、v4
baseline 与 v3 结果全部进入 legacy，不能与 v5 混表。新候选按以下流程：

```bash
# 1. 审计 split（必须 OK）
python3 -m ipd protocol audit

# 2. dev 选模：所有候选，五个 validation tier
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --seed 20261027 --baseline RL-PPO \
  --out results/protocol_v5/protocol_v5_dev.json

# 3. test_v8 final：一次性消费（写 usage ledger）
python3 -m ipd protocol eval --candidates <final> \
  --tiers test_v8 --fields 50 --n 2048 --workers 20 \
  --seed 20261027 --acknowledge-once \
  --out results/protocol_v5/protocol_v5_test_final.json

# 3.1 provisional 诊断 / pipeline 烟测：不消费 test_v8
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v8 --fields 50 --n 2048 --workers 20 \
  --seed 20261027 --no-consume \
  --out results/protocol_v5/protocol_v5_test_provisional.json
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v8 --fields 1 --n 256 --workers 4 \
  --no-consume --out /tmp/test_v8_smoke.json

# 4. 按思考等级 / 复杂度分级
python3 -m ipd protocol eval --candidates all \
  --tiers dev_level3 --fields 30 --n 2048 \
  --out results/protocol_v5/protocol_v5_dev_level3.json
python3 -m ipd protocol eval --candidates all \
  --tiers dev_complexity4 --fields 30 --n 2048 \
  --out results/protocol_v5/protocol_v5_dev_complexity4.json

# 5. red（diagnostic / final）
python3 -m ipd protocol red --candidates <top3> --tier dev_red --seeds 30
python3 -m ipd protocol red --candidates <top3> --tier test_red \
  --seeds 30 --no-consume
```

规则：

- `dev_param` 是主排序；`dev_family` / `dev_complex` / `dev_axelrod` /
  `dev_ood` 是门禁；`dev_level0..4` / `dev_complexity0..4` 用于分级诊断；
- 必须同时看 per-family、`min_norm`、`worst_family`；
- `configs/splits/protocol_v5/test_policy.json` 初始把 `test_v8` /
  `test_red` 都锁定；final 前显式解禁 `test_v8`，再用
  `--acknowledge-once` 消费。`--no-consume` 运行标记
  `selection_eligible=false`，不写 ledger；
- `--force` 只允许对已消费候选重跑并标记 `contaminated=true`，不能绕过锁定；
- usage ledger：`results/protocol_v5/usage.json`（首次 `test_v8` final 后创建）；
- manifest：`configs/splits/protocol_v5/manifest.json`；
- 旧 `official` / `blind` / v2 / v3 / v4 只能作为 legacy 对照，不能与
  新表混用。

详见 [../reference/protocol-v5.md](../reference/protocol-v5.md)。

---

## 1. 当前建议

> 下表依据仍是 **v4 legacy** 数字；v5 dev baseline 已跑（见 §2.0），
> 但旧 `min_norm ≥ 0.99` 门槛下没有 candidate 通过，暂不改变提交建议。
> v5 final 仍需要在解禁 `test_v8` 后单独跑。

| 场景 | 建议 | 依据 |
|---|---|---|
| 现在提交，允许新候选 | **RL-Switch** | dev_param 548.58、dev_complex 454.24；test 488.26/487.51；C++ 已移植 |
| 需要单模型 / 不想维护双权重开关 | **GRU-DAgger**（`rl-dagger-gru`） | dev_param 548.58、dev_complex 455.13；test provisional 487.89/487.02；Python + C++ |
| 需要 160/128 单模型部署 | **RL-Wide160** | dev_complex 454.31、complex min_norm 0.997；test 485.62 |
| 需要 128/128 部署 | **RL-Wide** | dev_complex 452.79、complex min_norm 0.973；test 485.50 |
| 需要 64/32 部署 / 更小模型 | **RL-Pattern** | dev_param 548.45；test 485.43 |
| meta 学习研究 | **RL-Gate** | 学习型 gate 近中性；**No-Go**，不用于提交 |
| 保守提交，保留 proven fallback | **RL-Repair** | v4 dev_param 547.96；test_v7 provisional 486.31 第一 |
| 保守提交，不替换主权重 | **RL-PPO（从零 + hard）** | 当前主权重；v3 legacy dev 524.37 |
| 需要第二候选 / fallback | **RL-PPO-Hard60（A60）** | v3 legacy test 476.54；min_norm 0.9989 |
| dev_complex / OOD 高 | **Adv-RL** | v3 legacy dev 526.58；complex 415.19 |
| 限制 21 维观测、无 PyTorch 推理 | **SelfPlay-GRU** | 固定 21 维；v3 legacy dev 513.85 |
| 需要 C++ | RL-Switch / GRU-DAgger / 主权重 RL-PPO / Adv-RL / Neural / GRU / Final / Resp3 / Bayes | Triple 没有 C++ 版 |
| 纯规则、无权重 | **Triple-v4**（`triple-v4`） | v4 dev_param 538.66、dev_complex 447.08、dev_ood 514.41；独立 seed 复现；未 promotion |
| PSRO / red 历史对照 | `psro` / `psro-scratch` / `psro-v4` 等 | 仅历史复现；**不再作为评估或 promotion 门槛**（ADR 0011） |

**Promotion 条件：** 先在 v5 的 `dev_param` / `dev_family` / `dev_complex` /
`dev_axelrod` / `dev_ood` 上用独立 seed 复现 RL-Switch / GRU-DAgger；
C++ 实现已完成并通过固定序列 action parity，但还需要第 1/100 天行为、
权重 checksum 和真实比赛接口验证。RL-Switch 和 GRU-DAgger 的 v4
`test_v7` 只做了两个 seed 的 provisional 诊断，不能迁移为 v5 final；
blue-red 已停用（ADR 0011），不再作为 promotion 证据。promotion 前应在
v5 上重跑 dev，解禁 `test_v8` 后用 `--acknowledge-once` 跑一次 final 并
写 usage ledger。得到确认后才把 `weights/rl_policy_weights.json` 替换为
RL-Switch、GRU-DAgger 或 RL-Wide160。当前文档不假设已经替换。

---

## 2. 评估口径（先看这一节）

**当前主口径是 Protocol v5**；下面 2.1 / 2.2 的 official/blind/league 仅用于
复现历史结果，不能作为新候选的 held-out 证据。Protocol v5 的定义和命令见
[../reference/protocol-v5.md](../reference/protocol-v5.md)。

项目里同时存在几种协议，数字略有差异是正常的，**不能混表**。

### 2.0 Protocol v5

| 层 | 用途 | 规模 |
|---|---|---|
| `dev_param` | 选模（标准 + Axelrod 参数 held-out） | 30 fields × 2048 |
| `dev_family` | 选模（family held-out） | 30 fields × 2048 |
| `dev_complex` | 选模（complex family） | 30 fields × 2048 |
| `dev_axelrod` | 选模（Axelrod） | 30 fields × 2048 |
| `dev_ood` | 选模（偏移混合） | 30 fields × 2048 |
| `dev_level0..4` | 思考等级分级诊断 | 30 fields × 2048 |
| `dev_complexity0..4` | 策略复杂度分级诊断 | 30 fields × 2048 |
| `test_v8` | held-out final（初始锁定；解禁后一次性消费） | 50 fields × 2048 |
| `dev_red` / `test_red` | exploitability | 30 seeds |

`test_v8` / `test_red` 初始都锁定；final 前显式解禁 `test_v8`，再传
`--acknowledge-once`，结果写 `results/protocol_v5/usage.json` 并标记
`selection_eligible=true`；消费后新候选需要 `test_v9`。`--no-consume`
仍只生成 provisional 结果。**`dev_red` / `test_red` 是历史口径**；
blue-red 对战已停用（ADR 0011），不再参与当前选模或 promotion。

v5 **dev baseline 已运行**（33 candidates，5 tiers × 30 fields × 2048，
默认顺序 sample RNG；总 wall 1662.15s）。按 `dev_param` 前 7：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | ood min_norm |
|---|---:|---:|---:|---:|---:|---:|
| **RL-Gate** | **532.82** | 582.95 | 477.00 | 567.78 | 522.11 | 0.899 |
| **RL-Repair** | 532.53 | 582.78 | 456.41 | 565.46 | 515.80 | 0.790 |
| **RL-Switch** | 532.37 | 582.91 | 477.57 | 566.53 | 521.73 | 0.934 |
| RL-Pattern | 532.30 | 583.09 | 476.06 | 568.00 | 521.94 | 0.932 |
| GRU-DAgger | 532.30 | 582.93 | 477.29 | 567.17 | 522.09 | 0.934 |
| RL-Wide | 531.89 | 583.09 | 476.94 | 568.01 | 522.07 | 0.932 |
| RL-Wide160 | 531.66 | 583.08 | **477.77** | 567.77 | **522.24** | 0.934 |
| RL-PPO（baseline） | 525.99 | 582.55 | 457.26 | 563.43 | 513.30 | 1.000 |

- v5 的共同弱项是 `probe_exploit`；RL-Switch / GRU-DAgger / RL-Wide /
  RL-Wide160 的 ood min_norm 约 0.932–0.934。RL-Gate 的 dev_param 第一，
  但 ood min_norm 0.899；RL-Repair 的 `pattern_breaker` min_norm 0.790。
- **旧 `min_norm ≥ 0.99` 门槛下没有 top candidate 通过**；本表只作
  v5 诊断，**不 promotion**。完整结果见
  `results/protocol_v5/protocol_v5_dev.json` 与
  `results/protocol_v5/README.md`。
- 该 baseline 在 977-config catalog 上运行，与当前 manifest 一致。

下表是 v4 legacy 的 top 7 dev 与 `test_v7` provisional，只用于理解候选，
不参与 v5 选模：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | test_v7 provisional |
|---|---:|---:|---:|---:|---:|---:|
| **RL-Switch** | **548.58** | 582.91 | 454.24 | **582.86** | **521.42** | **488.26** |
| **GRU-DAgger** | **548.58** | 582.90 | **455.13** | 582.49 | **521.54** | 487.89 |
| RL-Wide160 | 548.26 | **583.08** | **454.31** | 582.38 | 521.37 | 485.62 |
| RL-Wide | 548.24 | **583.08** | 452.79 | 582.30 | 520.71 | 485.50 |
| RL-Pattern | 548.45 | 583.07 | 447.64 | 582.22 | 519.46 | 485.43 |
| RL-Repair | 547.96 | 582.76 | 434.28 | 582.46 | 515.19 | 486.31 |
| RL-PPO | 540.35 | 582.53 | 440.31 | 574.64 | 513.48 | 477.26 |

- RL-Switch 相对 RL-Wide160：dev_param +0.32、dev_axelrod +0.48、
  dev_ood +0.05；dev_family -0.17、dev_complex -0.07；test_v7 +2.64。
- RL-Switch 独立 seed 20261029：test_v7 487.51 vs 484.61（+2.90）；
  Protocol catalog suite +0.75；158 bank 基本持平。
- GRU-DAgger 相对 RL-Switch：dev_param 0.00、dev_family -0.01、
  dev_complex +0.89、dev_axelrod -0.37、dev_ood +0.12；test_v7
  provisional -0.37（seed 27）/ -0.49（seed 29）。独立 seed
  20261029 dev：546.97 / 582.59 / 454.39 / 581.93 / 523.05
  （+0.16 / -0.03 / +0.12 / -0.20 / +0.21）。level/complexity
  全部 ±1.3 内。
- 下面是 legacy v3 baseline（20/40 fields × 299），仅用于历史对照；
  不能与上表混用：

| 候选 | dev_param | dev_family | dev_complex | dev_ood | test_v6 |
|---|---:|---:|---:|---:|---:|
| **RL-Repair** | **532.44** | 582.92 | **420.38** | **489.84** | **484.32** |
| RL-PPO-Hard60 | 527.97 | 582.94 | 410.09 | 485.12 | 476.54 |
| Adv-RL | 526.58 | 583.20 | 415.19 | 487.22 | 471.44 |
| RL-PPO | 524.37 | 582.72 | 397.90 | 478.61 | - |
| Final-tuned | 523.39 | 583.13 | 413.64 | 484.57 | - |

- 以上是 v3 数字，不能作为 v4/v5 排名；v4 加入 Axelrod 并扩大规模、
  v5 重新划分后都必须重跑。
- Red-v5 / Red-v6 对 top 3 仍 100% 胜率，anti-exploitation 未解决；
- 完整 v3 表：`results/protocol_v3/README.md`。

### 2.1 标准 CLI（legacy）

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

### 2.2 E0 / E1c 冻结 league（legacy）

```bash
python3 -m ipd league --config configs/eval/e0_league.json \
  --out results/e0_league_20261026/report.json
```

- 固定候选 spec、权重路径、suite、opponent seeds 和 policy seeds；
- official / blind 均为 20 fields（legacy）；
- 额外输出 validation、heldout_v2、min_norm、CVaR10、E4 red、cross-play、
  Nash / Elo；
- E1c 的 heldout_v3 final 用 `configs/eval/e1c_final_league.json` 单独跑一次。

### 2.3 结果表（legacy）

以下 E0/E1c 冻结协议数字只用于历史复现；新候选比较用 Protocol v5。

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
| `weights/rl_policy_weights_pattern.json` | RL-Pattern 64/32 部署权重（iter6 导出） |
| `weights/rl_policy_checkpoint_pattern.json` | RL-Pattern actor-critic checkpoint |
| `weights/rl_policy_weights_wide.json` | RL-Wide 128/128 部署权重（balance iter4 导出） |
| `weights/rl_policy_checkpoint_wide.json` | RL-Wide actor-critic checkpoint |
| `weights/rl_policy_weights_wide160.json` | RL-Wide160 160/128 部署权重（iter1 导出） |
| `weights/rl_policy_checkpoint_wide160.json` | RL-Wide160 actor-critic checkpoint |
| `rl-switch` | 无独立权重；组合 `wide160` + `repair`，开关逻辑在 `ipd/policies/meta.py` |
| `weights/rl_dagger_gru_weights.json` | GRU-DAgger 单模型：85→GRU(64)→64→tanh→1；1,063,718 bytes；sha256 `7b7bbe67…c80cca196` |
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
| **Triple-v4** | `ipd/policies/candidates/triple_v4.py` | Triple-majority + Final 噪声互惠分支；v4 最佳纯规则 |
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

Triple-v4 在 Final 上增加一条窄分支：当 `p_d >= 0.70`、
`0.20 <= p_c <= 0.62`、`p_d - p_c >= 0.30` 且 `d_rate >= 0.15`
（样本足够）时，把对手视为噪声互惠者，对手刚 D 时以 0.75 概率原谅。
该分支不会命中干净 TFT / AllD / 时间表 late defector，也没有主动
probe，因此 `dev_family` 不变。完整报告见
`docs/research/triple-v4.md`。

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

> **历史说明：** 以下 E4 / P3 / PSRO / red 相关结论只用于复现。
> blue-red 对战已停用（ADR 0011），不再作为当前评估或 promotion 门槛。

### Protocol v5（2026-10-30；当前主口径）

- catalog 仍为 977 configs，manifest `c7aab4da...`；
- `SPLIT_SALT=protocol_v5` 重新划分 train/dev/test，405/977 个 config 的 tier 与 v4 不同；
- 新 final 为 `test_v8`；初始锁定，`test_red` 继续锁定；
- **v5 dev baseline 已运行**（33 candidates，1662.15s）；top 见 §2.0；
  旧 min_norm 门槛下无 candidate 通过；v4 top 7 / test_v7 只作 legacy；
- 文档：`docs/reference/protocol-v5.md`，ADR 0013。

### Protocol v4（legacy，2026-10-27）

- 977 configs，manifest `beb594ab...`；
- 224 个 Axelrod 策略 + 15 个 AxelrodMix；14 个 complex family、10 个 policy
  opponent；每个 entry 带 `thinking_level` / `complexity` 0–4；
- 默认 2048 人 / dev 30 / test 50；`test_v7` 已解禁但只作历史；
- v4 结果全部冻结，不能与 v5 混表；
- 文档：`docs/reference/protocol-v4.md`，ADR 0007 / 0008。

### RL-Pattern（2026-10-29；实验候选，未 promotion）

- 训练：RL-Repair → 20 iter entropy v4 fine-tune（`train_complex` +
  `train_axelrod`）→ 8 iter `train_v4` + `train_pattern_breaker`
  （权重 1,1,1,2,2,4）；最终离线选 iter6，不是 `--out` 的 iter8。
- 新 profile：`train_pattern_breaker` / `train_weak_complex`，只含
  train-tier 参数；`tests/test_protocol.py` 有泄漏检查。
- dev（30×2048，sample）：dev_param **548.45** vs RL-Repair 547.96；
  dev_complex **447.64** vs 434.28；dev_ood **519.46** vs 515.19；
  dev_axelrod 582.22 vs 582.46；`pattern_breaker` 236.8 → 420.3。
- test_v7 provisional（50×2048）：485.43 vs RL-Repair 486.31
  （-0.88），仍高于 RL-PPO 477.26；`selection_eligible=false`。
- 代价：dev `extort` / `axelrod` 小幅回退；test `hyper_extort` /
  `axelrod_extra` 回退，说明定向 fine-tune 仍有过拟合训练 catalog 的迹象。
- 判定：注册 `rl-pattern`，**不 promotion**；主权重不变。报告
  `docs/research/rl-pattern.md`、`results/rl_v4_20261028/`。
- 附带修复：`protocol eval` 在全部候选走 GPU 且无 CPU task 时
  `budget` 变量覆盖导致 `TypeError`；已改用 `gpu_budget`。

### RL-Wide（2026-10-29；实验候选，未 promotion）

- 架构：64/32 → net2wider 128/64 → 128/128；扩展工具
  `ipd/training/expand.py`（`python -m ipd train expand`），带 4 个测试。
- 训练：128/64 gentle（legacy + `train_v4`，8 iter）→ expand 128/128 →
  128/128 gentle（8 iter）→ 128/128 balance（4 iter，PB 权重 1/11，
  lr 3e-5，KL 0.1）；最终取 `wide_balance_ckpts/iter_004.json`。
- dev（30×2048，sample）：dev_param **548.24**、dev_family **583.08**、
  dev_complex **452.79**、dev_axelrod **582.30**、dev_ood **520.71**；
  相对 RL-Pattern：dev_param -0.21、complex +5.15、ood +1.25。
- `min_norm`：dev_param 0.9968、complex **0.9726**、ood **0.9596**、
  axelrod 1.0072、family 1.0；RL-Pattern 为 0.9970 / 0.8448 / 0.8424。
- test_v7 provisional（50×2048）：485.50 vs RL-Repair 486.31
  （-0.81）、RL-Pattern 485.43（+0.07）。
- 独立 seed 20261029：dev_param 546.60、dev_complex 452.05、
  dev_ood 522.14（RL-Pattern 546.66 / 447.26 / 520.86），
  complex/OOD 收益可复现。
- `dev_red`（30 seeds）：blue 378.67 / red 711.33，red_win 100%；
  anti-exploitation 未改善，不能把 complex 提升当成 red 防御。
- 192/128 消融：dev_complex 459.57、dev_ood 522.32 更高，但
  dev_param 547.44、test_v7 484.90 回退；不注册，128/128 是当前
  综合最优宽度（`results/rl_v4_20261030/`）。
- 代价：`extort` / `temporal` / `probe_exploit` 小幅回退；test-only
  `hyper_extort` / `axelrod_extra` 仍未恢复。
- 判定：注册 `rl-wide`，**不 promotion**；主权重不变。报告
  `docs/research/rl-wide.md`、`results/rl_v4_20261029/`。

### RL-Wide160（2026-10-31；实验候选，未 promotion）

- 宽度：从 RL-Wide 128/128 只扩第一层到 160/128；2 iter balance
  PPO（PB 1/11，lr 3e-5，KL 0.1）；最终取 iter1。
- dev（30×2048，sample）：dev_param **548.26**、dev_family **583.08**、
  dev_complex **454.31**、dev_axelrod **582.38**、dev_ood **521.37**。
- `min_norm`：dev_param 0.9970、complex **0.9974**、ood **0.9966**、
  axelrod 1.0073、family 1.0；相对 RL-Wide 的 0.973 / 0.960
  基本消除了 worst-family 退化。
- 独立 seed 20261029：dev_param 546.39、dev_complex 454.02、
  dev_ood 522.99；complex/ood min_norm 0.9976 / 0.9971，收益可复现。
- test_v7 provisional：485.62 vs RL-Repair 486.31（-0.69）、
  RL-Wide 485.50（+0.12）。
- `dev_red`（30 seeds）：blue 378.40 / red 711.73，red_win 100%；
  `test_red` provisional：blue 319.43 / red 742.43，red_win 100%；
  anti-exploitation 未改善。
- 分级诊断：dev_level3 451.52、dev_level4 447.54、
  dev_complexity4 401.27；低难度 tier 有 0.1–0.8 的小幅回退。
- 判定：注册 `rl-wide160`，**不 promotion**；主权重不变。报告
  `docs/research/rl-wide160.md`、`results/rl_v4_20261032/`。

### RL-Switch（2026-10-29；当前最佳 meta，未 promotion）

- 窄 meta：默认 RL-Wide160；第 6 回合前双方各 D≥2 且对手最新 C 时
  切换 RL-Repair；其余安全规则不变。
- dev（seed 27）：dev_param **548.58**、dev_family 582.91、
  dev_complex 454.24、dev_axelrod **582.86**、dev_ood **521.42**；
  相对 RL-Wide160：+0.32 / -0.17 / -0.07 / +0.48 / +0.05。
- test_v7 provisional：seed 27 **488.26**（+2.64）、seed 29
  **487.51**（+2.90）；min_norm 0.994–0.998。
- 全量：Protocol catalog nine-suite avg 520.43 → **521.18**；
  158 bank 554.45 → 554.37（基本持平）。
- red：dev_red blue 378.40 相同；test_red blue 319.37 vs 319.43；
  red_win 仍 100%，anti-exploitation 未改善。
- 分级诊断：dev_level0 -2.68，level1–4 +0.19/+1.34/+0.42/+1.13；
  dev_complexity0 -2.68，complexity1–4 +0.07/+1.62/+0.26/+1.38。
- 限制：C++ 双权重实现已完成（`cpp/policies/rl_switch_policy.hpp`，
  action parity 通过），但尚未接入真实比赛接口；不能直接 promotion。
- 报告 `docs/research/rl-switch.md`、`results/meta_20261039/`。

### RL-Gate（2026-10-29；No-Go）

- 学习型 match-level gate：前 10 回合 RL-Wide160，25 维对手特征 +
  logistic regression 选择剩余回合 RL-Wide160 / RL-Repair。
- 1052 个 train-tier 反事实样本，train_acc 0.810、val_acc 0.816。
- 阈值 0.5：dev_param 548.76，但 dev_complex 435.55、dev_ood
  516.07、test 485.67（过度切换）。
- 阈值 0.9：dev_param 548.30、dev_complex 454.63、dev_ood 521.34、
  test 485.84；基本退化为 RL-Wide160，远低于 RL-Switch。
- cost-sensitive 版（complex 权重 2.5、margin 5）：dev_param
  548.62、dev_complex 454.61、dev_ood 521.40，接近 RL-Switch；
  test 486.98，仍低 1.28。
- 判定：**No-Go**，保留为实验候选，不进入提交推荐。报告
  `docs/research/rl-gate.md`、`results/meta_20261040/`。

### Distill-Switch（2026-10-29；旧结果 No-Go，已由 GRU-DAgger 修正）

- 从 RL-Switch 教师收集 96,196 个非强制状态，蒸馏到 160/128 MLP；
- 旧结果 dev_param 535.46，低于 RL-Switch 548.58；
- **复查发现 normalization bug**：初始化 checkpoint 的
  `feature_mean/std` 不是 0/1，旧脚本却用 identity 训练并导出；
  同一 wide160 权重在 dev_param（5×1024）上正确归一化 544.74，
  identity 仅 498.61，旧蒸馏 531.75；
- 修正后 MLP BC 官方 dev_param 548.84，但 dev_complex 446.59
  （pattern_breaker 383.19 vs RL-Switch 512.94）；冻结 trunk /
  DAgger 仍无法恢复 pattern_breaker；
- 判定：**MLP 单模型 No-Go**；后续用 GRU + DAgger 解决。报告
  `docs/research/distill-switch.md`、`docs/research/dagger-gru.md`。

### GRU-DAgger（2026-10-29；实验候选，未 promotion）

- 单模型 1 层 GRU（64 hidden），教师为 RL-Switch 的确定性开关；
- 训练分布 6 源：`train` / `train_complex` /
  `train_pattern_breaker` / `extort_hard` / `periodic_hard` /
  `hard`；BC 12 fields/source，2 轮 DAgger 6 fields/source，
  epsilon=0.05，软标签；
- 官方 dev seed 27：548.58 / 582.90 / **455.13** / 582.49 /
  **521.54**；seed 29：546.97 / 582.59 / 454.39 / 581.93 /
  523.05；与 RL-Switch 差在 ±0.4 内；
- level / complexity 全部 ±1.3 内，level4 +1.20、complexity4
  +1.26；
- test_v7 provisional 487.89 / 487.02，比 RL-Switch 低
  0.37 / 0.49（test 锁定，不选模）；
- 部署：`ipd-gru-policy-v1`，1,063,718 bytes，sha256
  `7b7bbe67…c80cca196`；C++ `RLDAggerGRUPolicy` 已通过 action
  parity；
- 判定：**实验候选，未 promotion**；单模型部署替代 RL-Switch。
  报告 `docs/research/dagger-gru.md`、`results/distill_20261043/`。

### Triple-v4（2026-10-29；规则候选，未 promotion）

- Triple-majority + Final 噪声互惠分支：`p_d >= 0.70`、
  `0.20 <= p_c <= 0.62`、`p_d - p_c >= 0.30`、`d_rate >= 0.15`
  时，对手刚 D 以 0.75 概率原谅；无主动 probe。
- 官方 dev（seed 27）：536.87 → **538.66**（+1.79）；
  dev_complex 441.17 → **447.08**（+5.91）；dev_ood 511.62 →
  **514.41**（+2.79）；dev_axelrod 566.93 → **568.38**；
  dev_family 583.06 → **583.07**（+0.01）。
- 独立 seed 20261029：dev_param +1.72、dev_complex +5.78、
  dev_ood +2.92；`dev_level4` +11.92、`dev_complexity4` +9.04。
- 收益主要来自 `pattern_breaker`（complex +65.7 / ood +77.9）、
  `lookahead`（+18.5 / +15.0）、`extort` / `noisy` / `temporal`
  （+3～6）；代价是 `adaptive_mix` dev_ood -2.3、`portfolio` -1.0；
  `min_norm` 0.992–1.000。
- test_v7 provisional 471.54 → **473.68**（+2.14，不选模）。
- 判定：**最佳纯规则，未 promotion**；仍低于 sample-mode
  RL-Repair 547.96 / RL-PPO 540.35（dev_param），不能替代学习策略。
  报告 `docs/research/triple-v4.md`、`results/rule_v4_20261043/`。

### Periodic-hard（2026-10-29；No-Go）

- 新增训练专用 `periodic_hard`：Kamikaze period 4–12、7 种短周期
  Periodic、DynamicWindow、RandomFSM + anchors。
- RL-Wide160 上 4 iter：dev_param 548.54、dev_complex 451.41、
  dev_ood 520.43、test_v7 486.39；frequent_kamikaze +8、
  temporal +2，但 pattern_breaker -41、complex min_norm 0.909。
- 判定：**No-Go，不注册候选**；不要继续加大权重。若继续，
  应训练 periodic specialist + meta gate。报告
  `docs/research/periodic-hard.md`、`results/rl_v4_20261041/`。

### 输入特征 v2（2026-10-29；No-Go）

- 新增 `ipd-neural-features-v2`（85→97）：对手/自己
  period-2/3/4/5、opp/self mirror、opp exploit/forgive。
- Python `extract_features_v2`、GPU `features97` /
  `FeatureState97`、C++ `extract_features_v2` 三端 parity（1e-12）；
  `train expand --features v2` 和 `train rl --features v2` 已接入；
  C++ 导出写 `IPD_FEATURES_V2` 宏。
- RL-Wide160 上 8 iter：v2 iter2 dev_param 547.90、dev_complex
  456.43、dev_ood 521.62、test 485.27；iter4 dev_complex 458.72
  但 dev_param 547.59、test 485.25。相对 RL-Wide160 的
  548.26 / 454.31 / 521.37 / 485.62，complex 略升、主排序和 test
  回退。
- 判定：**No-Go，不注册候选**；保留 v2 契约供后续 extortion /
  payoff-differential 特征实验。报告 `docs/research/feature-v2.md`、
  `results/rl_v4_20261035/`。

### RL-Wide160 全量对手扫描（2026-10-29；诊断）

- 覆盖 158 类 bank（16 reps）、977 Protocol configs（8 reps）、224
  Axelrod fast（8 reps），参考候选 RL-Repair / RL-Pattern / RL-Wide /
  Triple-majority。
- 总体：158 bank 547.35；protocol 507.66；Axelrod fast 575.39。
- 薄弱点优先级：
  1. extortion/ZD：`hyper_extort` 176、`extort` 402；
     ZDExtort2/3/4 diff -100～-364；
  2. periodic/Kamikaze：`frequent_kamikaze` 169、temporal Kamikaze
     198–208；
  3. Gap/FSM：`Gap0.55` 192 vs best ref 538、`FSM.F5` 569 vs 933；
  4. PatternBreaker 尾部：test threshold=.6 window=6 288；
  5. noisy/Joss：`Joss p=.52` 276；
  6. late defector 347：安全规则的结构性限制。
- 完整 243 Axelrod 的 19 个 long-run Meta 超时，使用 224 fast。
- 报告 `docs/research/weak-opponents-20261036.md`、
  `results/full_test_20261036/`；下一步优先做 extortion-aware 与
  periodic/Kamikaze 训练 profile。

### Extort-hard（2026-10-29；No-Go）

- 新增训练专用 `extort_hard`：ExtortZD chi 1.5–7.0、GenerousZD、
  Random、TFT/Grim/AllD/AllC/Bully anchors。
- strong（权重 2/9，6 iter）：dev_param 548.77、dev_complex 450.25、
  test_v7 486.44；test hyper_extort +13.2、extort +5.5，但
  pattern_breaker -58.2、complex min_norm 0.850。
- gentle（权重 1/10 + train_complex，3 iter）：dev_param 548.42、
  dev_complex 453.93、test 485.65；extort 只有 +1.7/+3.0。
- 判定：**No-Go，不注册候选**；不要继续加大 extort_hard 权重。
  下一步若继续 extortion，应做显式 ZD 识别 + expert / portfolio，
  而不是全局 PPO。报告 `docs/research/extort-hard.md`、
  `results/rl_v4_20261037/`、`results/rl_v4_20261038/`。

### Protocol v3 baseline（legacy）

- 532 configs；20 candidates dev；test_v6：
  RL-Repair 484.32、A60 476.54、Adv-RL 471.44；
- 报告：`results/protocol_v3/README.md`，ADR 0006。
- Protocol v2（224 configs、test_v5 477.05）保留为 legacy；
  `results/protocol_v2/`、ADR 0005。

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

### PSRO — full blue-vs-red meta-game（实验；未 promotion）

- 新增 `ipd/training/psro.py`：blue population + red exploiters；
  cross-play 估计 `blue_score - red_score`；blue 用 maximin、red 用对偶
  maximin；oracle 复用 `train rl --opponent-pool`。
- 新增 `ipd/policies/psro.py`（meta-policy，按 blue maximin 权重每场
  采样一个成员）和 `psro-br`（最后一轮 best response）。
- 配置 `configs/psro/psro_20261027.json`；结果
  `results/psro_20261027/`；2 轮 PPO oracle。
- Blue maximin support：Final-tuned 0.725 + Triple-majority 0.275；
  RL-Repair / PSRO-BR 权重 0。meta exploitability 284.797 → 284.797，
  oracle 未进入 support。
- 诊断（4 fields × 512）：PSRO-Meta dev_param **527.71** vs RL-Repair
  **543.35**；PSRO-BR 525.58。held-out Red-v5/v6 blue mean：
  PSRO-Meta **373.70**、PSRO-BR 354.45、RL-Repair 348.35；
  red-minus-blue 378.5 → 297.5（**-21.4%**，v5 0%、v6 -37.3%）；
  red 仍 100% 胜。
- **No-Go for promotion**：exploitability-reduction 目标达到，但
  官方/dev 掉约 15.6、family guardrail 失败，Gate D 不通过；保留
  `psro` / `psro-br` 为实验候选，不改主权重。下一步若继续 PSRO，
  需要更强 blue oracle（更多迭代 / 规则初始化）或 red oracle
  （double oracle），不要重复相同两轮配置。

### PSRO from scratch（现有模型只做 red；No-Go for promotion）

- 配置：`configs/psro/psro_scratch_20261027.json`；结果
  `results/psro_scratch_20261027/`；pilot/v2/v3 为 ablation。
- Blue：`Scratch-Init`（随机 MLP）+ `Scratch-Base`（从零官方 PPO，
  30 iter，dev_param 528.99）+ 从零 PPO BR；**没有已有模型进入 blue
  或 oracle 初始化**。
- Red：已有模型/规则（RL-Repair、A60、RL-PPO、Adv-RL、Final、Triple、
  SelfPlay、Blue-Meta-v2、Red-v1/v3/v4）仅作对手。
- 2 轮 PSRO：initial exploitability 62.81 → 0.108 → **0.0**；
  blue support PSRO-BR-02 0.9984。
- 诊断：PSRO-Scratch dev_param **480.43**（RL-Repair 543.35），
  min_norm 0.5501，worst family `noisy` 249.1；但 dev_red
  blue 436.2 / red 210.2、**red_win 0.0**，held-out Red-v5/v6
  blue 491.4 / red 197.4、**red_win 0.0**。这是项目首次翻转 red 结果。
- **No-Go for promotion**：official 掉约 63，Gate D 不通过；register
  `psro-scratch` / `psro-scratch-br` 为实验候选。Scratch-Base
  （528.99，red_win 1.0）是更保守的从零模型。
- 细节：`docs/research/psro-scratch.md`、`results/psro_scratch_20261027/`。

### PSRO v4 — stronger oracle（实验；No-Go for promotion）

- 配置：`configs/psro/psro_v4_20261029.json`；结果
  `results/psro_v4_20261029/`；3 轮 × 30 iter、40% red、低 KL
  （`bc_kl=0.02`），blue 加入 RL-Switch / RL-Wide160 / RL-Pattern。
- 初始 exploitability **290.00**；第 1 轮 BR_01 对 Red-v1/v3/v4
  score difference **+33.1 / -265.9 / -203.8**，直接进入 blue
  maximin support（权重 1.0），exploitability **0.00**；BR_02/03
  未进入 support。
- 部署 support 内 BR_01（不是最后一轮 BR_03）：`psro-v4-br`。
- 诊断：dev_param（10×1024）**486.93 vs RL-Repair 546.47**，
  min_norm 0.5599，worst family `noisy` 260.0；dev_complex（3×512）
  436.43 vs 434.28；dev_ood 477.72 vs 515.19；dev_axelrod 498.11
  vs 582.46；dev_red 与 held-out Red-v5/v6 **red_win 0.0**
  （Red-v5/v6 blue 530.55 vs RL-Repair 348.35）。
- **No-Go for promotion**：official 掉约 59.5，仍是 anti-exploitation
  specialist；register `psro-v4` / `psro-v4-br` 为实验候选。
- 细节：`docs/research/psro.md` §8、`results/psro_v4_20261029/`。

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

Protocol v5（当前主口径）：

```bash
python3 -m ipd protocol audit
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO
# 初始 test_v8 锁定；provisional 诊断用 --no-consume
python3 -m ipd protocol eval \
  --candidates RL-Switch,GRU-DAgger,RL-Wide160,RL-Pattern,RL-Repair,RL-PPO-Hard60,Adv-RL \
  --tiers test_v8 --fields 50 --n 2048 --no-consume
# final 前先解禁 test_v8，再用 --acknowledge-once 消费
# red / blue-red 命令仅作历史复现，当前流程不再执行（ADR 0011）
```

Legacy 四分布 / blind（只用于历史复现）：

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

# RL-Pattern（完整命令与 checkpoint 选择见 docs/research/rl-pattern.md）
# 1) 20 iter entropy v4 warm start；2) train_v4 + train_pattern_breaker
#    8 iter，最终离线选 pb_v4_ckpts/iter_006.json
python3 -m ipd train rl --device cuda --iterations 20 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261029 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --train-dists original,late,grim30,wide,train_complex,train_axelrod \
  --dist-weights 1,1,1,2,2,1 \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --no-validation \
  --out results/rl_v4_20261028/ent01_proper_weights.json \
  --checkpoint-out results/rl_v4_20261028/ent01_proper_ac.json

python3 -m ipd train rl --device cuda --iterations 8 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261032 \
  --init-weights results/rl_v4_20261028/ent01_proper_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,4 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261028/pb_v4_ckpts \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261028/pb_v4_weights.json \
  --checkpoint-out results/rl_v4_20261028/pb_v4_ac.json \
  --history-out results/rl_v4_20261028/pb_v4_history.json

# RL-Wide（完整命令见 docs/research/rl-wide.md）
# 1) expand 64/32 -> 128/64；2) 128/64 gentle 8 iter；
# 3) expand -> 128/128；4) 128/128 gentle 8 iter；
# 5) 128/128 balance 4 iter（PB 1/11）；最终取 iter_004
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_pattern.json --hidden 128 64 \
  --out results/rl_v4_20261029/wide_ac.json

# PSRO（完整 blue-vs-red + 2 轮 PPO oracle；实验）
python3 -m ipd train psro --config configs/psro/psro_20261027.json \
  --workers 20
bash results/psro_20261027/run_eval.sh

# PSRO v4（3 轮 × 30 iter、40% red、低 KL；实验）
python3 -m ipd train psro --config configs/psro/psro_v4_20261029.json \
  --workers 20
bash results/psro_v4_20261029/run_eval.sh

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
g++ -std=c++17 -O2 -Icpp cpp/rl_dagger_gru_demo.cpp -o rl_dagger_gru_demo
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
- `cpp/policies/rl_switch_policy.hpp`：RL-Switch 双 MLP + 开关；
- `cpp/policies/rl_dagger_gru_policy.hpp` /
  `cpp/rl_dagger_gru_demo.cpp`：GRU-DAgger 单模型；
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
5. **规模不可混用**：旧表多为 10/20-field blind 或 v3 的 299 人；v4/v5
   默认 2048 人、dev 30 fields、test 50 fields。
6. **split 已重划**：`heldout_v2` 已泄漏，`heldout_v3` 已被 E1c 使用，
   `heldout_v4` 未用但只覆盖单套 family；新候选统一用 Protocol v5。
   `test_v8` / `test_red` 初始锁定（ADR 0013）；v4 `test_v7`、v3
   `test_v6` 都只作历史。v5 与 v4 共用 catalog，旧 v4 候选在 v5 test
   上不是 held-out。
7. **SelfPlay 不是 anti-exploitation 证据**：final test 中的 held-out red
   是 RL-BC 互惠型对照，不能证明能防针对性 red。
8. **GRU / SelfPlay-GRU 有状态**：每场必须 `reset()`；同一时间步不要重复
   调用 `act()`。
9. **规则默认参数不是 tuned**：`ipd/core/machine.py`、`cpp/ipd_strategy.cpp`
   的默认参数是早期版本；提交时使用候选或 `FinalParams::tuned()`。
10. **权重文件必需**：Neural / GRU / RL / SelfPlay 都要带 JSON 权重；单文件
    提交时需要内嵌。
11. **Axelrod 是可选重量级依赖**：v4 full tiers 需要
    `requirements-axelrod.txt`；使用 50 个精选策略 + 15 个组合，不跑完整
    243；Darwin genome 是类变量，结果与顺序相关，不能当稳定强度。
12. **所有候选都应在第 100 天 D**；移植时漏掉会损失最后一天收益。
13. **NPU 路径未实机验证**：代码支持 `--device npu`，本轮只在 CUDA/CPU
    上完成训练和测试。
14. **PSRO / red 仅历史**：`psro` / `psro-scratch` / `psro-v4` 等
    只保留为历史实验和复现；blue-red 已停用（ADR 0011），不再作为
    promotion 证据，也不新增 red 任务。
15. **RL-Pattern / RL-Wide / RL-Wide160 仍有过拟合迹象**：test_v7
    provisional 分别比 RL-Repair 低 0.88 / 0.81 / 0.69，test-only 的
    `hyper_extort` / `axelrod_extra` 仍有回退。不要把 dev 第一当成
    promotion 证据；先用独立 seed 复现，并检查外部比赛。
16. **RL-Wide / RL-Wide160 部署文件更大**：128/128 权重约 890KB，
    160/128 约 1.1MB，C++ 导出已通过，但比 64/32 的约 249KB 更大；
    如果比赛有文件大小限制，优先用 RL-Pattern 或 RL-Repair。
17. **RL-Switch C++ 已移植但未实战**：`cpp/policies/rl_switch_policy.hpp`
    组合 160/128 与 64/32 两个 MLP，固定序列 action parity 通过；
    但尚未接入真实比赛接口；test 增益仍需外部比赛验证。
18. **RL-Gate 不是提交候选**：学习型 gate 阈值 0.5 过度切换、
    0.9 退化为 RL-Wide160；分类准确率不等于得分最优。不要用
    `rl-gate` 替换 `rl-switch`，除非重新设计 cost-sensitive gate。
19. **GRU-DAgger 是单模型替代，不是 promotion**：dev 两个 seed
    与 RL-Switch 持平，但 test_v7 provisional 低 0.37/0.49；单文件
    1.06MB；有状态，必须每场 `reset()`，同一时间步只调用一次
    `act()`；C++ 已通过固定序列 parity，但未接真实比赛接口。若比赛
    接口不能保证 reset，继续用 RL-Switch / RL-Wide160。不要用
    test provisional 宣称超越 RL-Switch。
20. **Triple-v4 是规则候选，不是学习策略替代**：dev_param 538.66
    仍低于 RL-Repair 547.96 / RL-PPO 540.35（sample）；收益集中在
    pattern_breaker / lookahead / noisy / extort，规则线仍有
    probe_exploit、late、axelrod_extra 大缺口。不要为了追分加入
    全局主动 probe（dev_family 的 DelayedGrim 会崩）；如需提交，
    先补 C++ 移植和 action parity。

---

## 8. 交接检查清单

- [ ] 确认比赛接口：`act` / `move` 签名、历史格式、是否允许 `reset()`。
- [ ] 确认是否允许每天运行三个子策略（Triple）。
- [ ] 确认是否允许提交权重文件；单文件限制下决定是否内嵌。
- [ ] 确定提交版本：RL-Switch / GRU-DAgger / RL-Wide160 / RL-Wide / RL-Pattern / RL-Repair / A60 / RL-PPO / Adv-RL / SelfPlay / 规则。
- [ ] 若提交 RL-Repair，先做独立 seed 复现，再决定是否替换主权重。
- [ ] 运行 `python3 -m ipd test` 和
      `python3 -m ipd self-test --candidates all`。
- [ ] 运行 `python3 -m ipd protocol audit`，确认 manifest / family / config
      无泄漏。
- [ ] 对学习策略跑 `protocol eval --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood`
      （2048 人 / 30 fields）；`test_v8` 初始锁定，provisional 诊断用
      `--no-consume`，final 前先解禁再传 `--acknowledge-once`。
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
| `ipd/evaluation/opponent_suite.py` | legacy validation / heldout_v2 / v3 / v4 |
| `ipd/evaluation/protocol.py` | Protocol v5 catalog / audit / manifest / ledger |
| `ipd/evaluation/protocol_cli.py` | `python -m ipd protocol` |
| `ipd/evaluation/policy_pool.py` | P3 frozen blue/red pool |
| `ipd/training/rl.py` | PPO 训练、rollout、验证、导出 |
| `ipd/training/psro.py` | PSRO meta-game / blue-vs-red maximin / oracle |
| `ipd/policies/psro.py` | PSRO meta-policy 推理（每场采样一个成员） |
| `ipd/training/risk.py` | E2 CVaR / Dirichlet / DRO |
| `ipd/training/opponent_model.py` | C2/C3 对手模型 / gate |
| `ipd/training/selfplay/` | 21 维自博弈训练器 |
| `ipd/policies/moe.py` | C3 three-way gate + MoE |
| `ipd/policies/belief_planner.py` | C1 belief planner |
| `ipd/policies/candidates/rl_dagger_gru.py` | GRU-DAgger 候选 |
| `cpp/policies/rl_dagger_gru_policy.hpp` | GRU-DAgger C++ 推理 |
| `docs/research/dagger-gru.md` | GRU-DAgger 方法与结果 |
| `results/distill_20261043/` | DAgger/GRU 实验数据与汇总 |
| `docs/README.md` | 文档导航与状态 |
| `docs/architecture/overview.md` | 分层与接口 |
| `results/*/README.md` | 各实验的完整报告 |
