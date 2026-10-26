# IPD 自博弈训练计划（下一 session 使用）

- **Status:** implemented (2026-10-26); see `docs/SELFPLAY.md` and `results/selfplay_final/README.md`
- **Repo:** `/home/frti64/pri_dile`
- **Date:** 2026-10-26
- **目标:** 训练一个 100 轮 IPD 蓝军策略，既能拿高分互惠基本盘，又能防御“有竞争力的基模狙击手”，并避免 adv-* 那种“只在少数合成对手上刷分”的偏科。

---

> 实施备注：完整训练器/测试/三阶段课程/red league 已落地到
> `ipd/training/selfplay/`。纯从零 15k self-play 在 held-out final 为 504.57、
> 2-field 四分布 524.77；最终选定策略使用 `selfplay-bc` 把现有 RL-PPO 蒸馏为
> 21 维 GRU，再做可选的 self-play 微调，四分布 552.52（8 fields）、blind
> 560.43 / 8.10、held-out final 505.06。详细结果见
> `results/selfplay_final/README.md`。

## 0. 背景结论（为什么要这样设计）

之前对 `adv-*` 的分析结论：

- adv-* 的四分布增益几乎全部来自 `LateD*/Harvest*` 和 `Bully`；
- 去掉这两类后，Adv-RL 的四分布约 `-0.15`、blind 约 `-1.02`，不再优于 base；
- `blind` 并不是真正的 OOD：`sample_blind_field == sample_wide(seed, 299, 0.15)`，只是换 seed；
- adv 调参 seed 与 benchmark seed 重叠，benchmark 不是完全独立；
- 因此 adv-* 更像 specialist，不是全面更强的 policy。

本计划的核心对策：

1. **对手构成是不确定集，不是固定配比**；
2. **red 是威胁种群，不是单点 best response**；
3. **train/validation/test 按 family/parameter 真正 held-out**；
4. **蓝军按绝对分优化，worst-family 作为一等公民**；
5. **先 contract + tests，再训练；不直接堆单文件。**

---

## 1. 技术路线

- **框架:** PyTorch 2.x + CleanRL 风格的自研 recurrent PPO。
- **不使用 Stable-Baselines3**：SB3 的 RecurrentPPO 主要是 LSTM + 单策略；本任务需要双策略、GRU、红蓝独立 buffer/optimizer、红蓝子轨迹和滑动窗口约束。
- **不使用 RLlib/TorchRL**：抽象过重，定制非对称双策略仍要写大量代码。
- **环境:** 自写 32-env NumPy 向量化 IPD；固定 100 步；不把 Gymnasium 作为核心依赖。
- **设备:** 环境模拟 CPU；两个 GRU/PPO 更新 GPU（如有）。单进程即可，不需要 Ray。
- **代码落点:** 新增 `ipd/training/selfplay/`，不改现有 `ipd/training/rl.py`、`ipd/policies/rl.py` 和已有结果。
- **对手库:** 优先复用 `ipd/core/opponents`、`ipd/policies/opponents` 的现有实现。

---

## 2. 先冻结的 Contract

### 2.1 21 维观测

按 `2 + 4 + 8 + 7 = 21` 冻结：

| 组 | 维度 | 内容 | 注意 |
|---|---:|---|---|
| 时间 | 2 | `t/100`, `(100-t)/100` | 明确 t=1..100 还是 0..99 |
| 上一轮 | 4 | CC/CD/DC/DD one-hot | t=0 用全零 |
| 微观历史 | 8 | 过去 4 天双方动作 | t<4 padding 规则 |
| 宏观 | 7 | 对手/自身累计背叛率、滑动 10 轮对手背叛率、Δ、CC 率、DD 率、t≥90 指示符 | 分母 `max(1,t-1)`；Δ 符号要定义 |

**禁止:** 把 opponent ID / opponent type / family one-hot 放进观测，否则会泄漏。

### 2.2 动作、收益与单位

- 双方同时决策，必须用同一 pre-step history；禁止 sequential leakage。
- 收益矩阵明确行/列是谁；蓝军 reward = 自身 payoff。
- 蓝军 reward 用每步 `0–10`；红军 reward 也用每步单位。
- `Score_FT`、`Score_TM` 如果是 100 步总分（约 600），必须除以 100 再和红军的每步均分比较。
- `TFT` 目标线 `594`，硬熔断线 `590`。

### 2.3 训练/验证/测试拆分

- **训练池:** 固定代表 + 参数化变体。
- **Validation:** held-out 参数区间 + 部分 held-out family；用于 fast/full validation、early stop、选模。
- **Final test:** 完全 held-out family/参数 + held-out red；只跑一次，不参与 early stop/选模。

不能只换 seed。至少要做到：

- 不同 family 权重；
- 不同 noise/late day/Bully/Extort 参数；
- 至少留 1–2 个 family 或一组参数变体完全不参与训练。

---

## 3. 对手池 v2

### 3.1 阶段配比

| 阵营 | Phase 1 | Phase 2 | Phase 3 |
|---|---:|---:|---:|
| 善意互惠：TFT / Pavlov / Grim / TFT-lastD / ContriteTFT | 35% | 25% | 25% |
| 顶尖规则：Final-tuned / Triple-majority | 20% | 15% | 15% |
| Red 威胁种群 | 0% | 20% | 25% |
| 噪声/误会：NoisyTFT.02-.20 / Joss / GTFT | 15% | 15% | 15% |
| 敲诈/压迫：Extort1.5/2/3 / Bully | 5% | 10% | 10% |
| 随机：Random.05-.90 | 15% | 10% | 5% |
| 极端/阶段鲶鱼：AllD / AllC / EarlyProbe / Periodic / Harvest 变体 | 10% | 5% | 5% |

- 所有比例不是固定值：用 Dirichlet 扰动；red 在 `15%–25%` 随机。
- 阶段式对手作为参数化变体嵌入 good/top/special，不只占 5%。
- Phase 1 早期注入 Random/Noisy；Extort 先给 5%，避免初期坍塌。
- Phase 3 降低 Random，提高 red/阶段式/Noisy。

### 3.2 动态/阶段对手生成器

不要只放固定 `TFT-lastD`、`Harvest45`、`LateD10`；生成参数化变体：

| 类型 | 参数 |
|---|---|
| 动态窗口自适应 | `W~U(3,8)`，`p_defect~U(0.15,0.35)`，retaliation/forgiveness 随机 |
| Harvest | 变脸天数 `T~U(25,75)`，背叛后是否可修复随机 |
| Probe+Phase | 前 3–5 天随机探针，然后 TFT 或 Harvest |
| Kamikaze | 每 `K~U(10,20)` 天偷一次，被反击永久报复 |
| Prey | AllC、Random.90、弱 TFT，用于测绝对收割 |
| Extort | ZD 参数 1.5/2/3，加随机扰动 |

训练用随机参数；validation 用 held-out 参数区间；final test 用 held-out family/参数。

---

## 4. Red 威胁种群

### 4.1 类型与比例

| Red 类型 | 默认比例 | 训练对手 | 模拟威胁 |
|---|---:|---|---|
| **Red-Base** | 40% | 冻结的 Final-tuned / Triple-majority / RL-final | 研究过基模的强队 |
| **Red-Blue** | 30% | 当前 blue 的旧快照（每 K 轮刷新） | 现场 scout 你的自适应对手 |
| **Red-Strong** | 30% | 全场，最大化绝对分 | 不专门针对你、但本身很强的队 |

阶段建议：

- Phase 2：Red-Base 50% / Red-Blue 30% / Red-Strong 20%；
- Phase 3：Red-Base 30% / Red-Blue 40% / Red-Strong 30%。

如果只能做一个 red，选 **Red-Base**。

### 4.2 Red 奖励

```text
R_t_red = (s_t_red - s_t_target)
          - λ * max(0, α * bar_s_base - bar_s_red)^2
```

- `target`: Red-Base 用基模；Red-Blue 用旧 blue 快照。
- `W ~ U(15, 25)`，只在 `t >= W` 时计算窗口惩罚；`t < W` 用 mask。
- Phase 2: `α=0.85, λ=5`；Phase 3: `α=0.90, λ=10`。
- 固定 λ 是 penalty method，不是严格 Lagrangian；必须记录 constraint violation；长期违反时再加 dual update。

### 4.3 竞争力约束

- 在 calibration suite 上预计算基模的 per-family 分数并冻结。
- 要求 red 在每个 family 上不低于 `α * base_family_score`；至少监控 worst-family。
- Red 自己训练只打 target，但每 N 轮在 calibration suite 上评估；低于门槛的 checkpoint 不进 league。
- 否则 red 会变成“只会打基模、打其他对手就崩”的偏科刺客。

### 4.4 Red League

- 保留最近 K 个 red checkpoint。
- 采样 current/past red，避免 blue 过拟合到单一 red。
- 留 1–2 个 red 完全不参与训练，只做 validation/test。
- 采用 block-wise alternating：blue 更新若干轮 → red 更新 → 刷新 league。
- Phase 2/3 可把 red 比例提高到 30%，或给 red 单独 rollout；20% 的 red 数据量偏少。

---

## 5. 网络与 PPO

### 5.1 网络

- 蓝军、红军各一个独立 `GRUActorCritic`。
- 2 层 GRU，hidden=64。
- Actor: `64→32→1→Sigmoid`；Critic: `64→32→1`。
- 蓝军 critic 预测蓝军 return；红军 critic 预测 red-target 差值。
- 两个独立 optimizer；对手 bot 全部 `requires_grad=False`；双方 hidden state 不跨图传递。

### 5.2 Rollout 与 PPO

- 32 envs × 100 steps = **3,200 transitions/iteration**。
- Recurrent buffer 按 episode 存储：obs、action、logp、value、reward、done、hidden state、mask、opponent type。
- GAE 逐 env 计算；固定 100 步，terminal value=0。
- **Gamma 默认 1.0**（有限 100 步，末日 D 重要）；如果 value loss 不稳定，再试 0.995。不要直接用 0.99。
- GAE lambda=0.95。
- PPO: 4 epochs，clip=0.2，value coef=0.5，entropy=0.01，grad clip=0.5。
- Minibatch 按 sequence/chunk 切：建议 8 个完整 episode（800 steps）或 32 个 25-step chunk；不能按单步 shuffle。
- 蓝军 lr=3e-4；红军 lr=1e-4~3e-4；Phase 3 余弦退火到 1e-5。
- 红军每轮约 640 步；red minibatch 建议 128，或累计 2–4 个 iteration 再更新。

---

## 6. 训练课程

总预算 15,000 episodes（约 1.5M env steps，约 469 个 iteration）。

| 阶段 | Episodes | Red | 重点 |
|---|---:|---|---|
| **1a 互惠奠基** | 0–3,000 | 关闭 | 只放合作/噪声，确立 600 分基本盘 |
| **1b 基础抗性** | 3,000–8,000 | 关闭 | 加 Bully/Extort/LateD/Random；学止损与收割 |
| **2 高压磨砺** | 8,000–13,000 | Red 20% | α=0.85，λ=5；Red-Base 为主 |
| **3 终局收敛** | 13,000–15,000 | Red 25% | α=0.90，λ=10；提高 Red-Blue/阶段式；LR 退火 |

阶段目标：

- Phase 1a: vs TFT 稳定 604/594；vs AllD 第 2–3 轮止损，得分 95+。
- Phase 1b: vs Bully 650+；vs Harvest/LateD 不被单边收割；vs NoisyTFT.02 不互掐。
- Phase 2: vs 动态窗口/Periodic 压制；vs Extort 不崩；vs Red-Base 不被打穿。
- Phase 3: 决策边界平滑；对红军的试探能 1–2 轮内反制；合作基本盘不丢。

---

## 7. 评估、早停与选模

### 7.1 三层评估

| 层级 | 频率 | 规模 | 用途 |
|---|---|---|---|
| Fast validation | 每 200 episodes | 10–15 个代表对手，N=5 | 看趋势、抓崩坏 |
| Full validation | 每 1,000–2,000 episodes | 50 对手，N=5，per-family | early stop、选模 |
| Final test | 全程一次 | held-out family/参数 + held-out red，N=20+ | 最终报告，不参与选择 |

50 个 validation 对手如果还是同一 85 类库、同一参数，只是换 seed，那不是 OOD。

### 7.2 指标

- 主指标：absolute score per family、worst-family、prey score、kamikaze tolerance score。
- 次指标：gap to contenders（FT/TM/Adv-GRU/RL-final）。
- 不把 win rate 作为主指标。

### 7.3 早停

- Warm-up：前 1,000 episodes 不启用早停；Phase 1 不检查 Bully。
- 用 20–50 局均值 + 置信下界；连续 3 次失败才触发。
- 红线：TFT < 590（目标 594）、Bully < 650、NoisyTFT.02 < 560、AllD < 95、min_family 低于阈值。
- 触发后优先 rollback 到 best checkpoint + 降 LR/调整配比；只有灾难性崩坏才 hard stop。

### 7.4 Checkpoint 选择

先过滤：

```text
TFT >= 594
Bully >= 650
NoisyTFT.02 >= 560
AllD >= 95
min_family >= 阈值
held-out red 不超过阈值
```

再排序：

```text
Score_Select = mean
             + 0.5 * S_vs_FT
             + 0.5 * S_vs_TM
             + 0.5 * min_family
             - 2.0 * sigma
```

`sigma` 明确为跨 family/多 seed 的标准差。保存 top 3；final test 只跑一次，不拿 test 反复选。

---

## 8. 工程结构

```text
ipd/training/selfplay/
  config.py
  obs.py            # 21 维特征
  env.py            # 32-env 向量化 IPD
  opponents.py      # 训练池 + validation suite
  red_pool.py       # Red-Base / Red-Blue / Red-Strong + league
  models.py         # 两个独立 GRUActorCritic
  rollout.py        # recurrent buffer、GAE、sequence minibatch
  ppo.py            # PPO-Clip + red 约束
  train.py          # 三阶段课程、交替更新、early stop
  evaluate.py       # per-family 评估、checkpoint 遴选
  tests/
```

- `ipd/policies/selfplay_gru.py`: 提交推理 wrapper，注册到 registry。
- `configs/selfplay/`: smoke / full 配置。
- `results/selfplay_<date>/`: 日志、checkpoint、评估报告。
- 不改现有 `ipd/training/rl.py` 和已有结果。
- Checkpoint 保存：模型、optimizer、config、RNG state、对手池版本、验证历史。
- 日志：per-episode + per-eval，per-family、red constraint violation、action distribution、specialization index（top-family 贡献、max-min gap）。

---

## 9. 测试与验收

必须先过这些测试，再跑 full training：

| 测试 | 验收标准 |
|---|---|
| 21 维观测 | t=0/1/4/99 无 NaN，padding 正确 |
| done mask | 只在 episode 边界重置 hidden state |
| 向量化一致性 | 32-env 与单 env 循环同 seed 结果一致 |
| 同时决策 | 双方动作都基于同一 pre-step history |
| GAE | 小样本与参考实现一致 |
| Recurrent PPO | minibatch 后 hidden state 正确 |
| 梯度隔离 | blue step 不改 red 参数，反之亦然；bots 无梯度 |
| 对手池 | 100k 采样比例在容差内 |
| Red 约束 | t<W 不 NaN；单位一致；per-family anchor 正确 |
| 确定性 | 同 seed 可复现 |
| Smoke run | 200 episodes 跑通，日志/checkpoint/resume 正常 |

---

## 10. 主要风险与对策

| 风险 | 对策 |
|---|---|
| Blue 过拟合 red | Red league + held-out red |
| Red 自己偏科 | per-family anchor + Red-Strong + calibration suite |
| Blue 过度防御、丢合作 | 纯 payoff + TFT/Grim 早停 + 配比随机化 |
| Validation 泄漏 | train/val/test 按 family/参数拆分 |
| Red 样本不足 | Phase 2/3 提高 red 比例或单独 rollout |
| 评估成本过高 | fast/full 两层评估 |
| 约束 gaming | W 随机 + per-family anchor + 监控 + 可选 dual update |
| 训练 cycling | block-wise alternating + league + entropy |
| Gamma 折现过重 | 用 1.0 或 0.995 |
| 合成库外推 | 报告限制；如允许，考虑 checkpoint ensemble 或 safety fallback |
| 硬编码启发式被反制 | 只作为特征/对手/评估，不直接写死策略 |
| 过早冻结特征层 | 用 layer-wise LR decay，不硬冻结 |

---

## 11. 实施顺序

1. **M0 Contract + tests**
   - 21 维表、单位、对手池、red 定义、验证拆分、测试骨架。
2. **M1 Env + opponents**
   - 32-env、比例采样、确定性、smoke。
3. **M2 Models + PPO**
   - 两个 GRU、recurrent buffer、GAE、200-episode smoke。
4. **M3 Red population + curriculum**
   - Red-Base/Blue/Strong、league、约束、三阶段。
5. **M4 Full run**
   - 15k episodes、fast/full validation、early stop、checkpoint。
6. **M5 Held-out test + submission**
   - 独立 test、选模、导出 wrapper、必要时 C++。

每个里程碑都要有可执行验收，不通过不进下一步。

---

## 12. 需要确认/可能调整

1. Red 配比：默认 40/30/30，Phase 2 50/30/20，Phase 3 30/40/30。
2. 是否复用当前 `ipd` 对手库（推荐），还是 standalone。
3. 训练设备：CPU 还是 GPU；CPU 能跑 smoke，full 15k 会慢。
4. Gamma：默认 1.0；如 value loss 不稳定再试 0.995。
5. Phase 2/3 是否允许 red 比例从 20% 提到 30%。
6. 比赛接口：是否允许 ensemble/fallback，还是只能单策略。
7. 是否需要 C++ 导出；训练阶段先不阻塞。

---

## 13. 不要做的事

- 不要直接用 Stable-Baselines3 的 PPO/RecurrentPPO 作为最终实现。
- 不要只训练固定 80/20 配比。
- 不要只用一个 red。
- 不要把 opponent ID/type 放进观测。
- 不要在 test set 上反复选模。
- 不要每 200 episodes 用 50 个对手做全量评估（成本太高）。
- 不要用 gamma=0.99 而不测试末日 D 的影响。
- 不要把“统计 TFT / 赤字触发 / 硬切 AllD / 动态抢跑”直接写死为策略逻辑。
- 不要在没有测试的情况下声称 done mask/梯度隔离/GAE 正确。
- 不要修改现有 `ipd/training/rl.py` 和已提交结果。
