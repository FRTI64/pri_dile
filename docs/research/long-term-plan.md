# IPD 策略长期改进计划（无 3 天限制）

- **Status:** E0 PASS; E1c `RL-Repair` Go candidate; E2 No-Go; P3 red pool / league-lite / expansion No-Go; P2 C1/C2/C3 offline No-Go; P2 C2/C3 excitation data Go / planner No-Go; P2 C3 three-way gate + MoE No-Go; P2 C3 repeated calibration No-Go; **Protocol v4 已定义**; **blue-red 已停用（ADR 0011）**
- **New split:** `train_v4` / `dev_param` / `dev_family` / `dev_complex` /
  `dev_axelrod` / `dev_ood` / `test_v7`，见
  [../reference/protocol-v4.md](../reference/protocol-v4.md)。旧 heldout / v2
  流程仅作历史记录。
- **Repo:** `/home/frti64/pri_dile`
- **Current best candidate (Protocol v4 dev/test top 7):**
  `RL-Switch` / `rl-switch`（窄 meta；test 488.26/487.51；
  Python + C++；未 promotion）；单模型替代 `GRU-DAgger` /
  `rl-dagger-gru`（dev 持平，complex +0.89，test provisional
  487.89/487.02；Python + C++；未 promotion）
- **Prior best / fallbacks:** `RL-Wide160` / `rl-wide160`、
  `RL-Wide` / `rl-wide`、`RL-Pattern` / `rl-pattern`、
  `RL-Repair` / `rl-repair`（未 promotion）；`RL-PPO`（主权重，`rl`）、
  `RL-PPO-Hard60`（A60，`rl-hard60`）
- **E0 harness:** `python3 -m ipd league` / `docs/research/e0-league.md`
- **目标:** 在未知对手分布、一次固定提交的比赛中，最大化期望总分，同时控制 worst-case 风险。**blue-red / exploitability 不再是评估或 promotion 门槛（ADR 0011）。**

---

## 0. 当前基线

E0/E1c 冻结协议（official/blind 20 fields；数字与标准 CLI 独立 seed 略有差异）：

| 策略 | 官方四分布 | blind 20 fields | heldout v2 | heldout v3 final | 备注 |
|---|---:|---:|---:|---:|---|
| RL-PPO | 553.22 | 562.80 | 494.97 | 502.31 | 当前主权重 |
| RL-PPO-Hard60 (A60) | 554.27 | 563.87 | 500.07 | 505.97 | 已注册 `rl-hard60` |
| **RL-Repair (E1c)** | **554.72** | **564.68** | **503.17** | **509.88** | 当前最佳候选；未 promotion |
| Adv-RL | 554.05 | 562.97 | 502.39 | 504.84 | 官方高、尾部/red 弱 |
| SelfPlay-GRU | 550.97 | 561.13 | 491.86 | - | BC 蒸馏；改进 No-Go |
| E4 red best response | — | — | — | — | blue ~300；100% 击败当前蓝军 |

标准 CLI、独立 policy seed：RL-PPO 553.33 / 562.83、A60 554.36 / 564.17、
RL-Repair 554.76 / 564.61。**10-field blind 与 20-field 不可混表。**

现状判断：

- RL-Repair 在 official、blind、heldout v2/v3 上均优于 A60 和主权重；
  但 anti-exploitation 和 CVaR 没有质变，仍需独立复现后才能 promotion；
- 继续调 PPO 超参/课程比例的边际收益很低；
- 真正的下一步是 **对手建模、策略种群、鲁棒目标**；
- anti-exploitation 仍未解决：E4 red 100% 击败所有当前策略。

---

## 1. 设计原则

1. **优化 unknown opponent distribution 下的期望分，而不是单一官方分布均值。**
2. **对手构成是不确定集**：训练配比、对手参数、动态窗口都要随机化。
3. **策略种群 > 单点 best response**：red 必须是一组对手，而不是一个。
4. **验证必须 held-out**：family/参数拆分，不能只换 seed。
5. **worst-family / CVaR 是一等公民**，不能只看 mean。
6. **先建评估，再做算法**：没有 cross-play / exploitability，无法判断改进真假。
7. **部署延迟和可复现性从一开始就纳入验收。**

---

## 2. 四条主线

### Track A：评估与元博弈（P0，必须先做）

目标：建立能判断“是否真的更稳”的评估系统。

- Cross-play matrix：所有候选两两对战。
- PSRO / Double Oracle / Nash averaging：估计 meta-game 权重。
- Exploitability：用 held-out red best response 测被针对时的损失。
- 多套 held-out：
  - 官方四分布（20 fields × 299）
  - blind 20–30 fields
  - held-out family/参数（86 × 20 + 新 family）
  - hard2 / dynamic / phase / extort / random / noisy
  - E4 级 red best response + 新 red
- 指标：
  - mean、per-family、min_family、min_norm
  - CVaR 10% family score
  - top-family contribution、max-min gap
  - cross-play Elo / Nash weight
  - exploitability / red win rate / blue mean under red
  - latency、参数量、推理依赖

建议新增：

```text
ipd/evaluation/league.py       # cross-play / Elo
ipd/evaluation/metagame.py     # PSRO / Nash averaging
ipd/evaluation/opponent_suite.py
```

**Gate A:** 所有 baseline 在同一 suite/seeds 上可复现；cross-play 和 exploitability 报告稳定。

---

### Track B：鲁棒训练（P1，最快见效）

目标：在不改架构的前提下，提高 unknown distribution 的稳健性。

#### B1. Domain-randomized robust fine-tune

- 从 A60 初始化。
- 随机化：
  - Extort chi ∈ [1.5, 3.8]
  - Random p ∈ [0.01, 0.95]
  - LateD T ∈ [20, 95]
  - NoisyTFT p ∈ [0.01, 0.5]
  - Joss p ∈ [0.01, 0.5]
  - DynamicWindow W ∈ [3, 10], p_defect, forgiveness
  - Kamikaze period ∈ [8, 30]
- 训练目标：
  - PPO + KL to A60
  - 可选 CVaR / worst-family 辅助项
- 预期：held-out +3–8，官方不降超过 1–2。

#### B2. Robust PPO / CVaR / DRO

- 优化 `mean - β * CVaR_tail` 或 `min_family` 约束。
- 用一组对手 mixture（Dirichlet 采样）而不是固定配比。
- 参考：RARL、PAIRED、CVaR PG、DRO。

#### B3. Multi-teacher distillation + gating

- 教师：RL-PPO、A60、Adv-RL、Bayes/Final/Triple、SelfPlay-GRU。
- 数据：多样对手池；每个状态用“当前最优教师”打标签，或训练 gate 选择教师。
- 学生：85 维 MLP / GRU。
- 训练：BC + DAgger + PPO fine-tune。
- 预期：把不同教师的强项合到一个策略；风险是教师冲突和 gate 过拟合。

**Gate B:** 至少一个候选在 held-out mean +5 或 min_norm +0.02，且官方不降超过 2。

---

### Track C：对手建模与规划（P2，最高上限）

目标：显式建模对手，而不是只靠 model-free PPO。

#### C1. Bayesian opponent model + planning

- 维护对手模型库：TFT、Grim、GTFT、Extort、Random、LateD、Harvest、DynamicWindow、Kamikaze、Probe/Phase。
- 每个模型带连续参数；用粒子滤波/贝叶斯更新 belief。
- 决策：
  - 有限深度 lookahead（10–20 步）或 expectimax；
  - 末端用学习到的 value function 收尾；
  - 或对 belief 采样 rollout。
- 优点：IPD 100 步、动作二元、对手库结构化，显式规划很适合。
- 风险：模型 misspecification；需要 fallback。
- 参考：Bayesian opponent modeling、POMDP、ReBeL、CFR。

#### C2. Auxiliary opponent prediction

- 在主策略上加辅助头：
  - 预测对手下一动作；
  - 预测对手类型/响应性；
  - 预测对手是否处于 phase switch。
- 辅助损失 + PPO 主损失联合训练。
- 预期：改善表示学习，尤其是 dynamic/phase 对手。
- 参考：UNREAL、opponent modeling survey。

#### C3. Opponent classifier + Mixture-of-Experts

- 分类器从历史推断对手类型/参数；
- gate 路由到专家策略（合作/防御/剥削/抗噪）；
- 专家可以是规则、RL、Bayesian planner。
- 风险：分类错误；需要 confidence fallback 到 generalist。

**Gate C:** C1 或 C2/C3 在 held-out red / dynamic / phase 上提升 ≥10，官方不降超过 2。

---

### Track D：League / PSRO（P3，中期；**已暂停**）

> **blue-red 已停用（ADR 0011）**：以下 Track D / red oracle /
> exploitability 内容只作历史设计记录，不再安排新实验。

目标：把“单个 red”升级成“策略种群 + meta-game”。

#### D1. PFSP / league-lite

- 蓝军 checkpoint 池；
- red best-response 池（E4 级 + 新 red）；
- main agents / exploiters / league exploiters；
- prioritized fictitious self-play：采样“难但可学”的对手；
- 用 cross-play / Elo 监控 forgetting。

#### D2. PSRO / Double Oracle

- 维护策略池；
- 每轮训练对 meta-strategy 的 best response；
- 更新 meta-game / Nash；
- 用 held-out policy 做 exploitability 测试。

参考：AlphaStar league、PSRO、NFSP、Double Oracle、Nash averaging。

**Gate D:** exploitability 降低 ≥20%，官方不降超过 2；否则停止 league 方向。

---

### Track E：架构与部署（P4，按需）

- 85 维 GRU / Transformer PPO：测试时序记忆是否值得；
- 更大 MLP（128/64） vs 当前 64/32；
- 推理延迟、C++/纯 Python 导出、单文件打包；
- checkpoint/checksum/determinism；
- 只在 Track B/C 有信号后再投入。

---

## 3. 16 周路线图

| 周 | 主线 | 交付 |
|---|---|---|
| 1–2 | Track A | cross-play / PSRO / exploitability harness；baseline 复现 |
| 3–5 | Track B1/B2 | domain-randomized A60 fine-tune；robust PPO |
| 4–6 | Track B3 | multi-teacher distillation + gating |
| 6–9 | Track C1/C2 | Bayesian belief planner；auxiliary opponent model |
| 8–10 | Track C3 | classifier + MoE |
| 10–13 | Track D | PFSP/league-lite + PSRO |
| 12–14 | Track E | 85 维 GRU/Transformer 对比 |
| 14–16 | 集成 | 最终候选、held-out test、部署、checksum |

可并行：Track B 和 Track C 使用同一评估 harness；Track D 在 B/C 有候选后启动。

---

## 4. 实验矩阵

| ID | 实验 | 假设 | 成功指标 | 优先级 |
|---|---|---|---|---|
| E0 | Cross-play + PSRO harness | 没有评估就没有改进 | 可复现、稳定 | P0 |
| E1 | Domain-randomized A60 | 参数随机化提升 unknown 分布 | held-out +5，官方不降 >2 | P1 |
| E2 | Robust PPO / CVaR | 最差 family 提升 | CVaR +10%，mean 不降 | P1 |
| E3 | Multi-teacher distillation | 合并强项 | held-out +5，无 family 崩 | P1 |
| E4 | Auxiliary opponent prediction | 表示更好 | dynamic/phase +10 | P2 |
| E5 | Bayesian belief planner | 显式建模优于 model-free | E4 red blue +30 | P2 |
| E6 | Classifier + MoE | 专家路由有效 | held-out +8 | P2 |
| E7 | PFSP / league-lite | 种群减少 cycling | exploitability -20% | P3 |
| E8 | PSRO / meta-Nash | 元博弈更稳 | cross-play Elo + | P3 |
| E9 | 85-dim GRU/Transformer | 时序记忆有用 | +3–5，延迟可接受 | P4 |
| E10 | Offline RL / Decision Transformer | 日志数据可用 | 研究项 | P4 |
| E11 | ES / CMA-ES | 无梯度优化 worst-case | 研究项 | P4 |

---

## 5. 评估与 Go/No-Go

### 主指标

- 官方四分布（20 fields × 299）
- blind 20–30 fields
- held-out final（86 × 20 + 新 family/参数）
- E4 级 / 新 red best response
- CVaR / worst-family / min_norm
- cross-play Elo / Nash weight / exploitability

### 硬约束

```text
TFT >= 594
Bully >= 650
NoisyTFT.02 >= 560
AllD >= 95
官方四分布不低于当前基线 - 2
无 family 崩塌
```

### 选模规则

1. 先过滤硬约束；
2. 比较 held-out CVaR / min_norm；
3. 再比较官方 mean；
4. 最后看 blind / cross-play / latency；
5. final test 只跑一次，不参与选择。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 合成对手库偏差 | 多套 held-out family/参数；报告限制 |
| Meta-game 过拟合 | held-out policy；Nash averaging；不复用 test |
| Red cycling | PFSP/league；冻结快照；entropy/KL |
| Bayesian 模型 misspecification | 模型库 + fallback；不确定性阈值 |
| 规划延迟 | depth-limited；value function 收尾；CPU benchmark |
| 计算成本 | 分阶段 gate；小预算 screening |
| 过度工程 | 先做 E0/E1；无信号就停 |
| 部署限制 | 早期做 latency/C++ 原型 |

---

## 7. 论文与资料

### IPD / 博弈论
- Axelrod, *The Evolution of Cooperation* (1984)
- Press & Dyson, “Iterated Prisoner’s Dilemma contains strategies that dominate any evolutionary opponent”, PNAS 2012
- Mailath & Samuelson, *Repeated Games and Reputations*

### 对手建模 / 规划
- Albrecht & Stone, “A Survey of Opponent Modeling in Adversarial Domains”, arXiv:1612.09074
- Brown & Sandholm, “Superhuman AI for multiplayer poker” (Pluribus), Science 2019
- Brown et al., “Combining Deep Reinforcement Learning and Search for Imperfect-Information Games” (ReBeL), arXiv:2007.13544
- Zinkevich et al., “Regret Minimization in Games with Incomplete Information” (CFR), NeurIPS 2007

### League / 自博弈
- Lanctot et al., “A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning” (PSRO), arXiv:1711.00832
- Heinrich & Silver, “Deep Reinforcement Learning from Self-Play in Imperfect-Information Games” (NFSP), arXiv:1603.01121
- Vinyals et al., “Grandmaster level in StarCraft II using multi-agent reinforcement learning” (AlphaStar), Nature 2019
- Bansal et al., “Emergent Complexity via Multi-Agent Competition”, arXiv:1710.03748
- Jaderberg et al., “Human-level performance in 3D multiplayer games with population-based reinforcement learning”, Science 2019

### Robust RL / 域随机化
- Pinto et al., “Robust Adversarial Reinforcement Learning” (RARL), arXiv:1703.02702
- Dennis et al., “Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design” (PAIRED), arXiv:2012.02096
- Jiang et al., “Prioritized Level Replay” (PLR), arXiv:2010.03934
- Tobin et al., “Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World”, arXiv:1703.06907
- Tamar et al., “Policy Gradient for Coherent Risk Measures” (CVaR), NeurIPS 2015

### 蒸馏 / 辅助任务 / 离线
- Rusu et al., “Policy Distillation”, arXiv:1511.06295
- Ross et al., “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning” (DAgger), arXiv:1011.0686
- Chen et al., “Decision Transformer”, arXiv:2106.01345
- Jaderberg et al., “Reinforcement Learning with Unsupervised Auxiliary Tasks” (UNREAL), arXiv:1611.05397

### 评估 / 进化策略
- Balduzzi et al., “Re-evaluating Evaluation” (Nash averaging), arXiv:1806.02643
- Salimans et al., “Evolution Strategies as a Scalable Alternative to Reinforcement Learning”, arXiv:1703.03864

---

## 8. 交付物

- `ipd/evaluation/league.py`、`metagame.py`、`opponent_suite.py`
- `ipd/training/rl_robust.py`、`distill.py`、`league.py`、`opponent_model.py`
- `ipd/policies/belief_planner.py`、`moe_policy.py`
- `configs/robust/`、`configs/league/`
- 评估报告：cross-play、PSRO、exploitability、per-family
- 最终候选 checkpoint / 可部署权重 / checksum
- 论文/实验记录：每次 Gate 的结论

---

## 9. 不要做

- 不要继续只调 PPO 超参 / self-play 课程比例。
- 不要用单个 red 作为“anti-exploitation 已解决”的证据。
- 不要只用官方四分布选模。
- 不要把 final test 用于反复选模。
- 不要宣称 raw min_family 跨 family 可比。
- 不要在 E0 评估没建好前就开新算法。
- 不要为了复杂度而放弃当前 RL-Repair / A60 / RL-PPO fallback；所有新方法都要保留 fallback。

---

## 10. 建议的默认顺序

```text
E0 评估/PSRO
  -> E1 Domain-randomized A60
  -> E2/E3 Robust PPO / Multi-teacher
  -> E4/C1 Opponent modeling / Bayesian planner
  -> E5/C3 MoE
  -> E7/E8 League / PSRO
  -> E9 架构对比
  -> 最终集成
```

如果只能做一条主线：**E0 + E1 + C1**。  
E0 保证判断可信，E1 是最快见效，C1 是最高上限。

---

## 11. E0 执行状态（2026-10-26）

- 新增 `python3 -m ipd league`：cross-play / Nash averaging / Elo /
  exploitability / 冻结 suite / held-out red。
- 新 held-out v2（74 对手、7 个新 family）已用于 E0；后续作为 validation。
  另新增 held-out v3，用于 E1 一次性 final test。
- E0 Gate PASS；主权重未替换。结果：
  - A60 overall 552.11 / held-out v2 500.07 / CVaR10 391.13；
  - RL-PPO 550.80 / 494.97 / 387.26；
  - A60 min_norm 0.9928（blind wild -0.72%），E4 red blue 299.20；
  - Adv-RL held-out v2 502.39，但 red/tail 更弱；
  - SelfPlay-GRU/BC 仍为 No-Go（min_norm 0.949）。
- 重要负结果：cross-play 蓝军池退化为互相合作（594–595），Elo/Nash
  不可区分；P3 league 必须加入 red/exploiter checkpoint 池。
- 详细报告：`results/e0_league_20261026/README.md`；
  harness 说明：`docs/research/e0-league.md`。
- 当时的下一阶段是 E1 domain-randomized A60（`robust` 训练分布已实现并有测试），
  选模用官方 validation + heldout_v2；heldout_v3 final test 只跑一次。
  E1a/E1b 已判 No-Go，E1c 为 Go candidate，见 §12、§13。

## 12. E1 执行状态（E1a/E1b No-Go；E1c Go candidate）

- `robust` 训练分布已实现（`ipd/evaluation/robust.py`，4 个测试）。
- 从 A60 fine-tune 40 iter：官方 validation 558.87→559.07。
- E0 风格评估：official 554.71、blind 564.63、heldout v2 501.43
  （A60 500.07，+1.36）、min_norm 0.9964（0.9928）、CVaR10 388.98
  （391.13）、E4 red blue 311.70（299.20）。
- 未达到预注册 Gate（held-out +3 或 min_norm +0.02），判 **No-Go**。
  per-family：extort +16.3、noisy +3.8，但 random -6.1、adaptive -1.7。
- E1b（KL 0.05, lr 2e-4）selection run heldout +3.05，但 full run +1.94，
  min_norm 0.9892、CVaR 388.81、random -6.3；两次 validation 不一致，
  仍判 **No-Go**。
- 不注册、不 promotion；权重保留为研究 artifact。
- **heldout_v3 未运行**（E1a/E1b 未通过 validation；后来 E1c 才一次性使用）。
- 下一步：E1c 调整 robust mixture（降低 random 权重/收窄 random 区间/
  加 cooperative anchor）或 E2 CVaR/worst-family 辅助目标；仍用
  heldout_v2 做 validation，连续两次过 Gate 才跑 heldout_v3。

### E1c repair（Go candidate）

- 从 E1b checkpoint 初始化。
- 新 `repair` 分布：Random 5.0（p∈[0.08,0.92]）、NoisyTFT 3.0、
  Joss 2.0、Dynamic 2.0、LateD 2.0、Extort 1.5、Kamikaze 1.0、
  Harvest 1.0；anchor TFT 3.0、GTFT 2.0、Pavlov 2.0、AllC 2.0、
  Grim/AllD/Bully 1.0。
- 20 iter、lr 1e-4、KL 0.05；heldout_v2 选 iter 20。
- E1c full：official 554.72、blind 564.68、validation 509.93、
  heldout_v2 503.17（+3.10 vs A60）、min_norm 0.9930、CVaR10 390.39、
  red blue 303.45。
- **heldout_v3 final：RL-Repair 509.88 vs A60 505.97（+3.91）**；
  extort +23.4、random +4.7，min_norm/CVaR 基本持平。
- 判定 **Go candidate**；注册 `rl-repair`，不覆盖主权重。
  `heldout_v3` 已使用，后续候选需 heldout_v4。
- 下一步：E2 CVaR/worst-family，或 P3 red/exploiter league。

## 13. E2 执行状态（2026-10-26，No-Go）

- 新增 `ipd/training/risk.py`：CVaR episode weights（worst quantile 加权，
  mean=1）和 Dirichlet mixture；`ipd/training/rl.py` 新增
  `--risk-objective/--risk-coef/--risk-quantile/--dirichlet-concentration`。
- E2a（cvar 1.0/q 0.2）、E2b（cvar 3.0/q 0.1）、E2c（DRO eta 0.5）
  从 RL-Repair 训练 20 iter。
- heldout_v2 最好 mean 505.30/505.78/505.33（RL-Repair 504.40）；
  CVaR item +1.2%/+0.5%/-0.5%，min_norm +0.0001/+0.0027/+0.0025；
  official validation 未超过 RL-Repair。**No-Go**。
- 不注册 `rl-risk`/`rl-dro`；保留 RL-Repair；`heldout_v4` 未运行。
- 诊断：episode-level CVaR 和训练分布 DRO 都没有直接约束 held-out
  family 尾部。下一步应直接用 family-level CVaR/exploitability 目标，
  或转 P3 red/exploiter league。

## 14. P3 red/exploiter pool（2026-10-26）

- 新增 `ipd/evaluation/policy_pool.py` 和 RL 训练 flags
  `--opponent-weights/--opponent-fraction/--opponent-penalty`。
- 85 维 `Red-RL-v1`（from scratch, safe=False）对 RL-PPO/A60/RL-Repair
  frozen pool 训练 20 iter；训练 reward 146.9→715.5。
- Red-RL-v1 在全部 held-out blue 上强于 E4 red（RL-Repair 727.6 vs
  703.8；Adv-RL 666.6 vs 611.8；red win 100%）→ red pool 第一里程碑 Go。
- League-lite blue response：50% official/repair + 50% red pool，
  reward=own-opponent；best iter 10。E4 exploitability 407.0→355.0
  （-12.8%），official 554.74、blind 563.80；未达 -20% Gate → **No-Go**。
- 不 promotion；E4 red 仍 100% 击败。`heldout_v4` 仍未使用。
- 下一步：扩大 red 池（Adv/SelfPlay exploiters、多 checkpoint）+ PSRO/
  meta-Nash 权重，而不是单 red response。

## 15. P3 expansion — multi-red + PSRO-lite（2026-10-26，No-Go）

- 新增 weighted `--opponent-pool`、CandidateSpec-aware policy pool、
  `python -m ipd meta`（maximin 权重 + Gate D）。
- 训练 Red-v2–v4（入 blue 训练池）和严格 held-out Red-v5/v6；
  red pool 扩展到 6 个 checkpoint。
- maximin 选 v4 0.70、v1/v2/v3 各 0.10（learned-blue columns）。
- Blue-Meta-v2（50% red，iter 5）：official 554.999、blind 564.345、
  heldout_v2 502.938、min_norm 0.9679。
- held-out red exploitability 390.33→367.33（**-5.9%**，Gate -20%）；
  E4 -10.5%、Red-v5 -1.7%、Red-v6 -4.7%；official +0.24 通过，
  但 `validation:extort` -3.21%，family Gate 失败。**Gate D FAIL**。
- 结论：多 red + maximin 没有超过单 red 的 E4 降幅；更激进的 60% red
  响应以 extort family 崩塌为代价。不注册 `rl-meta`，RL-Repair 保留。
  下一步 P2 C1 Bayesian belief planner 离线评估；不要继续调 E2。
- 报告：`results/p3_league_20261026/README.md`；
  计划/结果：`docs/research/p3-expansion.md`。

## 16. P2 C1 Bayesian belief planner（2026-10-26，offline No-Go）

- 新增 `ipd/policies/belief_planner.py`：38 类 opponent particle filter、
  Bayesian belief update、finite-horizon planning，RL-Repair 作为默认
  base/fallback；新增 `ipd/evaluation/planner_eval.py` 和
  `python -m ipd planner-eval`。
- **fair-RNG 复跑**（base 与 planner 使用独立同 seed 实例；早期共享 RNG
  的 E4 +6.6 是伪影）：激进版（horizon 6, top-k 8, confidence 0.75）
  validation -12.11、heldout_v2 -16.15，extort 严重回退。
- 保守版（horizon 4, top-k 6, confidence 0.9, margin 10）：
  validation -2.19、heldout_v2 +2.06、noisy -3.0、temporal +8.7；
  held-out red E4 +0.35、v5 0.00、v6 -0.10，均低于 Gate C 的 +10。
- 诊断：模型库无法区分 noisy/extort 与 non-responsive exploiter，
  planning 本身不是瓶颈。判 **No-Go**，不注册、不改权重。
- 文档：`docs/research/p2-c1-belief-planner.md`；
  结果：`results/p2_belief_20261026/`。

## 17. P2 C2/C3 learned opponent model + gate（2026-10-26，offline No-Go）

- 新增 `ipd/training/opponent_model.py`：family MLP（9 类）、binary
  gate MLP（allow/deny）、next-action linear；训练数据仅 `TYPE_BANK`，
  `heldout_v2` 只做评估。
- 指标：family MLP val 42.6% / heldout_v2 32.1%；gate MLP val 86.6% /
  heldout_v2 70.6%（precision 70.2%，recall 38.7%）；next-action 81.9%。
- heldout_v2 gate 的 false positives 全部来自 adaptive/noisy；
  random/late precision 1.0，但 recall 只有 75%/24%。
- fair-RNG planner：ungated conservative validation -2.19 / heldout_v2
  +2.06；gated conservative validation 0.00 / heldout_v2 +1.20 /
  min_norm 0.999 / held-out red 无变化；gated aggressive
  heldout_v2 -21.91、extort -107.6。
- Gate C 未达到（+10 缺失），判 **No-Go**。下一步用
  exploratory/counterfactual 轨迹训练 opponent model，或三路
  `safe/risky/unknown` gate + MoE；unknown 回退 RL-Repair。
- 文档：`docs/research/p2-c2c3-opponent-model.md`；
  结果：`results/p2_opponent_model_20261026/`。

## 18. P2 C2/C3 exploratory / counterfactual data（2026-10-26，data Go / planner No-Go）

- 新增 open-loop exploratory schedules（single/block/random/alternating/
  late/cooperate）和 common-RNG counterfactual C/D 分支采集。
- 训练 `excited` 率 25.8%→59.3%；cooperative/forgiving 0%→56.9%/48.6%，
  adaptive 23.7%→68.8%，noisy 33.9%→64.4%。536 个 counterfactual
  pair，43.7% responsive；cooperative/noisy 80.8%/89.6%，late/random
  0%/10%。
- gate：base+explore heldout_v2 72.0/71.1/43.7（旧 70.6/70.2/38.7）；
  mixed 73.1/69.0/52.7；counterfactual 77.1/73.7/61.5；responsiveness
  head 71.8%。
- Planner 仍 No-Go：mixed CF gate conservative 3-seed validation -0.44、
  heldout_v2 +0.88、temporal +0.45；1-seed +3.04/+11.0 不复现；
  held-out red 0.00；gated aggressive heldout_v2 -19.33、extort -107.6。
- 诊断：training excitation 已解决，deployment excitation 没有；
  RL-Repair 仍很少 early defect，adaptive/noisy false positives 仍在。
  下一步三路 `safe/risky/unknown` gate + MoE，或 early-game active
  probe；不要继续加 planner horizon。
- 文档：`docs/research/p2-c2c3-excitation.md`；
  结果：`results/p2_excitation_20261026/`。

## 19. P2 C3 three-way gate + MoE（2026-10-26，offline No-Go）

- 新增 `ThreeWayGate`（safe/risky/unknown）、`ThreeWayMoE`、
  `python -m ipd moe-eval`；safe→AllD 或 conservative planner，
  risky/unknown→RL-Repair。
- 三路 gate validation 94.3%；heldout_v2 base safe precision/recall
  65.6%/20.7%，explore 82.1%/36.9%，counterfactual 82.6%/31.7%。
- safe→AllD 1-seed heldout_v2 +3.64、3-seed +0.51、5-seed +4.04
  （validation +0.53）；best-validation 配置 +0.90/+1.41。
  safe→planner 3-seed heldout_v2 -1.28。Held-out red 全部 0.00。
- 诊断：三路 gate 在兴奋态 precision 够，但 safe recall 太低；
  MoE 收益集中在 random/noisy 且 seed 方差大；AllD 对 false positive
  极敏感。Gate C +10 未达到，判 **No-Go**。
- 无条件 early probe 成本 -10.7~-20.8（reciprocal/punitive
  单 family -35~-74），暂不实现；后续若做 probe，需要 Bayesian
  expected-value / information-gain 版本。
- 文档：`docs/research/p2-c3-threeway-moe.md`；
  结果：`results/p2_threeway_moe_20261026/`。

## 20. P2 C3 repeated counterfactual 校准（2026-10-26，offline No-Go）

- `collect_counterfactual_dataset` 新增 `repeats`：同一 branch state
  重复 10 次独立 C/D（common RNG within pair），得到 `p(response)`；
  新增 calibrated safe/risky/unknown gate。
- heldout counterfactual safe precision/recall **74.7%/62.1%**
  （旧 family+excitation 82.6%/31.7%）；calibration 提高 recall、
  降低 precision。
- no-veto AllD MoE 在 extort 崩坏：1-seed t0.6/day5 heldout -25.87、
  extort -137.12；day10 -9.46、extort -51.88。
- 加 C2 response veto 0.2：5-seed validation -0.31、heldout_v2 +2.93、
  min_norm 0.9988、safe rate 2.5%；held-out red E4 +0.35、v5/v6 0.00。
- One-shot Bayesian/EV probe screen：best 1-seed day30 +1.29、3-seed
  -5.22（random -33.1）；无条件 early probe -10.7~-20.8。
- 诊断：calibration 解决 recall 但不解决 precision；extort 的少量
  false positive 足以让 AllD 崩坏。继续 C3 需要显式
  value-of-information probe 或更好的后验/特征，而不是继续调
  threshold。判 **No-Go**，不注册、不改主权重。
- 文档：`docs/research/p2-c3-repeated-calib.md`；
  结果：`results/p2_repeated_calib_20261026/`。
