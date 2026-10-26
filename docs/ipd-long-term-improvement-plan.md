# IPD 策略长期改进计划（无 3 天限制）

- **Status:** proposed / research + competition roadmap
- **Repo:** `/home/frti64/pri_dile`
- **Current best candidate:** `RL-PPO-Hard60` / `rl-hard60`（A60）
- **Baseline:** `RL-PPO`（`rl`）
- **目标:** 在未知对手分布、一次固定提交的比赛中，最大化期望总分，同时控制 worst-case / 被针对性剥削的风险。

---

## 0. 当前基线

| 策略 | 官方四分布 | blind 20 fields | held-out final | 备注 |
|---|---:|---:|---:|---|
| RL-PPO | 553.33 | 562.83 / 5.80 | 506.49 | 当前主基线 |
| RL-PPO-Hard60 (A60) | **554.36** | **564.17 / 5.75** | **509.76** | 已注册 `rl-hard60` |
| Adv-RL | 554.29 | 562.91 / 8.00（10 fields） | 504.08 | 官方高、held-out 弱 |
| SelfPlay-GRU | 550.76 | 560.43 / 8.10 | 505.06 | BC 蒸馏，No-Go |
| E4 red best response | — | — | blue ~306 | 100% 击败 BC/RL/Adv |

现状判断：

- A60 是“hard 池用得更充分”的低风险改进；
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

### Track D：League / PSRO（P3，中期）

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
- 不要为了复杂度而放弃当前 A60 基线；所有新方法都要保留 fallback。

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
