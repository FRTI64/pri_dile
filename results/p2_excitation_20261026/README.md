# P2 C2/C3 — exploratory / counterfactual 数据（2026-10-26）

## 0. 结论

**数据层 Go：训练数据 excitation 已解决；planner 仍 No-Go。**

- 旧训练数据只有 RL-Repair 轨迹：cooperative / forgiving 的
  `excited` 样本率为 **0%**，adaptive 23.7%、noisy 33.9%。gate 在
  heldout_v2 的 false positives 全部来自 adaptive/noisy。
- 新增 open-loop exploratory 轨迹后，总体 `excited` 率从 **25.8%**
  升到 **59.3%**，adaptive/cooperative/forgiving/noisy 分别升到
  **68.8% / 56.9% / 48.6% / 64.4%**；counterfactual 分支进一步给出
  536 个 C/D 配对，其中 234 个（43.7%）对手行为“对蓝方动作敏感”。
- gate 本身变好：base+explore 在 heldout_v2 为 **72.0% / precision
  71.1% / recall 43.7%**（旧 70.6% / 70.2% / 38.7%）；mixed 为
  **73.1% / 69.0% / 52.7%**。counterfactual 轨迹上 gate 为
  **77.1% / 73.7% / 61.5%**，responsiveness head 在 heldout
  counterfactual 上 **71.8%**。
- 但部署轨迹仍然 excitation 不足：RL-Repair 很少主动 early defect，
  adaptive/noisy 的 false positive 仍分别为 21.6%/16.8%（base+explore）
  和 25.8%/27.8%（mixed）。
- fair-RNG conservative planner：mixed CF gate t0.6 单 seed 为
  validation **-1.04**、heldout_v2 **+3.04**（temporal +11.00）；
  **3 seeds 后只剩 validation -0.44、heldout_v2 +0.88、temporal
  +0.45**，单 seed 的 temporal 增益不可复现。gated aggressive 仍
  崩坏（heldout_v2 -19.33，extort -107.62）。held-out red E4/v5/v6
  全部 **0.00**。Gate C 未通过。
- **不注册候选、不 promotion、不覆盖主权重。** `heldout_v4` 仍未使用。

## 1. 数据设计

### 1.1 Base 轨迹

沿用 `collect_dataset`：RL-Repair 采样、67 个 TYPE_BANK 类型、
每类型 2 episodes。它代表部署分布，但 cooperative/forgiving 类型
几乎没有蓝方 D，conditional response 特征无法估计。

### 1.2 Exploratory 轨迹

`exploration_actions()` 生成**开环**蓝方动作表（不依赖对手），
每个 episode 从以下模板循环选择：

| 模板 | 行为 | 目的 |
|---|---|---|
| `single_probes` | 间隔 3–8 天的单步 D | 测即时报复/原谅 |
| `block_probes` | 2–4 天连续 D，再做 2–6 天 C | 测连续背叛阈值 |
| `random` | p(D) ∈ [0.08, 0.35] | 覆盖混合历史 |
| `alternating` | 10–30 天的 C/D 交替 | 测周期响应 |
| `late_block` | 55%–85% 时间后连续 D | 测 endgame / late 响应 |
| `cooperate` | 全 C | 保留低 excitation 基线 |

每类型 2 episodes；动作表与对手 RNG 分离，避免探索策略被对手反向
影响。`excited` 定义与特征 35 一致：历史中 `n_d >= 3` 且
`n_c >= 5`。

### 1.3 Counterfactual 分支

`collect_counterfactual_dataset()` 在每局分层选择 8 个 branch day
（覆盖 2–97 天），对同一对手状态做 `deepcopy`，蓝方分别执行 C 和 D，
随后双方继续 C 6 轮。两个分支共享对手 RNG 状态（common random
numbers），因此 reply 序列差异来自蓝方动作，而不是随机抽样差异。
prefix 交替使用 exploratory 和 RL-Repair，兼顾兴奋态与部署态。

分支同时产出：

- post-branch 样本（用于 family / gate 训练）；
- pre-branch 样本 + `responsive` / `unresponsive` 标签（C2
  responsiveness head）。标签判定：两个分支 6 轮内对手 D 的总数不同
  即为 responsive。

## 2. Excitation 指标

训练集（TYPE_BANK，2 episodes）：

| Class | Base excited | Explore excited | Base mean n_d | Explore mean n_d |
|---|---:|---:|---:|---:|
| adaptive | 23.7% | **68.8%** | 4.45 | 9.77 |
| cooperative | **0.0%** | **56.9%** | 0.08 | 7.55 |
| defect | 56.2% | 57.9% | 37.45 | 8.18 |
| forgiving | **0.0%** | **48.6%** | 0.00 | 6.69 |
| late | 20.6% | **59.4%** | 3.79 | 7.01 |
| noisy | 33.9% | **64.4%** | 1.84 | 8.34 |
| punitive | 11.1% | **63.7%** | 4.58 | 8.10 |
| random | 61.6% | 55.9% | 39.85 | 7.86 |
| wild | 77.7% | 63.7% | 10.32 | 9.14 |
| **overall** | **25.8%** | **59.3%** | 11.20 | 7.98 |

Base 的总体 `n_d` 不低，是因为 defect/random 长段背叛；关键指标是
conditional response 覆盖率和 cooperative/forgiving/adaptive/noisy
的 `excited` 率。

Counterfactual 分支（536 pairs）：

| Class | Pairs | Responsive rate |
|---|---:|---:|
| adaptive | 72 | 38.9% |
| cooperative | 104 | 80.8% |
| defect | 48 | 16.7% |
| forgiving | 64 | 46.9% |
| late | 48 | **0.0%** |
| noisy | 48 | 89.6% |
| punitive | 56 | 48.2% |
| random | 80 | 10.0% |
| wild | 16 | 37.5% |
| **overall** | **536** | **43.7%** |

这与 gate 的 safe/risky 语义一致：late/random/defect 大部分
unresponsive；cooperative/noisy/punitive 大部分 responsive；adaptive
混合（Extort/Probe 等）。

## 3. 模型指标

训练数据与 gate 指标（heldout_v2 为 base 轨迹；`:explore` 为
exploratory 轨迹；`:counterfactual` 为分支样本）：

| 训练数据 | 样本 | Gate val | heldout_v2 acc/prec/rec | :explore | :counterfactual | Response acc |
|---|---:|---:|---|---|---|---:|
| 旧 base | 13,400 | 86.6% | 70.6 / 70.2 / 38.7 | - | - | - |
| base+explore | 26,800 | 85.6% | **72.0 / 71.1 / 43.7** | 74.7 / 69.5 / 59.2 | - | - |
| mixed（+CF） | 39,880 | 85.2% | **73.1 / 69.0 / 52.7** | 73.5 / 66.2 / 61.3 | **77.1 / 73.7 / 61.5** | **71.8%** |

Mixed 的 gate 在 heldout_v2 上：

| Family | n | predicted allow | precision | recall |
|---|---:|---:|---:|---:|
| adaptive | 1,600 | 25.8% | 0.0% | 0.0% |
| cooperative | 1,300 | 0.0% | - | - |
| late | 2,000 | 37.1% | 100% | 37.1% |
| noisy | 900 | 27.8% | 0.0% | 0.0% |
| punitive | 800 | 0.0% | - | - |
| random | 800 | 91.6% | 100% | 91.6% |

Excitation 让 late/random recall 明显上升，但 **adaptive/noisy 的
false positives 仍然存在**；它们在低 excitation 历史上仍会被误判为
可剥削。Family MLP 在 mixed 数据上反而降到 20.5%（分布变化），说明
不能只看 family accuracy；gate 才是 C2/C3 的主头。

Counterfactual responsiveness head（mixed）：

| Truth | n | responsive rate | predicted responsive |
|---|---:|---:|---:|
| adaptive | 128 | 53.9% | 34.4% |
| cooperative | 104 | 76.9% | 40.4% |
| late | 160 | 16.3% | 12.5% |
| noisy | 72 | 86.1% | 55.6% |
| punitive | 64 | 29.7% | 3.1% |
| random | 64 | 0.0% | 10.9% |

它学到了方向，但对 adaptive/noisy 仍不够可靠；单独拿它当 gate
太保守（late/random recall 低），作为 family gate 的 veto 也没有
改变 planner 结果。

## 4. Planner screen（fair RNG）

Conservative：horizon 4 / top-k 6 / fallback 0.9 / margin 10。
1 seed 与 3 seeds 均使用独立同 seed 的 base/planner 实例。

| Planner | validation Δ | heldout_v2 Δ | min_norm | override |
|---|---:|---:|---:|---:|
| 旧 gated t0.7（1 seed） | 0.00 | +1.20 | 0.999 | 0.10% |
| base+explore gate t0.7（1 seed） | -0.34 | +1.15 | 0.998 | 0.11% |
| mixed CF gate t0.6（1 seed） | -1.04 | **+3.04** | 1.000 | 0.20% |
| mixed CF gate t0.6（**3 seeds**） | **-0.44** | **+0.88** | 0.994 | 0.12% |
| ungated conservative（3 seeds） | -0.51 | +0.95 | 0.993 | 0.22% |

Mixed CF gate t0.6 单 seed 的 heldout 家族：adaptive +8.75、
temporal **+11.00**、noisy +1.56、random/late/extort 0。3 seeds 后
temporal 只剩 **+0.45**、adaptive +8.62、noisy -2.59，说明单 seed
的 +11.0 是噪声。

Aggressive（horizon 6 / top-k 8 / fallback 0.75）仍然不安全：
mixed CF gate heldout_v2 **-19.33**、extort **-107.62**、random
-24.00、temporal -12.25。data 改善没有解决 C1 planner 的过度自信。

Held-out red（20 seeds，fair RNG）：E4-red-BC、Red-RL-v5、Red-RL-v6
的 blue delta 全部 **0.00**；override rate 0.0%，gate fallback
约 1,900/2,000 rounds。

### Gate C

- held-out red：0.00 — fail；
- dynamic/phase proxy（temporal）：3 seeds +0.45 — fail；
- heldout_v2 mean：3 seeds +0.88 — 远低于 +10；
- validation：-0.44（在 -2 容差内），min_norm 0.994，无 family 崩塌。

**No-Go。** 训练 excitation 已解决，但部署时 RL-Repair 仍然很少
制造 response event；静态 exploratory 数据不能凭空给 gate 提供
部署态证据。

## 5. 诊断与下一步

1. **Training excitation 已解决，deployment excitation 没有。**
   gate 需要的是“当前状态对手是否会对我的 D 反应”，不是更多
   offline 类型标签。下一步应做 **active probing**：在 early game
   用一次低成本 D 探测，再根据响应决定是否进入 exploit 模式；探测
   失败时回退 RL-Repair。
2. **三路 gate / MoE。** `safe_to_exploit / risky / unknown`；
   只有高置信 safe 才覆盖，unknown 回退。用 counterfactual 数据
   给 safe/risky 标签，用低 excitation 或模型分歧标 unknown。
3. **C2 head 联合训练。** 把 next-action / responsiveness 与 gate
   联合训练，评估 `our D → their next move` response event，而不是
   只看 marginal action accuracy；当前 response head 71.8% 还不足以
   单独 veto。
4. **不要继续加 planner horizon 或调 risk coef。** Aggressive 的
   extort/random/temporal 崩坏说明 C1 planner 仍是瓶颈；先解决
   gate/probe，再重开 planner。
5. 不改主权重；`heldout_v4` 保留。等 active-probe / 三路 gate 通过
   validation 后再做一次 final。

## 6. 复现

```bash
# 全套（含 1-seed / 3-seed planner 和 red screen）
bash results/p2_excitation_20261026/run.sh

# 只跑数据 + gate
.venv/bin/python -m ipd train opponent-model \
  --episodes 2 --explore-episodes 2 \
  --counterfactual-episodes 1 --counterfactual-branches 8 \
  --counterfactual-horizon 6 --counterfactual-prefix explore,base \
  --epochs 300 --seed 20261048 --data-mode mixed --eval-mode all \
  --out weights/opponent_classifier_cf.json \
  --gate-out weights/opponent_gate_cf.json \
  --next-out weights/opponent_next_action_cf.json \
  --response-out weights/opponent_responsiveness.json \
  --report-out results/p2_excitation_20261026/opponent_model_mixed_report.json
```

## 7. 产物

```text
ipd/training/opponent_model.py       # 数据生成 / gate / response head
ipd/evaluation/planner_eval.py       # --response-model / veto
tests/test_opponent_model.py         # 22 tests
weights/opponent_classifier_explore.json
weights/opponent_gate_explore.json
weights/opponent_next_action_explore.json
weights/opponent_classifier_cf.json
weights/opponent_gate_cf.json
weights/opponent_next_action_cf.json
weights/opponent_responsiveness.json
results/p2_excitation_20261026/
  README.md
  run.sh / red_screen.py
  opponent_model_explore_report.json
  opponent_model_mixed_report.json
  planner_explore_gated_t07.json
  planner_explore_gated_t07_seeds3.json
  planner_ungated_conservative_seeds3.json
  planner_cf_gated_t06.json
  planner_cf_gated_t06_seeds3.json
  planner_cf_gated_aggressive.json
  planner_cf_gated_red.json
  checksums.txt
```

主权重未变：

```text
bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4  weights/rl_policy_weights.json
a759a7b6e7eb5773aaf0884f9d6c6ca22587f82e1f4fc4c0cc16e983e3c1bd2b  weights/rl_policy_weights_repair.json
```
