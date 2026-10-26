# P2 C3 — three-way gate + MoE（2026-10-26）

## 0. 结论

**No-Go：三路 gate 是安全的，但 safe recall 太低，MoE 收益不足且不稳。**

- 三路 gate（safe/risky/unknown）在 heldout_v2 base 上 accuracy 84.5%，
  safe precision **65.6%**、safe recall **20.7%**；exploratory 轨迹上
  precision/recall 为 **82.1%/36.9%**，counterfactual 轨迹上
  **82.6%/31.7%**。
- `safe → AllD`、`risky/unknown → RL-Repair` 的 MoE：
  - 1 seed：validation +1.35、heldout_v2 **+3.64**；
  - 3 seeds：validation +2.56、heldout_v2 **+0.51**；
  - 5 seeds：validation +0.53、heldout_v2 **+4.04**；
  - best-validation 配置（t0.65/min-day10，5 seeds）：validation +0.90、
    heldout_v2 +1.41。
  - 结果随 seed 波动大，远低于 Gate C 的 +10。
- `safe → planner` 的 MoE 更差：3 seeds validation +1.18、heldout_v2
  **-1.28**；aggressive planner 仍然不可用。
- Held-out red E4/v5/v6 全部 **0.00**；safe rate 0，没有 red 误覆盖。
- 无条件 early probe 的成本 screen：单次 D 在 heldout_v2 上整体
  **-10.7 ~ -20.8**， reciprocal/punitive 单 family -35 ~ -74；在没有高精度
  safe gate 前，active probe 不作为候选。
- 不注册候选、不 promotion、不覆盖主权重；`heldout_v4` 仍未使用。

## 1. 设计

### 1.1 三路 gate

`three_way_gate_label()` 用 family + excitation 生成标签：

- `safe`：family ∈ {defect, late, random}，且 `n_d >= 3 and n_c >= 5`；
- `risky`：其他 family，且 excitation 足够；
- `unknown`：excitation 不足或低置信度。

`ThreeWayGate` 在推理时：

1. 低于 `--min-day` 或低置信度 → `unknown`；
2. 可选 C2 response model 预测“对手会对我们动作反应” → 把 `safe`
   降级为 `risky`；
3. 输出 `safe / risky / unknown`。

`ThreeWayMoE` 的路由：

- `safe` → `DefectExpert`（AllD）或 conservative planner；
- `risky` → RL-Repair fallback；
- `unknown` → RL-Repair fallback。

这是“先验证 safe 才允许剥削，其他状态回退”的保守 MoE。

### 1.2 数据与模型

沿用 `results/p2_excitation_20261026/` 的混合数据：

- base RL-Repair 2 episodes；
- exploratory 2 episodes；
- counterfactual 1 episode，8 branches × 6 rounds。

三路 gate：train accuracy 95.9%、validation 94.3%。

## 2. 三路 gate 指标

| Suite 轨迹 | 3-way accuracy | safe precision | safe recall |
|---|---:|---:|---:|
| heldout_v2 base | 84.5% | 65.6% | 20.7% |
| heldout_v2 explore | 88.4% | **82.1%** | **36.9%** |
| heldout_v2 counterfactual | 86.6% | **82.6%** | **31.7%** |

对比 binary gate（same mixed data）：heldout_v2 base 73.1/69.0/52.7；
explore 73.5/66.2/61.3；counterfactual 77.1/73.7/61.5。三路 gate 在
兴奋态上 precision 更高，但 **safe recall 明显更低**：它把大量
可剥削状态判成 unknown。

## 3. MoE screen（fair RNG）

`moe-eval` 使用独立同 seed 的 baseline / MoE base；MoE 只在 `safe`
路由调用 AllD 或 planner。

| 配置 | seeds | validation Δ | heldout_v2 Δ | min_norm | safe rate |
|---|---:|---:|---:|---:|---:|
| AllD, t0.6, day5 | 1 | +1.35 | **+3.64** | 0.996 | 13.4% |
| AllD, t0.6, day5 | 3 | +2.56 | +0.51 | 0.992 | 12.9% |
| AllD, t0.6, day10 | 3 | +2.56 | +0.62 | 0.998 | 13.4% |
| AllD, t0.6, day10 | **5** | +0.53 | **+4.04** | 0.997 | 13.2% |
| AllD, t0.65, day10 | **5** | **+0.90** | +1.41 | 0.998 | 11.8% |
| planner, t0.6, day5 | 3 | +1.18 | **-1.28** | 0.987 | 10.6% |
| ungated conservative | 3 | -0.51 | +0.95 | 0.993 | - |

5-seed AllD t0.6/day10 的 heldout family：random **+14.93**、
noisy **+9.73**、temporal +3.55、adaptive +1.30、extort -1.23；
validation random +3.00、noisy -0.14、extort +1.31。收益集中在
random/noisy，但 seed 间波动大；1-seed 的 +3.64 和 5-seed 的 +4.04
没有稳定复现到 validation 同方向的大幅提升。

`safe → planner` 反而把 noisy 抬起来、random 打下去（3 seeds
heldout -1.28），而且 extort/random 有回退，说明专家选择不是瓶颈，
safe 状态的覆盖率/精度才是。

## 4. Held-out red

`results/p2_threeway_moe_20261026/moe_alld_red.json`：

| Red | base blue | MoE blue | Δ | safe rate |
|---|---:|---:|---:|---:|
| E4-red-BC | 296.60 | 296.60 | 0.00 | 0.0% |
| Red-RL-v5-heldout | 375.10 | 375.10 | 0.00 | 0.0% |
| Red-RL-v6-heldout | 313.35 | 313.35 | 0.00 | 0.0% |

三路 gate 对 red 全程 risky/unknown，没有触发 AllD；安全，但也没有
anti-exploitation 收益。

## 5. Active probe control

`probe_cost.py` 在 heldout_v2 上强制一次 D，再回 RL-Repair：

| Probe day | overall Δ | reciprocal Δ | punitive Δ |
|---:|---:|---:|---:|
| 1 | -16.55 | -74.2 | -61.1 |
| 5 | -19.78 | -74.2 | -59.0 |
| 10 | -17.69 | -69.8 | -56.1 |
| 20 | -15.96 | -65.1 | -48.1 |
| 30 | -11.51 | -57.5 | -41.4 |
| 40 | -10.66 | -49.4 | -35.1 |

无条件 early probe 的期望成本明显为负；在没有高精度 safe gate /
Bayesian probe value 之前不实现为策略。

## 6. 诊断与下一步

1. **三路 gate 的 precision 在兴奋态上够用，但 recall 太低**：
   base 轨迹 safe recall 20.7%，大部分可剥削状态被标成 unknown 回退。
2. **MoE 收益集中在 random/noisy，seed 方差大**；Gate C 的 +10 没有
   达到，不能作为候选。
3. **AllD expert 对 false positive 很敏感**：extort 的 1-seed 回退提醒
   不能放宽 safe threshold；继续调 threshold 只是在 variance 里选点。
4. **下一步**：
   - 用 repeated counterfactual（同一状态多次 C/D 分支）降低 response
     标签噪声，训练 safe/risky/unknown 的概率校准模型；
   - 三路 gate 与 belief planner 的 posterior 联合，而不是只看 85 维
     历史特征；
   - 如果做 active probe，先实现 Bayesian expected-value probe（只在
     probe 期望收益为正时执行），不要无条件 early D；
   - 不改主权重，`heldout_v4` 保留。

## 7. 复现

```bash
bash results/p2_threeway_moe_20261026/run.sh
```

## 8. 产物

```text
ipd/training/opponent_model.py       # ThreeWayGate + 3-way labels
ipd/policies/moe.py                  # ThreeWayMoE / DefectExpert
ipd/evaluation/moe_eval.py           # moe-eval harness
tests/test_moe.py
weights/opponent_gate_threeway.json
results/p2_threeway_moe_20261026/
  README.md / run.sh / checksums.txt
  opponent_model_report.json
  moe_alld_t06_d5.json
  moe_alld_t06_d5_seeds3.json
  moe_alld_t06_d10_seeds3.json
  moe_alld_t06_d10_seeds5.json
  moe_alld_t065_d10_seeds5.json
  moe_planner_t06_d5.json
  moe_planner_t06_d5_seeds3.json
  moe_alld_red.json
  probe_cost.json / probe_cost.py / red_screen.py
```

主权重未变：

```text
bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4  weights/rl_policy_weights.json
a759a7b6e7eb5773aaf0884f9d6c6ca22587f82e1f4fc4c0cc16e983e3c1bd2b  weights/rl_policy_weights_repair.json
```
