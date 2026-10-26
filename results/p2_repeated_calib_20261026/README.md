# P2 C3 — repeated counterfactual calibration（2026-10-26）

## 0. 结论

**No-Go：calibration 提高了 safe recall，但没有解决 extort 误报；MoE 仍达不到 Gate C。**

- 每个 counterfactual branch 重复 **10 次**独立 C/D 采样，得到
  `p(response)`；用它训练 safe/risky/unknown gate。
- calibrated gate 在 heldout_v2 counterfactual 上 accuracy **67.7%**，
  safe precision **74.7%**、safe recall **62.1%**（旧 family+excitation
  gate 为 82.6%/31.7%）。calibration 明显提高了 recall，但 precision
  下降。
- 直接 `safe → AllD` 会在 extort 上崩坏：1-seed t0.6/day5
  heldout_v2 **-25.87**、extort **-137.12**；day10 -9.46、extort -51.88。
- 加 C2 response veto（`p(response) <= 0.2`）后：
  - 1 seed：validation -0.42、heldout_v2 +3.56；
  - **5 seeds：validation -0.31、heldout_v2 +2.93**，
    min_norm **0.9988**，safe rate 仅 **2.5%**；
  - heldout family：noisy +7.07、random +9.50、temporal +3.05、
    extort -0.55。
- Held-out red：E4 +0.35（safe rate 1.05%）、v5/v6 0.00；安全但无
  anti-exploitation 收益。
- One-shot Bayesian/EV probe screen：最佳 1-seed day30 +1.29，
  3-seed **-5.22**（random -33.1）；无条件 early probe -10.7~-20.8。
- 不注册、不 promotion、不改主权重；`heldout_v4` 未使用。

## 1. 数据与 gate

### 1.1 Repeated counterfactual

`collect_counterfactual_dataset(..., repeats=10)` 对同一 branch state：

- 每个 repeat 对 `opponent` 做 `deepcopy`；
- C/D 两个分支共享同一 RNG seed（common random numbers），repeat 之间
  换 seed；
- 记录 `delta = sum(reply_D) - sum(reply_C)`；
- `p(response) = count(delta != 0) / repeats`。

正式训练使用 mixed 数据（base 2 episodes + exploratory 2 episodes +
counterfactual 1 episode，8 branches × 6 rounds × 10 repeats）。

### 1.2 Calibrated 3-way gate

标签：

- `safe`：`p(response) <= 0.0`
- `risky`：`p(response) >= 0.5`
- `unknown`：中间值

训练：train 80.2%、validation 64.6%（比 family+excitation gate 低，
但任务不同）。

## 2. Gate 指标

| 轨迹 / 标签 | accuracy | safe precision | safe recall |
|---|---:|---:|---:|
| heldout_v2 counterfactual, calibrated | **67.7%** | **74.7%** | **62.1%** |
| heldout_v2 counterfactual, family+excitation（旧） | 86.6% | 82.6% | 31.7% |

Calibration 把 safe recall 从 31.7% 提到 62.1%，代价是 precision
从 82.6% 降到 74.7%；extort 的 false positives 仍然足以让 AllD
崩坏。

## 3. MoE screen

`safe → AllD`，`risky/unknown → RL-Repair`：

| 配置 | seeds | validation Δ | heldout_v2 Δ | min_norm | safe rate |
|---|---:|---:|---:|---:|---:|
| no veto, t0.6/day5 | 1 | -10.62 | **-25.87** | 0.704 | 40.4% |
| no veto, t0.6/day10 | 1 | -8.56 | **-9.46** | 0.888 | 34.7% |
| veto 0.2, t0.6/day10 | 1 | -0.42 | +3.56 | 0.985 | 2.0% |
| veto 0.2, t0.6/day10 | **5** | **-0.31** | **+2.93** | **0.9988** | 2.5% |

No-veto 的 extort 回退：day5 -137.12、day10 -51.88。Veto 0.2 把 safe
状态压到 2.5%，换来了 extort -0.55 和 heldout +2.93；validation
仍为负，远低于 Gate C 的 +10。

## 4. Held-out red

| Red | base blue | MoE blue | Δ | safe rate |
|---|---:|---:|---:|---:|
| E4-red-BC | 296.60 | 296.95 | +0.35 | 1.05% |
| Red-RL-v5-heldout | 375.10 | 375.10 | 0.00 | 0.0% |
| Red-RL-v6-heldout | 313.35 | 313.35 | 0.00 | 0.0% |

## 5. Bayesian / EV probe screen

`bayes_probe_screen.py` 让 C1 belief planner 只在 selected day 做一次
override，其他时间 RL-Repair：

| Case | seeds | heldout_v2 Δ | notes |
|---|---:|---:|---|
| h6 day10 | 1 | -0.68 | random -5.4 |
| h6 day20 | 1 | +0.25 | random +1.4 |
| h6 day30 | 1 | +1.29 | noisy +5.6, extort +4.4 |
| h6 day30 | 3 | **-5.22** | random -33.1 |
| h4 day30 | 1 | -0.01 | — |

One-shot EV probe 没有稳定收益；无条件 probe 的成本仍是
-10.7~-20.8（见 `results/p2_threeway_moe_20261026/probe_cost.json`）。
真正的 Bayesian 方案需要显式 information-gain / value-of-information，
而不是只看一步 planner EV。

## 6. 诊断与下一步

1. **Calibration 解决了 recall，但没有解决 precision。** extort 在
   部分状态看起来 non-responsive，AllD 一旦误判就造成 -100 级回退。
2. **Response veto 0.2 是安全的，但只覆盖 2.5% 状态**；收益主要来自
   noisy/random，validation 仍未转正。
3. **One-shot Bayesian probe 也不够**：day30 1-seed +1.29，3-seed
   -5.22，说明只做一次 planner override 不是“信息价值”方案。
4. 如果继续 C3，下一步必须是显式 **value-of-information**：
   比较 probe 的 expected payoff + 未来可剥削收益，并维护 uncertainty；
   或者重新设计特征/后验，而不是继续调 threshold。
5. 不改主权重；`heldout_v4` 保留。

## 7. 复现

```bash
bash results/p2_repeated_calib_20261026/run.sh
PYTHONPATH=. .venv/bin/python \
  results/p2_repeated_calib_20261026/bayes_probe_screen.py
```

## 8. 产物

```text
weights/opponent_gate_calibrated.json
ipd/training/opponent_model.py       # repeats / calibrated labels
results/p2_repeated_calib_20261026/
  README.md / run.sh / checksums.txt
  opponent_model_report.json
  moe_alld_t06_d5.json
  moe_alld_t06_d10.json
  moe_alld_t06_d10_resp02.json
  moe_alld_t06_d10_resp02_seeds5.json
  moe_alld_red.json
  bayes_probe_screen.py / bayes_probe_screen.json
```

主权重未变：

```text
bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4  weights/rl_policy_weights.json
a759a7b6e7eb5773aaf0884f9d6c6ca22587f82e1f4fc4c0cc16e983e3c1bd2b  weights/rl_policy_weights_repair.json
```
