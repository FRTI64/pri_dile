# Plan 6：Python 评估瓶颈优化

- **状态：** 计划（2026-10-28）
- **依据：** Plan 1–5 的 profile；`test_v7` v5 GPU sample 与 dev GPU argmax
- **目标：** 在不改变已有协议语义的前提下，继续降低 CPU Python 占比
- **前置：** Plan 5 v5 sample；`docs/research/plan5-gpu-sample.md`

## 1. 当前 profile

`test_v7` v5 GPU sample（wall 1365.63s，20,002 worker-seconds）：

| 部分 | 占比 | 说明 |
|---|---:|---|
| policy 对手 | **56.1%** | sample 下仍走 CPU Python |
| 规则 + anchor | 32.2% | test_v7 `fast-rules auto` 关闭 |
| Axelrod / Mix / extra | 11.7% | 全部 Python |
| field build | ~2% | 已复用，收益有限 |

`dev` GPU argmax（wall 695.57s，11,218 worker-seconds）：

| 部分 | 占比 |
|---|---:|
| Axelrod / Mix / extra | **65.6%** |
| 规则 + RuleBank | ~15% |
| 其他 | 剩余 |

结论：

- **test_v7 的第一瓶颈是 sample policy 对手**；
- **dev 的第一瓶颈是 Axelrod**；
- 规则 RuleBank 在 test_v7 自动关闭，当前不是首要问题。

## 2. 工作流

### 6.1 GPU sample policy 对手（最高优先）

把 `PolicyNeural/PolicyRL/PolicyGRU/PolicySelfPlay` 的 sample 推理也批量上
GPU。每个 policy 对手在一场中只使用一次，因此可以：

1. 在 `BatchOpponentField` 中按 `(type, weights)` 分组；
2. 从每个对象的 `rng` 状态生成 `[rows, 100]` 逐轮 uniform；
3. 复用现有 `BatchMLP.act_from_logits_sample` /
   `BatchGRU.act_from_logits_sample` /
   `BatchSelfPlayGRU.act_from_logits_sample`；
4. 每个 group 维护 `draw_idx`，hard-rule 轮不消耗 uniform；
5. `PolicyFinal` / `PolicyTriple` 保持 CPU，Adv/PSRO meta 仍走 CPU。

验收：

- CPU/GPU 逐场 `mine/theirs/score` parity（n=8/16，float64）；
- full `test_v7` v5 wall ≤ **900s**（当前 1365s，约 1.5×）；
- 候选 mean 与当前 v5 差 <0.3 分；
- 301 tests 全绿。

风险：policy 对手的 RNG 共享/快照必须逐对象处理；field 复用前后不能
泄漏 RNG 状态。

### 6.2 Worker / batch 调优（低成本）

小基准（5 fields × 256）：

| 配置 | wall |
|---|---:|
| 8 GPU / 12 CPU，field-batch 2 | 28.2s |
| 4 GPU / 16 CPU，field-batch 2 | 26.4s |
| 8 GPU / 12 CPU，field-batch 4 | 31.0s |
| 4 GPU / 16 CPU，field-batch 4 | 30.8s |

v5 sample 有 8 GPU / 16 CPU 候选，`--gpu-workers 4` 更均衡；
`field-batch 4` 在 n=256 反而更慢。full 50×2048 需重新测，不要直接套
小基准。

验收：记录 full test_v7 在 4/6/8 GPU workers、field-batch 2/4 下的 wall
和 profile；选最优默认，不改变结果。

### 6.3 Axelrod 加速（dev 优先）

dev 的 Axelrod 占 65.6%，但完整 224 策略 C++ port 工作量大。按收益/成本
排序：

1. **缓存/预构建**：field 复用已有；检查 Axelrod 对象构建成本和
   `set_rounds`/import 冷启动。
2. **子集**：日常 `axelrod-fast`；dev_axelrod 可只用高频 curated 策略。
3. **选择性 C++ port**：只覆盖高频/低复杂策略（TFT/Grim/GTFT/ZD/
   MemoryOne 等），其余 Python fallback。
4. **并行/调度**：Axelrod 任务单独进程池，避免拖慢 GPU 任务。

验收：dev GPU argmax wall ≤ **500s**（当前 696s）；Axelrod 分数与 v4
语义一致；不把 224 全 port 作为前置条件。

### 6.4 RuleBank 阈值复核（6.1 之后）

`test_v7` 规则行约 73%，`fast-rules auto` 关闭；Plan 3 全开实测 0.92×
（回归）。在 6.1 把 policy 对手移走后，重新测：

- `auto` vs `on` vs `off`；
- 阈值 0.90 / 0.75 / 0.60；
- 只在 wall 改善且分数不变时调整。

### 6.5 legacy suite / matchup（可选）

`matchup` / `suite` 仍是纯 Python：

- 158：约 3 分钟；
- catalog：约 10 分钟；
- suite 2048：约 1:15。

如果这些命令是日常高频，才考虑：

- 把 `suite` 的 field 评估接到 `evaluate_fields_gpu`；
- 或把 `matchup` 改成协议 field 采样 + GPU 批处理；
- 否则保持现状，优先 6.1/6.3。

## 3. 非目标

- 不为了 GPU 速度改变 Protocol v4 sample RNG；v5 仍是 opt-in。
- 不一次性 port 全部 Axelrod 224 策略。
- 不改 `Adv-*` / PSRO meta 的 v5 CPU 语义，除非单独设计逐场 RNG。
- 不用 argmax 结果替代 sample 排名。

## 4. 顺序与验收

| 顺序 | 任务 | 预计 | Gate |
|---:|---|---:|---|
| 1 | 6.1 GPU sample policy 对手 | 1–2 天 | v5 ≤900s；parity；301 tests |
| 2 | 6.2 worker/batch 调优 | 0.5 天 | full test_v7 最优配置 |
| 3 | 6.3 Axelrod 缓存/子集/选择性 port | 2–5 天 | dev ≤500s |
| 4 | 6.4 RuleBank 阈值复核 | 0.5 天 | 分数不变且 wall 改善 |
| 5 | 6.5 legacy GPU（可选） | 3–5 天 | 仅在有日常需求时 |

## 5. 复现

```bash
# 当前 v5 baseline
python3 -m ipd protocol eval --candidates all --tiers test_v7 \
  --fields 50 --n 2048 --workers 20 --gpu-workers 8 \
  --mode sample --sample-rng per-match --gpu on \
  --seed 20261027 --no-consume

# profile 文件
results/protocol_v4/profile_test_v7_all24_gpu_sample_v5.json
results/protocol_v4/profile_dev_all24_gpu_argmax.json
```
