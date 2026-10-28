# Plan 6：Python 评估瓶颈优化

- **状态：** 实现完成（2026-10-28）；6.1–6.4 完成，6.5 未做（无日常需求）
- **核心约束：** **24/24 候选分数必须与 Plan 5 v5 baseline（CPU
  Python policy 对手）逐位一致**；任何改变 RNG 消费顺序的加速都回退。
- **代码：**
  - `ipd/evaluation/gpu/policy_opponents.py`：sample policy 对手按
    `(type, weights)` 分组、逐对象 RNG uniform、批内 `draw_idx`；
  - `ipd/core/opponents/axelrod_bank.py`：Axelrod 历史增量维护；
  - `ipd/evaluation/protocol_cli.py`：`--fast-rules-threshold`、
    `--gpu-workers auto` 与 GPU-only 调度。
- **测试：** `tests/test_gpu_parity.py`（policy 对手 parity）、
  `tests/test_fast_opponents.py`；全量 303 tests 绿
- **结果：**
  - `results/protocol_v4/plan6_exact_w20_gpu4_fb4.json`（test_v7 v5 最佳）
  - `results/protocol_v4/protocol_v4_test_v7_all24_gpu_sample_v6.json`
    （6.1 单独，1310.57s，逐位一致）
  - `results/protocol_v4/plan6_migration_diagnostic_dev_argmax_w8fb2.json`
  - `results/protocol_v4/profile_plan6_*.json`
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

## 2. 工作流与结果

### 6.1 GPU sample policy 对手（完成）

实现：

1. `BatchOpponentField` 在 `policy_mode="sample"` 时也按
   `(type, weights, safe, temperature)` 分组，不再把学习型对手直接丢回
   CPU；
2. 每个对手从其 `random.Random` 状态预抽 `[rows, 100]` 逐轮 uniform；
   `draw_idx` 只在没有 hard-rule 的轮次前进，因此与 CPU 的惰性
   `rng.random()` 序列一致；
3. 复用 `BatchMLP/GRU/SelfPlayGRU.act_from_logits_sample`；
4. field 复用前后仍由 `snapshot_field_rng` / `restore_field_rng` 恢复
   每个对象的 RNG，group 不跨 candidate 泄漏状态；
5. `PolicyFinal` / `PolicyTriple` 保持 CPU。

`Adv-*` 的 GPU sample 实验已实现并做过 parity（用 per-match 契约时
CPU/GPU 逐动作一致），但它会把 `Adv-*` 从原来的 v4 顺序 RNG 换成 v5
per-match RNG，导致 `test_v7` 均分变化 `Adv-MLP +0.10`、`Adv-GRU
+0.23`、`Adv-RL +0.25`。**因为本计划的核心约束是分数逐位一致，这个
实验已回退；`Adv-*` 继续在 CPU 上使用原始顺序流。** PSRO meta 同理。

验收：

| 项 | 结果 |
|---|---|
| CPU/GPU 逐场 `mine/theirs/score` parity | **通过**：n=8 覆盖 `rl-repair`、`neural`、`gru`、`selfplay`、`psro-br`、`psro-scratch-br`，float64 全一致 |
| 全量 test_v7 v5 分数 | **通过**：24/24 候选与 Plan 5 v5 baseline 逐位一致 |
| 全量 test_v7 v5 wall | `1057.82s`（4 GPU / 16 CPU、field-batch 4） |
| 全量测试 | **通过**：303 tests |

6.1 单独（8 GPU / 12 CPU、field-batch 2）把 wall 从 1365.63s 降到
**1310.57s**，且 24/24 候选逐位一致；剩余时间几乎全部在 16 个 CPU
候选上，因此必须配合 6.2 的 worker split 继续压缩 wall。

在“分数逐位一致”约束下，最终 wall 为 **1057.82s**，未达到原 900s
目标。之前记录过的 621.94s 来自被回退的 `Adv-*` GPU 实验（会改变
分数），不作为本计划的交付结果。要继续逼近 900s，只能做不改变 RNG
和浮点求和顺序的 CPU 推理优化，而不是换采样契约。

### 6.2 Worker / batch 调优（完成）

全量 test_v7 v5、20 workers、`--candidate-chunk 4`、field 复用；
所有配置的 candidate mean 逐位一致，调优只改 wall：

| 配置 | wall |
|---|---:|
| 旧 v5 baseline（8 GPU / 12 CPU，fb2，Plan 5） | 1365.63s |
| 6.1 单独（8 GPU / 12 CPU，fb2） | 1310.57s |
| 3 GPU / 17 CPU，fb4 | 2100.95s（GPU 成为瓶颈） |
| **Plan 6 exact（4 GPU / 16 CPU，fb4）** | **1057.82s** |

- `Adv-*`、PSRO meta、Final/Triple/Bayes/Resp3/Harvest 仍在 CPU，
  v5 的 CPU 候选池比 GPU 任务更重，因此把 worker 从 8 调到 4；
  3 GPU / 17 CPU 反而退化到 2100.95s（GPU worker 不足、任务串行），
  说明 exact 配置下 4 GPU 是当前最优；
- field-batch 4 只改变 GPU 任务的聚合粒度，不改变逐场 RNG；full
  50×2048 下 batch 仍低于 `--gpu-max-batch 32768`；
- 默认改为 `--gpu-workers auto`：混合 sample 用 4、混合 argmax 用 8、
  GPU-only 用完整 worker budget；`--gpu-field-batch` 默认已是 4。
  因此 v5 test_v7 不传参即为 4 GPU / fb4；复现命令仍显式写出。

### 6.3 Axelrod 加速（完成）

没有做 224 策略 C++ port，先修掉 Python adapter 的 O(t²) 历史维护：
`AxelrodOpponent.act` 过去每轮重建整个 `axelrod.History`，现在只在
`_HistoryProxy` 上追加上一轮动作，`update_history` 仍逐轮调用。

- **逐动作 parity：** 224 个 fast Axelrod 策略 × 60–100 轮，对照旧
  adapter（TFT 与 RandomP 两种候选）0 差异；
- **微基准：** 224 策略 × 100 轮，0.495s → 0.125s（**3.96×**）；
- **dev family seconds：** `axelrod` 2656→869（3.06×）、
  `axelrod_extra` 2457→1022（2.40×）、`AxelrodMix` 2240→822（2.73×）；
  Axelrod 总 family 时间 7353→2713s；
- **dev 五层 30 fields（仅性能诊断）：** 用 baseline 相同的 8 GPU /
  12 CPU、field-batch 2 重跑：**475.38s**（baseline 695.57s → 1.46×）；
  24 候选全部层级 mean 与**同一 argmax 模式的 baseline 逐位一致**；
- `RuleBank` family 时间基本不变（455.5→468.3s），说明收益来自
  Axelrod adapter，而不是规则 bank。

> `dev argmax` 只用于 6.3 的受控性能诊断：前后都使用 argmax、同一 seed
> 和同一 RNG，因此它的**同模式**分数逐位一致不代表它与 sample 分数
> 一致。argmax 与 sample 是不同动作模式（Plan 5 已记录 10–17 分差异），
> 不能混表；它也不参与选模或 sample 一致性 gate。若只保留最小验收，
> 可以删掉这个端到端 gate，保留 224 策略 parity + 微基准即可；本文保留
> 它是因为它是 Axelrod 占比 65.6% 的唯一全量证据。

因此不需要 `axelrod-fast` 子集或选择性 port；`dev_axelrod` 的配置和
RNG 语义保持不变。

### 6.4 RuleBank 阈值复核（完成）

新增 `--fast-rules-threshold`（默认 0.90），保留 `auto` 语义：
仅当 field 的 fast-rule 行占比达到阈值时才构建 `RuleBank`。

6.1 + 6.2 后 CPU 候选池成为 wall 的瓶颈，而 `RuleBank` 只作用于 GPU
候选任务，因此全量 wall 不会从阈值调整中受益。10 fields × 2048 的
auto / threshold 0.60 对照（score 必须逐位一致）：

| `--fast-rules` | wall | 说明 |
|---|---:|---|
| `auto`（0.90） | 241.61s | test_v7 规则行约 73%，维持 Python 路径 |
| `auto --fast-rules-threshold 0.60` | 244.78s | 规则行全部进 C++ bank |

结论：**不调整**。阈值 0.60 反而略慢，24 候选 mean 逐位一致；
阈值 0.60 只改变 GPU 候选任务的规则路径，而 6.1/6.2 后 wall 由 CPU
候选池决定，所以不会带来收益。默认保持 `auto` + 0.90；0.75 对
test_v7 与 0.90 等价（73% < 75%），不再单独跑全量。

### 6.5 legacy suite / matchup（未做，可选）

`matchup` / `suite` 仍是纯 Python：

- 158：约 3 分钟；
- catalog：约 10 分钟；
- suite 2048：约 1:15。

这些命令不是当前日常高频路径，6.1–6.4 已把主评估瓶颈移走，因此本次
不接入 `evaluate_fields_gpu`，避免扩大改动面。

## 3. 非目标

- 不为 GPU 速度改变 Protocol v4 sample RNG；v5 仍是 opt-in。
- 不一次性 port 全部 Axelrod 224 策略。
- 不把 `PSRO-Meta` / `PSRO-Scratch` 的 v5 CPU 语义改成 GPU；它们仍
  逐场选 population 成员。
- `Adv-*` / PSRO meta 不接入 v5 GPU sample；它们必须保留原来的顺序
  RNG 消费，否则分数不再与 CPU baseline 逐位一致。
- 不用 argmax 结果替代 sample 排名。

## 4. 顺序与验收

| 顺序 | 任务 | 结果 |
|---:|---|---|
| 1 | 6.1 GPU sample policy 对手 | 完成；parity；全量 `1057.82s`，分数逐位一致 |
| 2 | 6.2 worker/batch 调优 | 完成；4 GPU / fb4 最优 |
| 3 | 6.3 Axelrod 历史增量 | 完成；224 策略 parity，3–4× family |
| 4 | 6.4 RuleBank 阈值复核 | 完成；保留 auto 0.90 |
| 5 | 6.5 legacy GPU | 未做；仅在有日常需求时 |

最终门槛：

| Gate | 目标 | 结果 |
|---|---|---|
| test_v7 v5 分数 | 与 Plan 5 v5 **逐位一致** | **通过**：24/24 候选 |
| test_v7 v5 wall | ≤900s（原目标） | **1057.82s（未达）**；exact 分数优先，未采用会改分的 `Adv-*` GPU 路径 |
| CPU/GPU parity | 逐场一致 | 通过 |
| 全量测试 | 全绿 | 303 tests OK |
| Axelrod 分数 | 与 v4 语义一致 | 逐动作 parity 通过 |

> **一次性迁移诊断（不属于长期 gate）：** `dev` argmax 的
> 695.57s→475.38s 只用于本轮 Plan 6 验证 Axelrod 提速；它不是默认流程、
> 不是 CI 项、也不进入长期验收。迁移结束后保留的常驻基准只有 sample
> 逐位一致性和 224 策略 Axelrod parity。历史结果文件仅作本轮记录。

## 5. 复现

```bash
# 当前最佳 test_v7 v5（4 GPU / 16 CPU，field-batch 4）
python3 -m ipd protocol eval --candidates all --tiers test_v7 \
  --fields 50 --n 2048 --workers 20 --gpu-workers 4 \
  --candidate-chunk 4 --gpu-field-batch 4 \
  --mode sample --sample-rng per-match --gpu on --gpu-dtype float32 \
  --seed 20261027 --no-consume

# 旧 v5 baseline（Plan 5）
python3 -m ipd protocol eval --candidates all --tiers test_v7 \
  --fields 50 --n 2048 --workers 20 --gpu-workers 8 \
  --candidate-chunk 4 --gpu-field-batch 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-dtype float32 \
  --seed 20261027 --no-consume

# dev argmax（baseline 相同调度）
python3 -m ipd protocol eval --candidates all --tiers dev \
  --fields 30 --n 2048 --workers 20 --gpu-workers 8 \
  --candidate-chunk 4 --gpu-field-batch 2 \
  --mode argmax --gpu on --gpu-dtype float32 \
  --seed 20261027 --no-consume

# profile 文件（4/2 与 dev；4/4 最佳 wall 见结果 JSON）
results/protocol_v4/profile_plan6_tune_w20_gpu4_fb2_adv.json
results/protocol_v4/profile_plan6_migration_diagnostic_dev_argmax_w8fb2.json
```
