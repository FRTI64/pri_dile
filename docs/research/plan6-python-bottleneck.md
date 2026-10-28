# Plan 6：Python 评估瓶颈优化

- **状态：** 实现完成（2026-10-28）；6.1–6.4 完成，6.5 未做（无日常需求）
- **代码：**
  - `ipd/evaluation/gpu/policy_opponents.py`：sample policy 对手按
    `(type, weights)` 分组、逐对象 RNG uniform、批内 `draw_idx`；
  - `ipd/evaluation/gpu/blue.py` / `policies.py`：`Adv-*` sample 的
    base RNG 抑制 mask；
  - `ipd/core/opponents/axelrod_bank.py`：Axelrod 历史增量维护；
  - `ipd/evaluation/protocol_cli.py`：`--fast-rules-threshold`、worker split
    与 GPU-only 调度。
- **测试：** `tests/test_gpu_parity.py`（policy 对手 + `Adv-*` sample
  parity）、`tests/test_fast_opponents.py`；全量 302 tests 绿
- **结果：**
  - `results/protocol_v4/plan6_tune_w20_gpu4_fb4.json`（test_v7 v5 最佳）
  - `results/protocol_v4/protocol_v4_test_v7_all24_gpu_sample_v6.json`
    （6.1 单独，1310.57s）
  - `results/protocol_v4/plan6_dev_all24_gpu_argmax_axelrod_fast_w8fb2.json`
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

在 6.1 基础上额外完成 `Adv-*` 的 v5 GPU sample：包装层在
punish / late / gap / final 轮覆盖动作时，通过 `need_mask` 抑制 base 的
RNG 抽取，但 GRU 仍推进 hidden state；CPU/GPU 逐动作一致。`Adv-*` 因此
从 v4 顺序流切换到 v5 per-match 契约，需要单独 v5 基线。

验收：

| 项 | 结果 |
|---|---|
| CPU/GPU 逐场 `mine/theirs/score` parity | **通过**：n=8 覆盖 `rl-repair`、`neural`、`gru`、`selfplay`、`adv-mlp`、`adv-gru`、`adv-rl`、`psro-br`、`psro-scratch-br`，float64 全一致 |
| 全量 test_v7 v5 wall | **通过**：4 GPU / 16 CPU、field-batch 4 = **621.94s**（1365.63s → 2.20×） |
| 候选 mean 与 v5 差 <0.3 | **通过**：21/24 候选逐位不变；`Adv-MLP` +0.10、`Adv-GRU` +0.23、`Adv-RL` +0.25 |
| 全量测试 | **通过**：302 tests |

6.1 单独（未启用 `Adv-*` GPU、8 GPU / 12 CPU、field-batch 2）把 wall
从 1365.63s 降到 **1310.57s**；剩余时间几乎全部在 16 个 CPU 候选上，
因此必须配合 6.2 的 worker split 才能达到 900s。最终 4/4 配置同时包含
6.1（policy 对手 + `Adv-*`）、6.2 和 6.3。

### 6.2 Worker / batch 调优（完成）

全量 test_v7 v5、20 workers、`--candidate-chunk 4`、field 复用：

| 配置 | wall |
|---|---:|
| 8 GPU / 12 CPU，field-batch 2 | 807.27s |
| 6 GPU / 14 CPU，field-batch 2 | 719.79s |
| 4 GPU / 16 CPU，field-batch 2 | 657.18s |
| **4 GPU / 16 CPU，field-batch 4** | **621.94s** |
| 旧 v5 baseline（8/12，fb2，Plan 5） | 1365.63s |

- 4 GPU workers 最优：v5 的剩余 CPU 候选（PSRO-Scratch、Final/Triple/
  Bayes/Resp3/Harvest、PSRO-Meta）比 GPU 任务更重；
- field-batch 4 在 full 50×2048 比 2 更快（621.94s vs 657.18s），与
  Plan 6 小基准（n=256）相反；batch 仍低于 `--gpu-max-batch 32768`；
- 四个配置的 candidate mean 完全相同（per-match RNG + field 复用），
  调优只改 wall，不改分数；
- 默认改为 `--gpu-workers auto`：混合 sample 用 4、混合 argmax 用 8、
  GPU-only 用完整 worker budget；`--gpu-field-batch` 默认已是 4。
  因此 v5 test_v7 不传参即为最优的 4 GPU / fb4；复现命令仍显式写出。

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
- **dev 五层 30 fields：** 用 baseline 相同的 8 GPU / 12 CPU、
  field-batch 2 重跑：**475.38s**（baseline 695.57s → 1.46×，目标
  ≤500s）；24 候选全部层级 mean 与 baseline **逐位一致**；
- `RuleBank` family 时间基本不变（455.5→468.3s），说明收益来自
  Axelrod adapter，而不是规则 bank。

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
| `auto`（0.90） | 163.42s | test_v7 规则行约 73%，维持 Python 路径 |
| `auto --fast-rules-threshold 0.60` | 162.86s | 规则行全部进 C++ bank |

结论：**不调整**。两者 wall 差异在噪声内，24 候选 mean 逐位一致；
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
- `Adv-*` 现在使用 v5 per-match 契约（逐场 parity），但不再沿用 v4
  顺序流；与 v4 sample 表不能逐位比较。
- 不用 argmax 结果替代 sample 排名。

## 4. 顺序与验收

| 顺序 | 任务 | 结果 |
|---:|---|---|
| 1 | 6.1 GPU sample policy 对手 | 完成；parity；全量 621.94s（含 6.2/Adv） |
| 2 | 6.2 worker/batch 调优 | 完成；4 GPU / fb4 最优 |
| 3 | 6.3 Axelrod 历史增量 | 完成；224 策略 parity，3–4× family |
| 4 | 6.4 RuleBank 阈值复核 | 完成；保留 auto 0.90 |
| 5 | 6.5 legacy GPU | 未做；仅在有日常需求时 |

最终门槛：

| Gate | 目标 | 结果 |
|---|---|---|
| test_v7 v5 wall | ≤900s | **621.94s** |
| test_v7 v5 分数 | 与 v5 差 <0.3 | 通过（最大 +0.25） |
| CPU/GPU parity | 逐场一致 | 通过 |
| 全量测试 | 全绿 | 302 tests OK |
| dev GPU argmax wall | ≤500s | **475.38s**；24 候选逐位一致 |
| Axelrod 分数 | 与 v4 语义一致 | 逐动作 parity 通过 |

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
results/protocol_v4/profile_plan6_dev_all24_gpu_argmax_axelrod_fast_w8fb2.json
```
