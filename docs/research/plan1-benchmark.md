# Plan 1 基准：CPU 评估管线提速

- **状态：** 实现完成；全量 `test_v7` 基准完成（2026-10-27）
- **代码：** `ipd/evaluation/protocol_cli.py` 调度重构，
  `ipd/evaluation/protocol.py` RNG 快照/恢复，
  `ipd/evaluation/scheduler.py` worker/进度/剖析，
  `tests/test_protocol_scheduler.py`
- **结果：** `results/protocol_v4/plan1_test_v7_all24.json` 与
  `results/protocol_v4/profile_plan1_test_v7_all24.json`
- **协议：** Protocol v4，`test_v7` seed=20261027，50 fields，n=2048，
  24 candidates，`--no-consume`

## 1. 环境

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 275HX（24 cores，混合架构） |
| RAM | 31 GB |
| GPU | RTX 5090 Laptop 24 GB（本计划未使用） |
| Python | 3.14.8 |
| torch | 2.11.0+cu128 |
| Axelrod | 4.14.0 |

## 2. 实现摘要

- **1.1 worker 自适应**：`--workers auto` =
  `min(20, cpu_count - 2, RAM 预算)`；预算为可用内存减去 2 GB 父进程预留，
  每个 worker 按 1 GB 计。显式整数不被截断。
- **1.2 field 级调度**：task 从 `(candidate, tier, seed)` 改为
  `(field, candidate_chunk)`。每个 task 构建一次 field，串行评估最多 6 个
  交错分组的候选。
- **RNG 契约**：构建后保存每个对手和学习策略的 `rng.getstate()`；每个候选
  开始前 `reset` 对手并恢复快照，候选 policy 仍按 `(candidate, field)` 重建。
  `Neural/GRU/SelfPlay` 不是 `Strategy` 子类，单独纳入快照。`Darwin` 等
  类级状态对手自动回退到逐候选重建。
- **1.4 进度/ETA**：输出 `fields done/total`、`tasks done/total`、当前
  tier/candidate、elapsed、ETA。
- **1.5 剖析**：结果 JSON 始终含 `profile` 摘要；`--profile` 另写
  `profile_*.json`，包含 per-candidate、per-tier、per-family 耗时。
- **内存**：`MemorySampler` 每秒采样进程树；同时记录 RSS（会重复计算共享
  库）与 PSS（物理内存估计）。结果文件含 `peak_rss_mb`、`peak_pss_mb`、
  `peak_anon_mb`、`workers_rss_mb` 和每个 worker 的峰值。

## 3. 基准结果

### 3.1 `test_v7` all24（n=2048，50 fields）

| Run | workers | wall time | speedup | 说明 |
|---|---:|---:|---:|---|
| 旧结果 `protocol_v4_test_v7_all24_provisional.json` | 未记录 | 9078.47s（2.52h） | 1.00× | `seconds` 来自 20-candidate 运行；4 个 PSRO 另一次 546.96s |
| 新调度 `plan1_test_v7_all24.json` | 20 | **3052.42s（0.85h）** | **2.97×** | 24 candidates；`--candidate-chunk 6`，field reuse on |

结论：20-worker 实测 **0.85h**，未达到 0.5–0.6h 的硬目标。原因不是
field 调度失效，而是本机 24 核为混合架构（8 P-core + 16 E-core）；
5→20 worker 的扩展低于 4×。旧结果的 workers 也未记录，2.97× 只是对仓库
现有结果文件的保守对比，不是严格同 worker 对照。

### 3.2 成本结构（新运行 profile）

profile 把每个 candidate 的 `eval_seconds` 按 opponent family 拆分；同一
field 的共享 `field_build_seconds` 单独记录。下表按总耗时排序；百分比以
所有 candidate family 累计时间 55,450s 为分母，`ms/match` 用该 family 的
总秒数除以匹配数。

| Family | 总秒数 | 占比 | ms/match |
|---|---:|---:|---:|
| `policy` | 14,971.1 | 27.0% | 61.00 |
| `anchor` | 4,018.5 | 7.2% | 16.40 |
| `axelrod` | 2,658.1 | 4.8% | 19.07 |
| `AxelrodMix` | 2,354.9 | 4.2% | 21.05 |
| `axelrod_extra` | 2,326.4 | 4.2% | 20.14 |
| `hyper_extort` | 1,784.0 | 3.2% | 21.22 |
| `early_probe` | 1,711.3 | 3.1% | 20.53 |
| `early_harvest` | 1,675.9 | 3.0% | 20.76 |

按 candidate 拆分后，**SelfPlay-GRU 一个候选占全部 candidate eval 时间的
36.8%**（20,393.5s）；RL-PPO / GRU / Adv-GRU 各约 4,426–4,857s，其余候选
约 1.5–1.7ks。Plan 2 首先要处理 SelfPlay/GRU 的蓝方推理，否则端到端收益会被
这一个候选拖住。

### 3.3 worker 扩展

旧结果文件没有记录 workers，因此不能把 9078.47s → 新 wall time 全部归因于
新调度。为区分两类收益，需要同 candidate set、同 fields 的 5-worker 对照；
本轮没有重跑这个对照，2.97× 只作参考。

硬件限制：24 核 Ultra 9 275HX 是混合架构（8 P-core + 16 E-core），20 个
worker 中只有 8 个能落在 P-core；5→20 worker 的实际扩展低于 4×。要严格
达到 0.5–0.6h，需要 20 个 P-core 级核心、或把长任务拆到更多进程/使用
Plan 2 的 GPU 路径；单纯增加 worker 数不会线性改善。

## 4. RNG / 结果一致性

回归覆盖：

- `tests/test_protocol_scheduler.py` 固定 `dev_param` seed=777/999，比较
  “复用 field + 恢复 RNG”与“重建 field”的逐场分数；
- 候选正序/逆序各跑一次，断言同一候选的 scores 完全一致；
- `workers=1` vs `workers=4` 跑 `dev_param --fields 2 --n 256`，断言
  `values`（每个 field 的 mean）逐位一致；
- `test_v7` 固定 seed，把 `Final-tuned / Harvest-tuned / GapStart / Neural`
  放在同一 chunk，与逐候选重建结果比对；`Neural` 覆盖 sample 模式；
- 单独断言快照包含 `NeuralPolicy / GRUPolicy / SelfPlayGRUPolicy` 的 RNG。

全量运行与旧结果文件逐候选比较：

- 24 个候选中 **23 个**的 `mean` / `min` / `families` / `family_norm` /
  `values` 完全一致；
- 唯一差异是 sample 模式的 **SelfPlay-GRU**：mean 466.04596 → 466.04819
  （**+0.00224**，相对 4.8×10⁻⁶）；50 个 field 中 30 个逐位一致，20 个有
  极小差异；policy family 均值差 +0.0224；
- 这是 sample 模式的 RNG 顺序差异，在预定容忍范围内，不影响 Plan 1 的
  确定性 parity 结论。确定性候选全部逐位一致。

Plan 2 的 sample 语义变更没有在本计划中引入：本计划仍保持每个对象一条顺序
Mersenne Twister 流；field 复用只恢复快照，不重排随机数。

## 5. field 复用收益

`ProfileAccumulator` 记录 `field_build_seconds` 和 `field_tasks`。按
`field_build_seconds / field_tasks` 得到平均热 build 时间，再与旧调度
（每个候选构建一次 field）比较：

```text
old_field_builds = fields × candidates
new_field_builds = field_tasks
saved_worker_seconds = (old_field_builds - new_field_builds) × avg_build
```

实测：

```text
field_build_seconds = 316.78s
field_tasks         = 200
avg build           = 1.58s（含每个 worker 首次 import Axelrod 的冷启动）
old builds          = 50 × 24 = 1200
new builds          = 200
saved worker-seconds = (1200 - 200) × 1.58 = 1583.9s
saved wall-seconds   ≈ 1583.9 / 20 = 79.2s
share of new wall    ≈ 79.2 / 3052.4 = 2.59%
```

**结论：field 构建复用只省约 2.6% wall time，低于 5% 门槛。** 因此不新增
“跨 field 对象缓存”（1.3）作为独立交付项；同一 field 内的 RNG-safe 复用
保留，因为它是后续 Plan 2/3 按 field 批量推理的接口基础。

## 6. 验收清单

| 验收项 | 结果 |
|---|---|
| `test_v7 all24` ≤0.5–0.6h（20 workers） | **未达**：3052.42s = 0.85h；混合 CPU 扩展不足 4× |
| dev 五层 all24 ≤2.3h | **未实测**：按 2.97× 外推约 3.1h（旧 9.2h），预计不达 |
| 240 tests 全绿 | **通过**：256 tests（原 240 + 新增 16）55.46s 全绿 |
| 确定性候选逐位一致 | **通过**：23 个非 SelfPlay 候选全部逐位一致 |
| sample 候选分布等价 | **通过**：SelfPlay mean +0.00224（4.8×10⁻⁶），其余 sample 逐位一致 |
| 内存峰值 <16GB | **通过（PSS）**：13,091MB；anon 12,825MB；RSS 19,089MB 含共享库重复计数 |
| field 缓存 ≥5% 才作为交付项 | **不启用**：实测 2.59% < 5%；不新增跨 field 缓存 |

> Gate 1 结论：**代码实现和确定性 parity 通过；性能硬指标未达标。**
> 合并代码不依赖性能数字达标，但文档必须保留这个偏差；Plan 2 的验收仍按
> 端到端 wall time 重新做。

## 7. 决策

- **合并代码，但不宣称 Gate 1 性能达标。** 调度、RNG 契约、可观测性和测试
  已满足 Plan 2 的接口依赖；`test_v7` 实测 0.85h，高于 0.5–0.6h 目标。
- 保留 field 级调度与 RNG 快照：它是唯一能在不改变结果的前提下复用 field
  的路径，也是 Plan 2/3 按 field 批量推理的接口基础。
- 不新增跨 field 对象缓存（1.3）：实测只省 2.59% wall time，未达到 5%
  门槛；继续增加缓存会扩大对象复用和类级状态的风险。
- 性能目标转移到 Plan 2：本机 CPU 的 5→20 worker 扩展约 3×，继续堆 worker
  不会到 0.5h；GPU 蓝方全 field 批量才是下一个数量级。
- `dev 五层` 没有全量重跑；若需要正式 Gate 1 数字，应在同一机器上补一次
  5-worker 对照和 dev 30-field 运行，并把结果追加到本文档。

## 8. 复现

```bash
# 全量 test_v7（provisional）
python3 -m ipd protocol eval --candidates all --tiers test_v7 \
  --fields 50 --n 2048 --workers 20 --candidate-chunk 6 \
  --field-reuse on --no-consume --profile --seed 20261027 \
  --out results/protocol_v4/plan1_test_v7_all24.json

# 单元测试
python3 -m unittest tests.test_protocol_scheduler -v
python3 -m unittest discover -s tests -t . -v

# dev 五层（若单独运行）
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 6 \
  --field-reuse on --profile --seed 20261027 \
  --out results/protocol_v4/plan1_dev_all24.json
```

## 9. Plan 2/3 交接提醒（2.5+）

给做 Plan 2.5 及之后的 session：

1. **先等 Plan 1 提交/合并。** 2.5 会改
   `ipd/evaluation/protocol_cli.py` / `protocol.py`，与 Plan 1 工作区冲突。
2. **任务接口**：`_eval_field_task(task)` 的 task 是
   `{specs: [CandidateSpec.dict], tier, seed, n, rounds, field_reuse}`；
   返回 `{results: [...], field_build_seconds, field_reuse, worker_pid,
   peak_rss_mb}`。GPU 路径应返回兼容的 `results` 行，并保留
   `timings.eval_seconds / candidate_build_seconds / family_seconds`。
3. **RNG 契约不能改**：field 构建后 `snapshot_field_rng()`；每个候选前
   `restore_field_rng()`；候选 policy 仍按 `(candidate, field)` 重建。
   `Neural/GRU/SelfPlay` 不是 `Strategy` 子类，但已纳入快照。
4. **sample 默认留 CPU**（方案 A）。不要为了 GPU 批量改成 per-match seed；
   那需要 Protocol v5 并重新基线。
5. **不要按单个 field 的 policy 子集（约 203 行）做 GPU 路由**；
   policy 对手要跨 field 聚合到 `B >= 2048`，或者显式测量小 batch 后接受
   回退。Plan 1 的 `(field, candidate_chunk)` 不够，需要额外的跨 field
   收集/合并层。
6. **结果 schema** 仍是 `ipd-protocol-eval-v1`；新增字段可以，不要删除或
   改名 `results` / `families` / `values`。`values` 必须保持 seed 顺序。
7. **性能基线**：`test_v7` all24 n=2048 50 fields，20 workers =
   **3052.42s（0.85h）**；profile 在
   `results/protocol_v4/profile_plan1_test_v7_all24.json`。
   `policy` family 占 27%，SelfPlay-GRU 单候选占 36.8%；Plan 2 优先做
   SelfPlay/GRU 蓝方全 field 批量，而不是只做 policy 对手。
8. **field 复用只省 2.59%**，不要把它当成 Plan 2 的端到端收益来源；它的
   价值是提供 field 级批处理接口和 RNG-safe 复用。
9. **回归**：`tests/test_protocol_scheduler.py` 的 16 个测试必须保留；
   全量 256 tests 当前全绿。加 GPU 路径后，先跑
   `tests.test_protocol tests.test_protocol_scheduler`，再跑全量。
10. **Gate 1 性能未达标**（0.85h vs 0.5–0.6h），但代码/接口可合并；
    Plan 2 的端到端验收要用自己的 GPU 基线重新做，不要复用这个 0.85h
    作为“Plan 2 加速前”的严格对照。
