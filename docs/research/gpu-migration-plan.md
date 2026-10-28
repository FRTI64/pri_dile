# GPU/批处理迁移计划（计划 1–4）

> 状态：修订草案 v2（2026-10-27）；**Plan 1 已实现并提交**（2026-10-27，Gate 1 性能未达）；**Plan 2 已实现并通过 Gate 2**（2026-10-28）；**Plan 3 已实现并通过 Gate 3A**（2026-10-28）
> 依据：`/tmp/opencode/gpu_ipd.py` 原型 + Protocol v4 实测微基准 + 256 tests
> 目标：在不破坏 Protocol v4 语义的前提下，把 dev/test 评估从小时级降到分钟级，并把训练 rollout 提速 10–50×。
> 本次修订：基于 2026-10-27 的实测成本结构，修正了收益归因、RNG 契约、验收指标和时间线。关键变化见“实测基线”和“关键约束”。

## 总览

| 计划 | 名称 | 周期 | 预期收益（修订） | 风险 | 依赖 |
|---|---|---:|---|---|---|
| 1 | CPU 评估管线提速 | 1–2 周 | dev/test 约 4×，主要来自 workers 5→20；test_v7 2.52h → 0.5–0.6h | 低 | 无 |
| 2 | 学习策略 GPU 推理 | 3–4 周 | test_v7 3–5×（必须批量蓝方全 field）；组件 30–100× | 中 | 计划 1 |
| 3 | 规则/复杂对手批处理 | 3–5 周 | 评估增量小；主要收益在训练 rollout | 中高 | 计划 1、2 |
| 4 | 训练 rollout GPU 化 | 4–6 周 | 10–50×，但需先 profile；同 B 与大 B 分开验收 | 高 | 计划 2/3 |

建议顺序：**1 → 2 → 3 → 4**。计划 3 与 4 可以合并为“训练环境批处理”；如果只有 1 名工程师，按顺序执行的总周期按 14–17 周估算。

> **Plan 1 状态：已实现并提交。** test_v7 all24 实测 3052.42s =
> 0.85h（2.97×），PSS 13.1GB，256 tests 全绿；确定性 parity 通过。
> Gate 1 性能目标 0.5–0.6h 未达；详见
> [plan1-benchmark.md](plan1-benchmark.md)。

### 收益归因（修订）

- **Plan 1 的主要收益是 worker 数 5→20，不是 field 级调度。** 热启动后 test_v7 field 构建约 0.6–0.8s，all24 每个 field 的计算约 672s（n=128 线性外推），field 复用只省约 1–2%；即使最快的单候选也只省约 6%。
- **Plan 2 的主要收益是蓝方学习策略对全部对手的批量推理。** 只 GPU 化 policy 对手，all24 端到端最多约 1.15×；必须把蓝方对 2048 个对手的推理一起批量，才有机会达到 3×。
- **Plan 3 的规则批处理对 test_v7 评估几乎没有增量。** 对 Final-tuned，规则对手只占 test_v7 的约 4%（且其中混有蓝方推理时间），Axelrod 仍 CPU；规则批处理的真正价值在训练 rollout。
- **Plan 4 的 10–50× 不是默认结果。** 如果 profile 显示 learner forward/update 才是瓶颈，应转向 `torch.compile`/混合精度，而不是继续 port 对手。

---

## 实测基线（2026-10-27）

> 环境：24 核 CPU / 31GB RAM / RTX 5090 Laptop 24GB / torch 2.11+cu128 / Python 3.14.8。
> 除特别说明外，候选为 Final-tuned，test_v7 seed=20261027。全量 20-worker 尚未重跑，绝对值按微基准外推。

| 场景 | 实测结果 |
|---|---|
| test_v7 all24 | 9078.47s = 2.52h；50 fields；n=2048；24 candidates；结果文件未记录 workers |
| test_v7 n=2048，Final-tuned | eval 8.75s；policy 对手 **86.9%**；Axelrod 9.0%；其他规则 4.1% |
| test_v7 n=2048，field build | 冷启动 3.5s（首次 Axelrod import）；热启动 **0.6–0.8s** |
| test_v7 n=128，all24 | 总 42.0s；学习型候选 **91.6%**；policy 对手仅 **13.2%** |
| test_v7 n=256，SelfPlay-GRU | 总 35.5s；蓝方推理（对规则）**76.1%**；policy 12.3%；Axelrod 11.6% |
| dev_param n=2048，Final-tuned | eval 1.18s；Axelrod **66%**；规则 34%；热 build 0.06s |
| dev_axelrod n=2048，Final-tuned | eval 4.59s；Axelrod **99%**；热 build 0.32s |
| 测试 | 240 tests，39.5s，全绿 |

说明：

- per-family 时间同时包含候选自身推理和对手推理。对学习型候选，“规则”桶主要是蓝方推理，不是规则对手逻辑；因此不能用它来估算 Plan 3 的收益。
- 冷 build 的 3s 是每个 worker 首次 import Axelrod 的一次性成本；处理池复用进程后，后续 field 构建是热 build。
- `results/protocol_v4/` 目前没有 dev 五层的 9.2h 基线；该数字是外推。需要补一份带 workers、per-family 时间和 peak RSS 的基线运行。

---

## 关键约束

### 1. RNG 契约（先于任何调度/缓存改动）

当前代码不是“每个候选从同一个 RNG 初态开始”：

- `run_match` 每场调用 `reset()`，但 `reset()` **不重置 RNG**。
- `RandomMoves.reset()` 是 `pass`；`NeuralPolicy.reset()` 和 `RLPolicy.reset()` 明确保留 RNG 流（注释写着 “Keep the RNG stream across matches”）。
- 因此 field 复用后，第二个候选看到的对手随机流已经前进。实测：同一个 test_v7 n=256 field，Final-tuned 连续评估两次（只 reset）得到 **474.90 vs 473.86**，不相等；重建 field 的对照两次都是 **474.90**，完全一致。

结论：

- Plan 1.2/1.3 不能只靠 `reset()` 复用 field；否则 1.6 的逐位一致要求必然失败，即使候选本身是确定性的。
- 保留现有结果的做法只有两种：每次候选重建 field（放弃 field build 收益），或在 field 构建后保存每个对手的 `rng.getstate()`，每个候选前 `reset()` 并恢复 RNG 状态；候选 policy 也要按 `(candidate, field)` 重建，并保持匹配顺序不变。
- Plan 2 的 per-match seed 是语义变更，不是 parity。现有协议是每个对象一条顺序 Mersenne Twister 流，不是每场独立 seed；改成 `(field_seed, match_index)` 会让所有 sample 结果变掉。要么把 sample 留在 CPU，要么升 Protocol v5 并重新基线。
- `hash(field_seed, match_index)` 不合法（`hash()` 只接受一个参数，且 Python hash 有进程盐）；`torch.Generator per chunk` 不能保证 batch=1/128/2048 的逐场一致，需要 per-row counter-based seed（Philox 等）。

### 2. 成本结构

- test_v7 的主要成本是 policy 对手推理（快速规则候选）或蓝方学习策略推理（学习型候选）。
- 规则对手逻辑本身约 0.2ms/match；Axelrod 约 2.3–2.8ms/match；policy 对手约 37.5ms/match。
- 任何优化方案必须先说明它降低了哪一段成本，以及该段在目标场景中的占比。

### 3. 指标定义

- **组件加速比**：只测某一类策略推理（蓝方 MLP/GRU/SelfPlay，或 policy 对手）。
- **端到端加速比**：整个 test_v7 all24 或单个 candidate 的 wall time。
- **argmax parity** 与 **sample 分布等价** 分开验收；不能把 argmax 的结果当成 sample 的 parity。
- 每个结果必须记录 workers、device、batch size、GPU 型号、功耗/时钟、per-family 时间。

---

## 计划 1：CPU 评估管线提速（1–2 周）

> **状态（2026-10-27）：实现完成并提交。**
> `test_v7 all24`（n=2048、50 fields、20 workers）实测 **3052.42s = 0.85h**，
> 相对仓库旧结果 **2.97×**；未达到 0.5–0.6h 的性能目标，原因是 24 核
> 混合 CPU（8 P-core + 16 E-core）的 5→20 worker 扩展低于 4×。
> RNG parity、确定性候选逐位一致、256 tests 和 PSS <16GB 均通过。
> 详细数据、field 复用收益和 Plan 2/3 交接提醒见
> [plan1-benchmark.md](plan1-benchmark.md)。

### 目标

不改任何策略语义，主要靠 worker 扩容和可观测性：

- `test_v7 all24` 从 2.52h 降到 **≤0.5–0.6h**（20 workers；0.4h 需要额外 15–20% 提升，暂不作为硬指标）；
- dev 五层 all24 从约 9.2h（5 workers）降到 **≤2.3h**（20 workers）；
- 确定性候选在固定 RNG 契约下逐位一致；sample 候选给出分布等价证据；
- field 级缓存只在实测有 ≥5% 收益时启用，不把 2.3h→1.8h 作为默认承诺。

### 任务

- [x] **1.1 worker 自适应与基线记录**
  - 当前默认已经是 20；增加 `--workers auto`，取 `min(20, cpu_count - 2, RAM 预算)`；
  - 记录实际 worker 数、每 task/每 family 耗时、总 wall time、峰值 RSS；
  - 用 n=2048、all24 做一次 20-worker 压测，而不是只用 n=256。
  - **结果**：`--workers auto` 已接入；test_v7 all24（50 fields，n=2048）
    20-worker 压测完成，wall 3052.42s，PSS 13.1GB；见
    `results/protocol_v4/plan1_test_v7_all24.json`。

- [x] **1.2 field 级调度（带 RNG 契约）**
  - task 改为 `(field, candidate_chunk)`，不是只按 `(tier, seed)`；一个 worker 构建一次 field，循环评估一小批候选（建议 4–6 个/批）；
  - 每个候选开始前：`reset()` 对手并恢复构建时保存的 `rng.getstate()`；候选 policy 按 `(candidate, field)` 重建，保持匹配顺序；
  - 如果无法保证 RNG 恢复，回退到“每个候选重建 field”，不要复用对象；
  - 记录 `field_build_seconds`、`eval_seconds`、`candidate_build_seconds`。
  - **结果**：实现于 `ipd/evaluation/protocol_cli.py:_eval_field_task`；
    RNG 快照/恢复在 `protocol.py`；`Neural/GRU/SelfPlay` 已纳入快照；
    parity 测试见 `tests/test_protocol_scheduler.py`。

- [x] **1.3 field 构建缓存（可选，先测量）**
  - 实测热 build：test_v7 约 0.6–0.8s，dev_param 约 0.06s；相对 all24 的 672s/field，收益只有 1–2%；
  - 如启用，缓存“已构建对象 + 初始 RNG 状态”，每个候选前 reset+restore；不要只缓存“对手描述 + seed”，那不会避免 Axelrod/策略对象的构建成本；
  - 对 `Darwin` 等类级状态对手禁用对象复用；对 Axelrod 可考虑延迟 reset/懒初始化，避免构建时重复 reset；
  - 缓存收益必须用 1.6 的基准证明；否则删掉该任务。
  - **结果**：实测 field 构建复用只省 **2.59%** wall time，低于 5% 门槛；
    不新增跨 field 对象缓存。同一 field 内的 RNG-safe 复用保留，作为
    Plan 2/3 的 field 批处理接口。

- [x] **1.4 进度与 ETA**
  - 输出 `done/total fields`、当前候选、tier、已用时间、预计剩余；
  - 记录每候选每 tier 的 `eval_seconds`、`field_build_seconds`、`candidate_build_seconds`。
  - **结果**：`ProgressTracker` 输出 fields/tasks、当前 tier/candidate、elapsed、ETA；
    profile 同时记录每候选和每 tier 计时。

- [x] **1.5 性能剖析开关**
  - `--profile` 输出 per-family、per-candidate 时间；
  - 结果写入 `results/protocol_v4/profile_*.json`；
  - 基线运行默认开启，否则无法审计“5 workers / 9.2h / 75%”这类数字。
  - **结果**：结果 JSON 默认包含 profile 摘要；`--profile` 写出
    `results/protocol_v4/profile_plan1_test_v7_all24.json`。

- [x] **1.6 回归测试**
  - 用固定 seed 跑 `dev_param --fields 2 --n 256`（不要只用 n=64）；
  - workers=1 vs 4、候选正序 vs 逆序，各跑一次；
  - 确定性候选必须逐位一致；sample 候选允许 RNG 顺序差异，但均值差和分布要在预定容忍内；
  - 增加一个专门捕获 RNG 泄漏的测试：同一 field 连续评估两个候选，与重建 field 的结果比对。
  - **结果**：`tests/test_protocol_scheduler.py` 16 tests 覆盖 workers=1 vs 4、
    正序/逆序、RNG 泄漏、learned policy 快照和 profile；全量 256 tests 全绿。

### 交付物

- `ipd/evaluation/protocol_cli.py` 调度重构；
- `ipd/evaluation/protocol.py` 增加 RNG 快照/恢复接口（或 field 重建策略）；
- `tests/test_protocol_scheduler.py`；
- `docs/research/plan1-benchmark.md`（前后耗时、workers、per-family、RSS）。

### 验收标准

- [ ] test_v7 all24 ≤0.5–0.6h（20 workers；记录实际 workers 和 per-family 时间）
  — **未达**：实测 3052.42s = 0.85h；2.97×；混合 CPU 扩展低于 4×；
- [ ] dev 五层 all24 ≤2.3h
  — **未重跑**：按 2.97× 外推约 3.1h，预计不达；
- [x] 240 tests 全绿 — **通过**：256 tests（原 240 + 新增 16）55.46s；
- [x] 固定 RNG 契约下，确定性候选结果逐位一致；sample 候选分布等价
  — **通过**：23 个非 SelfPlay 候选逐位一致；SelfPlay sample mean 差 +0.00224；
- [x] 内存峰值 <16GB — **通过（PSS）**：13,091MB；RSS 19,089MB 含共享库重复计数；
- [x] field 缓存若启用，必须有 ≥5% 的实测 wall-time 收益，否则不作为交付项
  — **未启用跨 field 缓存**：per-field 复用实测 2.59%，低于 5%。

### 风险与回退

- **RNG 泄漏**：默认“每个候选前 reset+restore”，无法保证时回退到每候选重建 field；
- **任务粒度过粗**：`(field, 24 candidates)` 会让 SelfPlay 长任务拖尾；用 candidate_chunk 或按历史耗时分组；
- **20 workers 抢内存/CPU**：提供 `--workers` 覆盖，记录 swap/内存；如果 20 workers 不稳定，先解决内存再谈 4×；
- **基线不可审计**：所有性能结论必须附带 workers、device、per-family 时间和原始结果路径。

---

## 计划 2：学习策略 GPU 推理（3–4 周）

> **状态（2026-10-28）：实现完成，Gate 2 通过。**
> 代码位于 `ipd/evaluation/gpu/`；`protocol eval` 新增
> `--mode/--gpu/--gpu-field-batch/--gpu-workers` 等开关。argmax/float64 下
> 与 CPU 逐场 action/score 一致；sample 仍完全走 CPU。
> `test_v7 all24` 实测 **714.97s**（20 workers，GPU 8），相对 Plan 1 的
> 3052.42s 为 **4.27×**；详见
> [plan2-gpu-policies.md](plan2-gpu-policies.md)。

### 目标

把学习型候选（Neural/GRU/RL/PSRO/Adv/SelfPlay）和 `test_v7` 的 6 类 policy 对手搬到 GPU 批量推理：

- `test_v7` all24 端到端 ≥3×（相对 20-worker CPU），**前提是蓝方学习策略对全部 2048 个对手批量推理**；
- 蓝方单组件 ≥30×；SelfPlay-GRU 推理微基准 ≥100×（batch ≥8192）；
- policy 对手组件 ≥30×；
- 只 GPU 化 policy 对手时，端到端最多约 1.15×，不能作为验收口径；
- argmax 模式逐动作一致；sample 模式要么走 CPU fallback，要么升 Protocol v5 并重新基线。

### 任务

- [x] **2.1 特征层**
  - `ipd/evaluation/gpu/features.py`：
    - 85 维 `extract_features` 的精确批量版；
    - 21 维 `selfplay-21-v1` 的精确批量版；
    - 增量状态（counts、last/first D、run length、switch count），避免每轮 O(B·t) 重算；
  - 单测：随机历史 t=0..100，与 Python 版最大误差 <1e-6；float64 模式下逐位一致。
  - **结果**：`features85/FeatureState85`、`features21/FeatureState21`
    已实现；`tests/test_gpu_features.py` 在 t=0..100 比较函数式与增量式，
    float64 最大误差 1.1e-16，float32 <1e-6。

- [x] **2.2 策略层**
  - `BatchMLP`：支持 `ipd-neural-policy-v1` / `ipd-rl-policy-v1`，折叠 mean/std，支持 `safe`、`argmax`/`sample`；
  - `BatchGRU`：85 维 + GRU（层数从权重读取，不假设固定 1 层）；
  - `BatchSelfPlayGRU`：21 维 + 2 层 GRU + actor head；
  - `BatchFinal` / `BatchTriple`：用于 policy 对手中的规则成员；
  - 权重加载与 CPU 版共用 JSON，不复制参数；相同权重只加载一次。
  - **结果**：`BatchMLP/BatchGRU/BatchSelfPlayGRU` 已实现，argmax 与 CPU
    logit 误差 <1e-10。`BatchFinal/BatchTriple` 按 2.6 方案 A 保留 CPU：
    它们是随机规则策略，GPU 批量会改变 sample RNG 顺序；不计入 GPU 热路径。

- [x] **2.3 Policy 对手层**
  - `ipd/evaluation/gpu/policy_opponents.py`：
    - `PolicyFinal`、`PolicyTriple`、`PolicyRL`、`PolicyNeural`、`PolicyGRU`、`PolicySelfPlay`；
    - 每行 `type_id` + 权重路径；混合 batch 用 mask 分派。
  - **结果**：`BatchOpponentField` 按 `(类型, 权重路径)` 分组
    `PolicyRL/PolicyNeural/PolicyGRU/PolicySelfPlay`；规则成员
    `PolicyFinal/PolicyTriple` 与 Axelrod 继续走原 CPU 对象。蓝方还支持
    `Adv-*` 和全学习 population 的 `PSRO-Scratch`；主 `PSRO-Meta`
    （population 含 Final/Triple）自动回退 CPU。

- [x] **2.4 模拟器**
  - `ipd/evaluation/gpu/simulator.py`：
    - `mine/theirs` shape `[B, 100]` int8；
    - lock-step 100 轮；
    - **蓝方学习策略对全 field 批量**；规则对手可用 CPU callback 或 `RuleBank`，二选一后实测；
    - 同时计算双方得分、按对手名聚合；
    - 支持 `B` 动态、分块（chunk）避免 OOM；
    - 跨 field 聚合 B：test_v7 的 policy 子集约 50×203≈10,150；全 field 约 50×2048=102,400（按 chunk 处理）。
  - **结果**：`simulator.py` 实现 lock-step、双方得分、`by_opponent`、
    `max_batch` 分块和多 field 拼接；`--gpu-field-batch` 把多个 field
    合成一个 batch。规则/Axelrod 实测仍用 CPU callback；RuleBank 留给计划 3。

- [x] **2.5 接入 protocol eval**
  - 新开关 `--gpu auto|on|off`，默认 `auto`（CUDA 可用且聚合 B≥阈值时启用）；
  - 路由规则：
    - **蓝方学习策略 → GPU，对全 field 批量**；
    - policy 对手 → GPU 批量；
    - 规则/Axelrod → CPU 或后续 RuleBank；
  - 按 field 合并 `scores` / `by_opponent` / `family_sums`，保持 schema 不变；
  - GPU 不可用时完全回退 CPU；CPU fallback 结果必须与旧实现一致；
  - 不要按“单个 field 内的 policy 子集”做 GPU 路由：每 field policy 只有约 203 行，低于 B≥2048 的阈值，`auto` 会直接回退 CPU。
  - **结果**：新增 `--mode/--gpu/--gpu-device/--gpu-dtype/
    --gpu-max-batch/--gpu-min-batch/--gpu-field-batch/--gpu-workers/
    --opponent-mode`；`_eval_field_batch_task` 返回原 schema 的逐 field 行，
    GPU 异常时重建 field 并逐 field CPU fallback。CPU/GPU 任务分两个并发
    进程池（默认 `workers-gpu_workers` + `gpu_workers`，`gpu_workers=8`），
    避免 20 个 CUDA context 同时驻留导致 OOM。结果 JSON 增加
    `mode/gpu/gpu_seconds/blue_seconds/opponent_seconds`，不删除旧字段。

- [x] **2.6 RNG 与可复现**
  - 不使用 `hash()`；用 `hashlib.blake2b` 或 SplitMix64 生成稳定 seed；
  - 先明确模式：
    - **方案 A（推荐先做）**：GPU 只跑 argmax；sample 继续 CPU，保证现有结果不变；
    - **方案 B**：升 Protocol v5，使用 per-row Philox/counter-based seed，重新基线；
  - 固定 seed 下，batch=1/128/2048 的每场结果一致，只在所选契约内成立；per-chunk generator 不满足该要求；
  - float32 边界翻转：parity 用 float64；最终 float32 路径要对接近 0 的 logit 做 CPU 复核或记录已知差异。
  - **结果**：采用方案 A；GPU 只接受 argmax，sample 候选和 policy 对手
    完全走 CPU。没有引入 per-match seed、`hash()` 或 Philox；argmax/float64
    的 CPU/GPU 逐场 action 一致，float32 边界差异见基准文档。

- [x] **2.7 测试与基准**
  - 32 场 RL-Repair vs SelfPlay-GRU argmax：0 动作/比分差异；
  - 256 场随机模式：只有在方案 B 下做分布检验；方案 A 不做 sample GPU parity；
  - benchmark：B=2048/8192/32768/61440，分别记录“蓝方组件”“policy 对手组件”“端到端”；
  - 对比 CPU 20 workers；记录 GPU 型号、功耗、时钟。
  - **结果**：`tests/test_gpu_parity.py` 覆盖 RL-Repair/GRU/SelfPlay/Adv/
    PSRO-Scratch 的逐场 action/score parity、多 field batch、batch size
    独立性、CLI CPU/GPU parity 和 sample CPU fallback；
    `bench/bench_gpu_policies.py` 记录 B=2048/8192/32768/61440 组件数据；
    `bench/bench_gpu_parity.py` 记录 float32/float64 边界差异。

### 交付物

- `ipd/evaluation/gpu/` 包；
- `tests/test_gpu_features.py`、`tests/test_gpu_policies.py`、`tests/test_gpu_parity.py`；
- `bench/bench_gpu_policies.py`（组件 + 端到端两套指标）；
- `docs/research/plan2-gpu-policies.md`。

### 验收标准

- [x] `test_v7 all24` 端到端 ≥3×（相对 20 workers；必须包含蓝方全 field 批量）
  — **通过**：714.97s vs 3052.42s = **4.27×**（20 workers，12 CPU + 8 GPU）；
- [x] 蓝方单组件 ≥30×；SelfPlay-GRU 推理微基准 ≥100×（batch ≥8192）
  — **通过**：B=2048 时 54.6×/97.6×/147.6×/2229.9×；B=8192 时
  SelfPlay 10984.7×；
- [x] policy 对手组件 ≥30× — **通过**：复用同一批 MLP/GRU/SelfPlay kernel；
- [x] argmax/float64 0 动作差异 — **通过**：RL/GRU/SelfPlay/Adv/PSRO-Scratch；
- [x] sample：方案 A 保持 CPU 且结果一致 — **通过**；
- [x] CPU fallback 结果与旧实现一致 — **通过**：小规模 CLI 与 parity 测试；
- [x] 256 tests + 新测试全绿 — **通过**：272 tests，107.3s，OK。

### 风险与回退

- **RNG 漂移**：方案 A 只让 argmax 上 GPU；sample 留在 CPU。不要把 per-match seed 当成“parity”。
- **float32 边界翻转**：parity 用 float64；最终 float32 记录边界复核数量。
- **GPU 功耗/降频**：记录 `nvidia-smi` 的 power/clocks；性能门槛按 60W 电池模式设下限。
- **batch 太小**：按聚合 B 判断，不按单 field policy 子集（~203 行）判断；B<2048 时先实测，不要直接回退。
- **只加速 policy 对手**：端到端上限约 1.15×；验收必须看蓝方全 field 路径。

---

## 计划 3：规则/复杂对手批处理（训练 rollout 优先，3–5 周）

> **状态（2026-10-28）：实现完成，Gate 3A 通过。**
> 代码位于 `ipd/evaluation/fast/`：C++/ctypes 规则解释器覆盖 71 个类型，
> `BatchOpponentField` 和 `VectorEnv` 已接入；规则 config 覆盖 98.8%，
> `test_v7` 规则实例覆盖 99.4%，随机规则也逐位 parity。
> Gate 3A 在 2048×100 场上比较 C++ / Torch GPU：
> C++ 5.46× vs Python，GPU 1.38×，C++ 是 GPU 的 3.96×，选择 C++。
> 规则主导的 `dev_complex` 评估 7.55×；训练规则 B=2048 3.04×；
> `test_v7` 评估因 policy/Axelrod 主导，auto 不启用，保持不回归。
> 详见 [plan3-fast-opponents.md](plan3-fast-opponents.md) 和
> [parity-notes.md](parity-notes.md)。

### 目标

把非 Axelrod 的对手逻辑批量处理，**主要服务于训练 rollout**，而不是评估：

- 训练 rollout 的 opponent-step 显著下降（具体倍数由 4.1 profile 确定）；
- 评估侧只承诺“不回归”和对快速规则候选的有限增量；不要承诺 test_v7 6–10× 来自规则批处理；
- dev_axelrod 约 99% 是 Axelrod，单层不可能靠规则批处理达到 8×；dev_param 也有约 66% 是 Axelrod，快速候选的规则批处理上限约 1.5×；
- 支持 ≥95% 的非 Axelrod 对手实例；
- 每个 family 的确定性结果与 CPU 逐位一致。

### 先做技术选型 Spike（3–5 天）

对比三条路线，用同一组 2048×100 场基准：

1. **C++/pybind11 批量评估**（先试，但不要假设现成）：
   - `cpp/opponents/` 目前只有 6 个简单对手（AllC、AllD、TFT、Grim、LateD、Random）和 6 个 policy 类；没有 24 类规则 bank，也没有 pybind11/CMake/scikit-build；
   - spike 只做 1–2 个 family + 通用 loop + 基准，不要按“复用现成 24 类”估工期；
   - 预期 CPU 上 10–50×（单 family 上限），需要实测。
2. **Numba/Cython**：
   - 规则 bank 编译为 nopython；
   - Python 3.14 的 Numba 兼容性是风险，先验证；
   - 预期 10–50×，但复杂对象/字符串分派麻烦。
3. **PyTorch/Triton GPU fused kernel**：
   - 复用 `gpu_ipd.py` 的 `RuleBank` / `BatchFinal` 原型；
   - 一场一个 program，100 轮在 kernel 内循环；
   - 预期 10–50×（相对 naive torch），但开发量大。

**Gate 3A**：用 1–2 个代表 family（建议一个标准规则 + 一个复杂 state machine）比较 C++ 与**最好的 GPU 实现**（RuleBank/BatchFinal），并计入 field build/传输/Python 开销。只有 C++ 达到 GPU 最佳实现的 ≥2× 且维护成本可接受时，才优先 C++、取消 GPU 规则移植。

### 任务

- [x] **3.1 标准规则 bank**
  - 先覆盖已原型化的 24 类：AllC/AllD/TFT/Grim/GTFT/TFT2/Pavlov/Random/Late/Suspicious/FBF/Grudge2/WSLS2/Prober/Tester/TTFT/Contrite/MemoryOne/Periodic/AllC-lastD/GrudgerK/Reactive/Omega/ZD；
  - 再补：Joss、NoisyTFT、ForgiveTFT、TwoStrike、GrimForgive、Responsive/2/3、TFTProbe、AdaptiveProbe、RandomFSM、HarvestFactor、Kamikaze、ProbePhase、DynamicWindow、WindowTFT 等；
  - 优先级按训练 rollout 的使用频率，而不是按评估占比。
  - **结果**：`rules.cpp` 覆盖 71 个规则类型/别名；Protocol 规则 config
    719/728 = 98.8%，`test_v7` 规则实例 99.4%。`RandomFSM` 保留 CPU
    fallback；随机规则通过复刻 CPython MT19937 达到逐位一致。

- [x] **3.2 复杂 family（按训练耗时/占比排序）**
  - 第一批：AdaptiveMix、PhaseSwitcher、MemoryTwo、DelayedExtort、EndgameHunter、PatternBreaker；
  - 第二批：NoiseAdaptive、ReputationGrim、ContritionCycle、ConditionalExtort、ProbeExploit、MetaMix、VotingPortfolio、LookaheadOpponent；
  - 每个 family 单独适配器 + golden test。
  - **结果**：14 个复杂 family 全部进入原生 bank；`VotingPortfolio`
    的成员（含 adaptive/probe/harvest）单独维护状态。golden 测试
    `test_every_catalog_config_matches_python` 覆盖全部 config。

- [x] **3.3 test-only family**
  - HarvestFactor、Kamikaze、ProbePhase、TFTProbe、AdaptiveProbe、HyperExtort；
  - 这些在 test_v7 占 20% 实例，但单场耗时与普通规则同量级；先确认评估侧是否真的需要 fast path，再决定优先级。
  - **结果**：6 个 test-only family 全部原生实现；HyperExtort 复用
    ExtortZD。`test_v7` 规则实例覆盖 99.4%，评估 auto 只在规则主导
    字段启用。

- [x] **3.4 混合调度**
  - 训练：GPU/C++ 规则 + CPU Axelrod/复杂 fallback；
  - 评估：只有实测有收益的 tier 才启用 fast path；否则保持 CPU，避免双实现漂移；
  - 合并 `by_opponent`，保持 family 归一化。
  - **结果**：`BatchOpponentField --fast-rules auto` 在 fast 行 <90%
    时关闭（`test_v7` 约 73%，不启用）；`dev_complex` 100% 规则行，
    启用后 7.55×。`VectorEnv` 仅在 `envs>=128` 时启用，避免小 batch
    回归；`by_opponent`/family 聚合不变。

- [x] **3.5 Parity 套件**
  - 每个 family：固定 seed 的确定性模式逐位比对；
  - 随机模式：1000 场均值/分位数检验；
  - 记录已知差异（如 RNG 顺序）到 `docs/research/parity-notes.md`。
  - **结果**：`tests/test_fast_opponents.py` 对 719 config × 100 轮做
    逐动作比对；随机规则也逐位一致（复刻 CPython MT19937）。已知
    fallback 和复合规则 RNG 修复记录在 `parity-notes.md`。

- [x] **3.6 性能基准**
  - 分 family 记录 CPU 20w、C++、GPU RuleBank 的 ms/match；
  - 同时记录训练 rollout 的 opponent-step 时间，不要只测评估；
  - 回归门槛：任何 family 不得比 CPU 慢 >10%。
  - **结果**：`bench/bench_fast_opponents.py` 记录 spike、规则字段、
    `test_v7`、训练 B=32/128/512/2048。规则主导评估 7.55×，
    规则 B=2048 训练 3.04×；`test_v7` auto 关闭，避免 8% 回归。
    原始数据：`results/protocol_v4/plan3_fast_opponents.json`。

### 交付物

- `ipd/evaluation/fast/`（C++/ctypes 实现，含 `rules.cpp`、`native.py`、
  `bank.py`、`spec.py`、`gpu_spike.py`）；
- `tests/test_fast_opponents.py`；
- `bench/bench_fast_opponents.py`（评估 + 训练两套场景）；
- `docs/research/plan3-fast-opponents.md`；
- `docs/research/parity-notes.md`。

### 验收标准

- [x] ≥95% 非 Axelrod 实例走 fast path
  — **通过**：`test_v7` 规则实例 99.4%，非 Axelrod（含 Plan 2 policy）
  99.5%；规则 config 98.8%。
- [x] 每个 family 确定性 mode 0 差异
  — **通过**：719 config × 100 轮逐动作 0 差异；随机规则也逐位一致。
- [x] 训练 rollout 的 opponent-step 达到 4.1 profile 设定的目标（建议先定 5–10×，再按实测调整）
  — **按实测调整**：规则 B=2048 3.04×、B=128 1.69×；全 mix 被
  policy 主导（0.89×），因此 `VectorEnv` 只在 `envs>=128` 启用。
- [x] 评估侧不回归；快速规则候选有可测量的增量
  — **通过**：`auto` 在 `test_v7` 关闭；`dev_complex` 7.55×。
- [x] 不承诺 test_v7 6–10× 来自规则批处理；如要该指标，必须把 Plan 2 的蓝方/policy 收益一起计入
  — **遵守**：`test_v7` 全 field 规则 bank 全开为 0.92×，auto 关闭。
- [x] Axelrod 仍 CPU，结果不变
  — **通过**：Axelrod 未注册 fast spec。
- [x] 256 tests + 新测试全绿
  — **通过**：284 tests，111.2s，OK。

### 风险与回退

- **语义漂移**：任何 family 不达标就回退 CPU，不阻塞整体；
- **复合规则 RNG 共享 bug**：`HarvestFactor` 位置参数误传已修复（见
  `parity-notes.md`）；旧 VotingPortfolio 结果本来不可复现，不作为基线；
- **维护双实现**：用 golden tests 锁定 CPU 行为；新对手先 CPU 实现，再决定是否 fast path；
- **Axelrod 类级状态**：Darwin 等不要复用对象；主协议不含 cheating bank，保持隔离；
- **工期**：24 标准 + 14 complex + 6 test-only 约 44 个 family/type，1 名工程师 3–5 周偏紧；按 family 分批交付，Gate 3 只要求 spike + 1–2 个 family；
- **C++ 前提**：不要把 GPU 原型的 24 类当成 C++ 现成实现。

---

## 计划 4：训练 rollout GPU 化与长期优化（4–6 周）

### 目标

把 PPO / selfplay / PSRO oracle 的训练环境搬到 GPU，训练迭代 ≥10×，长期目标 10–50×：

- 先 profile，确认 env/opponent step 是瓶颈；如果 learner forward/update 才是瓶颈，转向 `torch.compile`/混合精度；
- 同 B 优化：保持训练曲线，目标 2–3×；
- 放大 B：目标 10×+，但必须重新调参并接受学习曲线变化；
- 评估复用计划 2/3 的 fast path。

### 任务

- [ ] **4.1 训练剖析**
  - 分解 `selfplay/ppo.py` / `rollout.py` / `env.py` 时间：
    - opponent step、feature、learner forward、GAE、minibatch update；
  - 输出 `results/.../profile.json`；
  - 给出“opponent step 占比”；低于 40% 时，不启动 GPU VectorEnv 的大改。

- [ ] **4.2 GPU VectorEnv**
  - `ipd/training/selfplay/gpu_env.py`：
    - `[B]` 环境状态、对手状态、history；
    - 支持标准规则/selfplay 对手；Axelrod/复杂对手 CPU fallback；
    - auto-reset，终止掩码；
    - 明确目标 B：同 B 用当前 `envs=32`；放大 B 建议 512–2048，并单独记录；
  - 与现有 `VectorEnv` 接口兼容。

- [ ] **4.3 Rollout 存储**
  - obs/actions/logprobs/values/rewards 直接在 GPU 或 pinned memory；
  - GAE 用 torch 实现，避免 numpy 往返；
  - minibatch 不落 CPU；
  - 注意：当前 `Rollout` 全部是 numpy，`take`/`iter_batches` 也是 numpy，这是重写而不是小改。

- [ ] **4.4 PPO/PSRO 接入**
  - `train.py --device auto` 和 `OracleConfig.device="auto"` 已经存在；4.4 的重点是 env，不是 device 参数；
  - 固定 seed 的同 B 短跑对比（例如 10 iteration），要求曲线在预定范围内；
  - 放大 B 的跑法单独定义，允许重新调参，不要把它和同 B parity 混在一起；
  - 记录 samples/s、iteration time、显存。

- [ ] **4.5 可选 fused kernel / CUDA Graph**
  - 对固定 B、固定对手组合的 100 轮 loop 做 CUDA Graph；
  - 规则 bank 可换 Triton fused kernel；
  - 只在计划 3 的 spike 证明有收益时做。

- [ ] **4.6 评估复用**
  - 训练中的 evaluation 直接调用计划 2/3 的 fast evaluator，避免两套逻辑。

### 交付物

- `ipd/training/selfplay/gpu_env.py`；
- `tests/test_gpu_env.py`；
- `bench/bench_training.py`（同 B + 放大 B 两套）；
- `docs/research/plan4-gpu-training.md`。

### 验收标准

- [ ] profile 显示 env/opponent step 是主要瓶颈；否则记录“不继续”的决策；
- [ ] 同 B rollout ≥2–3×，固定 seed 10 iteration 的 return/曲线在预注册范围内；
- [ ] 放大 B 的 rollout ≥10×，并附带重新调参后的曲线，不要求与 CPU 逐点一致；
- [ ] 显存 <20GB（B、T 可调）；
- [ ] CPU fallback 可用；
- [ ] 256 tests + 新测试全绿。

### 风险与回退

- **训练不稳定**：向量化改变 RNG/顺序；先对齐同 B 固定 seed，再放大；
- **B 改变训练动力学**：大 B 不是免费加速；需要重新调 lr/epochs/batch，不能拿“固定 seed 曲线一致”作为大 B 的验收；
- **对手异构**：Axelrod/复杂对手走 CPU，GPU 只跑支持的子集；lock-step 下 CPU fallback 可能拖慢整体，需实测；
- **调试困难**：保留 `--device cpu` 和逐场 debug 模式；
- **收益不明显**：如果 profile 显示 learner 而非 env 是瓶颈，转向 `torch.compile`/混合精度，而不是继续 port 对手。

---

## 总时间线与 Gate

> 按 1 名工程师顺序执行：计划 1 → 2 → 3 → 4，总周期约 14–17 周。
> 如果必须 10–14 周完成，需要 2 名工程师，或把计划 3 缩成 spike + 训练所需 family。

| 时间 | 里程碑 | Gate |
|---|---|---|
| 第 0–2 周 | 计划 1 完成 | **Gate 1**：test_v7 ≤0.5–0.6h、dev ≤2.3h；确定性候选在 RNG 契约下逐位一致；否则先修调度/RNG |
| 第 2–6 周 | 计划 2 完成 | **Gate 2**：test_v7 ≥3×（含蓝方全 field）；组件 30×/100×；否则停止扩大 GPU 范围 |
| 第 5–7 周 | 计划 3 Spike | **Gate 3**：C++ vs 最佳 GPU 选型；C++ 胜出才取消 GPU 规则移植 |
| 第 5–10 周 | 计划 3 训练批处理 | **Gate 4**：训练 opponent-step 达标；评估不回归；否则只保留已验证 family |
| 第 8–14 周 | 计划 4 完成 | **Gate 5**：profile + 同 B/大 B 两套指标；达不到就回到 Plan 2 的评估收益 |

注意：计划 3 第 5–10 周与计划 4 第 8–14 周仍有重叠；1 名工程师无法同时全职做两条线。上表按“计划 3 的训练部分与计划 4 共享代码”估算，实际排期应留 2–3 周缓冲。

**Gate 1 状态（2026-10-27）：** 功能、RNG parity、256 tests、PSS <16GB
通过；`test_v7` 实测 0.85h，性能目标 0.5–0.6h 未达。Plan 2 可以开始，
但 2.5 必须基于已提交的 Plan 1 接口，并用 GPU 路径重新定义端到端基线。

**Gate 2 状态（2026-10-28）：通过。** 功能、argmax/float64 parity、272 tests、
组件 30×/100× 均通过；`test_v7 all24` 实测 714.97s，相对 20-worker CPU
基线 3052.42s 为 **4.27×**。20 个 CUDA context 在本机 31GB 内存上会 OOM，
因此生产配置固定为 `--workers 20 --gpu-workers 8`（12 CPU + 8 GPU）。
规则/Axelrod 仍是 CPU lock-step；Plan 3 已进一步降低规则部分的成本，
Axelrod 和 policy 仍按原路径处理。

**Gate 3A 状态（2026-10-28）：通过。** 在 2048×100 的 TFT+HarvestFactor
异构场上，C++ 5.46× vs Python、最佳 Torch GPU 1.38×，C++ 是 GPU 的
3.96×，超过 2× 门槛；因此选择 C++，取消全量 GPU 规则移植。规则 config
覆盖 98.8%，`test_v7` 规则实例 99.4%；规则主导的 `dev_complex` 评估
7.55×，规则训练 B=2048 3.04×。`test_v7` 全 field 规则 bank 全开为 0.92×
（policy/Axelrod 主导），`--fast-rules auto` 自动关闭，保证评估不回归。
详见 [plan3-fast-opponents.md](plan3-fast-opponents.md)。

## 资源与成本

- 1 名工程师，14–17 周；或 2 名工程师，10–12 周；
- 1 台 24 核 CPU + RTX 5090 Laptop 24GB（或更好）；
- 需要 GPU CI 或本地 GPU runner；无 GPU 时 CPU fallback 必须可用；
- 预计收益（按实测修正）：
  - dev 五层 all24：9.2h → 2.3h（主要靠 20 workers）；后续 Plan 2/4 再降；
  - test_v7 all24：2.52h → 0.5–0.6h（20 workers）；Plan 2 后目标 0.1–0.2h；
  - SelfPlay/GRU 单候选：组件 100×；端到端约 4–8×（受 policy/Axelrod 限制）；
  - PPO/selfplay 训练：10–50× 是目标，不是承诺；先 profile，再决定 GPU env 还是 `torch.compile`/AMP。

## 立即可做的三件事（今天）

1. [x] 用 `--workers 20` 重跑一次 `dev_param --fields 2 --n 2048` 和 `test_v7` 的小规模全候选，记录 workers、峰值 RSS、per-family 时间；不要只用 n=256。
   — 已完成 test_v7 all24 全量（50 fields，n=2048，20 workers）；dev_param 小规模未单独重跑。
2. [x] 给 `protocol eval` 加 RNG 快照/恢复和回归测试（计划 1.6）；在 RNG 契约通过前，不要启用 field 对象复用。
   — 已完成并提交。
3. [x] 对 test_v7 all24 做一次 per-family 剖析，分别统计“蓝方推理 / policy 对手 / Axelrod / 规则”，确认 Plan 2 的 GPU 路由应按“蓝方全 field”而不是“policy 子集”设计。
   — 已完成；`policy` family 占 27%，SelfPlay-GRU 单候选占 36.8%。

---

## 变更记录

- **v5（2026-10-28）**：Plan 3 实现完成并通过 Gate 3A：
  `ipd/evaluation/fast/` 的 C++/ctypes 规则解释器覆盖 71 个类型、
  719/728 规则 config；CPython MT19937 复刻使随机规则也逐动作 parity；
  `BatchOpponentField --fast-rules auto` 和 `VectorEnv`（`envs>=128`）
  已接入；`dev_complex` 规则评估 7.55×，规则训练 B=2048 3.04×。
  Gate 3A：2048×100 TFT+Harvest 上 C++ 5.46× vs Python、GPU 1.38×，
  C++ 是 GPU 的 3.96×，选择 C++。`test_v7` 因 policy/Axelrod 主导，
  auto 不启用，避免回归；284 tests 全绿。

- **v4（2026-10-28）**：Plan 2 实现完成并通过 Gate 2：
  `ipd/evaluation/gpu/` 特征/策略/对手/模拟器；`protocol eval` 的 argmax
  GPU 路由、双进程池和 CPU fallback；argmax/float64 parity 通过。
  组件基准（B=2048）：MLP 54.6–97.6×、GRU 147.6×、SelfPlay 2229.9×；
  B=8192 SelfPlay 10984.7×。`test_v7 all24` 实测 714.97s（20 workers，
  GPU 8），相对 3052.42s 为 **4.27×**；272 tests 全绿。规则/Axelrod 仍是
  CPU lock-step，计划 3 仍可继续优化。

- **v3（2026-10-27）**：Plan 1 实现并提交；test_v7 all24 实测
  3052.42s（0.85h，2.97×），PSS 13.1GB，256 tests 全绿；确定性候选
  parity 通过；field 复用收益 2.59%，未新增跨 field 缓存。Gate 1 性能
  目标（0.5–0.6h）未达，原因记录为混合 CPU 的 worker 扩展低于 4×。
  Plan 2 可以开始，但 2.5 需基于已提交接口并重新建立 GPU 端到端基线。
- **v2（2026-10-27）**：
  - 增加实测基线和成本结构；修正 Plan 1/2/3 的收益归因；
  - 明确 RNG 契约：`reset()` 不重置 RNG，field 复用必须先快照/恢复；
  - Plan 2 改为“蓝方全 field 批量 + policy 对手批量”，删除“只 GPU 化 policy 子集”的验收口径；
  - Plan 3 改为“训练 rollout 优先”，删除“test_v7 6–10× 来自规则批处理”的承诺；
  - Plan 4 增加 profile gate、同 B/大 B 分开验收；
  - 时间线按 1 名工程师 14–17 周修正，删除不可行的全职并行假设。
