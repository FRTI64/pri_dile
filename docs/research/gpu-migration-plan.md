# GPU/批处理迁移计划（计划 1–4）

> 状态：计划草案（2026-10-27）
> 依据：`/tmp/opencode/gpu_ipd.py` 原型 + Protocol v4 实测外推
> 目标：在不破坏 Protocol v4 可复现性的前提下，把 dev/test 评估从小时级降到分钟级，并把训练 rollout 提速 10–50×。

## 总览

| 计划 | 名称 | 周期 | 预期收益 | 风险 | 依赖 |
|---|---|---:|---|---|---|
| 1 | CPU 评估管线提速 | 1–2 周 | dev/test 3–4× | 低 | 无 |
| 2 | 学习策略 GPU 推理 | 2–3 周 | test_v7 3–5×；重候选 30–600× | 中 | 计划 1 |
| 3 | 规则/复杂对手批处理 | 3–6 周 | dev 8–12×；test 6–10× | 中高 | 计划 1、2 |
| 4 | 训练 rollout GPU 化 | 3–6 周 | PPO/selfplay 10–50× | 高 | 计划 2/3 |

建议顺序：**1 → 2 → 3 → 4**。计划 3 与 4 可部分并行；每个计划结束设一个 Go/No-Go gate。

---

## 计划 1：CPU 评估管线提速（1–2 周）

### 目标

不改任何策略语义，只改调度和缓存：

- `test_v7 all24` 从 2.52h（实际约 5 workers）降到 ≤0.4h；
- dev 五层 all24 从约 9.2h（5 workers）降到 ≤2.3h（20 workers），加缓存后 ≤1.8h；
- 输出 JSON 与当前结果逐位一致（确定性候选）。

### 任务

- [ ] **1.1 worker 自适应**
  - `protocol eval` 默认 `--workers = min(20, cpu_count - 2)`；
  - 增加 `--workers auto`；
  - 记录实际 worker 数、每 task 耗时、总 wall time。
- [ ] **1.2 按 field 调度**
  - 当前 task = `(candidate, tier, seed)`，每个 task 重新构建 field；
  - 改为 task = `(tier, seed)`，一个 worker 构建一次 field，循环评估该 field 上的所有候选；
  - 每个候选开始前重置所有对手状态；`run_match` 已调用 `reset()`，但需验证 Axelrod/有类级状态的对手。
- [ ] **1.3 Field 缓存**
  - 进程内 LRU：key = `(tier, seed, n, rounds, catalog_sha256)`；
  - 缓存“对手描述 + 每对手 seed”，不缓存有状态对象；或缓存对象但在每个候选前重置；
  - 对 `Darwin` 等类级状态对手禁用对象复用，单独 fresh build（Axelrod cheating 不在主协议，但要防回归）。
- [ ] **1.4 进度与 ETA**
  - 输出 `done/total fields`、候选、tier、已用时间、预计剩余；
  - 记录每候选每 tier 的 `eval_seconds`、`field_build_seconds`。
- [ ] **1.5 性能剖析开关**
  - `--profile` 输出 per-family、per-candidate 时间；
  - 结果写入 `results/protocol_v4/profile_*.json`。
- [ ] **1.6 回归测试**
  - 用固定 seed 跑 `dev_param --fields 2 --n 64`，与旧实现比对 mean/scores；
  - 确定性候选必须逐位一致；随机候选允许 RNG 顺序差异，但均值差 <0.05。

### 交付物

- `ipd/evaluation/protocol_cli.py` 调度重构；
- `ipd/evaluation/protocol.py` 增加 field 描述缓存接口；
- `tests/test_protocol_scheduler.py`；
- `docs/research/plan1-benchmark.md`（前后耗时对比）。

### 验收标准

- [ ] `test_v7 all24` ≤0.4h（20 workers + cache）；
- [ ] dev 五层 all24 ≤1.8h；
- [ ] 240 tests 全绿；
- [ ] 固定 seed 的确定性候选结果与旧结果逐位一致；
- [ ] 内存峰值 <16GB。

### 风险与回退

- 对手对象复用导致状态泄漏：默认“每个候选前重建/重置”，缓存仅缓存描述；如仍有问题，回退到每 task 构建。
- 20 workers 抢内存/CPU：提供 `--workers` 覆盖，记录 swap/内存。
- 结果顺序变化影响 RNG：确保每个 match 的 seed 由 `(field_seed, match_index)` 决定，而不是全局 RNG 流。

---

## 计划 2：学习策略 GPU 推理（2–3 周）

### 目标

把 24 个候选中的学习策略（MLP/GRU/SelfPlay）和 `test_v7` 的 6 类 policy 对手搬到 GPU 批量推理：

- `test_v7` 端到端 ≥3×；
- SelfPlay-GRU 单候选 ≥100×（batch ≥8192）；
- RL/GRU/MLP 单候选 ≥30×；
- 确定性 argmax 模式逐动作一致。

### 任务

- [ ] **2.1 特征层**
  - `ipd/evaluation/gpu/features.py`：
    - 85 维 `extract_features` 的精确批量版；
    - 21 维 `selfplay-21-v1` 的精确批量版；
    - 增量状态（counts、last/first D、run length、switch count），避免每轮 O(B·t) 重算；
  - 单测：随机历史 t=0..100，与 Python 版最大误差 <1e-6。
- [ ] **2.2 策略层**
  - `BatchMLP`：支持 `ipd-neural-policy-v1` / `ipd-rl-policy-v1`，折叠 mean/std，支持 `safe`、`argmax`/`sample`；
  - `BatchGRU`：85 维 + 1 层 GRU；
  - `BatchSelfPlayGRU`：21 维 + 2 层 GRU + actor head；
  - `BatchFinal` / `BatchTriple`：用于 policy 红方中的规则成员；
  - 权重加载与 CPU 版共用 JSON，不复制参数。
- [ ] **2.3 Policy 对手层**
  - `ipd/evaluation/gpu/policy_opponents.py`：
    - `PolicyFinal`、`PolicyTriple`、`PolicyRL`、`PolicyNeural`、`PolicyGRU`、`PolicySelfPlay`；
    - 每行 `type_id` + 权重路径；混合 batch 用 mask 分派。
- [ ] **2.4 模拟器**
  - `ipd/evaluation/gpu/simulator.py`：
    - `mine/theirs` shape `[B, 100]` int8；
    - lock-step 100 轮；
    - 同时计算双方得分、按对手名聚合；
    - 支持 `B` 动态、分块（chunk）避免 OOM。
- [ ] **2.5 接入 protocol eval**
  - 新开关 `--gpu auto|on|off`，默认 `auto`（CUDA 可用且 B≥阈值时启用）；
  - 把 field 拆成：
    - `policy` 子集 → GPU；
    - 其他 → CPU；
  - 合并 `scores` / `by_opponent` / `family_sums`；
  - GPU 不可用时完全回退 CPU，输出 schema 不变。
- [ ] **2.6 RNG 与可复现**
  - 每场比赛独立 seed：`seed = hash(field_seed, match_index)`；
  - 使用 `torch.Generator` per chunk 或 Philox；
  - 固定 seed 下，batch=1/128/2048 的每场结果一致（至少 argmax；sample 需同 RNG 流）。
- [ ] **2.7 测试与基准**
  - 32 场 RL-Repair vs SelfPlay-GRU argmax：0 动作/比分差异；
  - 256 场随机模式：均值差 <0.1，std 差 <0.05；
  - benchmark：B=2048/8192/32768/61440，记录 GPU 型号、功耗、时钟；
  - 对比 CPU 20 workers。

### 交付物

- `ipd/evaluation/gpu/` 包；
- `tests/test_gpu_features.py`、`tests/test_gpu_policies.py`、`tests/test_gpu_parity.py`；
- `bench/bench_gpu_policies.py`；
- `docs/research/plan2-gpu-policies.md`。

### 验收标准

- [ ] `test_v7 all24` 端到端 ≥3×（20 workers 对比）；
- [ ] SelfPlay-GRU 单候选 ≥100×（batch ≥8192）；
- [ ] RL/GRU/MLP 单候选 ≥30×；
- [ ] 确定性策略 0 动作差异；
- [ ] 随机策略均值差 <0.1；
- [ ] CPU fallback 结果与旧实现一致；
- [ ] 240 tests + 新测试全绿。

### 风险与回退

- **RNG 漂移**：先只支持 `argmax` 的 parity，再逐步放开 sample；提供 `--gpu-mode argmax|sample`。
- **float32 边界翻转**：调试模式用 float64 跑小 batch，定位接近 0 的 logit；必要时对边界动作做 CPU 复核。
- **GPU 功耗/降频**：记录 `nvidia-smi` 的 power/clocks；性能门槛按 60W 电池模式设下限。
- **batch 太小**：B<2048 时自动走 CPU，避免 GPU 更慢。

---

## 计划 3：规则/复杂对手批处理（3–6 周）

### 目标

把非 Axelrod 的对手逻辑批量处理，使 dev/test 不再被规则对手和 Python 解释器拖慢：

- dev 五层 all24 ≥8×；
- test_v7 all24 ≥6×（Axelrod 仍 CPU）；
- 支持 ≥95% 的非 Axelrod 对手实例；
- 每个 family 的确定性结果与 CPU 逐位一致。

### 先做技术选型 Spike（3–5 天）

对比三条路线，用同一组 2048×100 场基准：

1. **C++/pybind11 批量评估**（推荐先试）：
   - 复用 `cpp/` 现有策略实现；
   - 一个 loop 内跑多场，OpenMP 并行；
   - 预期 CPU 上 20–100×，风险低。
2. **Numba/Cython**：
   - 规则 bank 编译为 nopython；
   - 预期 10–50×，但复杂对象/字符串分派麻烦。
3. **PyTorch/Triton GPU fused kernel**：
   - 一场一个 program，100 轮在 kernel 内循环；
   - 预期 10–50×（相对 naive torch），但开发量大。

**Gate 3A**：如果 C++ 路线在 2048 场上达到 GPU naive 的 ≥2×，优先 C++，取消 GPU 规则移植。

### 任务

- [ ] **3.1 标准规则 bank**
  - 先覆盖已原型化的 24 类：AllC/AllD/TFT/Grim/GTFT/TFT2/Pavlov/Random/Late/Suspicious/FBF/Grudge2/WSLS2/Prober/Tester/TTFT/Contrite/MemoryOne/Periodic/AllC-lastD/GrudgerK/Reactive/Omega/ZD；
  - 再补：Joss、NoisyTFT、ForgiveTFT、TwoStrike、GrimForgive、Responsive/2/3、TFTProbe、AdaptiveProbe、RandomFSM、HarvestFactor、Kamikaze、ProbePhase、DynamicWindow、WindowTFT 等。
- [ ] **3.2 复杂 family（按耗时/占比排序）**
  - 第一批：AdaptiveMix、PhaseSwitcher、MemoryTwo、DelayedExtort、EndgameHunter、PatternBreaker；
  - 第二批：NoiseAdaptive、ReputationGrim、ContritionCycle、ConditionalExtort、ProbeExploit、MetaMix、VotingPortfolio、LookaheadOpponent；
  - 每个 family 单独适配器 + golden test。
- [ ] **3.3 test-only family**
  - HarvestFactor、Kamikaze、ProbePhase、TFTProbe、AdaptiveProbe、HyperExtort；
  - 这些在 test_v7 占 20%，优先级高于冷门 complex。
- [ ] **3.4 混合调度**
  - field 拆成 `gpu_supported` / `cpu_fallback`；
  - GPU 与 CPU 可并行：GPU 跑大批量规则/学习策略，CPU 跑 Axelrod；
  - 合并 `by_opponent`，保持 family 归一化。
- [ ] **3.5 Parity 套件**
  - 每个 family：固定 seed 的确定性模式逐位比对；
  - 随机模式：1000 场均值/分位数检验；
  - 记录已知差异（如 RNG 顺序）到 `docs/research/parity-notes.md`。
- [ ] **3.6 性能基准**
  - 分 family 记录 CPU 20w、GPU naive、GPU fused/C++ 的 ms/match；
  - 回归门槛：任何 family 不得比 CPU 慢 >10%。

### 交付物

- `ipd/evaluation/fast/`（C++/Numba 或 GPU 实现）；
- `tests/test_fast_opponents.py`；
- `bench/bench_opponents.py`；
- `docs/research/plan3-fast-opponents.md`。

### 验收标准

- [ ] ≥95% 非 Axelrod 实例走 fast path；
- [ ] 每个 family 确定性 mode 0 差异；
- [ ] dev all24 ≥8×，test_v7 all24 ≥6×；
- [ ] Axelrod 仍 CPU，结果不变；
- [ ] 240 tests + 新测试全绿。

### 风险与回退

- **语义漂移**：任何 family 不达标就回退 CPU，不阻塞整体；
- **维护双实现**：用 golden tests 锁定 CPU 行为；新对手先 CPU 实现，再决定是否 fast path；
- **Axelrod 类级状态**：Darwin 等不要复用对象；主协议不含 cheating bank，保持隔离。

---

## 计划 4：训练 rollout GPU 化与长期优化（3–6 周）

### 目标

把 PPO / selfplay / PSRO oracle 的训练环境搬到 GPU，训练迭代 ≥10×，长期目标 10–50×：

- 标准/selfplay 训练 profile 的 rollout 时间 ≥10×；
- 固定 seed 的学习曲线与 CPU 版一致（均值/方差在容忍范围内）；
- 评估复用计划 2/3 的 fast path。

### 任务

- [ ] **4.1 训练剖析**
  - 分解 `selfplay/ppo.py` / `rollout.py` / `env.py` 时间：
    - opponent step、feature、learner forward、GAE、minibatch update；
  - 输出 `results/.../profile.json`。
- [ ] **4.2 GPU VectorEnv**
  - `ipd/training/selfplay/gpu_env.py`：
    - `[B]` 环境状态、对手状态、history；
    - 支持标准规则/selfplay 对手；Axelrod/复杂对手 CPU fallback；
    - auto-reset，终止掩码；
  - 与现有 `VectorEnv` 接口兼容。
- [ ] **4.3 Rollout 存储**
  - obs/actions/logprobs/values/rewards 直接在 GPU 或 pinned memory；
  - GAE 用 torch 实现，避免 numpy 往返；
  - minibatch 不落 CPU。
- [ ] **4.4 PPO/PSRO 接入**
  - `selfplay/train.py`、`training/psro.py` 增加 `device=auto`；
  - 固定 seed 的短跑对比（例如 10 iteration）；
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
- `bench/bench_training.py`；
- `docs/research/plan4-gpu-training.md`。

### 验收标准

- [ ] 标准/selfplay profile 的 rollout ≥10×；
- [ ] 固定 seed 10 iteration 的 return/曲线与 CPU 版差异在预注册范围内；
- [ ] 显存 <20GB（B、T 可调）；
- [ ] CPU fallback 可用；
- [ ] 240 tests + 新测试全绿。

### 风险与回退

- **训练不稳定**：向量化改变 RNG/顺序；先对齐固定 seed 的小规模实验，再放大；
- **对手异构**：Axelrod/复杂对手走 CPU，GPU 只跑支持的子集；
- **调试困难**：保留 `--device cpu` 和逐场 debug 模式；
- **收益不明显**：如果 profile 显示 learner 而非 env 是瓶颈，转向 `torch.compile`/混合精度，而不是继续 port 对手。

---

## 总时间线与 Gate

| 时间 | 里程碑 | Gate |
|---|---|---|
| 第 0–2 周 | 计划 1 完成 | **Gate 1**：dev/test ≥3× 且结果一致；否则先修调度 |
| 第 2–5 周 | 计划 2 完成 | **Gate 2**：test_v7 ≥3×、重候选 ≥30×；否则停止扩大 GPU 范围 |
| 第 5–7 周 | 计划 3 Spike | **Gate 3**：C++ vs GPU 选型；C++ 胜则取消 GPU 规则移植 |
| 第 5–11 周 | 计划 3 完成 | **Gate 4**：dev ≥8×、test ≥6×；否则只保留已验证 family |
| 第 8–14 周 | 计划 4 完成 | **Gate 5**：训练 ≥10×；否则回到计划 2 的评估收益 |

## 资源与成本

- 1 名工程师，10–14 周；
- 1 台 24 核 CPU + RTX 5090 Laptop 24GB（或更好）；
- 需要 GPU CI 或本地 GPU runner；无 GPU 时 CPU fallback 必须可用；
- 预计收益：
  - dev 五层 all24：2.3h → 0.2–0.6h；
  - test_v7 all24：0.7h → 0.1–0.2h；
  - SelfPlay/GRU 单候选：0.2h → 秒级；
  - PPO/selfplay 训练：10–50×。

## 立即可做的三件事（今天）

1. 用 `--workers 20` 重跑一次 `dev_param --fields 2 --n 256`，确认 20 workers 的稳定性和内存。
2. 给 `protocol eval` 加 field 级调度（计划 1.2），这是无 GPU 风险的最大单点收益。
3. 把 `test_v7` 的 policy 子集单独统计耗时，确认 75% 的占比在目标机器上成立。
