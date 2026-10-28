# Plan 2 基准与实现：学习策略 GPU 推理

- **状态：** 实现完成；组件与端到端全量基准通过（2026-10-28）
- **代码：** `ipd/evaluation/gpu/`（features / policies / blue / policy_opponents / simulator），
  `ipd/evaluation/protocol_cli.py` 的 `--mode/--gpu/--gpu-field-batch/
  --gpu-workers` 路由，
  `ipd/evaluation/scheduler.py` 的多字段任务统计
- **测试：** `tests/test_gpu_features.py`、`tests/test_gpu_policies.py`、
  `tests/test_gpu_parity.py`
- **基准：** `bench/bench_gpu_policies.py`；组件数据在
  `results/protocol_v4/plan2_gpu_component.json`；端到端在
  `results/protocol_v4/plan2_test_v7_all24_gpu.json`；5-field per-family
  profile 在 `results/protocol_v4/profile_plan2_gpu_5fields.json`；
  float32 parity 在 `results/protocol_v4/plan2_gpu_parity*.json`

## 1. 实现摘要

### 1.1 特征层（2.1）

`ipd/evaluation/gpu/features.py` 提供：

- `features85` / `FeatureState85`：85 维 `ipd-neural-features-v1` 的精确批量版；
- `features21` / `FeatureState21`：`selfplay-21-v1` 与 `selfplay-21-v2` 的批量版；
- 增量状态保存 lifetime counters、first/last D、run length、switch count、
  joint counts 和累计得分，每轮只处理当前动作与窗口特征。

`extract_features` 的每个分组都被覆盖，包括 conditional response、两次
joint-frequency 窗口、recency/first-defection timing、score context 和
recent conditional。单测在 t=0..100 上同时比较函数式和增量式实现：

- float64：最大误差 ≤ `1.1e-16`（浮点结合顺序）；
- float32：最大误差 < `1e-6`；
- 常量、全 D、交替、CC/CD 等边界历史与 Python 版一致。

### 1.2 策略层（2.2）

`ipd/evaluation/gpu/policies.py`：

- `BatchMLP`：读取 `ipd-neural-policy-v1` 与 `ipd-rl-policy-v1`，
  把 feature mean/std 折进首层；支持 `safe` 与最终轮强制 D；
- `BatchGRU`：85 维输入、层数从权重读取、把标准化折进 `w_ih`；
- `BatchSelfPlayGRU`：21 维输入、2 层 GRU + actor head；
- 权重 JSON 与 CPU 版共用，不复制参数；argmax 是唯一 GPU 模式。

`BatchFinal` / `BatchTriple` 没有作为 GPU 组件启用：它们是随机规则策略，
Protocol v4 的 sample 语义依赖每个对象一条顺序 Mersenne Twister 流。
按 2.6 的方案 A，它们保留在 CPU；见 §5。

### 1.3 Policy 对手层（2.3）

`ipd/evaluation/gpu/policy_opponents.py` 的 `BatchOpponentField`：

- 按 `(类型, 权重路径)` 分组 `PolicyNeural/PolicyRL/PolicyGRU/PolicySelfPlay`；
- 每组一个批量网络和独立的增量特征/隐状态；
- 规则/Axelrod/`PolicyFinal`/`PolicyTriple` 仍调用原 CPU 对象；
- 每行的历史、reset 顺序和名称保持不变。

`ipd/evaluation/gpu/blue.py` 支持：

- 普通 `Neural/RL/GRU/SelfPlay` 候选；
- `Adv-MLP/Adv-GRU/Adv-RL` 的约束规则向量化；
- 蓝方 population 全为学习策略的 `PSRO-Scratch`；
- 主 `PSRO-Meta`（population 含 Final/Triple）自动回退 CPU。

### 1.4 模拟器（2.4）

`ipd/evaluation/gpu/simulator.py` 是 lock-step 100 轮批处理：

- `mine/theirs` shape `[B, 100]` int8；
- 蓝方每轮对整行 batch 推理；
- policy 对手按组批量，规则对手逐行 CPU callback；
- 同时计算双方 payoff、按对手名保留逐场分数；
- `max_batch` 分块，避免显存峰值；
- 支持多 field 拼接，`--gpu-field-batch` 让 policy 对手跨 field 聚合；
- 输出与原 `evaluate_field_details` 相同的 `scores/by_opponent/names`。

### 1.5 接入 protocol eval（2.5）

新增开关：

```text
--mode sample|argmax         候选动作模式（默认 sample）
--gpu auto|on|off            默认 auto；只有 --mode argmax 才启用
--gpu-device cuda
--gpu-dtype float32|float64  默认 float32；parity 用 float64
--gpu-max-batch 32768
--gpu-min-batch 2048         auto 模式下聚合 B 的下限
--gpu-field-batch 4          GPU 任务一次模拟的 field 数
--gpu-workers 8              CPU/GPU 混合时允许并发的 CUDA worker 数
--opponent-mode sample|argmax 学习型 policy 对手的模式（默认跟随 --mode）
```

路由：

- 蓝方学习策略 → GPU 全 field 批量；
- 学习型 policy 对手 → 同 batch GPU；
- 规则/Axelrod/`PolicyFinal`/`PolicyTriple` → CPU；
- GPU 初始化、显存或 CUDA 失败 → 重建 field 并逐 field CPU fallback；
- `--mode sample` 或 `--gpu off` → 完全走 Plan 1 的 CPU 路径，结果不变。

CPU 与 GPU 任务分别进入两个并发的进程池：CPU 池用
`workers - gpu_workers` 个进程，GPU 池用 `--gpu-workers` 个进程。这样既避免
20 个 CUDA context 同时驻留导致的主机内存 OOM，也让规则任务和 GPU 任务
重叠执行。GPU 任务内部仍是 `(field,batch × candidate chunk)` 粒度。

结果 schema 保持 `ipd-protocol-eval-v1`，新增 `mode` / `gpu` / `gpu_seconds` /
`blue_seconds` / `opponent_seconds` 字段；`results`、`families`、`values`
不变。

### 1.6 RNG 与可复现（2.6）

采用方案 A：

- GPU 只跑 `argmax`；
- sample 候选和 sample policy 对手留在 CPU；`--gpu on --mode sample`
  会输出警告并回退；
- argmax/float64 下，batch=1/24/2048 的逐场 action 与 CPU 一致；
- 不使用 `hash()`、不做 per-match seed；不改变 Protocol v4 的 RNG 流；
- float32 用于生产，float64 用于 parity；边界差异见 §4。

## 2. 组件基准（2.7）

环境：

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 275HX（24 cores） |
| RAM | 31 GB |
| GPU | NVIDIA GeForce RTX 5090 Laptop 24 GB |
| torch | 2.11.0+cu128 |
| Python | 3.14.8 |
| dtype | float32，100 轮 |

命令：

```bash
python3 bench/bench_gpu_policies.py \
  --cpu-batch 512 --batches 2048,8192,32768,61440 \
  --out results/protocol_v4/plan2_gpu_component.json
```

### 2.1 结果（2026-10-28）

完整脚本已跑通：CPU 用 `--cpu-batch 512` 逐条测量后线性外推；GPU 使用
float32、100 轮、每次测量前 `cuda.synchronize()`。B=2048 的第一次 GPU
调用包含 CUDA context 预热，因此更高 batch 的加速比反而更大；这不是
多线程/异步计时造成的假象。

| 策略 | B=2048 | B=8192 | B=32768 | B=61440 |
|---|---:|---:|---:|---:|
| RL-Repair (MLP) | 54.6× | 702.5× | 2980.6× | 5268.2× |
| Neural (MLP) | 97.6× | 360.4× | 1321.3× | 2689.4× |
| GRU (85 维) | 147.6× | 554.7× | 2021.4× | 4361.1× |
| SelfPlay-GRU | 2229.9× | 10984.7× | 12629.5× | 31831.5× |

原始数据：`results/protocol_v4/plan2_gpu_component.json`。

结论：蓝方单组件和 policy 对手（同一套 MLP/GRU/SelfPlay kernel）在
B=2048 就超过 30×；SelfPlay-GRU 在 B≥8192 远超 100×。

## 3. 端到端基准

命令：

```bash
python3 -m ipd protocol eval --candidates all --tiers test_v7 \
  --fields 50 --n 2048 --workers 20 --gpu-workers 8 \
  --candidate-chunk 4 --field-reuse on --mode argmax \
  --opponent-mode argmax --gpu on --gpu-dtype float32 \
  --gpu-field-batch 2 --gpu-max-batch 32768 --no-consume \
  --seed 20261027 --out results/protocol_v4/plan2_test_v7_all24_gpu.json
```

### 3.1 结果

**10-field 预跑（24 workers = 16 CPU + 8 GPU，无 profile）：**

- 10 fields / 45 tasks，总 **140.6s**；
- 按 Plan 1 的 50-field 基线外推：`3052.42 × 10/50 = 610.48s`；
- 对应 **4.34×**。

**全量 50-field（20 workers = 12 CPU + 8 GPU，无 profile）：**

| 项 | 结果 |
|---|---:|
| wall time | **714.97s** |
| Plan 1 基线（50 fields，20 workers） | 3052.42s |
| 端到端加速比 | **4.27×** |
| candidates | 24（GPU 12 / CPU 12） |
| field_tasks | 225 |
| peak PSS | 20,646MB |
| peak RSS（含共享库重复计数） | 26,974MB |

原始数据：`results/protocol_v4/plan2_test_v7_all24_gpu.json`。

5-field `--profile` 复核（`profile_plan2_gpu_5fields.json`）中，
`policy` family 累计 814.2s（24,216 场，33.6ms/match），
`AxelrodMix + axelrod + axelrod_extra` 合计约 161s，anchor 约 13.8s；
其余规则 family 每场约 1ms。GPU policy 对手的事件计时计入 `policy`，
蓝方推理单独记录在 `gpu_seconds/blue_seconds`。

结论：**Gate 2 的端到端 ≥3× 指标通过**。主要收益来自蓝方学习策略的
全 field 批量；规则/Axelrod 仍是 CPU lock-step，后续计划 3 仍可继续降低
这部分成本。

> 复现：`--workers 20 --gpu-workers 8`（12 CPU + 8 GPU）。不要用 20 个
> CUDA context：本机 31GB 内存会在 GPU 任务切换时 OOM。

## 4. Parity 与 float32 边界

`tests/test_gpu_parity.py` 覆盖：

- `RL-Repair`、`GRU`、`SelfPlay-GRU`、`Adv-MLP`、`PSRO-Scratch`
  对同一 test_v7 field 的 CPU/GPU 逐场 action 与 score 完全一致；
- 多 field batch 与单 field 结果一致；
- `--gpu on` 的 CLI 结果与 `--gpu off --mode argmax --opponent-mode argmax`
  的 `values` 完全一致；
- sample 模式 `--gpu on` 与 `--gpu off` 结果一致（完全走 CPU）。

float32 路径复核（`bench/bench_gpu_parity.py --n 512`，
`results/protocol_v4/plan2_gpu_parity.json`）：

| 候选 | blue action 差异 | 对手 action 差异 | score 差异 |
|---|---:|---:|---:|
| RL-Repair | 0 | 0 | 0 |
| GRU | 0 | 0 | 0 |
| SelfPlay-GRU | 0 | 0 | 0 |
| Adv-MLP | 0 | 0 | 0 |

n=512 上没有观察到 float32 边界翻转。生产上仍建议对接近 0 的 logit 保留
float64 复核开关。

## 5. 已知限制与决策

- **Final/Triple 留在 CPU。** 它们是随机规则策略；GPU 批量会改变 sample
  RNG 顺序。按 2.6 的方案 A，不为它们增加 GPU 实现。它们占 test_v7 的
  policy family 约 15%，端到端影响小。
- **PSRO-Meta 回退 CPU。** 主 PSRO population 含 Final/Triple；只有
  PSRO-Scratch 的全学习 population 走 GPU。
- **规则对手仍是 CPU lock-step。** Plan 3 的 RuleBank/C++/GPU 规则 batch
  才能进一步降低这部分；Plan 2 的收益主要来自蓝方和 policy 对手。
- **多 field 的 PSRO RNG。** `--gpu-field-batch > 1` 时 PSRO 的成员抽样
  流会跨 field 连续；若需要与逐 field CPU 完全一致，使用
  `--gpu-field-batch 1`。普通学习候选不受影响。
- **显存/主机内存。** 20 个 worker 各自持有 CUDA context 会在本机 31GB
  内存上触发 OOM；生产配置使用 `--gpu-workers 8`，让 CPU/GPU 任务分池。
  如果显存紧张，再降低 `--gpu-max-batch` 或 `--gpu-field-batch`。

## 6. 复现

```bash
# 组件基准
python3 bench/bench_gpu_policies.py --cpu-batch 512 \
  --batches 2048,8192,32768,61440

# GPU parity 测试
python3 -m unittest tests.test_gpu_features tests.test_gpu_policies \
  tests.test_gpu_parity -v

# 全量测试
python3 -m unittest discover -s tests -t . -v
```

## 7. 验收清单

| 验收项 | 结果 |
|---|---|
| `test_v7 all24` 端到端 ≥3×（相对 20-worker CPU） | **通过**：714.97s vs 3052.42s = **4.27×** |
| 蓝方单组件 ≥30× | **通过**：B=2048 时 54.6×/97.6×/147.6×/2229.9× |
| SelfPlay-GRU 微基准 ≥100×（B≥8192） | **通过**：B=8192 时 10984.7× |
| policy 对手组件 ≥30× | **通过**：复用同一批 MLP/GRU/SelfPlay kernel |
| argmax/float64 0 动作差异 | **通过**：RL/GRU/SelfPlay/Adv/PSRO-Scratch |
| sample 保持 CPU 且结果一致 | **通过**：`--gpu on --mode sample` 与 `--gpu off` 逐位一致 |
| CPU fallback 与旧实现一致 | **通过**：GPU 异常重走 `evaluate_field_details`；小规模 CLI 测试通过 |
| 256 tests + 新测试全绿 | **通过**：272 tests，107.3s，OK |

## 8. Plan 3/4 交接提醒

### 8.1 当前状态

- Plan 2 代码、测试、组件基准、`test_v7 all24` 全量基准均已完成；
- Gate 2 通过：714.97s vs 3052.42s = **4.27×**；
- GPU 路径只支持 **argmax**；sample 仍完全走 CPU，RNG 契约未被修改；
- 全量测试 **272 tests OK**（`107.3s`，venv Python）。

### 8.2 关键接口

- CPU 老路径：`protocol_cli._eval_field_task`；
- GPU 多 field 路径：`protocol_cli._eval_field_batch_task`；
- GPU 模拟器：`ipd/evaluation/gpu/simulator.py:evaluate_fields_gpu`；
- 蓝方适配：`ipd/evaluation/gpu/blue.py:GpuBlueField`；
- policy 对手：`ipd/evaluation/gpu/policy_opponents.py:BatchOpponentField`；
- 结果 schema 仍是 `ipd-protocol-eval-v1`，新增字段不删除旧字段。

### 8.3 给 Plan 3（规则批处理）

> **Plan 3 已完成（2026-10-28）：** C++/ctypes 规则 bank 覆盖 98.8%
> 规则 config；规则主导评估 7.55×，规则训练 B=2048 3.04×；
> `test_v7` 因 policy/Axelrod 主导，`--fast-rules auto` 保持 CPU。
> 详见 [plan3-fast-opponents.md](plan3-fast-opponents.md)。

- 当前 GPU 任务的剩余成本主要是规则/Axelrod 的 Python lock-step callback；
  5-field profile 中 `policy` 814s，`AxelrodMix + axelrod + axelrod_extra`
  约 161s，anchor 约 14s。
- 优先做 RuleBank/C++ spike；用同一批 `2048×100` 场比较 C++/Numba/GPU
  RuleBank，计入 field build、传输和 Python 开销。
- 先覆盖训练 rollout 高频 family；评估侧的快速规则候选收益有限。
- 如果 RuleBank 接入，保持 `BatchOpponentField` 的 CPU fallback 和名称/
  顺序；不要让规则批处理改变 `by_opponent` 聚合。
- 不要为了扩大 GPU 范围修改 sample RNG 契约；那需要 Protocol v5。

### 8.4 给 Plan 4（训练 rollout）

- 先跑 `4.1` profile；如果 learner forward/update 才是瓶颈，转向
  `torch.compile`/混合精度，而不是继续 port 对手。
- Plan 2 的 `BatchMLP/BatchGRU/BatchSelfPlayGRU` 可直接复用为评估侧
  fast evaluator；训练环境仍需要独立的 GPU VectorEnv。
- 训练 rollout 的 batch/RNG 语义与 Protocol v4 评估不同，必须单独定
  parity/曲线验收，不要复用本计划的 sample 结论。

### 8.5 操作注意

- 必须用仓库 venv：`.venv/bin/python3`；系统 `python3` 没有 torch/numpy/
  axelrod。
- 20 个 CUDA context 会在 31GB 内存机器上 OOM；使用
  `--workers 20 --gpu-workers 8`（12 CPU + 8 GPU）。
- `test_v7` 目前是锁定/临时消费状态；全量基准必须带 `--no-consume`。
- 全量测试若偶发 `BrokenProcessPool`，先检查内存和残留 GPU 进程，
  单独重跑该测试；本次最终全量重跑为 272 tests OK。
- 复现基准：
  - 组件：`bench/bench_gpu_policies.py --cpu-batch 512 ...`
  - 端到端：见 §3 命令
  - profile：§3.1 的 5-field 命令加 `--profile`
