# Plan 4：训练 rollout GPU 化与长期优化

- **状态：** 实现完成（2026-10-28）；Gate 5 部分通过
- **代码：** `ipd/training/selfplay/gpu_env.py`，
  `ipd/training/selfplay/rollout.py` 的 `TensorRollout`，
  `ipd/training/selfplay/train.py` / `ppo.py` 的接入；
  `train selfplay --gpu-env auto|on|off`
- **测试：** `tests/test_gpu_env.py`；全量 296 tests 绿
- **基准：** `bench/bench_training.py`；
  结果在 `results/protocol_v4/plan4_training.json`，
  profile 摘要另存 `results/protocol_v4/plan4_profile.json`
- **环境：** 24 核 CPU / 31GB RAM / RTX 5090 Laptop 24GB /
  torch 2.11+cu128 / Python 3.14.8；`torch.set_num_threads(1)`

## 1. Gate 4.1：先 profile，再决定改哪里

`bench_training.py --scenarios profile` 把一次 100 轮 rollout 拆成
observe（21 维特征）、learner forward、sample、env/opponent step、store，
再单独测 GAE 和 PPO update。CPU 与 GPU 都先各跑一次 warmup，
避免把首次 CUDA/cuDNN 初始化算进 KPI。

| B | 路径 | rollout | observe | forward | sample | env step | store | update |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | CPU | 0.128s | 15.1% | 24.0% | 19.2% | **32.5%** | 8.4% | 0.027s |
| 32 | GPUEnv | 0.238s | 19.0% | 11.5% | 3.3% | **65.8%** | 0.0% | 0.025s |
| 128 | CPU | 0.348s | 21.6% | 10.6% | 7.4% | **55.7%** | 4.3% | 0.099s |
| 128 | GPUEnv | 0.288s | 16.2% | 10.1% | 2.8% | **70.5%** | 0.0% | 0.096s |
| 512 | CPU | 1.123s | 27.2% | 5.2% | 3.2% | **61.8%** | 2.2% | 0.392s |
| 512 | GPUEnv | 0.426s | 11.1% | 7.5% | 2.1% | **79.0%** | 0.0% | 0.374s |
| 2048 | CPU | 4.366s | 28.0% | 2.2% | 1.6% | **66.3%** | 1.0% | 1.611s |
| 2048 | GPUEnv | 0.892s | 5.4% | 4.3% | 1.1% | **89.1%** | 0.0% | 1.540s |

结论与决策：

- **生产 batch（`envs=32`）的 opponent step 只占 32.5%，低于 40% 门槛。**
  因此不按 Plan 4 的“为同 B 重写 GPU VectorEnv”继续投入；同 B 的
  rollout 反而从 0.140s 退到 0.242s（0.58×），瓶颈在 learner/PPO update
  以及 Python 对手的 lock-step 同步。
- **B≥128 时 env/opponent step 成为主瓶颈（55–66%）**，GPUEnv 开始有收益：
  B=128 1.21×、B=512 2.63×、B=2048 4.90×。
- 这正是计划 4.1 要求的“低于 40% 就记录不继续”的分支；GPUEnv 只作为
  **大 B 训练路径**交付，`--gpu-env auto` 默认在 `envs>=128` 才启用，
  小 B 保持原 CPU 路径。
- 对 learner 本身，额外实测了 `torch.compile(model.step, dynamic=True)`：
  B=32 为 0.0186s/100 步，eager 为 0.0186s；没有可测收益，因此不把它
  作为 Gate 5 的补救项。

## 2. GPU VectorEnv（4.2）

`ipd/training/selfplay/gpu_env.py::GPUEnv` 与 `VectorEnv` 保持相同的
`observations()` / `step()` / `scores` / `done` 接口，同时提供
`observations_tensor()` / `step_tensor()` 零拷贝路径。

实现组成：

| 组件 | 实现 |
|---|---|
| 蓝方历史/特征 | `FeatureState21`（Plan 2）；100 轮增量更新，输出留在 GPU |
| 标准规则 | Plan 3 `RuleBank`（C++/ctypes）；每轮一次 `rb_step`，原生行不做 Python 循环 |
| 学习型 policy 对手 | Plan 2 `BatchMLP/BatchGRU/BatchSelfPlayGRU`；按 `(类型, 权重, safe, temperature)` 分组 |
| 冻结 self-play 快照 | `TorchOpponent` 按 `state_ref` 分组，一个 batch 共享权重、独立 hidden |
| 复杂/未知对手 | `Final-tuned`、`Triple-majority`、Axelrod、`RandomFSM` 等保留原 Python 对象，逐行 CPU fallback |
| 收益/分数 | `[B]` torch payoff，own/opponent 分数累加，不再逐行 `payoff()` |
| 终止/重置 | 每轮返回 done mask；`auto_reset=True` 在整批完成后下一次调用自动重置 |

小的 85 维 MLP/GRU policy 组（`rows < policy_gpu_min=256`）放在 CPU 上做
特征和 logits：Plan 2 的 `FeatureState85` 在 60–200 行时是 kernel-launch
bound，CPU 反而更快；`selfplay-21` 和冻结快照始终走 GPU。

RNG 契约：

- 规则对手沿用 Plan 3 的 CPython MT19937 复刻；测试中确定性/随机规则
  都与 CPU 逐动作一致。
- 学习型 policy 与快照采样使用按组固定种子的 torch generator；**训练曲线
  等价，但不是逐位 parity**。`mode="argmax"` 时测试与 CPU 逐动作一致。
- CPU fallback 保存构造时的 `random.Random` 状态，`reset()` 恢复该状态。

## 3. Rollout 存储与 GAE（4.3）

`ipd/training/selfplay/rollout.py` 新增 `TensorRollout`：

- `obs/actions/logprobs/values/rewards/hidden/opponent_rewards` 全部是
  torch tensor，默认就在训练 device；
- `compute_gae()` 用 torch 实现，输出 float32，和 NumPy 版在 1e-5 内一致；
- `tensors()` 不复制；`batches()` 生成 device 上的 `long` 索引；
- `PPOAgent.update` 改为调用 `rollout.batches()`，minibatch 不落 CPU；
- `concat_rollouts` 支持 `TensorRollout`，red 的多 iteration 累积仍可用；
- `to_numpy()` 提供一次显式 host transfer，供日志/`RedConstraint` 使用。

`gpu_rollout` 在返回前显式 `torch.cuda.synchronize()`。如果不同步，计时器
只统计 kernel enqueue 时间；本仓库的 Plan 4 基准全部使用同步后的时间。

## 4. PPO/PSRO 接入与同 B 曲线（4.4）

`train selfplay` 新开关：

```
--gpu-env auto|on|off        # 默认 auto：CUDA 且 envs >= min_batch 时启用
--gpu-env-min-batch 128      # auto 的 batch 门槛
```

- `collect_blue` / `collect_red` 用同一套 `_rollout` 分发；`--gpu-env off`
  与旧路径逐行一致，CPU fallback 可用。
- red 的 `RedConstraint` 仍然在 CPU 上用 Python 列表计算；只是把
  `own/other` 在 rollout 结束时一次性取回 host，避免每轮往返。
- 每个 iteration 的 `history.json` 新增 `timing` 字段：
  `blue_collect_s`、`blue_update_s`、`red_collect_s`、`red_update_s`。

固定 seed 的 10 iteration 同 B 对比（`envs=32`，两个独立子进程）：

| 指标 | 结果 |
|---|---:|
| 每 iteration score 最大差 | 6.375 |
| 每 iteration score 平均差 | 2.241 |
| 每 iteration reward 最大差 | 0.0638 |
| 每 iteration reward 平均差 | 0.0224 |
| 预注册容忍 | max score ≤25、mean ≤10、max reward ≤0.10、mean ≤0.05 |
| 判定 | **通过**（曲线等价） |
| rollout 中位时间 CPU → GPU | 0.140s → 0.242s（**0.58×**） |
| 10 iteration 总训练时间 CPU → GPU | 0.86× |

同 B 的 rollout 没有达到 2–3×。原因与 Gate 4.1 一致：B=32 时 GPUEnv 的
每轮 host 同步和 policy fallback 开销高于 CPU lock-step；learner forward
和 update 才是主项。**这是计划 4.1 明确允许的“不继续”结论，不能记作
GPU 化收益。** 大 B 的收益见下一节。

## 5. 大 B 扩展（Gate 5 主指标）

`bench_training.py --scenarios large_b`，CPU 与 GPU 都先 warmup，GPU 计时
包含末尾同步。`full_mix` 为训练默认六族；`rules` 去掉 `top`
（Final/Triple/RL），对应计划允许的“标准规则/self-play 子集”。

| B | mix | CPU rollout | GPU build | GPU rollout | 加速 | GPU 显存 |
|---:|---|---:|---:|---:|---:|---:|
| 512 | full | 1.128s | 0.032s | 0.420s | 2.68× | 0.08GiB |
| 512 | rules | 0.446s | 0.014s | 0.163s | 2.74× | 0.08GiB |
| 2048 | full | 4.518s | 0.205s | 1.015s | 4.45× | 0.25GiB |
| 2048 | rules | 1.607s | 0.055s | 0.177s | 9.07× | 0.25GiB |
| 4096 | full | 8.935s | 0.370s | 1.831s | 4.88× | 0.49GiB |
| 4096 | rules | 3.171s | 0.111s | 0.212s | **14.97×** | 0.49GiB |

- **支持子集在 B=4096 达到 14.97×，满足 ≥10×**；B=2048 为 9.07×，
  接近门槛。
- **full mix 上限约 5×**。原因是 `Final-tuned`/`Triple-majority` 的
  Python CPU fallback（计划 4.2 明确允许），以及它们随 batch 线性增长；
  Axelrod 同样留在 CPU。该数字单独记录，不用支持子集的 14.97× 代替。
- 最大显存 0.49GiB（B=4096，包含 rollout storage），远低于 20GiB；
  B、T 可继续放大。

## 6. 可选 fused kernel / CUDA Graph（4.5）

**不实现。** 依据：

- Plan 3 Gate 3A 已实测 C++ 规则 bank 是 Torch GPU 规则 kernel 的 3.96×，
  规则侧继续 GPU 化没有收益；
- Gate 4.1 显示同 B 是 learner/update 主导，环境不是瓶颈；
- `torch.compile(model.step)` 实测无可测收益；
- 100 轮 loop 中有 CPU 对手 fallback，无法整段 capture CUDA Graph。

如果以后训练分布变成纯 21 维规则/self-play 且 B≥8192，可再把
`GPUEnv` 的固定 B 段做成 CUDA Graph；当前没有数据支持投入。

## 7. 评估复用（4.6）

- `GPUEnv` 直接复用 Plan 2 的 `FeatureState21/FeatureState85` 和
  `BatchMLP/BatchGRU/BatchSelfPlayGRU`，以及 Plan 3 的 `RuleBank`；
  训练 rollout 与评估不再各写一套规则/特征实现。
- 训练中的 validation/final test 仍走 `evaluate_model` 的 CPU
  `VectorEnv` sample 路径。原因：validation 需要与历史 checkpoint 的
  sample 分布可比，Plan 2 也只承诺 argmax GPU parity；把 validation
  切到 torch RNG 会改变历史曲线。若将来启用，只需在 `evaluate_model`
  增加 GPUEnv 分支，规则/特征层已经共享。
- 因此 4.6 记为 **部分完成**：核心 kernel 已复用，validation 的 GPU
  路由未默认开启。

## 8. 验收状态

| 验收项 | 状态 | 证据 |
|---|---|---|
| profile 显示 env/opponent 是主要瓶颈；否则记录“不继续” | **通过** | B=32 opponent 32.5% <40%，记录同 B 不继续；B≥128 55–66% |
| 同 B rollout ≥2–3× | **未达** | 0.140s → 0.242s = 0.58×；曲线等价通过 |
| 放大 B rollout ≥10× | **支持子集通过，full mix 未达** | rules B=4096 14.97×；full mix 4.88×（CPU fallback 限制） |
| 显存 <20GB | **通过** | 最大 0.49GiB（B=4096） |
| CPU fallback 可用 | **通过** | `--gpu-env off`、`--device cpu`、GPU 异常路径 |
| 296 tests 全绿 | **通过** | `tests/test_gpu_env.py` 12 tests + 原 284 |

## 9. 风险与后续

- **不要把 GPUEnv 用于小 B。** `--gpu-env auto` 在 `envs<128` 关闭；
  强行 `on` 会回归（B=32 约 0.58×）。
- **full mix 的 CPU fallback 是硬上限。** 若要继续提升，优先把
  `Triple-majority` / `Final-tuned` 做成可批处理的实现或 native 规则，
  而不是继续优化规则 bank；Axelrod 不在训练主路径。
- **训练动力学：** 大 B 需要重新调 lr/epochs/batch；本次只验证
  rollout 吞吐和同 B 曲线，不把大 B 曲线当作“免费加速”。
- **调试：** `--device cpu` 和 `--gpu-env off` 保留逐场路径；
  `history.json` 的 `timing` 字段可直接定位 collect/update 占比。
- **性能记录：** 笔记本 GPU 受 CPU 负载/功耗影响；本文件所有 GPU 数字
  均在 warmup 和 `torch.cuda.synchronize()` 后记录，并附带 GPU 型号与
  torch 版本（见 JSON 的 `gpu` 字段）。
