# Plan 3 基准与实现：规则/复杂对手批处理

- **状态：** 实现完成；Gate 3A 选型完成（2026-10-28）
- **代码：** `ipd/evaluation/fast/`（`rules.cpp`、`native.py`、`bank.py`、`spec.py`），
  `ipd/evaluation/gpu/policy_opponents.py` 的规则路由，
  `ipd/training/selfplay/env.py` 的 `VectorEnv` 接入
- **测试：** `tests/test_fast_opponents.py`；全量 284 tests 绿
- **基准：** `bench/bench_fast_opponents.py`；
  结果在 `results/protocol_v4/plan3_fast_opponents.json`
- **协议：** Protocol v4；`test_v7` seed=20261027；`dev_complex` n=2048；
  C++ 用系统 `g++ -O3` 现场编译，无 pybind11/CMake/Numba 依赖

## 1. 决策：C++/ctypes，而不是完整 GPU RuleBank

Plan 3 要求先比较 C++ 与最佳 GPU 实现。当前环境没有 pybind11、CMake 或
Numba，GPU 原型也只存在日志中；因此 spike 用：

- 一个原生 C++ 批量解释器（`rules.cpp`，通过 ctypes 调用）；
- 一个 Torch per-round kernel（`ipd/evaluation/fast/gpu_spike.py`），
  同时实现 TFT 和 HarvestFactor，并用 mask 跑异构 batch。

同一组 2048×100 场、50% TFT + 50% HarvestFactor、计入 setup/传输：

| 实现 | 时间 | 相对 Python |
|---|---:|---:|
| Python 规则 | 0.0230s | 1.00× |
| C++/ctypes | 0.0042s | **5.46×** |
| Torch GPU（预热后） | 0.0167s | 1.38× |
| Torch GPU（首次） | 0.2242s | 0.10× |

C++ 是 GPU 的 **3.96×**，超过 Gate 3A 的 2× 门槛；因此选择 C++，
不继续做全量 GPU 规则移植。GPU 仍然负责蓝方学习策略和 policy 对手
（Plan 2），规则对手留在 CPU。

> GPU 规则 kernel 每轮一次 launch；100 轮的小 kernel 被 launch/同步开销
> 主导。C++ 在同一个进程内循环 100 轮，正好避开这部分成本。

## 2. 实现摘要

### 2.1 原生规则 bank（3.1–3.3）

`rules.cpp` 覆盖 71 个规则类型/别名，包括：

- 标准：AllC/AllD/TFT/Grim/GTFT/TFT2/TFT-lastD/Pavlov/ContriteTFT/
  WSLS2/TFTForgiveOnce/TFT2-lastD/TTFT/sTFT/Majority/TwoStrike/
  ForgiveTFT/TFTNoisyForgive/GDK/ExtortZD/Grudger2/FirmButFair/Mistrust/
  Gradual/GrudgerK/GrimForgive/ProbabilisticGrim/ResetGrim/SpitefulGrim/
  DelayedPunisher/Bully/Prober/Tester/RemorsefulProber/EarlyD/LateD/
  Random/Periodic/MemoryOne/Reactive/Responsive/2/3/ProbeBot 等；
- 复杂：PhaseSwitcher、AdaptiveMix、ProbeExploit、DelayedExtort、
  MemoryTwo、MetaMix、NoiseAdaptive、ReputationGrim、ContritionCycle、
  EndgameHunter、PatternBreaker、ConditionalExtort、LookaheadOpponent、
  VotingPortfolio；
- test-only：HarvestFactor、Kamikaze、ProbePhase、TFTProbe、
  AdaptiveProbe、ExtortZD；
- dev-only：MirrorAdaptive、DelayedGrim。

覆盖统计：

| 口径 | 覆盖 |
|---|---:|
| Protocol v4 规则 config（排除 Axelrod/policy） | **719 / 728 = 98.8%** |
| `test_v7` 规则实例（按权重） | **99.4%** |
| `test_v7` 非 Axelrod 实例（含 policy，policy 由 Plan 2 处理） | **99.5%** |
| 未覆盖 | `RandomFSM`（0.44%）、`ProbeBot`（train/dev 少量） |

`RandomFSM` 和 `ProbeBot` 保留 Python 对象；`RuleBank` 的 fallback 逐行
调用原实现，名字、顺序、分数聚合都不变。

### 2.2 RNG 契约：逐位一致，而不是“分布等价”

原生 bank 捕获每个对象的 CPython `random.Random` 状态（624 个 MT19937
word + index），在 C++ 内复刻 CPython 的 `random()`、`getrandbits()` 和
`_randbelow()`。因此：

- 支持的确定性规则逐动作一致；
- 支持的**随机**规则也逐动作一致（Joss、GTFT、NoisyTFT、MemoryOne、
  ExtortZD、Responsive、VotingPortfolio 等）；
- 测试对 719 个 config × 100 轮做逐动作比对，0 差异；
- `RuleBank.reset()` 恢复构建时捕获的初始 RNG，与 Plan 1 的 field 复用
  语义一致。

这比 Plan 3 的最低要求（随机模式只做分布检验）更强，也不需要 per-match
seed 或 Protocol v5。

### 2.3 修复复合规则的 RNG 共享 bug

实现 parity 时发现 `VotingPortfolio`/`PhaseSwitcher`/`AdaptiveMix`/
`ProbeExploit`/`MetaMix` 把 `rng` 作为 `HarvestFactor` 的第 4 个位置参数
传入；该参数实际是 `last_d`，导致子规则拿到一个新的 OS-seeded RNG。
`VotingPortfolio` 的 harvest 成员因此与 adaptive 成员使用不同 RNG 流，
旧结果不可复现。

已改为 `rng=` 关键字，并记录在
[parity-notes.md](parity-notes.md)。这是可复现性修复；受影响的
VotingPortfolio 等概率规则本来就没有稳定基线。

### 2.4 混合调度（3.4）

- **评估**：`BatchOpponentField` 先按 `(类型, 权重路径)` 路由学习型
  policy 对手；其余有 `_fast_spec` 的行进入 `RuleBank`；Axelrod/
  RandomFSM/ProbeBot 走 CPU fallback。
  - `--fast-rules auto`（默认）只在 fast 行 ≥90% 时启用；
  - `test_v7` 的规则行约 73%，且成本被 policy/Axelrod 主导，auto 会
    保持原 CPU 路径，避免回归；
  - `dev_complex` 100% 规则行，auto 启用，实测 **7.55×**；
  - `--fast-rules on/off` 可强制。
- **训练**：`VectorEnv` 在 `envs >= --fast-rules-min-envs`（默认 128）
  时构建 `RuleBank`，用一次 `rb_step` 完成 act+observe；小 batch 保持
  Python 路径，因为 ctypes 每轮开销大于收益。默认训练 `envs=32` 不会
  启用；Plan 4 放大 B 后收益才出现。
- **聚合**：`by_opponent`、family 归一化、field 顺序、名字均不变。

### 2.5 性能基准（3.6）

环境：24 核 CPU / 31GB RAM / RTX 5090 Laptop 24GB / torch 2.11+cu128 /
Python 3.14.8。

命令：

```bash
python3 bench/bench_fast_opponents.py \
  --out results/protocol_v4/plan3_fast_opponents.json
```

结果（`--rounds 100`）：

| 场景 | Python | 原生 | 加速 |
|---|---:|---:|---:|
| spike 2048×100，TFT+Harvest | 0.0230s | 0.0042s | **5.46×** |
| `dev_complex` n=2048（全规则） | 0.0800s | 0.0106s | **7.55×** |
| `test_v7` n=2048（含 policy/Axelrod） | 8.50s | 9.20s | 0.92×（auto 不启用） |
| 训练规则 B=128 | 1.56ms | 0.92ms | 1.69× |
| 训练规则 B=512 | 6.67ms | 2.32ms | 2.88× |
| 训练规则 B=2048 | 26.95ms | 8.88ms | **3.04×** |
| 训练全 mix B=2048（含 top policy） | 2.46s | 2.75s | 0.89× |

结论：

- 规则批处理在**规则主导**的评估和训练中有效（3–7.5×），但
  `test_v7` 评估的成本被 policy/Axelrod 主导，不能承诺 6–10×；
- 训练全 mix 被 `top` 学习型对手主导，规则 bank 不能替代 Plan 2/4；
- 小 batch（B=32）ctypes 开销大于原生收益，默认不启用；
- 这正是 Plan 3 预先声明的“评估增量小、主要收益在训练 rollout”的
  实测结论。

## 3. 接入方式

### 评估

```bash
python3 -m ipd protocol eval --candidates rl-repair --tiers dev_complex \
  --fields 1 --n 2048 --workers 1 --mode argmax --gpu on \
  --fast-rules auto --no-consume
```

结果 JSON 增加：

```json
"fast_rules": {"available": true, "requested": "auto",
               "reason": "...", "auto_threshold": 0.9}
```

`--fast-rules off` 完全走 Plan 1/2 原路径；`auto` 在规则占比低时自动
关闭，保证 `test_v7` 不回归。

### 训练

```bash
python3 -m ipd train selfplay --config configs/selfplay/full.json \
  --envs 512 --fast-rules --fast-rules-min-envs 128
```

`--no-fast-rules` 关闭；默认 `fast_rules=true`，但 `envs<128` 不构建 bank。

## 4. 验收清单

| 验收项 | 结果 |
|---|---|
| ≥95% 非 Axelrod 实例走 fast path | **通过**：`test_v7` 实例 99.5%（policy 由 Plan 2）；规则 config 98.8% |
| 每个 family 确定性 mode 0 差异 | **通过**：719 configs × 100 轮逐动作 0 差异；随机规则也逐位一致 |
| 训练 rollout opponent-step 目标 | **实测后调整**：规则 B=2048 3.04×，B=128 1.69×；全 mix 被 policy 主导（0.89×），不虚报 5–10× |
| 评估侧不回归 | **通过**：`auto` 在 `test_v7` 关闭，结果与 Plan 2 一致；`dev_complex` 7.55× |
| 快速规则候选有可测量增量 | **通过**：`dev_complex` 7.55×；规则 B=2048 3.04× |
| 不承诺 test_v7 6–10× 来自规则批处理 | **遵守**：文档明确说明 |
| Axelrod 仍 CPU，结果不变 | **通过**：未注册 fast spec，走原对象 |
| 256 tests + 新测试全绿 | **通过**：284 tests，111.2s，OK |

## 5. 已知限制与回退

- **RandomFSM / ProbeBot** 没有原生实现；fallback 到 Python，结果不变。
- **Axelrod / policy** 仍走各自路径；Plan 2 的 GPU policy 对手不受影响。
- **小 batch**：B<128 时 ctypes/每轮开销大于收益，`VectorEnv` 不启用。
- **评估 full test_v7**：规则 bank 全开时 0.92×；`auto` 自动关闭，避免回归。
- **多进程**：每个 worker 独立编译/加载同一 `.so`；首次编译约 0.1s，
  之后命中 `.build/fast_rules/` 缓存。
- **无 C++ 编译器**：`available()` 返回 false，所有调用回退 Python；
  `--fast-rules on` 会打印警告。

## 6. 复现

```bash
# 全量基准
python3 bench/bench_fast_opponents.py \
  --out results/protocol_v4/plan3_fast_opponents.json

# 只跑规则/复杂场景
python3 bench/bench_fast_opponents.py \
  --scenarios spike,eval_rule,training_rules

# Plan 3 测试
python3 -m unittest tests.test_fast_opponents -v

# 全量测试
python3 -m unittest discover -s tests -t . -v
```
