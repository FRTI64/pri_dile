# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/)；日期为 UTC。
未发布改动放在 `Unreleased`。

## [Unreleased]

### Added

- **RL-Wide（`rl-wide`）**：用新增的 `python -m ipd train expand`
  （net2wider）把 RL-Pattern 的 64/32 trunk 扩到 128/64、128/128，
  再做 128/64 gentle → 128/128 gentle → 128/128 balance（PB 1/11）
  的小步 PPO fine-tune。全量 dev（30×2048，sample）：dev_param 548.24、
  dev_family 583.08、dev_complex **452.79**、dev_axelrod 582.30、
  dev_ood **520.71**；complex min_norm **0.973**、ood min_norm
  **0.960**（RL-Pattern 0.845/0.842）。`test_v7` provisional 485.50
  vs RL-Repair 486.31（未消费、`selection_eligible=false`）。注册为
  实验候选，**未 promotion**；权重
  `weights/rl_policy_weights_wide.json`，报告
  `docs/research/rl-wide.md`、`results/rl_v4_20261029/`。
- **`train expand`**：`ipd/training/expand.py`，支持
  `ipd-rl-policy-v1` / `ipd-neural-policy-v1` 的 net2wider 扩展；
  `tests/test_expand.py` 覆盖形状、函数近似与确定性。
- **RL-Wide 验证与宽度消融**：独立 seed 20261029 复现 dev_param
  546.60 / dev_complex 452.05 / dev_ood 522.14；`dev_red` 仍
  red_win 100%；192/128 消融 dev_complex 459.57，但 dev_param
  547.44、test_v7 484.90 回退，不注册。报告
  `results/rl_v4_20261029/`（复现/red）和
  `results/rl_v4_20261030/`（192/128）。
- **RL-Pattern（`rl-pattern`）**：从 RL-Repair + entropy v4 fine-tune
  出发，混入 `train_v4` 与新的 `train_pattern_breaker` profile，
  修复 Protocol v4 `pattern_breaker` 漏洞。全量 dev（30×2048，sample）：
  dev_param **548.45**（RL-Repair 547.96）、dev_complex **447.64**
  （434.28）、dev_ood **519.46**（515.19）、dev_axelrod 582.22
  （582.46）、dev_family 583.07（582.76）；`pattern_breaker`
  236.8 → 420.3。`test_v7` provisional 485.43 vs RL-Repair 486.31
  （未消费、`selection_eligible=false`）。注册为实验候选，**未
  promotion**，主权重不变。权重
  `weights/rl_policy_weights_pattern.json`，实验报告
  `docs/research/rl-pattern.md`、`results/rl_v4_20261028/`。
- **定向训练 profile**：`train_pattern_breaker`（仅 pattern_breaker
  train_complex 参数）和 `train_weak_complex`（pattern_breaker /
  adaptive_mix / probe_exploit / conditional_extort / lookahead）；
  `tests/test_protocol.py` 增加 train-tier 泄漏检查。
- **Plan 6 Python 瓶颈优化**：v5 sample 的学习型 policy 对手在
  `BatchOpponentField` 中按权重分组，逐对象预抽 uniform 并批量上 GPU；
  CPU/GPU 与 Plan 5 v5 baseline **逐位一致**。Axelrod adapter 改为增量
  维护 `History`（O(t²)→O(t)，224 策略逐动作 parity，family 2.5–3.1×）。
  新增严格同序 C++ CPU policy 前向（`ipd/policies/policy_ops.cpp`，
  Neumaier `sum()`、同分支 sigmoid、禁用 FMA/重排），MLP/GRU/
  SelfPlay 加速 4.6–18× 且逐位一致。新增 `--fast-rules-threshold` 与
  `--gpu-workers auto`。`Adv-*` 与 PSRO meta 保持 CPU，避免改变 v5
  分数。exact 模式下 `test_v7` v5 从 1365.63s 降到 **272.77s**
  （5.01×，24/24 候选逐位一致）；dev argmax 仅作迁移期加速诊断，
  不进入长期 gate；307 tests 全绿。文档
  `docs/research/plan6-python-bottleneck.md`。
- **Plan 5 GPU sample（opt-in）**：`protocol eval --sample-rng per-match`
  使用 per-match RNG 契约，把学习型蓝方的 sample 推理批量上 GPU；
  CPU/GPU 逐场 parity（`tests/test_gpu_parity.py`），默认 `v4` 不变。
  结果标记 `sample_rng=per-match` / `gpu_sample=true` / `protocol_version=5`；
  当时 `Adv-*` 与 PSRO meta-policy 仍走 CPU（Plan 6 已把 `Adv-*`
  接入 v5 GPU sample）。文档 `docs/research/plan5-gpu-sample.md`。
- **Plan 4 训练 rollout GPU 化**：`ipd/training/selfplay/gpu_env.py`
  （GPU 历史/21 维特征/收益，Plan 3 C++ 规则 bank，Plan 2 batch policy，
  `TorchOpponent` 分组，复杂对手 CPU fallback）；`rollout.py` 新增
  `TensorRollout`（GPU GAE/device 索引）；`train selfplay` 新增
  `--gpu-env auto|on|off` / `--gpu-env-min-batch`；`history.json` 增加
  timing；`bench/bench_training.py`、`tests/test_gpu_env.py` 与
  `docs/research/plan4-gpu-training.md`。Gate 5 部分通过：同 B 曲线通过
  但 rollout 0.58×；大 B rules 14.97×（B=4096）、full mix 4.88×；
  显存 0.49GiB；296 tests 全绿。
- **Plan 1 CPU 评估管线**：`protocol eval` 支持 `--workers auto`、
  `--candidate-chunk`（默认 6）、`--field-reuse auto|on|off` 和
  `--profile`；新增 `ipd/evaluation/scheduler.py`（worker/RAM 预算、进度
  ETA、per-family 剖析、RSS/PSS 采样）；field 复用前保存/恢复对手与学习策略
  RNG 快照，保持逐位一致；新增 `tests/test_protocol_scheduler.py` 与
  `docs/research/plan1-benchmark.md`。全量 `test_v7` all24：3052.42s
  （0.85h，2.97×），确定性 parity 通过，256 tests 全绿；Gate 1 性能目标
  0.5–0.6h 未达（混合 CPU 扩展不足 4×）。
- **PSRO**：`ipd/training/psro.py`（blue-vs-red 元博弈、maximin、
  PPO best-response oracle、population/report/checksum）；`ipd/policies/psro.py`
  （meta-policy，每场按 blue maximin 采样）；候选 `psro` / `psro-br`；
  `configs/psro/`；`results/psro_20261027/`；测试 `tests/test_psro.py` /
  `tests/test_psro_policy.py`。2 轮结果：blue support 为 Final-tuned +
  Triple-majority，oracle 未进入 support；dev_param 527.71 vs RL-Repair
  543.35，held-out Red-v5/v6 red-minus-blue -21.4%（blue +25.4），
  **未 promotion**。
- **从零 PSRO**：`configs/psro/psro_scratch_20261027.json`；blue 只有
  random init + 从零官方 base + 从零 BR，现有模型/规则只做 red；
  候选 `psro-scratch` / `psro-scratch-br`；结果
  `results/psro_scratch_20261027/`。dev_param 480.43 / min_norm 0.5501，
  但 dev_red 与 held-out Red-v5/v6 **red_win 0.0**（首次翻转）；
  **未 promotion**。
- **Protocol v4**：977-config catalog；224 个 Axelrod 4.14.0 策略 + 15 个
  `AxelrodMix` 组合；6 个新复杂风格 family（noise_adaptive /
  reputation_grim / contrition_cycle / endgame_hunter / pattern_breaker /
  conditional_extort）+ 2 个高阶 style（lookahead / portfolio），并为旧
  family 增加新对手；新增 `train_axelrod` / `dev_axelrod` /
  `test_axelrod`；每个 entry 带 `thinking_level` / `complexity` 0–4，
  提供 `dev_level0..4` / `dev_complexity0..4` 和对应训练 profile；
  默认 2048 人 / dev 30 fields / test 50 fields；test tier 为 `test_v7`。
  test 消费永久锁定：`configs/splits/protocol_v4/test_policy.json`
  （`consume_test=false`）；CLI 无覆盖开关，test 运行必须显式传
  `--no-consume`，一律 provisional、不写 usage ledger，结果标记
  `selection_eligible=false`。
  v4 baseline 尚未运行。
- **Protocol v3**（legacy）：`ipd/evaluation/protocol.py` + `python3 -m ipd protocol`，
  532-config catalog、6 个 complex family、10 个 policy opponent；
  `train` / `dev_param` / `dev_family` / `dev_complex` / `dev_ood` /
  `test_v6` split、manifest、泄漏审计和 test usage ledger。
- `configs/splits/protocol_v3/manifest.json`（532 configs）和
  `results/protocol_v3/` dev/test baseline。
- baseline 结果：RL-Repair test 484.32 > A60 476.54 > Adv-RL 471.44；
  Red-v5/v6 仍 100% 击败 top 3。
- 训练 profile：`train_v3` / `train_easy` / `train_hard` / `train_complex` /
  `train_policy`；validation 支持 `dev_param` / `dev_family` /
  `dev_complex` / `dev_ood`。
- 红队评估：`protocol red --tier dev_red|test_red`。
- Protocol v2（224 configs、test_v5）保留在 `results/protocol_v2/` 作为 legacy。
- `LICENSE`（MIT，与 `pyproject.toml` 一致）。
- 文档重组为现代工程结构：`architecture/`、`guides/`、`reference/`、
  `policies/`、`research/`、`operations/`、`history/`、`decisions/`。
- 根 README 精简为项目入口；新增 `CONTRIBUTING.md`、`CHANGELOG.md`、
  文档索引和 ADR。
- 统一当前候选表为 E0/E1c 冻结协议（20 fields），旧 10-field 结果标为历史基线。
- 更新 `docs/operations/handover.md`：RL-Repair 为最佳候选但未 promotion；
  `heldout_v3` 已使用，`heldout_v4` 未使用。

### Changed

- 注册候选从 20 增至 26（新增 `rl-pattern` / `rl-wide`；此前 24 含
  4 个 PSRO）；完整测试 **312 OK**。
- Protocol v4 top 4（RL-PPO / RL-Repair / RL-Pattern / RL-Wide）的 dev
  已运行；RL-Wide 综合最均衡，但 test_v7 provisional 仍低 0.81，
  暂不 promotion。
- `official` / `blind` 标记为 legacy in-distribution 诊断；新候选比较使用
  Protocol v4 dev/test，旧表不再作为 held-out 证据。
- `heldout_v2/v3/v4` 与 Protocol v2/v3 的选模流程被 v4 取代；历史结果保留为
  legacy。

### Fixed

- `protocol eval` 在“全部候选都走 GPU、没有 CPU task”且显式设置
  `--gpu-workers` 时，局部变量 `budget` 覆盖 worker 预算字典，导致
  `TypeError: 'int' object is not subscriptable`；现在改用独立的
  `gpu_budget`。
- `Reactive.F0..F7` 工厂在每次调用时继续消耗共享 RNG，导致对手参数随
  候选数量和顺序变化；现在改为构造时只抽样一次。回归测试：
  `tests/test_core.py::OpponentFactoryTests`。这会使 34cb9b0 中这 8 类
  对手的旧结果不可复现，属于修正而非回归。
- 修正 `heldout_v3` 的过期状态、17/20 候选数量和 `--arch` 旧命令说明。
- 修正 C++ `AdvRLParams` 示例为当前 aggressive 配置。
- 修正文档间相对链接，移动后链接检查 0 broken。

## [2026-10-27]

### Added

- 可选 Axelrod 对手库：`axelrod`（243 普通）、`axelrod-fast`（224，剔除
  long-run）、`axelrod-cheating`（4.14.0 仅 Darwin）。
- `requirements-axelrod.txt`；适配器延迟导入，默认安装仍零第三方依赖。
- Axelrod benchmark / matchup / eval-split 支持；224 fast 和 243 smoke 结果。

### Changed

- 默认四分布、85/158 对手集合和主权重保持不变。

## [2026-10-26]

### Added

- E0 冻结 league harness：cross-play、Nash averaging、Elo、exploitability、
  validation / heldout_v2、E4 red。
- E1 `robust` / `repair` 训练分布；E1c `rl-repair` 注册为 Go candidate。
- E2 CVaR episode weighting、Dirichlet mixture、worst-family DRO。
- P3 frozen policy pool、red/exploiter 训练、league-lite blue response、
  maximin manifest。
- P2 C1 Bayesian belief planner；C2/C3 opponent model / gate / excitation /
  counterfactual / three-way MoE / repeated calibration。
- 21 维 SelfPlay-GRU、BC 蒸馏、self-play 改进实验。
- 66 候选 × 158 类对手全量扫描。

### Changed

- A60（`rl-hard60`）注册为候选；RL-Repair 注册为 `rl-repair`，但未 promotion。
- E2、P3 league-lite/expansion、P2 C1–C3、SelfPlay 改进判 No-Go。

## [2026-10-25]

### Added

- 初始 IPD 模拟器、规则策略、85 类对手库、四分布和盲测框架。
- Final / Responsive3 / BayesPlus / Triple / Hybrid 规则策略与参数搜索。
- MLP / GRU / RL / Adv 的 C++ 推理、权重导出和测试。
- 四分布、blind、9-suite、matchup 的统一评估入口。
- retrain_20261025 拆分评估配置和结果。

### Changed

- 主权重切换为从零 + hard 的 RL-PPO（run6）。
