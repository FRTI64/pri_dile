# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/)；日期为 UTC。
未发布改动放在 `Unreleased`。

## [Unreleased]

### Added

- **Plan 5 GPU sample（opt-in）**：`protocol eval --sample-rng per-match`
  使用 per-match RNG 契约，把学习型蓝方的 sample 推理批量上 GPU；
  CPU/GPU 逐场 parity（`tests/test_gpu_parity.py`），默认 `v4` 不变。
  结果标记 `sample_rng=per-match` / `gpu_sample=true` / `protocol_version=5`；
  `Adv-*` 与 PSRO meta-policy 仍走 CPU。文档
  `docs/research/plan5-gpu-sample.md`。
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

- 注册候选从 20 增至 24（`psro` / `psro-br` / `psro-scratch` /
  `psro-scratch-br`）；完整测试 240 OK。
- `official` / `blind` 标记为 legacy in-distribution 诊断；新候选比较使用
  Protocol v4 dev/test，旧表不再作为 held-out 证据。
- `heldout_v2/v3/v4` 与 Protocol v2/v3 的选模流程被 v4 取代；历史结果保留为
  legacy。

### Fixed

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
