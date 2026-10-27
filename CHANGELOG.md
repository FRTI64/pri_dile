# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/)；日期为 UTC。
未发布改动放在 `Unreleased`。

## [Unreleased]

### Added

- **Protocol v2**：`ipd/evaluation/protocol.py` + `python3 -m ipd protocol`，
  版本化 `train` / `dev_param` / `dev_family` / `test_v5` split、catalog
  manifest、泄漏审计和 test usage ledger。
- `configs/splits/protocol_v2/manifest.json`（224 configs）和
  `results/protocol_v2/` dev/test baseline。
- baseline 结果：RL-Repair test 477.05 > A60 469.91 > Adv-RL 467.95；
  Red-v5/v6 仍 100% 击败 top 3。
- `train_v2` 训练分布；`dev_param` / `dev_family` validation 支持。
- 红队评估：`protocol red --tier dev_red|test_red`。
- `LICENSE`（MIT，与 `pyproject.toml` 一致）。
- 文档重组为现代工程结构：`architecture/`、`guides/`、`reference/`、
  `policies/`、`research/`、`operations/`、`history/`、`decisions/`。
- 根 README 精简为项目入口；新增 `CONTRIBUTING.md`、`CHANGELOG.md`、
  文档索引和 ADR。
- 统一当前候选表为 E0/E1c 冻结协议（20 fields），旧 10-field 结果标为历史基线。
- 更新 `docs/operations/handover.md`：RL-Repair 为最佳候选但未 promotion；
  `heldout_v3` 已使用，`heldout_v4` 未使用。

### Changed

- `official` / `blind` 标记为 legacy in-distribution 诊断；新候选比较使用
  Protocol v2 dev/test，旧表不再作为 held-out 证据。
- `heldout_v2/v3/v4` 的选模流程被 Protocol v2 取代；历史结果保留为 legacy。

### Fixed

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
