# 评估指南

本文说明如何在当前仓库中跑评估、选择协议、读取结果。指标定义见
[../reference/metrics.md](../reference/metrics.md)，候选列表见
[../reference/candidates.md](../reference/candidates.md)。

所有命令在仓库根目录执行；从其他目录运行时先 `pip install -e .`。

## 1. 先选协议

**新候选默认走 Protocol v3**：`dev_param` / `dev_family` / `dev_complex` /
`dev_ood` 选模，`test_v6` 只给预注册 top 3 跑一次。

```bash
python3 -m ipd protocol audit
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_ood \
  --fields 20 --baseline RL-PPO
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v6 --fields 40 --acknowledge-once
```

| 目标 | 协议 | 命令 |
|---|---|---|
| 正式选模 | Protocol v3 dev | `protocol eval --tiers dev_param,dev_family,dev_complex,dev_ood` |
| 一次性 final | Protocol v3 test | `protocol eval --tiers test_v6 --acknowledge-once` |
| 快速烟测 | 四分布（legacy），少 fields | `benchmark --fields 1..8` |
| 历史四分布 | 20 fields × 299（legacy） | `benchmark --fields 20` |
| 历史盲测 | 20 个未见 seed（legacy） | `blind --fields 20` |
| 9 个场景加权 | suite | `suite --fields 12` |
| 逐对手类型 | 158 类 × 10 家族 | `matchup --opponent-set all` |
| 候选冻结对比 | E0/E1c league | `league --config ...` |
| 每个候选独立参数 | eval-split | `eval-split --config ...` |

Protocol v3 的 manifest 在 `configs/splits/protocol_v3/manifest.json`，
test 使用记录在 `results/protocol_v3/usage.json`。审计失败（family 泄漏、
config 重叠、manifest hash 不匹配）时不要继续跑 test。`test_v6` 尚未消费；
`test_v6` 已在 2026-10-27 对 top 3 消费一次；不要复用。详见
[../reference/protocol-v3.md](../reference/protocol-v3.md)。

**不要混表：**

- Protocol v3 与 Protocol v2 / 旧 `official` / `blind` 不可直接比较；
- 10-field 与 20-field blind 不可直接比较；
- 标准 CLI（独立 policy seed）与 E0/E1c 冻结 league（固定 policy seed）会有
  0.0x～1 分差异；
- `suite` 是家族加权期望分，不是单次 299 人赛场；
- `matchup` 是逐类型等权/家族加权，不是官方四分布。

## 2. 四分布 benchmark

```bash
# 烟测
python3 -m ipd benchmark --candidates core --fields 1 --workers 4

# 当前头部候选
python3 -m ipd benchmark --candidates rl,rl-hard60,rl-repair \
  --fields 20 --workers 20

# 指定权重 / 配置
python3 -m ipd benchmark --candidates rl-repair \
  --weights weights/rl_policy_weights_repair.json --fields 20
python3 -m ipd benchmark --candidates adv-rl \
  --config configs/adversarial_rl_policy_config_aggressive.json --fields 20
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--candidates` | 逗号分隔名称或集合 `core/base/classic/suite/learned/all` |
| `--fields` | 每个分布的赛场数；正式比较用 20 |
| `--n` | 每场对手数，默认 299 |
| `--workers` | 并行进程数 |
| `--seed` | 赛场 seed，默认 20261026 |
| `--neural-mode` | `sample`（默认）或 `argmax` |
| `--out` | 输出 JSON；默认 `results/benchmark.json` |

输出是每个候选 × 每个分布的均分，以及 `avg / worst / min`。默认 `--out`
会覆盖 `results/benchmark.json`；正式实验请显式指定输出路径。

## 3. 盲测

```bash
python3 -m ipd blind --candidates rl,rl-hard60,rl-repair \
  --fields 20 --workers 10
python3 -m ipd blind --candidates selfplay \
  --weights weights/selfplay_gru_policy.json --fields 20 --workers 10
```

盲测使用未见 seed 的 wide/smart 赛场，输出均分、平均排名、第一名次数、
前 10 次数和最差排名。默认输出 `results/blind_results.json`。

## 4. 9 个测试集

```bash
python3 -m ipd suite --fields 12 --workers 24
python3 -m ipd suite --candidates core --suites original,late,grim --fields 12
```

输出每个候选 × 9 个 suite 的均值和最差 suite，默认写
`results/suite_results.json`。

## 5. 逐对手矩阵

```bash
# 20 个注册候选 × 158 类对手
python3 -m ipd matchup --candidates all --opponent-set all \
  --reps 16 --workers 24

# 66 个候选/权重变体，使用 eval-split 配置
python3 -m ipd matchup \
  --spec-config configs/eval/sweep_all_weights_20261026.json \
  --opponent-set all --reps 16 --workers 24 \
  --out results/sweep_20261026/matchup.csv
```

输出：

- `matchup.csv`：逐候选 × 逐对手；
- `matchup_wide.csv`：透视表（由 `ipd/tools/pivot_matchup.py` 生成）；
- `matchup_suite_weighted.csv`：9 测试集加权；
- `matchup_families.csv`：候选 × 10 家族。

当前 66 候选结果见 `results/sweep_20261026/README.md`。

## 6. 冻结 league

E0/E1c 的主口径，固定候选 spec、权重、suite、seed 和 policy seed：

```bash
python3 -m ipd league --config configs/eval/e0_league.json --workers 20 \
  --out results/e0_league_20261026/report.json \
  --md-out results/e0_league_20261026/README.md

# 只跑部分候选 / suite，做烟测
python3 -m ipd league --config configs/eval/e0_league.json \
  --only RL-PPO,RL-Repair --suites heldout_v2 \
  --cross-play-seeds 2 --red-seeds 2 --workers 4 \
  --out /tmp/e0_smoke/report.json
```

输出包含 official / blind / validation / heldout_v2 / cross-play / red，
以及 `min_norm_family`、CVaR、Nash/Elo/exploitability。详见
[../research/e0-league.md](../research/e0-league.md)。

以上 league 输出是 legacy 口径。新候选使用 Protocol v3：`test_v6` 尚未消费，
下一个通过 dev 的候选可以直接用它做一次 final。

## 7. 拆分评估

```bash
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json
python3 -m ipd eval-split --config configs/eval/retrain_20261025_fast.json
```

适合每个候选使用不同权重/配置/fields。输出、配置格式和 `--only` 用法见
[split-evaluation.md](split-evaluation.md)。

## 8. Axelrod 可选对手库

```bash
pip install -r requirements-axelrod.txt
python3 -m ipd benchmark --candidates core --dists axelrod-fast \
  --fields 1 --n 224 --workers 5
python3 -m ipd matchup --candidates final-tuned --opponent-set axelrod-fast \
  --reps 4 --workers 12
```

默认四分布和 85/158 对手集合不变。完整 243 很慢，日常用 `axelrod-fast`。
详见 [axelrod.md](axelrod.md)。

## 9. 结果检查清单

- [ ] 显式写出协议（fields、seed、policy seed、suite）；
- [ ] 学习策略使用 `sample`，`argmax` 只作对照；
- [ ] 不把 10-field 与 20-field、标准 CLI 与 league 混表；
- [ ] 报告 per-family、`min_norm`、CVaR、held-out red，而不只看均值；
- [ ] 正式输出写到 `results/<run>/`，不要覆盖 `results/benchmark.json` 历史结果；
- [ ] 公开结论附上权重 checksum 和配置路径。
