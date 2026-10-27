# 按策略/模型/对手拆分评估

`ipd eval-split` 与 `ipd benchmark` / `ipd blind` 的区别：

| 需求 | 命令 |
|---|---|
| 一个候选、统一参数、合并结果 | `python3 -m ipd benchmark` / `blind` |
| 多个候选、每个候选不同参数、结果拆文件 | `python3 -m ipd eval-split` |

## 1. 配置文件

配置放在 `configs/eval/*.json`。`defaults` 提供默认参数，`candidates` 中的
`benchmark` / `blind` 只写需要覆盖的字段。

```json
{
  "name": "my_run",
  "defaults": {
    "benchmark": {"fields": 20, "n": 299, "seed": 20261026, "workers": 20},
    "blind":     {"fields": 10, "n": 299, "seed": 20261025, "workers": 10}
  },
  "candidates": [
    {
      "name": "Neural-A",
      "kind": "neural",
      "params": {"weights": "weights/neural_policy_weights.json", "mode": "sample"},
      "benchmark": {"fields": 20},
      "blind": {"fields": 10}
    },
    {
      "name": "GRU-A",
      "kind": "gru",
      "params": {"weights": "weights/gru_policy_weights.json", "mode": "sample"},
      "benchmark": {"fields": 12, "workers": 12},
      "blind": {"fields": 6, "workers": 6}
    }
  ]
}
```

- `kind` 与 `ipd benchmark --candidates` 使用的注册名一致；
- `params` 支持 `weights`、`config`、`mode`；
- `benchmark` / `blind` 可以分别覆盖 `fields`、`n`、`seed`、`workers`、
  `rounds`；benchmark 还支持 `dists`；
- 相对路径优先按仓库根目录解析，其次按配置文件所在目录解析。

仓库自带两份示例：

- `configs/eval/retrain_20261025.json`：所有候选统一 20/10 场，适合公平比较；
- `configs/eval/retrain_20261025_fast.json`：GRU/Adv-GRU 使用更少赛场，适合快速迭代。

已生成一份示例结果：`results/split_retrain_20261025/`。

## 2. 输出布局

```text
results/split_<name>/
  config.json                 本次运行的完整配置
  summary.json                所有候选的汇总（不含逐对手明细）
  README.md                   自动生成的摘要表
  by_candidate/
    <candidate>/
      benchmark.json          四分布汇总
      blind.json              盲测汇总
      opponents.json          该候选的逐对手均值/最小值/分位数/原始分数
  by_opponent/
    <opponent>.json           该对手类型下所有候选的汇总
```

盲测会把 `blind` 参数完全相同的候选放在同一组，复用同一赛场的对手循环赛；
参数不同的候选会进入不同组。这样既能利用缓存，又允许不同模型使用不同的
赛场数量或随机种子。

## 3. 常用命令

```bash
# 完整配置
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json

# 快速配置
python3 -m ipd eval-split --config configs/eval/retrain_20261025_fast.json

# 只跑一个候选，覆盖参数做烟测
python3 -m ipd eval-split \
  --config configs/eval/retrain_20261025.json \
  --only Neural-retrain \
  --fields 1 --n 32 --workers 2 \
  --blind-fields 1 --blind-n 32 --blind-workers 2 \
  --out-dir results/smoke --force

# 只重跑盲测
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json \
  --skip-benchmark --force
```

## 4. 逐对手统计

`opponents.json` 的每个对手条目包含：

```json
{
  "mean": 604.0,
  "min": 604.0,
  "max": 604.0,
  "p05": 604.0,
  "count": 20,
  "values": [604.0, "..."]
}
```

- benchmark 的 `count` 是该对手类型在 `fields × 每场副本数` 中的总对局数；
- blind 的 `count` 是 `fields × 每场副本数`；
- `by_opponent/<opponent>.json` 只保留统计量，不重复保存原始分数，便于跨候选比较。

## 5. 用拆分配置跑逐对手/逐家族对阵

`matchup` 可以直接复用 `configs/eval/*.json`，因此每个候选可以带自己的权重和
对抗配置；不同模型、不同权重不需要拆成多次命令：

```bash
python3 -m ipd matchup \
  --spec-config configs/eval/sweep_all_weights_20261026.json \
  --opponent-set all --reps 16 --workers 24 \
  --out results/sweep_20261026/matchup.csv
```

新增输出：

- `matchup_families.csv`：候选 × 对手家族均值（另有 `--family-out` 可改路径）；
- `matchup_suite_weighted.csv`：候选 × 9 个测试集加权；
- `matchup.json` 中的 `family_means` 和
  `candidate_summary.<candidate>.family_means`。

`--only` 可按候选名或 `kind` 过滤。`--spec-config` 与 `--candidates` 不能同时使用。
完整示例和 66 候选结果见 `results/sweep_20261026/README.md`。

## 6. 与旧脚本的关系

`results/retrain_20261025/evaluate.py` 仍保留为历史的一次性合并评估脚本。
新的拆分评估请使用 `python3 -m ipd eval-split`。
