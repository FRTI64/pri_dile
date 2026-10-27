# CLI 参考

入口：`python3 -m ipd <command> [options]`。也可以安装后使用 `ipd`。
所有命令默认从仓库根目录解析路径；从其他目录运行时先 `pip install -e .`。

```bash
python3 -m ipd --help
python3 -m ipd <command> --help
```

## 1. 命令

| 命令 | 用途 |
|---|---|
| `protocol` | **Protocol v3**：train/dev/test split 审计与评估 |
| `benchmark` | legacy 四分布基准；也支持 Axelrod 分布 |
| `blind` | legacy blind 排名（只换 seed） |
| `suite` | 9 个家族加权测试集 |
| `matchup` | 逐对手类型矩阵 + 9 测试集加权 |
| `eval-split` | 每个候选独立参数的拆分评估 |
| `league` | E0/E1c 冻结 league（cross-play / Nash / CVaR / red） |
| `meta` | 从 red league 报告生成 maximin / opponent manifest |
| `planner-eval` | C1 belief planner 离线评估 |
| `moe-eval` | C3 three-way gate + MoE 离线评估 |
| `train` | 训练/搜索任务；`train --list` 查看全部 |
| `export` | 权重导出为 C++ 头文件 |
| `self-test` | 策略接口快速自检 |
| `test` | 运行单元测试 |
| `list-policies` | 列出注册候选、显示名、集合 |
| `paths` | 打印仓库目录 |

## 2. 评估命令

### protocol

```bash
python3 -m ipd protocol audit
python3 -m ipd protocol manifest
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_ood \
  --fields 20 --n 299 --workers 20 --baseline RL-PPO \
  --out results/protocol_v3/protocol_v3_dev.json
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v6 --fields 40 --acknowledge-once \
  --out results/protocol_v3/protocol_v3_test.json
python3 -m ipd protocol red --candidates all --tier dev_red --seeds 20
python3 -m ipd protocol red --candidates <top3> --tier test_red \
  --seeds 20 --acknowledge-once
```

参数：`--candidates`、`--tiers`（`dev_param` / `dev_family` /
`dev_complex` / `dev_ood` / `test_v6`）、`--fields`、`--n`、`--rounds`、
`--workers`、`--seed`、`--baseline`、`--acknowledge-once`、`--force`、
`--out`。`test_v6` 必须显式确认，并写入 usage ledger。详见
[protocol-v3.md](protocol-v3.md)。

### benchmark（legacy）

```bash
python3 -m ipd benchmark --candidates core --fields 20 --workers 20
python3 -m ipd benchmark --candidates rl,rl-hard60,rl-repair --fields 20
python3 -m ipd benchmark --candidates core --dists axelrod-fast --fields 1 --n 224
```

常用参数：`--candidates`、`--weights`、`--config`、`--dists`、`--fields`、
`--n`、`--workers`、`--seed`、`--neural-mode sample|argmax`、`--out`。
`--arch` / `--neural*` 是旧兼容参数，新命令用 `--candidates`。

### blind

```bash
python3 -m ipd blind --candidates rl,rl-hard60,rl-repair --fields 20 --workers 10
```

参数：`--candidates`、`--weights`、`--config`、`--fields`、`--n`、
`--workers`、`--seed`、`--neural-mode`、`--out`。默认 10 fields；
正式比较建议 20。

### suite

```bash
python3 -m ipd suite --fields 12 --workers 24
python3 -m ipd suite --candidates core --suites original,late,grim --fields 12
```

参数：`--candidates`、`--suites`、`--fields`、`--n`、`--workers`、`--seed`、
`--out`。

### matchup

```bash
python3 -m ipd matchup --candidates all --opponent-set all --reps 16
python3 -m ipd matchup --spec-config configs/eval/sweep_all_weights_20261026.json \
  --opponent-set all --reps 16 --workers 24 --out results/sweep_20261026/matchup.csv
```

参数：`--candidates` 或 `--spec-config`（二者互斥）、`--only`、
`--opponent-set core|all|axelrod|axelrod-fast|axelrod-cheating`、`--reps`、
`--rounds`、`--workers`、`--seed`、`--out`、`--json-out`、`--suite-out`、
`--family-out`。

### eval-split

```bash
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json
python3 -m ipd eval-split --config configs/eval/retrain_20261025.json \
  --only Neural-retrain --fields 1 --n 32 --workers 2 --force
```

参数：`--config`、`--out-dir`、`--only`、`--skip-benchmark`、`--skip-blind`、
`--no-details`、`--force`，以及 `--fields` / `--blind-fields` 等覆盖参数。
格式见 [../guides/split-evaluation.md](../guides/split-evaluation.md)。

### league

```bash
python3 -m ipd league --config configs/eval/e0_league.json --workers 20 \
  --out results/e0_league_20261026/report.json \
  --md-out results/e0_league_20261026/README.md
```

参数：`--config`、`--out`、`--md-out`、`--only`、`--suites`、`--workers`、
`--cross-play-seeds`、`--red-seeds`、`--nash-iterations`、`--skip-suites`、
`--skip-cross-play`、`--skip-red`。

## 3. 训练命令

```bash
python3 -m ipd train --list
python3 -m ipd train neural --device auto
python3 -m ipd train gru --device auto
python3 -m ipd train rl --device auto --iterations 100
python3 -m ipd train adversarial --arch mlp
python3 -m ipd train opponent-model --data-mode mixed
python3 -m ipd train selfplay --config configs/selfplay/full.json --device auto
python3 -m ipd train selfplay-bc --device auto --episodes 4000 --epochs 10
```

完整任务见 [../guides/training.md](../guides/training.md)。

## 4. 导出与自检

```bash
python3 -m ipd self-test --candidates all
python3 -m ipd self-test --candidates rl --weights weights/rl_policy_weights.json
python3 -m ipd export --arch mlp
python3 -m ipd export --arch gru
python3 -m ipd export --weights weights/rl_policy_weights.json \
  --out cpp/rl_policy_weights.hpp
```

## 5. 测试与路径

```bash
python3 -m ipd test
python3 -m ipd test test_core.py
python3 -m ipd list-policies
python3 -m ipd paths
```

`test` 的额外参数是 `tests/` 下的文件名/glob；默认运行 `test_*.py`。
