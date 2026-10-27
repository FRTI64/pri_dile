# 配置与目录参考

## 1. 仓库目录

```text
ipd/           Python 包（core / policies / evaluation / training / tools）
cpp/           C++17 推理实现与生成头文件
configs/       评估、对抗、league、self-play 配置
weights/       训练权重与 checkpoint
results/       基准、盲测、实验报告、训练曲线
tests/         分层测试
docs/          文档（见 ../README.md）
```

路径统一由 `ipd/paths.py` 解析：`ROOT`、`CONFIG_DIR`、`WEIGHTS_DIR`、
`RESULTS_DIR`、`CPP_DIR`、`TESTS_DIR`。默认从仓库根目录运行；从其他目录运行
前先 `pip install -e .`。

## 2. configs/

```text
configs/
├── adversarial_policy_config*.json          MLP 保守/激进
├── adversarial_gru_policy_config*.json      GRU 保守/激进
├── adversarial_rl_policy_config*.json       RL 保守/激进
├── eval/                                    eval-split / league / sweep 配置
├── splits/protocol_v4/manifest.json         Protocol v4 catalog manifest
├── league/                                  P3 red pool blue manifests
├── selfplay/                                full / smoke / BC / E4 red
└── retrain_20261025/                        复训实验配置
```

### eval-split 配置

```json
{
  "name": "my_run",
  "defaults": {
    "benchmark": {"fields": 20, "n": 299, "seed": 20261026, "workers": 20},
    "blind": {"fields": 10, "n": 299, "seed": 20261025, "workers": 10}
  },
  "candidates": [
    {
      "name": "RL-Repair",
      "kind": "rl-repair",
      "params": {"weights": "weights/rl_policy_weights_repair.json", "mode": "sample"},
      "benchmark": {"fields": 20},
      "blind": {"fields": 20}
    }
  ]
}
```

- `kind` 是注册名；`params` 支持 `weights`、`config`、`mode`；
- `benchmark` / `blind` 可分别覆盖 `fields`、`n`、`seed`、`workers`、
  `rounds`；benchmark 还支持 `dists`；
- 相对路径优先按仓库根目录解析，其次按配置文件目录解析。

完整格式见 [../guides/split-evaluation.md](../guides/split-evaluation.md)。

### league 配置

`configs/eval/e0_league.json` 定义候选 spec、权重路径、suite、seed、
cross-play、red 和 Nash 预算。关键字段：

```json
{
  "defaults": {"workers": 20, "rounds": 100, "n": 299, "fields": 20,
                "blind_fields": 20, "opponent_seeds": 20,
                "cross_play_seeds": 20, "red_seeds": 20},
  "suites": {
    "official": {"kind": "official", "fields": 20, "n": 299, "seed": 20261026},
    "blind": {"kind": "blind", "fields": 20, "n": 299, "seed": 20261025},
    "validation": {"kind": "opponents", "suite": "validation", "seeds": 20},
    "heldout_v2": {"kind": "opponents", "suite": "heldout_v2", "seeds": 20}
  },
  "candidates": [{"name": "RL-PPO", "kind": "rl",
                  "params": {"weights": "weights/rl_policy_weights.json"}}],
  "red_policies": [{"name": "E4-red-BC", "kind": "selfplay",
                    "params": {"weights": "weights/selfplay_improvement/E4_red_policy.json"}}]
}
```

## 3. weights/

| 文件 | 内容 |
|---|---|
| `neural_policy_weights.json` | MLP 部署权重 |
| `gru_policy_weights.json` | GRU 部署权重 |
| `rl_policy_weights.json` | **主权重**（RL-PPO，从零 + hard） |
| `rl_policy_checkpoint.json` | 主权重 actor-critic checkpoint |
| `rl_policy_weights_hard60.json` | A60 |
| `rl_policy_weights_repair.json` | RL-Repair |
| `rl_policy_weights_bc.json` | Adv-RL base |
| `selfplay_gru_policy.json` | 21 维 SelfPlay-GRU |
| `opponent_*` / `red_pool/` | C2/C3、P3 研究 artifact |
| `retrain_20261025/` / `hard2_20261026/` | 复训实验权重 |

`.pt` checkpoint 被 `.gitignore` 忽略；提交 JSON 权重和报告即可。

## 4. results/

```text
results/
├── benchmark.json / blind_results.json / suite_results.json
├── split_<name>/                 eval-split 输出
├── protocol_v4/                  Protocol v4 manifest 使用记录 / dev / test
├── e0_league_<date>/             E0
├── robust_<date>/                E1
├── risk_<date>/                  E2
├── red_pool_<date>/ p3_league_<date>/
├── p2_*_<date>/
├── selfplay_* / sweep_<date> /
└── <experiment>/README.md        实验报告
```

默认 `--out` 会写 `results/benchmark.json`、`results/blind_results.json`
等；正式实验请显式指定路径，避免覆盖历史结果。

## 5. 环境

- Python ≥ 3.10（`pyproject.toml`）；
- 默认安装无第三方依赖；
- 训练：`pip install -r requirements-train.txt`（torch/numpy）；
- Axelrod：`pip install -r requirements-axelrod.txt`（`axelrod==4.14.0`）；
- 可选安装：`pip install -e '.[train]'` / `.[axelrod]`。

最近完整测试为 v3 的 204 tests；v4 改动后未运行测试（按要求）。
