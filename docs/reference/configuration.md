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
├── splits/protocol_v4/test_policy.json      test 消费锁定策略（consume_test=false）
├── league/                                  P3 red pool blue manifests
├── psro/                                    PSRO population / oracle / red check
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

### PSRO 配置

`configs/psro/psro_20261027.json` 定义 blue/red population、cross-play
seed、元博弈 solver、PPO oracle 和输出路径：

```json
{
  "name": "psro_20261027",
  "rounds": 2,
  "output_dir": "results/psro_20261027",
  "deploy_path": "weights/psro/psro_policy.json",
  "deploy_policy": "last",
  "cross_play": {"seeds": 8, "seed_base": 21161027,
                  "policy_seed_base": 21061027, "workers": 20, "rounds": 100},
  "meta": {"iterations": 20000, "seed": 20261027, "uniform_mix": 0.15},
  "oracle": {"init_weights": "weights/rl_policy_checkpoint_repair.json",
              "iterations": 12, "fields_per_dist": 4, "n": 64,
              "opponent_fraction": 0.6, "opponent_penalty": 1.0},
  "population": [
    {"name": "Final-tuned", "kind": "final-tuned", "role": "blue"},
    {"name": "Red-RL-v4", "kind": "rl", "role": "red",
     "params": {"weights": "weights/red_pool/red_rl_v4.json"}}
  ]
}
```

- `role=blue` 是待部署策略；`role=red` 是 exploiter；
- `meta.uniform_mix` 只混合 red 训练权重，不用于 evaluation；
- `oracle` 复用 `train rl --opponent-pool`；`--no-validation` 是 PSRO 的
  默认语义（best response，而不是验证选模）；
- `deploy_policy=last|meta_best` 选择最后一轮 BR 或 blue 期望最高的成员；
- `psro` 候选读取 `weights/psro/meta_strategy.json`，每场按 blue maximin
  采样一个成员。
- 从零配置 `configs/psro/psro_scratch_20261027.json`：
  blue 只有 `Scratch-Init`（随机 MLP）、`Scratch-Base`（从零官方 PPO）
  和 `PSRO-BR-*`；已有模型/规则全部是 `role=red`。oracle 的
  `init_weights` 是 scratch base checkpoint，不是已有候选。
  `psro-scratch` / `psro-scratch-br` 分别读取
  `weights/psro_scratch/meta_strategy.json` 和
  `weights/psro_scratch/psro_policy.json`。
- `psro_scratch_pilot_20261027.json`（激进 pilot）和
  `psro_scratch_v2/v3` 是 ablation；不要和主配置混用。

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
| `psro/meta_strategy.json` | PSRO reference blue/red population + 元策略 |
| `psro/psro_policy.json` | reference PSRO best response（`psro-br`） |
| `psro_scratch/init_policy.json` | 从零随机初始策略 |
| `psro_scratch/base_policy.json` / `_checkpoint.json` | 从零官方 PPO base |
| `psro_scratch/meta_strategy.json` | 从零 PSRO blue/red population + 元策略 |
| `psro_scratch/psro_policy.json` | 从零 PSRO 最终 best response |
| `opponent_*` / `red_pool/` | C2/C3、P3 研究 artifact |
| `retrain_20261025/` / `hard2_20261026/` | 复训实验权重 |

`.pt` checkpoint 被 `.gitignore` 忽略；提交 JSON 权重和报告即可。

## 4. results/

```text
results/
├── benchmark.json / blind_results.json / suite_results.json
├── split_<name>/                 eval-split 输出
├── protocol_v4/                  Protocol v4 dev / provisional test 结果
├── e0_league_<date>/             E0
├── robust_<date>/                E1
├── risk_<date>/                  E2
├── red_pool_<date>/ p3_league_<date>/
├── psro_<date>/                  PSRO 元博弈 + oracle + 诊断
├── psro_scratch_<date>/          从零 PSRO + scratch base + 诊断
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

最近完整测试：**312 tests OK**（本轮全量运行）。
