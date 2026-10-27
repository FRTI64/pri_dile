# 候选参考

候选由 `ipd/policies/registry.py` 注册，实际实现在
`ipd/policies/candidates/<file>.py`。每个候选一个独立文件，候选文件之间不
互相 import。

```bash
python3 -m ipd list-policies
```

## 1. 当前候选

> 新候选比较使用 Protocol v4（747 configs + Axelrod，2048/30/50）。
> v4 baseline 尚未运行；下表状态仍基于 legacy v3 baseline（299/20/40）。

| 注册名 | 显示名 | 类型 | 权重 / 配置 | 状态 |
|---|---|---|---|---|
| `final-base` | Final-base | 规则 | - | 基线 |
| `final-tuned` | Final-tuned | 规则 | 参数内嵌 | 规则基线 |
| `resp3-base` | Resp3-base | 规则 | - | 基线 |
| `resp3-tuned` | Resp3-tuned | 规则 | 参数内嵌 | 规则 |
| `bayes-base` | Bayes-base | 规则 | - | 基线 |
| `bayes-tuned` | Bayes-tuned | 规则 | 参数内嵌 | 规则；宽/盲测弱 |
| `hybrid` | Final+Resp3 | 规则 | - | 规则融合 |
| `harvest45` | Harvest45 | 规则 | - | late 特化 |
| `harvest-tuned` | Harvest-tuned | 规则 | 参数内嵌 | late 特化 |
| `triple-majority` | Triple-majority | 规则 | - | 规则中最均衡 |
| `gap-start` | GapStart | 规则 | - | 搜索骨架；不用于提交 |
| `neural` | Neural | MLP | `weights/neural_policy_weights.json` | 可部署；低于 RL 系列 |
| `gru` | GRU | GRU | `weights/gru_policy_weights.json` | 有状态；低于 RL 系列 |
| `rl` | RL-PPO | RL | `weights/rl_policy_weights.json` | **当前主权重** |
| `rl-hard60` | RL-PPO-Hard60 | RL | `weights/rl_policy_weights_hard60.json` | A60；v3 test 第二（legacy） |
| `rl-repair` | RL-Repair | RL | `weights/rl_policy_weights_repair.json` | **v3 test 第一（legacy）；未 promotion** |
| `adv-mlp` | Adv-MLP | MLP + 规则 | base 权重 + `configs/adversarial_policy_config*.json` | 只有 Python |
| `adv-gru` | Adv-GRU | GRU + 规则 | base 权重 + `configs/adversarial_gru_policy_config*.json` | 只有 Python |
| `adv-rl` | Adv-RL | RL + 规则 | `rl_policy_weights_bc.json` + `configs/adversarial_rl_policy_config*.json` | v3 dev 高；test 第三；red 弱（legacy） |
| `selfplay` | SelfPlay-GRU | 21 维 GRU | `weights/selfplay_gru_policy.json` | 21 维备选；改进 No-Go |

## 2. 集合别名

| 集合 | 成员 |
|---|---|
| `core` | `final-tuned`, `resp3-tuned`, `bayes-tuned`, `triple-majority` |
| `base` / `legacy` | Final/Resp3/Bayes（base+tuned）+ `hybrid` |
| `classic` | `base` + `harvest45`, `harvest-tuned`, `gap-start` |
| `suite` | `base` + `harvest45`, `triple-majority` |
| `learned` | `neural`, `gru`, `rl`, `rl-hard60`, `rl-repair`, `adv-mlp`, `adv-gru`, `adv-rl`, `selfplay` |
| `all` | 全部 20 个注册候选 |

## 3. 常用别名

| 输入 | 解析为 |
|---|---|
| `mlp`, `neural-sample`, `neural-argmax` | `neural` |
| `gru-sample`, `gru-argmax` | `gru` |
| `rl-ppo`, `rl-sample`, `rl-argmax` | `rl` |
| `rl-hard`, `hard60`, `rl-ppo-hard60` | `rl-hard60` |
| `repair`, `rl-robust-repair` | `rl-repair` |
| `adv`, `adversarial`, `advmlp` | `adv-mlp` |
| `advgru` / `advrl` | `adv-gru` / `adv-rl` |
| `self-play`, `selfplay-gru`, `sp` | `selfplay` |
| `final`, `finalbase` | `final-base` |
| `finaltuned`, `final-param` | `final-tuned` |
| `resp`, `resp3`, `respbase` | `resp3-base` |
| `resptuned`, `resp3tuned` | `resp3-tuned` |
| `bayes` | `bayes-tuned` |
| `harvest` | `harvest-tuned` |
| `triple`, `triplemajority` | `triple-majority` |
| `gap`, `gapstart` | `gap-start` |

名称解析会忽略大小写、空格、下划线和连字符差异；未知名称直接报错。

## 4. 常用命令

```bash
# 只跑头部候选
python3 -m ipd benchmark --candidates rl,rl-hard60,rl-repair --fields 20

# 全候选
python3 -m ipd benchmark --candidates all --fields 8

# 指定权重/配置
python3 -m ipd benchmark --candidates rl --weights path/to/weights.json
python3 -m ipd benchmark --candidates adv-rl \
  --config configs/adversarial_rl_policy_config_aggressive.json
```

新增候选流程见 [../architecture/overview.md](../architecture/overview.md) 和
[../../CONTRIBUTING.md](../../CONTRIBUTING.md)。
