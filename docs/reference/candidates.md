# 候选参考

候选由 `ipd/policies/registry.py` 注册，实际实现在
`ipd/policies/candidates/<file>.py`。每个候选一个独立文件，候选文件之间不
互相 import。

```bash
python3 -m ipd list-policies
```

## 1. 当前候选

> 新候选比较使用 Protocol v5（977 configs + Axelrod，2048/30/50，
> train/dev/test 已重新划分）。v5 baseline 尚未运行；下表的 v4 top 7
> 与 v3 数字只作 legacy，提交前必须在 v5 上重跑。

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
| `triple-v4` | Triple-v4 | 规则 | 参数内嵌 | 最佳纯规则；Protocol v4 校准的 Triple 变体；未 promotion |
| `gap-start` | GapStart | 规则 | - | 搜索骨架；不用于提交 |
| `neural` | Neural | MLP | `weights/neural_policy_weights.json` | 可部署；低于 RL 系列 |
| `gru` | GRU | GRU | `weights/gru_policy_weights.json` | 有状态；低于 RL 系列 |
| `rl` | RL-PPO | RL | `weights/rl_policy_weights.json` | **当前主权重** |
| `rl-hard60` | RL-PPO-Hard60 | RL | `weights/rl_policy_weights_hard60.json` | A60；v3 test 第二（legacy） |
| `rl-repair` | RL-Repair | RL | `weights/rl_policy_weights_repair.json` | v3 test 第一（legacy）；**v4 dev 次席；未 promotion** |
| `rl-pattern` | RL-Pattern | RL | `weights/rl_policy_weights_pattern.json` | v4 dev 前列；pattern_breaker 修复版；未 promotion |
| `rl-wide` | RL-Wide | RL 128/128 | `weights/rl_policy_weights_wide.json` | v4 dev 前列；complex min_norm 0.973；未 promotion |
| `rl-wide160` | RL-Wide160 | RL 160/128 | `weights/rl_policy_weights_wide160.json` | v4 dev_complex 454.31；complex/ood min_norm 0.997；未 promotion |
| `rl-switch` | RL-Switch | meta（2 RL） | `weights/rl_policy_weights_wide160.json` + `rl_policy_weights_repair.json` | **v4 dev/test 最佳（top 7）；Python + C++；未 promotion** |
| `rl-dagger-gru` | GRU-DAgger | GRU | `weights/rl_dagger_gru_weights.json` | 单模型 RL-Switch 蒸馏；v4 dev 与 RL-Switch 持平；Python + C++；未 promotion |
| `rl-gate` | RL-Gate | meta（learned gate） | 同上 + `weights/meta_gate_wide160_repair.json` | 实验；阈值 0.9 近中性；**No-Go** |
| `adv-mlp` | Adv-MLP | MLP + 规则 | base 权重 + `configs/adversarial_policy_config*.json` | 只有 Python |
| `adv-gru` | Adv-GRU | GRU + 规则 | base 权重 + `configs/adversarial_gru_policy_config*.json` | 只有 Python |
| `adv-rl` | Adv-RL | RL + 规则 | `rl_policy_weights_bc.json` + `configs/adversarial_rl_policy_config*.json` | v3 dev 高；test 第三；red 弱（legacy） |
| `selfplay` | SelfPlay-GRU | 21 维 GRU | `weights/selfplay_gru_policy.json` | 21 维备选；改进 No-Go |
| `psro` | PSRO-Meta | 策略组合 | `weights/psro/meta_strategy.json` | 实验；reference blue maximin 混合；未 promotion |
| `psro-br` | PSRO-BR | RL | `weights/psro/psro_policy.json` | 实验；最后一轮 best response；未 promotion |
| `psro-scratch` | PSRO-Scratch | 策略组合 | `weights/psro_scratch/meta_strategy.json` | 实验；blue 仅 random/scratch/BR；red_win 0.0；未 promotion |
| `psro-scratch-br` | PSRO-Scratch-BR | RL | `weights/psro_scratch/psro_policy.json` | 实验；从零最终 BR；未 promotion |
| `psro-v4` | PSRO-V4-Meta | 策略组合 | `weights/psro_v4/meta_strategy.json` | 实验；更强 oracle；BR_01 入 support；未 promotion |
| `psro-v4-br` | PSRO-V4-BR | RL | `weights/psro_v4/psro_policy.json` | 实验；red_win 0.0，dev_param -59.5；未 promotion |

## 2. 集合别名

| 集合 | 成员 |
|---|---|
| `core` | `final-tuned`, `resp3-tuned`, `bayes-tuned`, `triple-majority`, `triple-v4` |
| `base` / `legacy` | Final/Resp3/Bayes（base+tuned）+ `hybrid` |
| `classic` | `base` + `harvest45`, `harvest-tuned`, `gap-start` |
| `suite` | `base` + `harvest45`, `triple-majority`, `triple-v4` |
| `learned` | `neural`, `gru`, `rl`, `rl-hard60`, `rl-repair`, `rl-pattern`, `rl-wide`, `rl-wide160`, `rl-switch`, `rl-dagger-gru`, `rl-gate`, `psro`, `psro-br`, `psro-scratch`, `psro-scratch-br`, `psro-v4`, `psro-v4-br`, `adv-mlp`, `adv-gru`, `adv-rl`, `selfplay` |
| `all` | 全部 33 个注册候选 |

## 3. 常用别名

| 输入 | 解析为 |
|---|---|
| `mlp`, `neural-sample`, `neural-argmax` | `neural` |
| `gru-sample`, `gru-argmax` | `gru` |
| `rl-ppo`, `rl-sample`, `rl-argmax` | `rl` |
| `rl-hard`, `hard60`, `rl-ppo-hard60` | `rl-hard60` |
| `repair`, `rl-robust-repair` | `rl-repair` |
| `pattern`, `rl-pb` | `rl-pattern` |
| `wide-rl`, `rl-128` | `rl-wide` |
| `wide160`, `rl-160` | `rl-wide160` |
| `switch`, `meta-switch` | `rl-switch` |
| `dagger-gru`, `gru-dagger`, `switch-gru` | `rl-dagger-gru` |
| `gate`, `meta-gate` | `rl-gate` |
| `adv`, `adversarial`, `advmlp` | `adv-mlp` |
| `advgru` / `advrl` | `adv-gru` / `adv-rl` |
| `self-play`, `selfplay-gru`, `sp` | `selfplay` |
| `psro`, `psro-meta` | `psro` |
| `psro-v4`, `psro4` | `psro-v4` |
| `psro-v4-br`, `psro4-br` | `psro-v4-br` |
| `psro-br`, `psro-response` | `psro-br` |
| `psro-scratch`, `psro-zero` | `psro-scratch` |
| `psro-scratch-br`, `psro-zero-br` | `psro-scratch-br` |
| `final`, `finalbase` | `final-base` |
| `finaltuned`, `final-param` | `final-tuned` |
| `resp`, `resp3`, `respbase` | `resp3-base` |
| `resptuned`, `resp3tuned` | `resp3-tuned` |
| `bayes` | `bayes-tuned` |
| `harvest` | `harvest-tuned` |
| `triple`, `triplemajority` | `triple-majority` |
| `triplev4`, `v4-rule`, `rule-v4` | `triple-v4` |
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
