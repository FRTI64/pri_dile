# 指标与评估协议

本文定义分数、分布、指标和协议差异。跑命令见
[../guides/evaluation.md](../guides/evaluation.md)。

## 1. 基础收益

每轮收益：

```text
CC = 6, CD = 0, DC = 10, DD = 1
```

100 轮总分为每轮收益之和。四分布、blind、suite、matchup 中的“分数”都是
每对手平均分（除特别注明）。

## 2. 官方四分布

| 分布 | 含义 |
|---|---|
| `original` | 原始混合场景 |
| `late` | 末期收割 |
| `grim30` | 高 Grim |
| `wide` | wide + smart，`sample_wide(..., smart_share=0.15)` |

标准 `benchmark --fields N` 在每个分布生成 N 个 299 人赛场（默认 8；
正式比较用 20），对每个候选计算每个分布均分和四分布均值。

## 3. blind

未见 seed 的 wide/smart 赛场，299 人。输出：

- `score`：均分；
- `mean_rank`：平均排名；
- `rank1` / `top10`：第一名/前 10 次数；
- `worst`：最差排名。

默认 10 fields；当前正式比较建议 20。**10-field 与 20-field 不可直接比较。**

## 4. suite

9 个家族加权测试集：`original`、`late`、`grim`、`noise`、`complex`、
`smart`、`punitive`、`random`、`wide`。每个 suite 用 `FAMILY_MULT` 对 10 个
对手家族加权，输出每个候选 × suite 的均值和最差 suite。

`suite` 是“按测试集家族分布抽样的期望分”，不是一次 299 人赛场的实现分数。

## 5. matchup

158 类对手、10 个家族：`nice`、`forgiving`、`punitive`、`defect`、`late`、
`random`、`wild`、`adaptive`、`plus`、`smart`。

输出：

- 类型等权：158 类直接平均；
- 家族等权：先按家族平均，再对 10 家族平均；
- 9 测试集加权：用 `FAMILY_MULT` 归一化后加权；
- 逐类型明细、透视表、家族表。

## 6. 冻结 league 指标

| 指标 | 定义 |
|---|---|
| `mean` | 每场平均分 |
| `families` | 每个 suite/family 的均分 |
| `min_family_raw` | 最差家族原始均分；跨家族不可比 |
| `min_norm_family` | 最差家族相对基线（默认 RL-PPO）的归一化分数 |
| `cvar10_family` | 最差 10% 家族均值的平均 |
| `cvar10_item` | 最差 10% 单场分数的平均 |
| `p05_item` | 单场分数 5% 分位 |
| `sigma_family` | 家族均值的标准差 |
| `max_minus_min` | 最好/最差家族差 |
| `top_family_share` | 最高家族对总分的贡献占比 |
| `red blue` | 蓝军对 red 的均分，**越高越好** |
| `red win` | red 胜率，越低越好 |
| `exploitability` | 池内/red 对候选的最大可获利差 |

`heldout_v2` 是 validation；`heldout_v3` 已被 E1c 一次性使用；
`heldout_v4` 未使用，只留给下一个通过 validation 的候选做 final test。

## 7. 协议差异速查

| 协议 | official | blind | policy seed | 用途 |
|---|---|---|---|---|
| 标准 CLI | 20 fields × 299 | 10/20 fields | 默认独立 seed | 日常/复现 |
| E0/E1c league | 20 fields × 299 | 20 fields | 固定 index-based seed | 候选冻结比较 |
| suite | 9 个家族加权 | - | 默认 | 场景鲁棒性 |
| matchup | 158 类 | - | 默认 | 逐类型/家族诊断 |
| split | 每候选独立 | 每候选独立 | 可配置 | 多权重比较 |

同一候选在标准 CLI 与 league 上可能差 0.0x～1 分，这是 policy RNG 不同，
不是数据错误。**任何跨表比较都必须先确认协议。**

## 8. 报告最低要求

- 协议：fields、n、seed、policy seed、suite；
- official / blind / held-out；
- per-family、`min_norm`、CVaR10；
- E4 red blue / red win；
- 权重 checksum 和配置路径；
- Go/No-Go 结论和失败原因。
