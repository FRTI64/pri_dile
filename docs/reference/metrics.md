# 指标与评估协议

本文定义分数、分布、指标和协议差异。跑命令见
[../guides/evaluation.md](../guides/evaluation.md)。

> **2026-10-27 起：Protocol v4 是当前主口径。** `official` / `blind` 只保留
> 为 legacy 诊断，不再作为 held-out 证据；详细 split 见
> [protocol-v4.md](protocol-v4.md)。

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

## 3.1 Protocol v4（当前主口径）

| 层 | 含义 | 用途 |
|---|---|---|
| `train` | 标准 + complex + policy + Axelrod config | 训练 |
| `dev_param` | 标准 family + Axelrod 的参数 held-out | 选模 |
| `dev_family` | `mirror_adaptive` / `delayed_grim` family held-out | 选模 / 病理检测 |
| `dev_complex` | complex family 的参数 held-out | 选模 / 复杂对手 |
| `dev_axelrod` | Axelrod 策略 / 组合 | 选模 / Axelrod 泛化 |
| `dev_ood` | dev / complex / family / Axelrod 的偏移混合 | 选模 / OOD |
| `dev_level0..4` | 按思考等级划分的派生对手集 | 选模 / 分级诊断 |
| `dev_complexity0..4` | 按策略复杂度划分的派生对手集 | 选模 / 复杂度诊断 |
| `test_v7` | 30% seen + 20% test-only + 15% complex + 10% policy + 15% Axelrod + 10% anchors | 永久 held-out（provisional only） |

- 默认 2048 人，dev 30 fields，test 50 fields。
- 具体 config 不跨层；`test_v7` 的 test-only family 不出现在 train/dev。
- complex family：`phase_switch`、`adaptive_mix`、`probe_exploit`、
  `delayed_extort`、`memory_two`、`meta_mix`、`noise_adaptive`、
  `reputation_grim`、`contrition_cycle`、`endgame_hunter`、
  `pattern_breaker`、`conditional_extort`、`lookahead`、`portfolio`。
- Axelrod：224 个 fast 策略（66 curated + 158 extra）+ 15 个
  `AxelrodMix` 组合，需安装 `requirements-axelrod.txt`。
- thinking level：0 非策略 / 1 固定启发式 / 2 反应自适应 /
  3 对手感知 meta / 4 搜索/集成/学习。
- complexity：0 无状态 / 1 简单反应 / 2 有状态 / 3 复合 / 4 高阶。
- policy opponent：训练用 RL-BC / SelfPlay-BC / Neural / GRU；test 用
  Final-tuned / Triple / RL-scratch / SelfPlay / Neural-retrain / GRU-retrain。
- 报告 `mean`、`min`、`p05`、per-family、`worst_family`、`min_norm`。
- 旧的 `official` / `blind` / v2 / v3 数字不能与 v4 混表。

## 4. suite

9 个家族加权测试集：`original`、`late`、`grim`、`noise`、`complex`、
`smart`、`punitive`、`random`、`wide`。每个 suite 用 `FAMILY_MULT` 对 10 个
对手家族加权，输出每个候选 × suite 的均值和最差 suite。

`suite` 是“按测试集家族分布抽样的期望分”，不是一次 299 人赛场的实现分数。

## 5. matchup

158 类对手、10 个家族：`nice`、`forgiving`、`punitive`、`defect`、`late`、
`random`、`wild`、`adaptive`、`plus`、`smart`。`--opponent-set protocol`
使用 Protocol v4 全 catalog（977 configs，38 个 family），用于逐 config
覆盖诊断，不与 158 类 legacy 表混用。

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

### 6.1 PSRO 元博弈

PSRO 使用 **blue × red** 矩形收益矩阵：

```text
M[i][j] = blue_i score - red_j score
```

- `blue_meta.weights` / `red_meta.weights`：blue maximin 和 red 对偶
  maximin 混合；`red_meta.training_weights` 额外混入 uniform，仅用于
  oracle 采样；
- `blue_worst_case`：blue 混合在 red 列上的最小收益（越高越好）；
- `exploitability`：`max(0, -blue_worst_case)`，即 red 对 blue 混合的
  最大收益（越低越好）；
- `red_meta.expected`：每个 red 对 blue 混合的收益，用于检查是否有
  单一 exploiter 主导；
- 元博弈 exploitability 只是有限 population 内的估计，**不能替代**
  held-out Red-v5/v6 或 Protocol v4 red；报告时必须同时给出外部 red。
- 符号约定：`protocol red` 的 `diff` 是 **blue - red**（正数 = blue
  优势）；league red report 的 `diff_mean` 是 **red - blue**（负数 = blue
  优势）。跨表比较前先确认符号，不要把两者直接相加或排名。

旧的 `heldout_v2/v3/v4`、`official/blind` 与 Protocol v2/v3 已进入 legacy。
新候选使用 Protocol v4：`dev_param` / `dev_family` / `dev_complex` /
`dev_axelrod` / `dev_ood` 选模；test 消费永久锁定，只能做 provisional 诊断。

## 7. 协议差异速查

| 协议 | 评估集 | fields | policy seed | 用途 |
|---|---|---|---|---|
| **Protocol v4** | `dev_param` / `dev_family` / `dev_complex` / `dev_axelrod` / `dev_ood` / `test_v7` | 30 / 30 / 30 / 30 / 30 / 50 | 默认独立 seed | **当前主口径** |
| 标准 CLI（legacy） | official 四分布 | 20 | 默认独立 seed | 历史复现 |
| E0/E1c league（legacy） | official + blind + heldout | 20 | 固定 index-based seed | 历史比较 |
| suite | 9 个家族加权 | 12 | 默认 | 场景诊断 |
| matchup | 158 类 | - | 默认 | 逐类型诊断 |
| eval-split | 每候选独立 | 可配置 | 可配置 | 多权重比较 |

同一候选在标准 CLI 与 league 上可能差 0.0x～1 分，这是 policy RNG 不同，
不是数据错误。**任何跨表比较都必须先确认协议。** Protocol v4 与 legacy
official/blind 的第一位数字不同是 split 不同，不是实现差异。

## 8. 报告最低要求

- 协议：`protocol_v4` / legacy，fields、n、seed、policy seed、suite；
- Protocol v4：`dev_param` / `dev_family` / `dev_complex` / `dev_axelrod` /
  `dev_ood` / `test_v7`，manifest hash；
- per-family、`min_norm`、`worst_family`；
- red：blue mean / red win / diff；
- 权重 checksum 和配置路径；
- test usage ledger 路径和是否 contaminated；
- Go/No-Go 结论和失败原因。
