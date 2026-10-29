# RL-Wide160 全量对手扫描（2026-10-29）

> 目的：找出 RL-Wide160 的薄弱对手，而不是替代 Protocol v4 排名。
> `matchup` 使用单对手类型 × N 次重复；与 2048 人 field 的绝对分数
> 不可直接比较。

## 覆盖范围

| 集合 | 数量 | reps | 结果 |
|---|---:|---:|---|
| 158 类完整对手库 `all` | 158 | 16 | `matchup_all.*` |
| Protocol v4 catalog `protocol` | 977 configs | 8 | `matchup_protocol.*` |
| Axelrod fast | 224 | 8 | `matchup_axelrod_fast.*` |
| 参考候选（158 / protocol） | 4 | 16 / 8 | `matchup_*_refs.*` |
| Axelrod 完整 243 | 243 | 4 | **未完成**：19 个 long-run Meta 超时 |

参考候选：`rl-repair`、`rl-pattern`、`rl-wide`、`triple-majority`。

## 总体均值

| 集合 | 均值 |
|---|---:|
| 158 类 bank | **547.35** |
| 158 类 nine-suite avg / worst | 554.45 / 530.23 |
| Protocol catalog | 507.66 |
| Protocol nine-suite avg / worst | 520.43 / 484.55 |
| Axelrod fast | **575.39** |

## 最弱 family

Protocol catalog：

| family | 分数 |
|---|---:|
| early_harvest | 129.00 |
| frequent_kamikaze | 169.17 |
| hyper_extort | 176.27 |
| probe_exploit | 206.36 |
| adaptive_mix | 291.88 |
| conditional_extort | 323.63 |
| late | 347.10 |
| phase_switch | 385.44 |
| endgame_hunter | 397.25 |
| extort | 402.10 |
| delayed_extort | 415.40 |
| probe_harvest | 429.27 |
| noisy | 462.40 |
| pattern_breaker | 468.33 |
| temporal | 478.93 |

158 类 bank：

| family | 分数 |
|---|---:|
| late | 476.63 |
| random | 507.23 |
| wild | 517.29 |
| plus | 518.95 |
| adaptive | 526.28 |
| forgiving | 570.97 |
| smart | 573.62 |

## 最弱的 held-out test 对手

按“比参考候选最好分低多少”排序（`weak_test.csv`）：

| tier | family | 对手 | our | best ref | gap |
|---|---|---|---:|---:|---:|
| test_axelrod | axelrod_extra | GradualKiller | 115.00 | 946.00 | **-831.00** |
| test_complex | pattern_breaker | threshold=.6 window=6 | 288.00 | 804.00 | **-516.00** |
| test_only | frequent_kamikaze | p_steal=1.0 period=9 | 208.00 | 571.00 | **-363.00** |
| test | temporal | Kamikaze p=.75 period=8 | 198.12 | 338.38 | -140.25 |
| test_only | frequent_kamikaze | p_steal=1.0 period=8 | 193.00 | 270.00 | -77.00 |
| test | temporal | DynamicWindow p=.08 w=12 | 434.38 | 497.00 | -62.62 |
| test_only | hyper_extort | ExtortZD chi=5.5 | 156.12 | 218.25 | -62.12 |
| test_complex | probe_exploit | extort probe=4 | 257.50 | 319.12 | -61.62 |
| test_axelrod | axelrod_mix | ZDExtort2/3/Extortion | 185.50 | 241.25 | -55.75 |
| test_axelrod | axelrod_extra | FirstByTullock | 285.25 | 339.88 | -54.62 |
| test_only | hyper_extort | ExtortZD chi=4.5 | 152.62 | 202.12 | -49.50 |
| test_only | hyper_extort | ExtortZD chi=6.0 | 169.38 | 218.50 | -49.12 |
| test_complex | phase_switch | coop→extort switch=55 | 424.62 | 473.25 | -48.62 |
| test | temporal | DynamicWindow p=.38 w=14 | 211.88 | 258.75 | -46.88 |
| test_complex | conditional_extort | coop=.75 window=20 | 329.50 | 371.75 | -42.25 |

完整 179 个 test-tier 对手见 `weak_test.csv`；按 `gap_to_best` 排序。

## 最弱的 158 类 bank 对手

| family | 对手 | our | best ref | gap |
|---|---|---:|---:|---:|
| plus | FSM.F5 | 569.00 | 933.00 | **-364.00** |
| smart | Gap0.55 | 192.00 | 538.00 | **-346.00** |
| smart | Gap0.64 | 137.00 | 247.00 | -110.00 |
| plus | FSM.F1 | 866.00 | 951.00 | -85.00 |
| random | Rand3->TFT | 529.62 | 587.44 | -57.81 |
| adaptive | ProbeBot | 544.88 | 593.00 | -48.12 |
| smart | Harvest21 | 547.75 | 588.00 | -40.25 |
| smart | EarlyProbe | 606.00 | 641.94 | -35.94 |
| adaptive | Extort3 | 274.75 | 301.50 | -26.75 |
| plus | Extort.F3.0 | 282.94 | 308.75 | -25.81 |

完整 158 行见 `weak_all.csv`；按 `gap_to_best` 排序。

## Axelrod 薄弱点

`weak_axelrod_fast.csv` 中 diff 最差：

- `ZDExtort4` our 312 / opp 676（-364）
- `TF2` our 240 / opp 490（-250）
- `ZDExtort3` our 265 / opp 469（-204）
- `ZDExtortion` our 158 / opp 353（-195）
- `ZDExtort2v2` our 357 / opp 504（-148）
- `ZDMischief` our 132 / opp 274（-141）
- `ZDSet2` our 376 / opp 514（-139）
- `ZDMem2` our 520 / opp 638（-118）
- `FirstByFeld` / `FirstByJoss` / `DoubleResurrection` / `BurnBothEnds`
  / `NaiveProber` / `RemorsefulProber` 也有 -70 到 -140 的差距。

## 文件

| 文件 | 内容 |
|---|---|
| `matchup_all.csv/json` | 158 类 bank 原始结果 |
| `matchup_protocol.csv/json` | 977 configs 原始结果 |
| `matchup_axelrod_fast.csv/json` | 224 Axelrod 原始结果 |
| `matchup_*_refs.csv` | 参考候选结果 |
| `weak_test.csv` | test-tier 弱点排序 |
| `weak_all.csv` | 158 类弱点排序 |
| `weak_axelrod_fast.csv` | Axelrod 按 diff 排序 |
| `weak_protocol_families.csv` | family 对比 |

## 复现

```bash
python3 -m ipd matchup --candidates rl-wide160 --opponent-set all \
  --reps 16 --rounds 100 --workers 20 --seed 20261036 \
  --out results/full_test_20261036/matchup_all.csv

python3 -m ipd matchup --candidates rl-wide160 --opponent-set protocol \
  --reps 8 --rounds 100 --workers 20 --seed 20261036 \
  --out results/full_test_20261036/matchup_protocol.csv

python3 -m ipd matchup --candidates rl-wide160 --opponent-set axelrod-fast \
  --reps 8 --rounds 100 --workers 20 --seed 20261036 \
  --out results/full_test_20261036/matchup_axelrod_fast.csv

python3 -m ipd matchup \
  --candidates rl-repair,rl-pattern,rl-wide,triple-majority \
  --opponent-set protocol --reps 8 --rounds 100 --workers 20 \
  --seed 20261036 \
  --out results/full_test_20261036/matchup_protocol_refs.csv
```
