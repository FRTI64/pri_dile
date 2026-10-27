# Axelrod 对手库与 cheating 测试集

状态：可选功能，已接入 benchmark/matchup/eval-split；不改变原有四分布和
85/158 对手集合的默认口径。

依赖：`axelrod==4.14.0`（`requirements-axelrod.txt` 或
`pip install -e '.[axelrod]'`）。Axelrod 会带入 torch/scipy/pandas/dask，
因此不放进默认依赖。

## 1. 三个对手集合

| 名称 | 数量 | 内容 |
|---|---:|---|
| `axelrod` | 243 | `axelrod.strategies`，服从 Axelrod 锦标赛规则的普通策略 |
| `axelrod-fast` | 224 | 从 243 中剔除 19 个 `long_run_time` 策略（17 个 Meta + `DBS` + `EvolvedAttention`） |
| `axelrod-cheating` | 1 | `axelrod.cheating_strategies`；4.14.0 只有 `Darwin` |

`axelrod` 不包含 `Darwin`；cheating 集是独立的额外测试集，不混入普通集。
`axelrod-cheating` 按 `axelrod.cheating_strategies` 动态构建，但旧版
`Geller`/`MindReader`/`MindController` 系列在 4.13/4.14 发布包中已移除，
当前适配器也没有实现它们所需的“读取候选下一步/改写候选策略”能力。若未来
Axelrod 恢复这些类，需要先补适配器，不能假设它们能直接跑。

## 2. 适配器

文件：`ipd/core/opponents/axelrod_bank.py`

- `AxelrodOpponent(Strategy)` 把 `axelrod.Player` 的 `strategy(opponent)`
  接口接到本项目的 `act(t, mine, theirs)`。
- 延迟导入 Axelrod：默认 `import ipd` 不会导入 torch/scipy。
- 每回合用 `Player.update_history` 增量维护历史；Meta 策略的 team history
  也会随之更新。
- `set_rounds(rounds)` 把比赛长度写进 `match_attributes["length"]`；
  `ipd/evaluation/common.py::run_match` 和 `ipd/core/simulator.py::play_match`
  已加可选 hook。
- 随机策略通过派生 seed 调用 `set_seed`，同一 adapter 的 `reset()` 可复现。
- 内部收益矩阵使用 Axelrod 标准 **3/0/5/1**：零行列式策略的四向量按该矩阵
  校准，直接用本项目的 6/0/10/1 会产生负概率。实际对局得分仍由
  `ipd/core/simulator.py` 的 **6/0/10/1** 计算。

## 3. 命令

```bash
# 安装
python3 -m pip install -r requirements-axelrod.txt

# 224 个非 long-run 策略，适合日常跑分
python3 -m ipd benchmark --candidates core \
  --dists axelrod-fast --fields 1 --n 224 --workers 5
python3 -m ipd matchup --candidates final-tuned \
  --opponent-set axelrod-fast --reps 4 --workers 12

# 完整 243 个普通策略（含 19 个很慢的 long-run 策略）
python3 -m ipd benchmark --candidates final-tuned \
  --dists axelrod --fields 1 --n 243
python3 -m ipd matchup --candidates final-tuned \
  --opponent-set axelrod --reps 1 --rounds 10 --workers 12

# cheating 额外测试集（4.14.0: Darwin）
python3 -m ipd benchmark --candidates core \
  --dists axelrod-cheating --fields 1 --n 1 --workers 4
python3 -m ipd matchup --candidates final-tuned \
  --opponent-set axelrod-cheating --reps 4
```

`benchmark --dists` 可以一次传多个集合，例如
`--dists axelrod,axelrod-cheating`；这是在同一个 run 里分别生成两个分布，
不是把 244 个策略混成一个 field。若需要混合 field，需要新增
`axelrod-all` 采样器。

`matchup` 新增 `--rounds`（默认 100），用于全量 243 的快速 smoke。

`eval-split` 的配置也接受上述分布名（`EVAL_DISTS`），但默认仍是四分布。

## 4. 结果（2026-10-27）

目录：`results/axelrod_20261027/`

### 224 fast，100 回合，1 个 field

| 候选 | 平均分/对手 |
|---|---:|
| Triple-majority | 560.95 |
| Resp3-tuned | 555.37 |
| Final-tuned | 555.27 |
| Bayes-tuned | 546.25 |

这些分数不能和官方四分布的 554.x 直接比较：对手池和采样方式不同。

### cheating（Darwin），100 回合

| 候选 | 平均分/对手 |
|---|---:|
| Final-tuned | 604.00 |
| Resp3-tuned | 604.00 |
| Bayes-tuned | 604.00 |
| Triple-majority | 604.00 |

604 是 99 回合 CC + 最后一天候选出 D 的固定收益，Darwin 的 genome 在这些
对局中保持全 C；它是机制/鲁棒性检查，不是区分策略强弱的指标。

### 完整 243 的 smoke

`matchup --opponent-set axelrod --reps 1 --rounds 10 --workers 12`：
243/243 类型跑通，Final-tuned 平均 53.46（10 回合，不可与 100 回合分数比较）。

## 5. 性能

- 224 fast、1 个 field、100 回合：约 12 秒；core 四个候选并行约 12 秒。
- `MetaMajority` 100 回合单场约 30 秒，19 个 long-run 策略会主导完整 243
  的耗时。
- 完整 243、10 回合、12 workers：约 4 分 42 秒；100 回合预计几十分钟。
  日常跑分请用 `axelrod-fast`。

## 6. 风险与交接注意

1. **完整 243 很慢**：主要来自 Meta 每回合模拟整个策略库；不要把它放进
   常规 CI 或日常四分布循环。
2. **cheating 集目前只有 Darwin**：旧版 Geller/MindReader 系列不在 4.14
   发布包中，且需要 harness 级 peek/改写支持；不要把它们写进“已支持”。
3. **Darwin 是跨对局/顺序相关的**：`genome` 是类变量，重复运行、
   worker 调度和对手顺序都会影响结果；不要把单次 604 当作稳定强度。
4. **内部用标准 3/0/5/1，外部用 6/0/10/1**：这是有意为之；若有人改成
   6/0/10/1，ZD 策略会因四向量越界而失败。
5. **可选依赖较重**：Axelrod 会带入 torch/scipy/pandas/dask；默认安装仍
   零第三方依赖，只有显式使用 Axelrod 集时才需要。
6. **版本固定为 4.14.0**：243/224/1 的数量和测试断言都按该版本写死；升级
   Axelrod 前先跑 `tests/test_axelrod_opponents.py` 并重新确认 cheating 类
   是否可适配。

## 7. 验证

```bash
python3 -m ipd test
# 183 tests, OK
```

新增测试：`tests/test_axelrod_opponents.py`，覆盖 243/224/1 数量、
`Darwin` 排除/单独集合、适配器对局与 reset、随机种子、
`set_rounds`、采样、benchmark CLI、matchup worker。

## 8. 后续可选工作

- 若要“244 全量混合 field”，新增 `axelrod-all` 采样器。
- 若要旧版 Geller/MindReader，先设计 candidate peek/monkey-patch 的隔离
  harness，再移植；不要直接复用当前 history-proxy 适配器。
- 若要长期跟踪全量 243，建议把结果拆成 per-opponent 文件或用
  `matchup --opponent-set axelrod`，避免单个 benchmark field 被 Meta 拖慢。
