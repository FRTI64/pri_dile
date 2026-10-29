# RL-Wide160 全量对手扫描：薄弱点与优先级

> **状态（2026-10-29）：诊断完成。**
> 覆盖 158 类对手库、977 个 Protocol v4 config、224 个 Axelrod fast
> 策略，并与 RL-Repair / RL-Pattern / RL-Wide / Triple-majority 对照。
> 结论：最需要修的是 **extortion/ZD、periodic/Kamikaze、Gap/FSM、
> PatternBreaker 尾部、noisy/Joss**；late defector 是安全规则的
> 结构性限制。

## 1. 口径

- `matchup` 对每个对手类型跑 N 次独立 100 回合，不是 2048 人 field；
  分数用于相对比较，不替代 Protocol v4 排名。
- 158 类 bank：16 reps；protocol catalog：8 reps；Axelrod fast：8 reps。
- 参考候选：`rl-repair`、`rl-pattern`、`rl-wide`、`triple-majority`。
- Axelrod 完整 243 的 19 个 long-run Meta 策略在 1h 预算内超时；
  本报告使用 224 fast，不把 long-run 结果当作缺失证据。

## 2. 总体位置

| 集合 | RL-Wide160 均值 |
|---|---:|
| 158 类 bank | 547.35 |
| 158 类 nine-suite avg / worst | 554.45 / 530.23 |
| Protocol catalog | 507.66 |
| Protocol nine-suite avg / worst | 520.43 / 484.55 |
| Axelrod fast | 575.39 |

在 158 / protocol 上与 RL-Pattern、RL-Wide 基本同水平；RL-Repair 在
`smart` / `adaptive` / `extort` 上略好，Triple-majority 在
`frequent_kamikaze` / Gap / GradualKiller 上明显更好。

## 3. 薄弱 family

### Protocol catalog（绝对值）

`early_harvest` 129、`frequent_kamikaze` 169、`hyper_extort` 176、
`probe_exploit` 206、`adaptive_mix` 292、`conditional_extort` 324、
`late` 347、`phase_switch` 385、`endgame_hunter` 397、
`extort` 402、`delayed_extort` 415、`probe_harvest` 429、
`noisy` 462、`pattern_breaker` 468、`temporal` 479。

### 158 类 bank（绝对值）

`late` 477、`random` 507、`wild` 517、`plus` 519、`adaptive` 526。
其中 `late` / `random` 的低分是安全规则和随机噪声的结构性结果；
`plus` / `smart` 的差距更值得关注。

## 4. 最值得修的对手

### P0：Extortion / ZD

- test `hyper_extort`：176；test-only `ExtortZD chi=4.5/5.5/6.0/6.5`
  只有 152–169，而对手 354–437；
- test `extort`：402；`ExtortZD chi=3.75` 267 vs 562；
- test `conditional_extort`：324；`DelayedExtort` 415；
- Axelrod：`ZDExtort2/3/4`、`ZDExtortion`、`ZDMischief`、`ZDSet2`、
  `ZDMem2`、`TF2` 的 diff 在 -100 到 -364；
- 说明当前策略仍会被零行列式 extortionist 稳定剥削，且训练分布中的
  普通 extort 参数不足以覆盖强 chi。

### P0：Periodic / Kamikaze

- test-only `frequent_kamikaze` period=4..9：129–208；Triple 在
  period=9 上能到 571；
- test `temporal:Kamikaze p=.75 period=8`：198；
- 158 bank 的 `Gap0.55` 192、`Gap0.64` 137 也是类似“可预测/可剥削”
  结构；
- 对手按固定周期 D，最优策略应主动利用周期；当前模型仍以合作/报复
  为主，说明缺少“预测到周期后转为收割”的行为。

### P1：Gap / FSM / smart

- 158 bank `Gap0.55` 192 vs best ref 538；`Gap0.64` 137 vs 247；
- `FSM.F5` 569 vs 933，`FSM.F1` 866 vs 951；
- `ProbeBot` 545 vs 593，`Harvest21` 548 vs 588，
  `EarlyProbe` 606 vs 642；
- 这些属于 smart/plus/adaptive，和 Protocol 的 `adaptive_mix` /
  `probe_exploit` / `phase_switch` 弱点一致。

### P1：PatternBreaker 尾部

- 总体 PB 已从 236.8 提升到 511.6；但 test `threshold=.6 window=6`
  仍只有 288（best ref 804）；
- Protocol catalog 还有 `threshold=.75 window=10` 等变体只有 450；
- 说明训练覆盖了部分 PB 参数，但没有覆盖全部可检测周期。

### P1：Noisy / Joss

- test `noisy:Joss p=.52` 276 vs 666；`p=.46` 288 vs 637；
- 158 bank `Random.05` 144、`Random.10` 179、`Random.20` 270；
- 这是“低频随机背叛 → 长期互相 D”的典型问题。所有参考候选也有
  类似低分，说明是安全策略的结构性弱点；但可以通过一次性原谅 /
  短冷却窗口改善。

### P2：Late defector

- test `late` 347；`LateDefector k=75` 214；158 `LateD50` 346；
- 安全规则要求对手先 D，无法在 k 之前收割；这是设计取舍，不建议
  为了 late 单独去掉安全规则。

## 5. 参考候选对比

- **RL-Repair**：在 `adaptive` / `smart` / `extort` 上略好；适合作为
  extortion 方向的对照。
- **RL-Pattern**：PB 更好（protocol 520 vs 468），但其他 family 接近。
- **Triple-majority**：在 `frequent_kamikaze`（263 vs 169）、
  `Gap0.55`（538 vs 192）、`GradualKiller`（946 vs 115）上明显更好；
  但整体均值和大量 family 低于 RL-Wide160。它更像一个“特定结构
  专家”，可作为 gating / portfolio 的候选成员。
- **RL-Wide**：与 RL-Wide160 基本同水平；160/128 的 complex 收益
  没有转化为全量对手优势。

## 6. 建议的下一步（按 ROI）

1. **Extortion-aware 训练**（最高 ROI）
   - 新增 `train_extort_hard`：覆盖 `ExtortZD chi=1.5..7`、
     `ConditionalExtort`、`DelayedExtort`、`ProbeExploit mode=extort`；
   - 或加入 opponent payoff differential / extortion-detection 特征；
   - 目标：test `hyper_extort` +50，test `extort` +30，不牺牲
     dev_param / noisy。
2. **Periodic/Kamikaze 训练 profile**
   - 新增 `train_periodic`：`Kamikaze p_steal=1.0/0.75 period=4..12`、
     `Periodic` patterns、`DynamicWindow` p_defect=.38；
   - 目标：test-only `frequent_kamikaze` 从 169 提到 250+；
     可参考 Triple 的行为。
3. **Gap/FSM hard mining**
   - 把 `Gap0.45/0.55/0.64`、`FSM.F1/F5`、`ProbeBot`、`Harvest21`
     加入 `train_hard` 或新 profile；
   - 目标：158 bank `Gap0.55` 从 192 提到 350+。
4. **PatternBreaker 参数扩展**
   - 训练集目前覆盖 window 6/8/10、threshold .6/.75/.9；test 的
     `.6 window=6` 仍差；增加更多组合 / 用 v2 周期特征做辅助。
5. **Noisy/Joss 原谅规则**
   - 研究“单次 D 后原谅一次/短冷却”的后处理规则；
   - 目标：`Random.05`/`Joss p=.5` 不出现长期 D 螺旋，同时不触发
     Grim 永久惩罚。
6. **Portfolio / gating（最后考虑）**
   - Triple 在 Kamikaze/Gap 上强，RL-Wide160 在多数 family 上强；
     若能低成本识别这些结构，可用 portfolio；但历史 P2/C3 gating
     经验是 No-Go，先做数据/规则验证。

## 7. 复现

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
```

完整表格与 CSV：`results/full_test_20261036/`。
