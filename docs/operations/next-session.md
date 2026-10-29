# 交班 Prompt（2026-10-27，文档结构重构后）

> 下一 session 直接复制下面这一段作为起始 prompt。

---

你在 `/home/frti64/pri_dile` 继续 IPD（100 轮重复囚徒困境）策略项目。先读：

- `README.md`（当前总览与推荐）
- `docs/operations/handover.md`（交接结论、协议、权重、风险）
- `docs/README.md`（文档状态与阅读顺序）
- `docs/research/long-term-plan.md`
- `docs/architecture/overview.md`
- `results/e0_league_20261026/README.md`
- `results/robust_20261026/README.md`
- `results/hard2_20261026/README.md`
- `results/p3_league_20261026/README.md`
- `docs/research/e1-robust-finetune.md` / `docs/research/e2-risk-ppo.md`
- `docs/research/p2-c1-belief-planner.md`、`docs/research/p2-c2c3-*.md`、`docs/research/p2-c3-*.md`
- `docs/guides/axelrod.md`
- `docs/reference/protocol-v4.md`（**新主口径**）
- `results/protocol_v4/README.md`

## 当前状态

- 最近提交：`git log -1`（RL-Switch 窄 meta 与全量验证）。
- 最近完整测试：**327 tests OK**；`python3 -m ipd list-policies` 为
  **29 个注册候选**。
- 主权重未替换：`weights/rl_policy_weights.json`，sha256
  `bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。
- **RL-Switch（`rl-switch`）是 v4 dev/test 最佳（top 6）**，但未
  promotion，且是 Python-only：默认 RL-Wide160，早期 mutual-D 后
  切换 RL-Repair；test_v7 488.26/487.51。
- **前最佳 / fallback：RL-Wide160 / RL-Wide / RL-Pattern /
  RL-Repair**，均未 promotion。
- RL-Switch 独立 seed 20261029 已复现 test +2.90；Protocol catalog
  suite +0.75；158 bank 持平；red 仍 100%。C++ 双权重移植待做。
- 默认评估口径：**Protocol v4**（`dev_param` / `dev_family` /
  `dev_complex` / `dev_axelrod` / `dev_ood` 选模；test 永久 provisional；
  另有 `dev_level0..4` / `dev_complexity0..4` 分级集；2048 人 / dev 30 /
  test 50）；旧四分布 / blind / v2 / v3 是 legacy。
- `test_v7` / `test_red` **永久锁定**（`consume_test=false`），CLI 无覆盖
  开关；所有 test 运行都是 provisional，不写 usage ledger。
- anti-exploitation 未解决：Red-v5 / Red-v6 对 top 3 仍 100% 胜率。
- **PSRO 已引入：** `psro`（meta-policy）/ `psro-br` 为实验候选；
  PSRO-Meta 在 held-out Red-v5/v6 上 red diff -21.4%，但 dev_param
  527.71 vs 543.35，未 promotion。详见 `docs/research/psro.md`。
- **从零 PSRO：** `psro-scratch` / `psro-scratch-br`；blue 只有
  random/scratch base/BR，已有模型只做 red；dev 480.43，但
  dev_red / Red-v5/v6 red_win 0.0；未 promotion。详见
  `docs/research/psro-scratch.md`。

### 头部候选（Protocol v4 dev top 6；30×2048 sample）

| 策略 | 注册名 | 权重 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | test_v7 prov. |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **RL-Switch** | `rl-switch` | meta（wide160 + repair） | **548.58** | 582.91 | 454.24 | **582.86** | **521.42** | **488.26** |
| RL-Wide160 | `rl-wide160` | `weights/rl_policy_weights_wide160.json` | 548.26 | **583.08** | **454.31** | 582.38 | 521.37 | 485.62 |
| RL-Wide | `rl-wide` | `weights/rl_policy_weights_wide.json` | 548.24 | **583.08** | 452.79 | 582.30 | 520.71 | 485.50 |
| RL-Pattern | `rl-pattern` | `weights/rl_policy_weights_pattern.json` | **548.45** | 583.07 | 447.64 | 582.22 | 519.46 | 485.43 |
| RL-Repair | `rl-repair` | `weights/rl_policy_weights_repair.json` | 547.96 | 582.76 | 434.28 | 582.46 | 515.19 | **486.31** |
| RL-PPO | `rl` | `weights/rl_policy_weights.json` | 540.35 | 582.53 | 440.31 | 574.64 | 513.48 | 477.26 |

`rl-gate` 是学习型上层 gate 实验（阈值 0.9 近中性，test 485.84）；
**No-Go**，不进入提交推荐，详见 `docs/research/rl-gate.md`。

下表是 legacy v3 baseline（299 人 / 20 / 40 fields），只作历史对照：

| 策略 | 注册名 | 权重 | dev_param | dev_family | dev_complex | dev_ood | test_v6 |
|---|---|---|---:|---:|---:|---:|---:|
| **RL-Repair** | `rl-repair` | `weights/rl_policy_weights_repair.json` | **532.44** | 582.92 | **420.38** | **489.84** | **484.32** |
| RL-PPO-Hard60 | `rl-hard60` | `weights/rl_policy_weights_hard60.json` | 527.97 | 582.94 | 410.09 | 485.12 | 476.54 |
| Adv-RL | `adv-rl` | aggressive config | 526.58 | 583.20 | 415.19 | 487.22 | 471.44 |
| RL-PPO（主权重） | `rl` | `weights/rl_policy_weights.json` | 524.37 | 582.72 | 397.90 | 478.61 | - |

Red-v5 / Red-v6 对 top 3 仍 100% 胜率。旧 official/blind/heldout/v2 表已进入
legacy，不能与新表混用。

### 已完成的 Gate 结论

- **E0：PASS。** cross-play / Nash / Elo / exploitability / heldout red
  完整；报告 `results/e0_league_20261026/`。
- **E1c：Go candidate。** repair fine-tune，heldout v2 +3.10、heldout v3
  final +3.91，official/blind 不降；注册 `rl-repair`，未覆盖主权重。
- **E2：No-Go。** CVaR/DRO 没有抬起 heldout CVaR；不注册 `rl-risk`/`rl-dro`。
- **P3：red pool Go；league-lite / expansion No-Go。** league-lite E4
  -12.8%；expansion held-out exploitability 只 -5.9%，Gate D FAIL；E4 red
  仍 100% 击败。
- **PSRO：实现完成；No-Go for promotion。** blue maximin 选
  Final-tuned + Triple-majority；两轮 PPO oracle 未进入 support；
  held-out red diff -21.4%，但 dev 掉约 15.6、family guardrail 失败。
  不要重复同一 2 轮配置；下一步需要更强 oracle 或 red oracle
  （double oracle）。
- **从零 PSRO：No-Go for promotion；anti-exploitation 首次成功。**
  `psro-scratch` dev 480.43、min_norm 0.55，但 dev_red / Red-v5/v6
  red_win 0.0；Scratch-Base 528.99 但 red_win 1.0。下一步不要重复
  同一 fine-tune；应做约束/多目标 oracle（official family floor +
  red objective）或在新 held-out red 上复核。
- **P2 C1/C2/C3：offline No-Go。** 包括 belief planner、binary gate、
  excitation/counterfactual、three-way MoE、repeated calibration / EV probe。
  不要继续扫 C3 threshold 或 planner horizon。
- **SelfPlay 改进：No-Go。** 保留 21 维部署备选。
- **Axelrod：可选功能，已提交。** 243 / 224 / 1 cheating；不要混入常规 CI。

## 下一 session 的任务

按用户优先级选择：

0. **Plan 6：Python 评估瓶颈（已完成，2026-10-28）**
   - v5 sample policy 对手已上 GPU，24/24 候分数逐位一致；
   - Axelrod adapter 历史增量维护；`--fast-rules-threshold` 已加；
   - 严格同序 C++ CPU policy 前向；test_v7 v5 exact 272.77s；
   - `Adv-*` / PSRO meta 保持 CPU；见
     `docs/research/plan6-python-bottleneck.md`。
1. **比赛提交硬化（若近期提交）**
   - 在 v4 上复现 RL-Switch / RL-Wide160 / RL-Wide / RL-Pattern /
     RL-Repair 的 `dev_param` / `dev_family` / `dev_complex` /
     `dev_axelrod` / `dev_ood`；补跑 `dev_level0..4` /
     `dev_complexity0..4` 分级诊断；
   - RL-Switch 已用两个 seed 复现 test +2.64/+2.90，但仍是
     Python-only；先移植 C++ 双权重 + 开关逻辑，再跑 test_red；
   - RL-Wide160 的 test_v7 provisional 低 0.69，作为单模型 fallback；
   - `test_v7` 只做 provisional 诊断，不能当作独立 final；
   - 检查 C++ 导出、主权重 checksum、第 1/100 天行为；
   - 只有用户确认后才替换 `weights/rl_policy_weights.json`。
2. **按全量扫描结果做定向训练（研究首选）**
   - `extort_hard` 已实现但 **No-Go**（PB/complex 回退）；不要再
     加大权重，改做显式 ZD 识别 + expert / portfolio；
   - `periodic_hard` 也已实现但 **No-Go**（test +0.77，PB -41）；
     若继续，应训练 periodic specialist + meta gate；
   - 加入 Gap0.45/0.55/0.64、FSM.F1/F5、ProbeBot、Harvest21 到
     hard mining；
   - 参考 `docs/research/weak-opponents-20261036.md` 和
     `docs/research/extort-hard.md`。
3. **E3 multi-teacher distillation + gating（研究备选）**
   - 教师：RL-Repair / A60 / Adv-RL / SelfPlay / 规则策略；
   - 学生：85 维 MLP / GRU；BC + DAgger + PPO fine-tune；
   - 报告 per-family、min_norm、CVaR、held-out red，而不只看官方均值。
4. **P3 / PSRO anti-exploitation（仅在有新 held-out red 时）**
   - 之前的 red pool / PSRO-lite / 完整 PSRO 都未过 Gate D；不要复读相同实验。
   - 若继续 PSRO，只做更强 blue oracle（更多迭代/规则初始化）或加入 red
     oracle（double oracle）；不要重复当前 2 轮配置。
5. **E9 容量/特征 ablation（可选，低优先级）**
   - 85→128→64 vs 85→64→32，同 init/seed/data；
   - 或一次只加一组新特征（response gap / phase / first-D timing）；
   - 预期收益小；先看 heldout_v2 +3 或 min_norm +0.02。

## 数据/评估纪律

- 新候选用 `dev_param` / `dev_family` / `dev_complex` / `dev_axelrod` /
  `dev_ood` 选模；`dev_level0..4` / `dev_complexity0..4` 是分级诊断；
  test 消费永久锁定。
- `test_v7` / `test_red` 运行结果标记 `selection_eligible=false`，
  不写 usage ledger；不要用 test 结果排名或选模。
- 不把“对训练 red 有效”当成 anti-exploitation 解决；必须有 held-out red
  best response。
- `min_family` 跨 family 不可比；看 `min_norm_family` / per-family /
  per-opponent。
- 任何提升都要同时报告 official、blind、heldout、CVaR、min_norm 和 red。

## 硬性要求

- 先读代码和文档，再写短计划；新模块先写 tests，再训练。
- 不只用官方四分布选模；必须报告 per-family、min_norm、CVaR、held-out red。
- 保留 RL-PPO / A60 / RL-Repair 作为 fallback；**未确认前不要覆盖**
  `weights/rl_policy_weights.json`。
- **Axelrod 已提交，不要 revert 或覆盖**；它是可选功能，不是主线权重。
  Axelrod 是重量级可选依赖（torch/scipy/pandas/dask），版本固定 `4.14.0`。
- **公平 RNG 是硬要求**：planner/opponent-model 评估时，base 与 planner
  必须使用独立、同 seed 的 policy 实例；C1 早期共享 RNG 曾产生 E4 +6.6
  的伪影，已修正。
- `.pt` checkpoint 被 `*.pt` 忽略；提交代码、JSON 权重和报告即可。
- 结束时更新 `README.md` / `docs/operations/handover.md` / `docs/README.md`、checksum、
  Go/No-Go 结论，并提交。

## 建议的第一个动作

```bash
cd /home/frti64/pri_dile
git status                    # 文档重组提交后应为 clean
python3 -m ipd test           # 全量测试
python3 -m ipd list-policies  # 27 candidates
python3 -m ipd protocol audit
python3 -m ipd protocol eval --candidates core --tiers dev_param \
  --fields 1 --n 32 --workers 4
```

确认默认口径和测试；之后根据用户优先级选择“比赛提交硬化”、E3，
或可选的 E9 容量/特征 ablation。不要回到 C3 threshold sweep。

## 复现命令

```bash
# tests
python3 -m unittest discover -s tests -t . -v

# 当前 v4 dev 头部 / 前最佳
python3 -m ipd self-test --candidates rl-wide160,rl-wide,rl-pattern,rl-repair
python3 -m ipd benchmark --candidates rl-wide160,rl-wide,rl-pattern,rl-repair --fields 20 --workers 20
python3 -m ipd blind --candidates rl-wide160,rl-wide,rl-pattern,rl-repair --fields 20 --workers 10

# E0 / E1c 冻结评估
python3 -m ipd league --config configs/eval/e0_league.json --workers 20 \
  --out results/e0_league_20261026/report.json
python3 -m ipd league --config configs/eval/e1c_final_league.json --workers 20 \
  --out results/robust_20261026/e1c_final_v3.json

# Axelrod（可选）
python3 -m pip install -r requirements-axelrod.txt
python3 -m ipd benchmark --candidates core --dists axelrod-fast \
  --fields 1 --n 224 --workers 5
python3 -m ipd matchup --candidates final-tuned --opponent-set axelrod-fast \
  --reps 4 --workers 12

# C3 复现（只用于对照，不要继续调参）
bash results/p2_threeway_moe_20261026/run.sh
bash results/p2_repeated_calib_20261026/run.sh
```

## 已知坑

- **Axelrod 完整 243 很慢**：19 个 `long_run_time`（主要是 Meta）主导
  100 回合耗时；日常用 `axelrod-fast`（224）。
- **Axelrod 内部 3/0/5/1，外部 6/0/10/1**：不要为了统一改内部矩阵，ZD
  四向量会越界。
- **Darwin genome 是类变量**：跨对局/重复/worker 顺序相关；604 只是机制
  检查，不进入正式排名。
- **Axelrod cheating 不是完整旧版家族**：4.14.0 只有 Darwin；
  Geller/MindReader 需要 candidate peek/改写 harness。
- **P2 C3 的信息瓶颈**：calibration 提高 recall 但不提高 precision；
  extort 的少量 false positive 足以让 AllD 崩坏。
- **P3 cross-play 退化**：所有 blue 互相合作（594–595），Elo/Nash 不可区分；
  必须加入 red/exploiter pool。
- **E2 CVaR/DRO 无信号**：不要继续盲调 coef/quantile。
- **规模不可混用**：旧表多为 10/20-field blind 或 v3 的 299 人；v4 用
  2048 人 / dev 30 fields / test 50 fields。
- **test 消费永久锁定**：`test_v7` / `test_red` 只做 provisional 诊断，
  不写 usage ledger，不能用于排名或 final。
- **legacy 表不可混用**：`official` / `blind` / `heldout_v2` / v2 / v3
  只做历史对照。
