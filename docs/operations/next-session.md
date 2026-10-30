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
- `results/protocol_v5/README.md`

## 当前状态

- 最近提交：`git log -1`（GRU-DAgger 单模型蒸馏 + C++）。
- 最近完整测试：**346 tests OK**；`python3 -m ipd list-policies` 为
  **33 个注册候选（all 集合）**。
- 主权重未替换：`weights/rl_policy_weights.json`，sha256
  `bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。
- **v5 dev baseline 已运行**（33 candidates，5 tiers × 30 fields × 2048，
  1662.15s）：RL-Gate dev_param 532.82 第一，但 ood min_norm 0.899；
  RL-Switch / GRU-DAgger / RL-Wide / RL-Wide160 约 532 / 477 / 522，
  共同弱项 `probe_exploit`（min_norm ~0.934）。旧 `min_norm ≥ 0.99`
  门槛下无 candidate 通过，**不 promotion**。
- **v5 分级诊断已完成**（`protocol_v5_dev_levels.json`，1605.51s）：
  `probe_exploit` 是共同瓶颈（头部学习策略约 116，RL-PPO 124）；
  `pattern_breaker` 对 RL-Wide160 / RL-Switch / GRU-DAgger 已高于
  RL-PPO（632–639 vs 449–452），RL-Repair 的 PB 才偏低（357）。
  已新增 `train_probe_pattern`，以 RL-Wide160 做定向 fine-tune。
- **定向 fine-tune**：ProbePattern-v2-iter4（RL-Wide160 起点）把
  `dev_complex` 477.77 → **481.81**、`dev_ood` 522.24 → **523.14**、
  `pattern_breaker` 632 → **762**；但 `dev_param` -0.55、
  `probe_exploit` 仍约 **118.1**、min_norm ~0.951。PB 可训，
  probe_exploit 需要新特征 / 显式反 probe / specialist + gate。
  实验产物在 `results/protocol_v5/probe_pattern_v2_20261030/`，
  未注册、未 promotion。
- **probe 专项按 3→1→2 试过**：probe-only specialist 与 v2 特征都没有
  增益；精确规则（first D=5/7 切 AllD）把 probe 拉到 123–129，但
  `dev_param` -1.9～-4.5、`dev_axelrod` -1.1～-2.7，净收益为负。
  probe 专项暂停：**不采用 unsafe specialist / `--allow-probe` /
  预判 gate**，也不为 probe 修改 protocol 或 safety 规则；v5 保持
  977 configs。
- **RL-Switch（`rl-switch`）是 v4 legacy dev/test 最佳（top 7）**，
  但未 promotion；Python + C++ 双权重实现：默认 RL-Wide160，早期
  mutual-D 后切换 RL-Repair；v4 `test_v7` 488.26/487.51；v5 dev
  532.37 / 477.57 / 521.73。
- **GRU-DAgger（`rl-dagger-gru`）是单模型替代**：1 层 GRU（64
  hidden）+ 2 轮 DAgger；v5 dev 532.30 / 477.29 / 522.09；
  Python + C++；未 promotion。详见 `docs/research/dagger-gru.md`。
- **前最佳 / fallback：RL-Wide160 / RL-Wide / RL-Pattern /
  RL-Repair**，均未 promotion。
- **最佳纯规则：Triple-v4（`triple-v4`）**：v4 legacy dev_param
  538.66 / dev_complex 447.08 / dev_ood 514.41；v5 dev 518.88 /
  488.37 / 518.90；v4 `test_v7` provisional 473.68；无权重、无 probe；
  未 promotion。v5 的 `dev_complex` 高于学习策略，但 `dev_param`
  明显低约 14 分。
- RL-Switch 独立 seed 20261029 已复现 test +2.90；Protocol catalog
  suite +0.75；158 bank 持平；level/complexity 高档 tier 小幅提升；
  blue-red 已停用。C++ 双权重已完成 action parity，待接真实接口。
- 默认评估口径：**Protocol v5**（`dev_param` / `dev_family` /
  `dev_complex` / `dev_axelrod` / `dev_ood` 选模；`test_v8` 新 final；
  另有 `dev_level0..4` / `dev_complexity0..4` 分级集；2048 人 / dev 30 /
  test 50）；v4 / v3 / 旧四分布 / blind 都是 legacy。
- `test_v8` / `test_red` **初始锁定**（ADR 0013）：provisional 用
  `--no-consume`；final 前先在 `configs/splits/protocol_v5/test_policy.json`
  解禁 `test_v8`，再用 `--acknowledge-once` 写
  `results/protocol_v5/usage.json`。消费后新候选需要 `test_v9`。
- **blue-red 已停用（ADR 0011）：** 不再用 red / exploitability 作为
  评估或 promotion 门槛，也不新增 red 任务。
- **PSRO / red 历史：** `psro` / `psro-br` / `psro-scratch` /
  `psro-scratch-br` / `psro-v4` / `psro-v4-br` 仍注册为历史实验
  候选；dev 与红队数字只用于复现，不参与当前推荐。

### 头部候选（v4 legacy dev top 7；30×2048 sample；v5 待重跑）

| 策略 | 注册名 | 权重 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | test_v7 prov. |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **RL-Switch** | `rl-switch` | meta（wide160 + repair） | **548.58** | 582.91 | 454.24 | **582.86** | **521.42** | **488.26** |
| **GRU-DAgger** | `rl-dagger-gru` | `weights/rl_dagger_gru_weights.json` | **548.58** | 582.90 | **455.13** | 582.49 | **521.54** | 487.89 |
| RL-Wide160 | `rl-wide160` | `weights/rl_policy_weights_wide160.json` | 548.26 | **583.08** | **454.31** | 582.38 | 521.37 | 485.62 |
| RL-Wide | `rl-wide` | `weights/rl_policy_weights_wide.json` | 548.24 | **583.08** | 452.79 | 582.30 | 520.71 | 485.50 |
| RL-Pattern | `rl-pattern` | `weights/rl_policy_weights_pattern.json` | **548.45** | 583.07 | 447.64 | 582.22 | 519.46 | 485.43 |
| RL-Repair | `rl-repair` | `weights/rl_policy_weights_repair.json` | 547.96 | 582.76 | 434.28 | 582.46 | 515.19 | **486.31** |
| RL-PPO | `rl` | `weights/rl_policy_weights.json` | 540.35 | 582.53 | 440.31 | 574.64 | 513.48 | 477.26 |

`rl-gate` 是学习型上层 gate 实验（阈值 0.9 近中性，test 485.84）；
**No-Go**，不进入提交推荐，详见 `docs/research/rl-gate.md`。

纯规则头部是 **Triple-v4**（`triple-v4`）：536.87 → 538.66
（dev_param）、441.17 → 447.08（dev_complex）、511.62 → 514.41
（dev_ood），dev_family 不变；独立 seed 复现。它仍低于
RL-Repair / RL-PPO，只作纯规则 fallback。报告
`docs/research/triple-v4.md`、`results/rule_v4_20261043/`。

下表是 legacy v3 baseline（299 人 / 20 / 40 fields），只作历史对照：

| 策略 | 注册名 | 权重 | dev_param | dev_family | dev_complex | dev_ood | test_v6 |
|---|---|---|---:|---:|---:|---:|---:|
| **RL-Repair** | `rl-repair` | `weights/rl_policy_weights_repair.json` | **532.44** | 582.92 | **420.38** | **489.84** | **484.32** |
| RL-PPO-Hard60 | `rl-hard60` | `weights/rl_policy_weights_hard60.json` | 527.97 | 582.94 | 410.09 | 485.12 | 476.54 |
| Adv-RL | `adv-rl` | aggressive config | 526.58 | 583.20 | 415.19 | 487.22 | 471.44 |
| RL-PPO（主权重） | `rl` | `weights/rl_policy_weights.json` | 524.37 | 582.72 | 397.90 | 478.61 | - |

blue-red 已停用（ADR 0011）；旧 official/blind/heldout/v2 表和
red/PSRO 数字都只作历史存档，不能与新表混用。

### 已完成的 Gate 结论（red/PSRO 为历史）

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
- **PSRO v4：No-Go for promotion；learned BR 首次入 support。**
  BR_01 权重 1.0，exploitability 290.00 → 0.00；`psro-v4-br`
  dev_param 486.93（-59.5 vs RL-Repair）、min_norm 0.56，但
  dev_red / Red-v5/v6 red_win 0.0。下一步同样应做约束/多目标 oracle
  或 red oracle；不要重复同一 3 轮 blue-only 配置。
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
   - 严格同序 C++ CPU policy 前向；v4 `test_v7` exact 272.77s；
   - `Adv-*` / PSRO meta 保持 CPU；见
     `docs/research/plan6-python-bottleneck.md`。
1. **比赛提交硬化（若近期提交）**
   - v5 dev baseline 已跑；下一步先分析 `probe_exploit`（RL-Switch /
     GRU-DAgger / RL-Wide / RL-Wide160 的 ood min_norm ~0.934）和
     `pattern_breaker`（RL-Repair 0.790），决定 v5 family gate 或做定向
     training；补跑 `dev_level0..4` / `dev_complexity0..4` 分级诊断；
   - RL-Switch 已用两个 seed 在 v4 复现 test +2.64/+2.90；C++ 双权重 +
     开关逻辑已完成 action parity；GRU-DAgger 两个 seed dev 持平，
     C++ 单模型也已完成 parity；接下来接真实比赛接口（不再跑
     test_red，blue-red 已停用）；
   - RL-Wide160 的 v4 `test_v7` provisional 低 0.69，作为单模型 fallback；
   - 选定最终候选后，解禁 `test_v8` 并跑一次 final
     （`--acknowledge-once`，写 usage ledger）；v4 provisional 数字不能
     当作 v5 final；
   - 检查 C++ 导出、主权重 checksum、第 1/100 天行为；
   - 只有用户确认后才替换 `weights/rl_policy_weights.json`。
2. **按全量扫描结果做定向训练（研究首选）**
   - `extort_hard` 已实现但 **No-Go**（PB/complex 回退）；不要再
     加大权重，改做显式 ZD 识别 + expert / portfolio；
   - `periodic_hard` 也已实现但 **No-Go**（test +0.77，PB -41）；
     若继续，应训练 periodic specialist + meta gate；
   - **probe_exploit 已试 specialist / v2 特征 / 显式规则三条线**：
     `train_probe_pattern` 把 `pattern_breaker` 提升到 762，但
     `probe_exploit` 仍约 118；probe-only specialist 和 v2 特征无增益；
     精确规则可到 123–129 但整体 dev 回退。probe 专项暂停：不采用
     unsafe specialist / `--allow-probe` / 预判 gate，也不为 probe
     修改 protocol 或 safety 规则；v5 保持 977 configs；
   - 旧 MLP 蒸馏 **No-Go**（normalization bug；修正后仍受
     pattern_breaker 限制）；GRU + 2 轮 DAgger 已得到单模型
     `rl-dagger-gru`，dev 与 RL-Switch 持平；下一步是外部比赛/
     独立 seed 验证，而不是继续调 dev；
   - 加入 Gap0.45/0.55/0.64、FSM.F1/F5、ProbeBot、Harvest21 到
     hard mining；
   - **规则线已完成一轮**：`triple-v4` 在 Final 加入噪声互惠分支，
     dev_param +1.79 / dev_complex +5.91 / dev_ood +2.79，独立 seed
     复现；若继续，做显式 ZD / probe-style 响应，不要扫同一组
     Triple 参数，也不要加入全局主动 probe；
   - 参考 `docs/research/weak-opponents-20261036.md` 和
     `docs/research/extort-hard.md`。
3. **E3 multi-teacher distillation + gating（研究备选）**
   - 教师：RL-Repair / A60 / Adv-RL / SelfPlay / 规则策略；
   - 学生：85 维 MLP / GRU；BC + DAgger + PPO fine-tune；
   - 报告 per-family、min_norm、CVaR 和外部比赛结果，而不只看官方均值。
4. **blue-red / PSRO anti-exploitation：不安排**
   - 用户已明确停用 blue-red 对战（ADR 0011）；不要新增 red pool、
     PSRO anti-exploitation 或 red oracle 任务。
5. **E9 容量/特征 ablation（可选，低优先级）**
   - 85→128→64 vs 85→64→32，同 init/seed/data；
   - 或一次只加一组新特征（response gap / phase / first-D timing）；
   - 预期收益小；先看 heldout_v2 +3 或 min_norm +0.02。

## 数据/评估纪律

- 新候选用 `dev_param` / `dev_family` / `dev_complex` / `dev_axelrod` /
  `dev_ood` 选模；`dev_level0..4` / `dev_complexity0..4` 是分级诊断。
- `test_v8` 初始锁定；provisional 诊断用 `--no-consume`。只在最终候选
  确定后解禁，再用 `--acknowledge-once` 写 usage ledger；不参与排名、
  调参或选模。`test_red` 继续锁定。
- blue-red 已停用；不再要求 red best response，也不把 red 结果当成
  评估或 promotion 证据。
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

# Protocol v5：审计 + dev 选模
python3 -m ipd protocol audit
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v5/protocol_v5_dev.json

# v4 legacy dev 头部 / 前最佳（只作历史复现）
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
- **Axelrod 内部与外部统一 6/0/10/1**：adapter 传入项目矩阵；ZD 的
  `phi` 会自动降到可行范围，斜率/基线不变；6 组历史阈值已按策略原理
  用 `T-S`、`R`、`P` 重写。
- **Darwin genome 是类变量**：跨对局/重复/worker 顺序相关；604 只是机制
  检查，不进入正式排名。
- **Axelrod cheating 不是完整旧版家族**：4.14.0 只有 Darwin；
  Geller/MindReader 需要 candidate peek/改写 harness。
- **P2 C3 的信息瓶颈**：calibration 提高 recall 但不提高 precision；
  extort 的少量 false positive 足以让 AllD 崩坏。
- **P3 cross-play 退化**：所有 blue 互相合作（594–595），Elo/Nash 不可区分；
  必须加入 red/exploiter pool。
- **E2 CVaR/DRO 无信号**：不要继续盲调 coef/quantile。
- **规模不可混用**：旧表多为 10/20-field blind 或 v3 的 299 人；v4/v5
  用 2048 人 / dev 30 fields / test 50 fields。
- **test 消费**：`test_v8` / `test_red` 初始锁定；provisional 用
  `--no-consume`，final 前先解禁 `test_v8` 再传 `--acknowledge-once`。
  消费后新候选需要 `test_v9`。
- **legacy 表不可混用**：`official` / `blind` / `heldout_v2` / v2 / v3 / v4
  只做历史对照。
