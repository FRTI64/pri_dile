# 交班 Prompt（2026-10-27，Axelrod + P2 C3 之后）

> 下一 session 直接复制下面这一段作为起始 prompt。

---

你在 `/home/frti64/pri_dile` 继续 IPD（100 轮重复囚徒困境）策略项目。先读：

- `HANDOVER.md`
- `docs/ipd-long-term-improvement-plan.md`
- `results/p3_league_20261026/README.md`
- `docs/p3-expansion-plan.md`
- `docs/AXELROD_OPPONENTS.md`
- `results/axelrod_20261027/README.md`
- `docs/p2-c3-threeway-moe.md`
- `results/p2_threeway_moe_20261026/README.md`
- `docs/p2-c3-repeated-calib.md`
- `results/p2_repeated_calib_20261026/README.md`
- `docs/p2-c2c3-excitation.md`
- `results/p2_excitation_20261026/README.md`
- `docs/p2-c2c3-opponent-model.md`
- `docs/p2-c1-belief-planner.md`

## 当前状态

- 最新研究提交：`973de9c Add optional Axelrod opponent bank`。
- **工作区在交班提交后应为 clean**：Axelrod 可选对手库已提交；
  `HANDOVER.md` 顶部状态 note 与本 prompt 一起提交。
  当前 `python3 -m ipd test` 为 **183 tests OK**。
- 主权重未替换：
  `weights/rl_policy_weights.json` sha256
  `bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`；
  `weights/rl_policy_weights_repair.json` sha256
  `a759a7b6e7eb5773aaf0884f9d6c6ca22587f82e1f4fc4c0cc16e983e3c1bd2b`。
- 默认评估口径仍是四分布（original/late/grim30/wide）和 85/158 对手集合；
  Axelrod 是**可选**对手库，不改变默认 benchmark/blind 结果。

### 基线与候选

| 策略 | 注册名 | 权重 | official 20-field | blind 20-field | heldout v2 | E4 red blue |
|---|---|---|---:|---:|---:|---:|
| RL-PPO（主基线） | `rl` | `weights/rl_policy_weights.json` | 553.33 | 563.23 / 8.00 | 494.97 | 310.55 |
| RL-PPO-Hard60（A60 fallback） | `rl-hard60` | `weights/rl_policy_weights_hard60.json` | 554.36 | 564.17 / 5.75 | 500.07 | 299.20 |
| **RL-Repair（当前最佳候选）** | `rl-repair` | `weights/rl_policy_weights_repair.json` | **554.76** | **564.61 / 5.75** | **503.17** | **303.45** |
| Adv-RL | `adv-rl` | aggressive config | 554.29 | 562.91 / 8.00 | 502.39 | 325.55 |
| SelfPlay-GRU | `selfplay` | `weights/selfplay_gru_policy.json` | 550.76 | 560.43 / 8.10 | 491.86 | 301.65 |

- **RL-Repair 仍是 Go candidate，但未 promotion**；heldout v3 final
  509.88（A60 505.97）。如果要替换主权重，先做独立 seed 复现并得到用户确认。
- **21 维不是硬约束**；主线继续用 85 维 RL-PPO / RL-Repair。SelfPlay-GRU
  只作为 21 维/无 PyTorch 部署备选。

### 已完成的 Gate 结论

- **E0：PASS。** `python3 -m ipd league` 冻结 cross-play / Nash / Elo /
  exploitability / heldout red；报告 `results/e0_league_20261026/`。
- **E1c：Go candidate。** `repair` fine-tune，heldout v2 +3.10、heldout v3
  final +3.91，official/blind 不降；注册 `rl-repair`，未覆盖主权重。
- **E2：No-Go。** CVaR/DRO 没有把 heldout CVaR 抬起；不注册 `rl-risk`/`rl-dro`。
- **P3 red pool：第一里程碑 Go。** Red-RL-v1 在全部 held-out blue 上强于
  E4 red；但 **P3 league-lite / expansion 均 No-Go**：
  league-lite E4 -12.8%；expansion 用 v1-v6 red + maximin 权重，held-out
  exploitability 只 390.33→367.33（**-5.9%**），official +0.24 通过但
  `validation:extort` -3.21%，Gate D FAIL。E4 red 仍 100% 击败当前 blue。
  不要继续盲调 red fraction / red 数量。
- **P2 C1：No-Go（fair RNG）。** belief planner 激进版 validation -12.11 /
  heldout_v2 -16.15；保守版 validation -2.19 / heldout_v2 +2.06，
  held-out red E4 +0.35、v5 0.00、v6 -0.10。早期共享 base RNG 的
  E4 +6.6 是伪影，已修正。诊断是 opponent model discrimination / excitation，
  不是 planning horizon。
- **P2 C2/C3 binary gate：No-Go。** heldout_v2 gate 70.6%，false positives
  全部来自 adaptive/noisy；gated conservative +1.20，红色无变化。
- **C2/C3 exploratory/counterfactual 数据：数据层 Go / planner No-Go。**
  `excited` 率 25.8%→59.3%；calibrated/mixed gate 改善，但部署轨迹仍
  excitation 不足；conservative planner 3-seed +0.88。
- **C3 three-way gate + MoE：No-Go。** safe/risky/unknown + safe→AllD；
  5-seed heldout +4.04（validation +0.53），best-validation +0.90/+1.41；
  safe→planner 3-seed -1.28；红色 0.00。
- **C3 repeated counterfactual 校准：No-Go。** 每 state 重复 10 次 C/D；
  calibrated safe precision/recall 74.7%/62.1%（旧 82.6%/31.7%）。
  no-veto AllD 在 extort 崩坏（-137.1）；response veto 0.2 后 5-seed
  validation -0.31、heldout +2.93、safe rate 2.5%。
- **C3 EV probe screen：No-Go。** 一次性 belief planner override 最好
  1-seed +1.29，3-seed -5.22；无条件 early probe -10.7~-20.8。
  不要继续调 C3 threshold；需要显式 value-of-information 或更好的后验/特征。
- **Axelrod 可选对手库（已提交 973de9c）**：243 普通 / 224 fast / 1
  cheating (Darwin)；默认依赖不变，`axelrod==4.14.0` 仅在
  `requirements-axelrod.txt` 或 `pip install -e '.[axelrod]'` 时安装。
  224 fast × 100 回合：Triple 560.95、Resp3 555.37、Final 555.27、
  Bayes 546.25（不能与四分布 554.x 直接比较）。

## 任务（下一 session）

### 1. Axelrod 已提交，保留为可选评估

`973de9c` 已提交 Axelrod 243/224/cheating 对手库和 smoke 结果；默认四分布、
85/158 对手集合、主权重不变。不要把它混入常规 CI 或主线选模。

可选后续：
- 用 `axelrod-fast` 评估 RL-Repair / A60 / Adv-RL / SelfPlay（目前只有
  core 规则策略的 smoke 结果）；
- 结果只作为额外 validation / 诊断，不要当 final test；
- 完整 `axelrod`（243）很慢，日常用 `axelrod-fast`（224）。

### 2. 再决定主线

P2 C1/C2/C3 已经连续 No-Go，**不要继续调 C3 threshold / planner horizon**。
优先级建议：

1. **比赛提交硬化（若近期要提交）**
   - 用独立 seeds 复现 RL-Repair 的 official/blind/heldout v2/v3；
   - 检查 C++ 导出、主权重 checksum、第 1/100 天行为；
   - 只有用户确认后才替换 `weights/rl_policy_weights.json`。
2. **E3 multi-teacher distillation + gating（研究首选）**
   - 教师：RL-Repair / A60 / Adv-RL / SelfPlay / 规则策略；
   - 学生：85 维 MLP / GRU；BC + DAgger + PPO fine-tune；
   - Gate C 之外也要报告 per-family、min_norm、CVaR、held-out red。
3. **P3 anti-exploitation（仅在有新 held-out red 时）**
   - 之前的 red pool / PSRO-lite 都未过 Gate D；不要复读相同实验。
4. **E9 容量/特征 ablation（可选，低优先级）**
   - 85→128→64 vs 85→64→32，同 init/seed/data；
   - 或一次只加一组新特征（response gap / phase / first-D timing），
     不要只重复 last moves；
   - 预期收益小；先看 heldout_v2 +3 或 min_norm +0.02。

### 3. 数据/评估纪律

- `heldout_v2` 是 validation；`heldout_v3` 已被 E1c 使用；
  **`heldout_v4` 仍未使用**，下一个通过 validation 的候选只能用它做一次
  final test，之后新建 heldout_v5。
- `final` suite 已多次用于 A/B/C，现在更像 validation，不要再当 final。
- 不把“对训练 red 有效”当成 anti-exploitation 解决；必须有 held-out red
  best response。

## 硬性要求

- 先读代码和文档，再写短计划；新模块先写 tests，再训练。
- 不只用官方四分布选模；必须报告 per-family、min_norm、CVaR、held-out red。
- 保留 RL-PPO / A60 / RL-Repair 作为 fallback；**未确认前不要覆盖**
  `weights/rl_policy_weights.json`。
- **Axelrod 已提交（973de9c），不要 revert 或覆盖**；它是可选功能，不是
  主线权重。
- Axelrod 是重量级可选依赖（torch/scipy/pandas/dask）；默认安装保持零第三方
  依赖，版本固定 `4.14.0`。
- **公平 RNG 是硬要求**：planner/opponent-model 评估时，base 与 planner
  必须使用独立、同 seed 的 policy 实例；C1 早期共享 RNG 曾产生 E4 +6.6
  的伪影，已修正。
- `.pt` checkpoint 被 `*.pt` 忽略；提交代码、JSON 权重和报告即可。
- 结束时更新 README/HANDOVER、checksum、Go/No-Go 结论，并提交。

## 建议的第一个动作

```bash
cd /home/frti64/pri_dile
git status                    # 交班提交后应为 clean
python3 -m ipd test           # 183 tests OK
python3 -m ipd benchmark --candidates core --fields 8 --workers 8
```

确认默认口径和 183 tests；之后根据用户优先级选择“比赛提交硬化”、E3，
或可选的 E9 容量/特征 ablation。不要回到 C3 threshold sweep。

## 复现命令

```bash
# tests
.venv/bin/python -m unittest discover -s tests -v

# 当前最佳候选
.venv/bin/python -m ipd self-test --candidates rl-repair
.venv/bin/python -m ipd benchmark --candidates rl-repair --fields 20 --workers 20
.venv/bin/python -m ipd blind --candidates rl-repair --fields 20 --workers 10

# Axelrod（可选）
python3 -m pip install -r requirements-axelrod.txt
python3 -m ipd benchmark --candidates core --dists axelrod-fast \
  --fields 1 --n 224 --workers 5
python3 -m ipd matchup --candidates final-tuned --opponent-set axelrod-fast \
  --reps 4 --workers 12
python3 -m ipd benchmark --candidates core --dists axelrod-cheating \
  --fields 1 --n 1 --workers 4

# C3 复现（只用于对照，不要继续调参）
bash results/p2_threeway_moe_20261026/run.sh
bash results/p2_repeated_calib_20261026/run.sh
PYTHONPATH=. .venv/bin/python \
  results/p2_repeated_calib_20261026/bayes_probe_screen.py
```

## 已知坑

- **Axelrod 完整 243 很慢**：19 个 `long_run_time`（主要是 Meta）会主导
  100 回合耗时；日常用 `axelrod-fast`（224），完整集只做定向分析或
  `matchup --rounds 10` smoke。
- **Axelrod 内部 3/0/5/1，外部 6/0/10/1**：不要为了统一改内部矩阵，ZD
  四向量会越界。
- **Darwin genome 是类变量**：跨对局/重复/worker 顺序相关；604 只是机制
  检查，不进入正式排名。
- **Axelrod cheating 不是完整旧版家族**：4.14.0 只有 Darwin；
  Geller/MindReader 需要 candidate peek/改写 harness。
- **P2 C3 的信息瓶颈**：calibration 提高 recall 但不提高 precision；
  extort 的少量 false positive 足以让 AllD 崩坏。不要继续扫 threshold。
- **P3 cross-play 退化**：所有 blue 互相合作（594–595），Elo/Nash 不可区分；
  必须加入 red/exploiter pool。
- **E2 CVaR/DRO 无信号**：不要继续盲调 coef/quantile。
- **`min_family` 跨 family 不可比**：看 `min_norm_family` / per-family /
  per-opponent。

## 已提交 commits

```text
9a4c931 Update handover prompt after Axelrod and P2 C3
973de9c Add optional Axelrod opponent bank (243/224/cheating)
e7475e0 P2 C3 repeated counterfactual calibration (No-Go)
601426c P2 C3 three-way gate + MoE (No-Go)
b19d4f9 P2 C2/C3 exploratory/counterfactual data (data Go / planner No-Go)
7ab85fb P2 C2/C3 learned opponent model + gate (No-Go); fair-RNG C1 correction
b9abfbe P2 C1 belief planner offline prototype (No-Go)
c4da6c0 P3 expansion: multi-red pool + PSRO-lite (Gate D No-Go)
5a62daa Update handover prompt after E0/E1/E2/P3
09a9ec2 Add P3 red pool and league-lite blue response
467c42b E2c worst-family DRO / No-Go
b445969 E2 CVaR risk PPO / No-Go
c95371b E1c repair fine-tune + register rl-repair
501fe46 E0 league harness + E1 robust fine-tune evaluation
```
