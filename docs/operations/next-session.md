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

## 当前状态

- 文档重组前最新提交：`efa9fbb Finalize handover after Axelrod and P2 C3`。
- 工作区在文档重组提交后应为 clean；`python3 -m ipd test` 为
  **183 tests OK**；`python3 -m ipd list-policies` 为 **20 个注册候选**。
- 主权重未替换：`weights/rl_policy_weights.json`，sha256
  `bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。
- **RL-Repair（`rl-repair`）是当前最佳候选**，但未 promotion：
  `weights/rl_policy_weights_repair.json`，sha256
  `a759a7b6e7eb5773aaf0884f9d6c6ca22587f82e1f4fc4c0cc16e983e3c1bd2b`。
- 默认评估口径：四分布 `original/late/grim30/wide`，299 人；正式比较用
  20 fields。Axelrod 是可选库，不改变默认 benchmark/blind。
- anti-exploitation 未解决：E4 red 对当前蓝军仍接近 100% 胜率。

### 头部候选（E0/E1c 冻结协议，20 fields）

| 策略 | 注册名 | 权重 | official | blind 20 | heldout v2 | heldout v3 | E4 red blue |
|---|---|---|---:|---:|---:|---:|---:|
| **RL-Repair** | `rl-repair` | `weights/rl_policy_weights_repair.json` | **554.72** | **564.68** | **503.17** | **509.88** | 303.45 |
| RL-PPO-Hard60 | `rl-hard60` | `weights/rl_policy_weights_hard60.json` | 554.27 | 563.87 | 500.07 | 505.97 | 299.20 |
| RL-PPO（主权重） | `rl` | `weights/rl_policy_weights.json` | 553.22 | 562.80 | 494.97 | 502.31 | 310.55 |
| Adv-RL | `adv-rl` | aggressive config | 554.05 | 562.97 | 502.39 | 504.84 | 325.55 |
| SelfPlay-GRU | `selfplay` | `weights/selfplay_gru_policy.json` | 550.97 | 561.13 | 491.86 | - | 301.65 |

标准 CLI（独立 policy seed）：RL-PPO 553.33 / 562.83、A60 554.36 / 564.17、
RL-Repair 554.76 / 564.61。**10-field blind 与 20-field 不能混表。**

### 已完成的 Gate 结论

- **E0：PASS。** cross-play / Nash / Elo / exploitability / heldout red
  完整；报告 `results/e0_league_20261026/`。
- **E1c：Go candidate。** repair fine-tune，heldout v2 +3.10、heldout v3
  final +3.91，official/blind 不降；注册 `rl-repair`，未覆盖主权重。
- **E2：No-Go。** CVaR/DRO 没有抬起 heldout CVaR；不注册 `rl-risk`/`rl-dro`。
- **P3：red pool Go；league-lite / expansion No-Go。** league-lite E4
  -12.8%；expansion held-out exploitability 只 -5.9%，Gate D FAIL；E4 red
  仍 100% 击败。
- **P2 C1/C2/C3：offline No-Go。** 包括 belief planner、binary gate、
  excitation/counterfactual、three-way MoE、repeated calibration / EV probe。
  不要继续扫 C3 threshold 或 planner horizon。
- **SelfPlay 改进：No-Go。** 保留 21 维部署备选。
- **Axelrod：可选功能，已提交。** 243 / 224 / 1 cheating；不要混入常规 CI。

## 下一 session 的任务

按用户优先级选择：

1. **比赛提交硬化（若近期提交）**
   - 用独立 seeds 复现 RL-Repair 的 official / blind / heldout v2；
   - 检查 C++ 导出、主权重 checksum、第 1/100 天行为；
   - 只有用户确认后才替换 `weights/rl_policy_weights.json`。
2. **E3 multi-teacher distillation + gating（研究首选）**
   - 教师：RL-Repair / A60 / Adv-RL / SelfPlay / 规则策略；
   - 学生：85 维 MLP / GRU；BC + DAgger + PPO fine-tune；
   - 报告 per-family、min_norm、CVaR、held-out red，而不只看官方均值。
3. **P3 anti-exploitation（仅在有新 held-out red 时）**
   - 之前的 red pool / PSRO-lite 都未过 Gate D；不要复读相同实验。
4. **E9 容量/特征 ablation（可选，低优先级）**
   - 85→128→64 vs 85→64→32，同 init/seed/data；
   - 或一次只加一组新特征（response gap / phase / first-D timing）；
   - 预期收益小；先看 heldout_v2 +3 或 min_norm +0.02。

## 数据/评估纪律

- `heldout_v2` 是 validation；`heldout_v3` 已被 E1c 使用；
  **`heldout_v4` 未使用**，下一个通过 validation 的候选只能用它做一次
  final test，之后新建 heldout_v5。
- `final` suite 已多次用于 A/B/C，现在更像 validation，不要再当 final。
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
python3 -m ipd test           # 183 tests OK
python3 -m ipd list-policies  # 20 candidates
python3 -m ipd benchmark --candidates core --fields 8 --workers 8
```

确认默认口径和 183 tests；之后根据用户优先级选择“比赛提交硬化”、E3，
或可选的 E9 容量/特征 ablation。不要回到 C3 threshold sweep。

## 复现命令

```bash
# tests
python3 -m unittest discover -s tests -t . -v

# 当前最佳候选
python3 -m ipd self-test --candidates rl-repair
python3 -m ipd benchmark --candidates rl-repair --fields 20 --workers 20
python3 -m ipd blind --candidates rl-repair --fields 20 --workers 10

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
- **10 vs 20 fields 不可混用**：旧表多为 10-field blind；当前主表用 20-field
  或 E0 冻结协议。
