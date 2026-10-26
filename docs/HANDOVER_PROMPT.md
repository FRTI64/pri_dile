# 交班 Prompt（2026-10-26，E0/E1/E2/P3 之后）

> 下一 session 直接复制下面这一段作为起始 prompt。

---

你在 `/home/frti64/pri_dile` 继续 IPD（100 轮重复囚徒困境）策略项目。先读：

- `docs/ipd-long-term-improvement-plan.md`
- `HANDOVER.md`
- `docs/E0_LEAGUE.md`
- `docs/e1-robust-finetune.md`
- `docs/e2-risk-ppo.md`
- `docs/p3-red-league.md`
- `docs/RL_POLICY.md`
- `results/e0_league_20261026/README.md`
- `results/robust_20261026/README.md`
- `results/risk_20261026/README.md`
- `results/red_pool_20261026/README.md`

## 当前状态

### 基线与候选

| 策略 | 注册名 | 权重 | official 20-field | blind 20-field | heldout v2 | E4 red blue |
|---|---|---|---:|---:|---:|---:|
| RL-PPO（主基线） | `rl` | `weights/rl_policy_weights.json` | 553.33 | 563.23 / 8.00 | 494.97 | 310.55 |
| RL-PPO-Hard60（A60 fallback） | `rl-hard60` | `weights/rl_policy_weights_hard60.json` | 554.36 | 564.17 / 5.75 | 500.07 | 299.20 |
| **RL-Repair（当前最佳候选）** | `rl-repair` | `weights/rl_policy_weights_repair.json` | **554.76** | **564.61 / 5.75** | **503.17** | **303.45** |
| Adv-RL | `adv-rl` | aggressive config | 554.29 | 562.91 / 8.00 | 502.39 | 325.55 |
| SelfPlay-GRU | `selfplay` | `weights/selfplay_gru_policy.json` | 550.76 | 560.43 / 8.10 | 491.86 | 301.65 |

- **主权重未替换**：`weights/rl_policy_weights.json` 的 sha256 仍是
  `bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4`。
- **RL-Repair（E1c）是 Go candidate**：heldout v2 503.17（A60 +3.10）、
  heldout v3 final 509.88（A60 +3.91）；official +0.45、blind +0.80；
  min_norm 0.9930、CVaR10 390.39、E4 red blue 303.45。已注册但未
  promotion；如果要替换主权重，先做独立 seed 复现并得到用户确认。
- **21 维不是硬约束**。主线继续用 85 维 RL-PPO / RL-Repair；SelfPlay-GRU
  只作为 21 维/无 PyTorch 部署备选。

### 已完成的 Gate 结论

- **E0：PASS。** `python3 -m ipd league` 冻结 cross-play / Nash / Elo /
  exploitability / heldout red。9 候选同 suite/seeds；报告
  `results/e0_league_20261026/report.json`。
- **E1a：No-Go。** `robust` domain-randomized A60 fine-tune，heldout v2 +1.36，
  min_norm +0.0036，CVaR -2.15。
- **E1b：No-Go。** 更强更新（KL 0.05, lr 2e-4）selection +3.05，但 full
  +1.94，min_norm -0.0036，CVaR -2.32。
- **E1c：Go candidate。** `repair` 分布修复 random 尾部，heldout v2 +3.10，
  heldout v3 final +3.91，official/blind 不降；注册 `rl-repair`。
- **E2a/E2b/E2c：No-Go。** CVaR episode weighting（coef 1.0/3.0）和
  worst-family DRO 都没有把 heldout CVaR 抬起：
  CVaR item +1.2% / +0.5% / -0.5%，min_norm +0.0001/+0.0027/+0.0025。
  不注册 `rl-risk`/`rl-dro`；E2 代码保留为研究 artifact。
- **P3 red pool：第一里程碑 Go。** 85 维 `Red-RL-v1`（from scratch,
  `safe=False`）在全部 held-out blue 上都比 E4 red 更强：
  RL-Repair red 727.6 vs 703.8；Adv-RL 666.6 vs 611.8；red win 100%。
  这说明当前 blue 比 E4 结果更易被针对，**不等于 anti-exploitation 已解决**。
- **P3 league-lite：No-Go。** competitive blue response（50% official/repair
  + 50% Red-RL-v1，reward=own-opponent）把 E4 exploitability
  407.0→355.0（-12.8%），official 554.74、blind 563.80；
  未达 Gate D 的 -20%，不 promotion。E4 red 仍 100% 击败。

### 已提交 commits

```text
09a9ec2 P3 red pool + league-lite blue response
467c42b E2c worst-family DRO / No-Go
b445969 E2 CVaR risk PPO / No-Go
c95371b E1c repair fine-tune + register rl-repair
501fe46 E0 league harness + E1 robust fine-tune
efb4924 long-term plan
```

当前工作区应为 clean；测试 **112 tests OK**。

## 任务（下一 session）

按 `docs/ipd-long-term-improvement-plan.md` 继续。优先级建议：

1. **P3 扩展（首选）**
   - 扩大 red pool：`Red-RL-v1` + E4 + 新 red（对 Adv-RL / SelfPlay /
     规则池训练，至少 3–5 个 red checkpoint）；
   - 用 PSRO / meta-Nash 权重训练 blue response，而不是只针对单个 red；
   - held-out red 必须不参与 blue 训练；
   - Gate D：exploitability 降低 ≥20%，official 不降 >2，无 family 崩塌。
2. **P2 对手建模（最高上限）**
   - C1：Bayesian opponent model + finite-horizon planning；
   - C2：auxiliary opponent prediction；
   - C3：opponent classifier + MoE；
   - 先做 belief/planner 的离线评估和 fallback，不要直接改主策略。
3. **E3 multi-teacher distillation + gating**
   - 教师：RL-Repair / A60 / Adv-RL / SelfPlay / 规则策略；
   - 学生：85 维 MLP / GRU；BC + DAgger + PPO fine-tune。
4. **E1c promotion（可选）**
   - 用新 seeds 独立复现 RL-Repair 的 official/blind/heldout v2/v3；
   - 只有确认稳定后才考虑替换主权重。

**不建议**继续调 E2 的 risk coef/quantile；三次 No-Go 已经说明
episode-level CVaR/DRO 不是有效信号。

## 硬性要求

- 先读代码和文档，再写短计划；不要直接堆功能。
- 所有新模块先写 tests，再训练。
- 不只用官方四分布选模；必须报告 per-family、min_norm、CVaR、held-out red。
- 保留当前 RL-PPO / A60 / RL-Repair 作为 fallback；**未确认前不要覆盖**
  `weights/rl_policy_weights.json`。
- `heldout_v2` 已作为 validation；`heldout_v3` 已被 E1c 使用；
  **`heldout_v4` 仍未使用**，下一个通过 validation 的候选只能用它做一次
  final test。之后必须新建 heldout_v5。
- 不把 final test 用于反复选模；每次新候选先过 validation，再跑 final。
- 不要把“对训练 red 有效”当成 anti-exploitation 解决；必须有 held-out
  red best response。E4 red 只是 21 维压力测试，Red-RL-v1 是更强的 85 维
  exploiter，但两者都仍 100% 击败当前 blue。
- 不要硬编码 statistical TFT / 硬切 AllD / 动态抢跑；先作为特征/对手/评估。
- `.pt` checkpoint 被 `*.pt` 忽略；提交代码、JSON 权重和报告即可，必要时
  `git add -f`。
- 结束时更新 README/HANDOVER、checksum、Go/No-Go 结论，并提交。

## 建议的第一个动作

先读 `results/red_pool_20261026/README.md` 和 `docs/p3-red-league.md`，
检查 `ipd/training/rl.py` 的 opponent-pool flags 和
`ipd/evaluation/policy_pool.py`。然后写一个 **P3 扩展短计划**：

- 多 red 池（不同 blue 训练池 + 不同 seed）；
- PSRO/meta-Nash blue response；
- held-out red 评估和 Gate D（-20% exploitability，official -2 内）。

如果 P3 扩展仍无信号，转 P2 C1 Bayesian planner；不要回头继续调 E2。

## 复现命令

```bash
# tests
.venv/bin/python -m unittest discover -s tests -v

# 当前最佳候选
.venv/bin/python -m ipd self-test --candidates rl-repair
.venv/bin/python -m ipd benchmark --candidates rl-repair --fields 20 --workers 20
.venv/bin/python -m ipd blind --candidates rl-repair --fields 20 --workers 10

# E0 league（全量约 11 分钟；含 SelfPlay）
.venv/bin/python -m ipd league --config configs/eval/e0_league.json \
  --workers 20 --out /tmp/e0_full/report.json --md-out /tmp/e0_full/README.md

# red pool 评估
.venv/bin/python -m ipd league --config configs/eval/red_pool_eval.json \
  --skip-suites --skip-cross-play --workers 20 \
  --out /tmp/red_eval.json

# 训练 Red-RL（示例）
.venv/bin/python -m ipd train rl --device auto --seed 20261036 --from-scratch \
  --iterations 20 --fields-per-dist 20 --n 80 --workers 20 \
  --opponent-weights weights/rl_policy_weights.json,weights/rl_policy_weights_hard60.json,weights/rl_policy_weights_repair.json \
  --allow-probe --no-validation \
  --lr 2e-4 --ent-coef 0.01 --target-kl 0.03 \
  --out /tmp/red_pool/red_rl_v2.json \
  --checkpoint-out /tmp/red_pool/red_rl_checkpoint_v2.json

# league-lite blue response（混合 official + red pool）
.venv/bin/python -m ipd train rl --device auto --seed 20261038 \
  --iterations 20 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,repair \
  --dist-weights 1,1,1,2,2 \
  --opponent-weights weights/red_pool/red_rl_v1.json \
  --opponent-fraction 0.5 --opponent-penalty 1.0 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --out /tmp/blue_vs_red_comp_v2.json \
  --checkpoint-out /tmp/blue_vs_red_comp_checkpoint_v2.json
```

## 已知坑

- raw `min_family` 跨 family 不可比；看 `min_norm_family` / per-family /
  per-opponent。
- `final` suite 已用于 A/B/C 比较，现在更像 validation；不要再当 final。
- `heldout_v2` 是 validation；`heldout_v3` 已被 E1c 使用；
  `heldout_v4` 未使用；不要重复使用已消费的 final。
- cross-play 蓝军池退化：所有 blue 互相合作（594–595），Elo/Nash 不可区分。
  必须加入 red/exploiter pool；不要只靠蓝军 cross-play 选模。
- E4 red（21 维 GRU）是 upper-bound 压力测试，不代表真实比赛一定 100%
  被击败；Red-RL-v1（85 维）更强，但同样不是 solved anti-exploitation。
- E2 的 CVaR/DRO 没有改善 heldout CVaR；不要继续盲调 coef/quantile。
- DRO 权重归一化后必须按初始 `dist_weights` 总和缩放，否则 rollout field
  总数会缩水（这个 bug 已修复）。
- `--opponent-fraction` / `--opponent-penalty` 只在有
  `--opponent-weights` 时生效；red-only 用 fraction=1.0。
- `RL-Repair` 的 official 554.76 是独立 benchmark；E0/E1 报告的
  554.72 是 league 同 suite 结果，两者差在 policy seed，不要混用。
- 主权重 checksum 未变；A60 仍是 fallback；E1c/ P3 的权重都是 JSON，
  可直接审计。
