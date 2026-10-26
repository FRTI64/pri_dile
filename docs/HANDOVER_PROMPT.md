# 交班 Prompt

> 下一 session 直接复制下面这一段作为起始 prompt。

---

你在 `/home/frti64/pri_dile` 继续 IPD（100 轮重复囚徒困境）策略项目。先读：

- `docs/ipd-long-term-improvement-plan.md`
- `results/hard2_20261026/README.md`
- `results/selfplay_improvement/final_decision.md`
- `docs/HANDOVER.md`
- `docs/SELFPLAY.md`

## 当前状态

- 主基线：`RL-PPO`，`weights/rl_policy_weights.json`，官方四分布 **553.33**，blind **563.23**。
- 新候选：`RL-PPO-Hard60`，注册名 **`rl-hard60`**，权重 `weights/rl_policy_weights_hard60.json`。
  - 官方 20-field：**554.36**
  - blind 20-field：**564.17 / 5.75**
  - held-out final：**509.76**（baseline 506.49）
  - 未替换主权重，保留为候选/fallback。
- self-play / red league 已 **No-Go**；E4 red best response 100% 击败 BC/RL/Adv，anti-exploitation 尚未解决。
- **21 维不是硬约束**。后续主线和模型选择以 **85 维 RL-PPO** 为主；SelfPlay-GRU 只作为 21 维/部署备选。
- 已提交 commits：
  - `efb4924` long-term plan
  - `532156c` register `rl-hard60`
  - `24f76ce` hard-opponent A60
  - `ea6c007` self-play No-Go / sanity

## 任务

按 `docs/ipd-long-term-improvement-plan.md` 推进，**不要考虑 3 天限制**。优先级：

1. **P0：评估/元博弈**
   - cross-play matrix、PSRO/Nash averaging、exploitability；
   - 固定 baseline suite/seeds；
   - 新增 held-out family/参数，final test 不参与选模。
2. **P1：鲁棒训练**
   - domain-randomized A60 fine-tune；
   - robust PPO / CVaR / DRO；
   - multi-teacher distillation + gating。
3. **P2：对手建模**
   - Bayesian opponent model + finite-horizon planning；
   - auxiliary opponent prediction；
   - opponent classifier + MoE。
4. **P3：League/PSRO**
   - PFSP / league-lite；
   - 蓝军 checkpoint 池 + red best-response 池；
   - 用 exploitability/cross-play 判断收益。
5. **P4：架构**
   - 85 维 GRU/Transformer 只在 P1/P2 有信号后再做。

## 硬性要求

- 先读代码和文档，再写计划；不要直接堆功能。
- 所有新模块先写 tests，再训练。
- 不只用官方四分布选模；必须报告 per-family、min_norm、CVaR、held-out red。
- 保留当前 A60/RL-PPO 作为 fallback；**未确认前不要覆盖** `weights/rl_policy_weights.json`。
- 不把 `final_test` 用于反复选模；如果已经用过，必须新建 held-out。
- `.pt` checkpoint 被 `*.pt` 忽略；提交代码、JSON 权重和报告即可，必要时 `git add -f`。
- 不要把“对训练 red 有效”当成 anti-exploitation 解决；必须有 held-out red best response。
- 不要硬编码 statistical TFT / 硬切 AllD / 动态抢跑；先作为特征/对手/评估。
- 结束时更新 README/HANDOVER、checksum、Go/No-Go 结论，并提交。

## 建议的第一个动作

先做 **E0**：检查现有 `ipd/training/selfplay/compare.py`、`red_eval.py` 和 evaluation 代码，设计一个 cross-play/PSRO/exploitability harness，冻结 RL-PPO、A60、Adv-RL、SelfPlay、规则基线的同 suite/seeds 对照。输出一份短计划后再实现。

---

## 附：复现当前候选

```bash
# tests
python3 -m unittest discover -s tests -v

# A60 / RL-PPO-Hard60
python3 -m ipd self-test --candidates rl-hard60
python3 -m ipd benchmark --candidates rl-hard60 --fields 20 --workers 20
python3 -m ipd blind --candidates rl-hard60 --fields 20 --workers 10

# 与原 RL-PPO 对比
python3 -m ipd train selfplay-compare \
  --candidates rl-ppo,rl-hard60 --suite final --seeds 20 --workers 20

# E4 red 压力测试
python3 -m ipd train selfplay-red-eval \
  --red weights/selfplay_improvement/E4_red_policy.json \
  --blue base=weights/rl_policy_weights.json \
  --blue hard60=weights/rl_policy_weights_hard60.json --seeds 20
```

## 附：已知坑

- raw `min_family` 跨 family 不可比；看 `min_norm_family` / per-opponent。
- `final` suite 已被用于 A/B/C 比较，严格说现在更像 validation；继续选模要新建 held-out。
- E4 red 是针对 frozen blue 的 upper-bound 压力测试，不代表真实比赛一定 100% 被击败。
- A60 的官方/blind/held-out 提升是 +1/+1.3/+3.3 量级，不是质变；需要新的方法才有上限提升。
- 不要再用 21 维优化；21 维不是硬约束。
