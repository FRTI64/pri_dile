# 非对称自博弈训练（Self-Play PPO）

> **状态（2026-10-27）**：训练器已实现，最终策略是 21 维 GRU + BC 蒸馏；
> self-play / red league 的改进实验为 **No-Go**，不替换主权重。SelfPlay-GRU
> 仍作为 21 维 / 无 PyTorch 部署备选；当前最佳候选是 RL-Repair。当前候选
> 比较见 [README.md](../../README.md)。

实现 `docs/research/selfplay-plan.md` 的训练系统。入口：

```bash
# 200 episode 冒烟（日志/checkpoint/resume）
python3 -m ipd train selfplay --smoke --device auto

# 完整 15,000 episode 三阶段课程（约 1.5M env steps）
python3 -m ipd train selfplay --config configs/selfplay/full.json \
  --device auto --final-test

# 从 checkpoint 续跑
python3 -m ipd train selfplay --config configs/selfplay/full.json \
  --device auto --resume weights/selfplay_checkpoint.pt --final-test

# 可选：把现有 RL-PPO 蒸馏成 21 维 GRU，再做 self-play 微调
python3 -m ipd train selfplay-bc --device auto --episodes 4000 --epochs 10
python3 -m ipd train selfplay --config configs/selfplay/full_bc.json \
  --device auto --init-checkpoint weights/selfplay_bc_init.pt --final-test
```

训练产物：

```text
weights/selfplay_gru_policy.json      # 可提交推理策略（无 PyTorch 依赖）
weights/selfplay_checkpoint.pt        # 可续跑 checkpoint（蓝/红/优化器/league）
results/selfplay_20261026/
  train_log.jsonl                     # 每轮日志
  history.json                        # 完整训练历史
  eval_full_XXXX.json                 # 每 1000 episode 的 held-out 验证
  red_base_anchor.json                # red 竞争力约束的 per-family anchor
  best_val.json, checkpoints/best_*.pt
  final_test.json                     # 只跑一次的 held-out final test
```

## 1. 冻结的 21 维观测

实现位于 `ipd/policies/selfplay_gru_obs.py`（纯标准库），训练端
`ipd/training/selfplay/obs.py` 只做 NumPy 批量包装。双方观测都只使用
**当前回合之前**的历史。

| 索引 | 内容 |
|---:|---|
| 0 | `t/100` |
| 1 | `(100-t)/100` |
| 2–5 | 上一轮联合结果 one-hot `CC/CD/DC/DD`；`t=0` 全零 |
| 6–13 | 最近 4 轮双方动作，新→旧；缺失轮 padding `-1` |
| 14 | 对手累计背叛率 |
| 15 | 自己累计背叛率 |
| 16 | 最近 10 轮对手背叛率 |
| 17 | `自己的累计背叛率 - 对手累计背叛率`（v1 契约） |
| 18 | 累计 CC 率 |
| 19 | 累计 DD 率 |
| 20 | `t>=90` 终局指示 |

分母统一为 `max(1, t)`，`t` 是已完成轮数（0-based 下一动作索引）。观测中
没有 opponent ID / family / type / latent parameter。

## 2. 训练结构

- 32 个 lock-step 环境；蓝军和红军各 100 步，双方动作基于同一 pre-step history。
- 每步收益 `CC=6, CD=0, DC=10, DD=1`；训练时只做 `0.01` 数值缩放，日志与阈值
  仍使用 100 步总分。
- 蓝军/红军各一个独立 `GRUActorCritic`：2 层 GRU，hidden=64；actor/critic 都是
  `64→32→1`。两个独立 optimizer，机器人无梯度。
- PPO：4 epochs、clip=0.2、value coef=0.5、entropy=0.01、grad clip=0.5；
  gamma=1.0、lambda=0.95；按完整 episode 切 minibatch（默认 8 个 episode）。
- 曲线：Phase 1a 互惠奠基（0–3k，混入 15% Random、10% special、5% Extort）→ 1b 抗性（3k–8k）→ 2 高压红队
  （8k–13k，α=0.85/λ=5）→ 3 终局（13k–15k，α=0.90/λ=10，LR 余弦到 1e-5）。
- 对手池按 family 先验做 Dirichlet 扰动；随机化动态窗口、Harvest 变脸日、
  Probe+Phase、Kamikaze、Extort 参数。
- 实验分支 `feature_version=selfplay-21-v2` 把第 17 维换成响应差 `pD-pC`
  （`pC=P(opp D | 我 C)`，`pD=P(opp D | 我 D)`）。它在早期能更快学会收割
  Bully/Random，但 15k episode 内更容易在“合作/剥削”之间震荡；默认仍使用 v1。
- Red：Red-Base（Final/Triple/RL-final）、Red-Blue（当前蓝军快照）、
  Red-Strong（绝对分场）。红队使用滚动窗口竞争力惩罚：

  ```text
  R_t = red_t - target_t
        - λ * max(0, α * mean_W(target) - mean_W(red))^2
  ```

  `W~U(15,25)`，`t<W` 不计算惩罚；每个 episode 重新采样 W。
- Red league 保留最近 8 个快照；蓝军的一部分环境直接对阵 league 快照。
- 验证/测试拆分：
  - 训练：全参数区间 + 全部训练 family；
  - validation：51 个固定对手，参数区间与训练不相交，并包含 held-out 参数；
  - final test：86 个新参数对手 + 两个 **训练中完全不出现** 的 family
    `MirrorAdaptive` / `DelayedGrim` + held-out red（现有 RL-PPO checkpoint）。
  - final test 只在训练结束后跑一次，不参与选模。

## 3. 验证与选模

- Fast validation：每 200 episode，12 个代表对手，N=3，用于看趋势。
- Full validation：每 1000 episode，51 个对手，N=5，per-family；包含 TFT、
  Bully、NoisyTFT.02、AllD 和 held-out red。
- 红线：TFT≥590、Bully≥650、NoisyTFT.02≥560、AllD≥95、min_family≥350、
  held-out red≥430。
- 过滤红线后按以下分数选 top-3：

  ```text
  mean + 0.5*S_vs_FT + 0.5*S_vs_TM + 0.5*min_family - 2.0*sigma
  ```

  `sigma` 是跨 family 均值的标准差。若整个训练没有任何 checkpoint 通过全部
  红线，则回退到未过滤的 best validation checkpoint，并在日志中标记
  `eligible=False`。
- 早停只在相对 best 分数回退超过 30 分时累计；连续 3 次才 rollback + 降 LR。
- 红队 snapshot 入 league 前会跑一次小型 per-family calibration；低于当前阶段
  `alpha` 的 snapshot 会被 quarantine，不进入蓝军对手池。

## 4. 推理

```bash
python3 -m ipd benchmark --candidates selfplay --weights weights/selfplay_gru_policy.json \
  --fields 20 --workers 20
python3 -m ipd blind --candidates selfplay --weights weights/selfplay_gru_policy.json \
  --fields 10 --workers 12
```

`ipd.policies.selfplay_gru.SelfPlayGRUPolicy` 只用标准库实现 GRU 前向；每场
必须 `reset()`。注册名为 `selfplay`（显示名 `SelfPlay-GRU`）。旧 `--arch
selfplay` 仍可用，但新命令请用 `--candidates selfplay`。

## 5. 测试

```bash
python3 -m unittest tests.test_selfplay -v
```

覆盖：21 维观测与 padding、向量化/单环境一致性、同时决策、GRU step/sequence
一致性、GAE 参考值、梯度隔离、对手比例、red 约束、确定性、导出推理一致性，
以及 BC 蒸馏、200-episode smoke + checkpoint resume。


## 6. 实际训练结果（2026-10-26）

最终选定策略是 **21 维 GRU + BC 蒸馏（教师为现有 RL-PPO）**，导出在
`weights/selfplay_gru_policy.json`。参数量：GRU 可部署 43,777（含 critic
45,890），RL-PPO 7,617；GRU 并不更小。选择它而不是纯从零 self-play 的原因：

| 策略 | 四分布 | blind | held-out final test |
|---|---:|---:|---:|
| **SelfPlay-GRU（BC warm start，最终）** | **550.76（20 fields）** | **560.43 / 8.10（10 fields）** | **505.06** |
| Self-play 从零 15k（`selfplay_checkpoint.pt`） | 524.77（2 fields） | - | 504.57 |
| 8-field 快速对照（最终策略） | 552.52 | - | - |
| 现有 RL-PPO | 553.33（20 fields） | 563.23 / 8.00（10 fields） | - |
| Adv-RL 激进 | 554.29（20 fields） | 562.91 / 8.00（10 fields） | - |

> E0 冻结协议的 20-field blind：SelfPlay-GRU 561.13、RL-PPO 562.80、
> A60 563.87、RL-Repair 564.68。不要和上表的 10-field 数字混排。

最终策略的 held-out per-opponent 验收：

```text
TFT 604, Bully 878, NoisyTFT.02 604, AllD 99,
min_family 370 (阈值 350), held-out red 595 (阈值 430；RL-BC 互惠型，
不是 anti-exploitation 证据)
```

- 纯从零 self-play 在 15k episode 内会在“对 TFT/Grim 合作”和“对
  Random/Bully/AllD 收割”之间震荡，无法稳定满足全部红线；因此最终提交
  使用 BC 蒸馏得到的 21 维 GRU。
- 完整 self-play 训练器、red league、three-phase curriculum、held-out
  test 都已实现并跑通；结果保存在 `results/selfplay_20261026/`。
- 最终评估、复现命令和对照表见 `results/selfplay_final/README.md`。
- `--init-checkpoint weights/selfplay_bc_init.pt` 可以把该 BC 策略作为
  self-play 的初始化继续训练；`bc_kl_coef` 可控制偏离 BC 的 KL 惩罚。


## 7. 改进实验结论（2026-10-26，No-Go）

按 `docs/research/selfplay-improvement.md` 做了公平对照、BC+red fine-tune
和 red best response。结论：**self-play / red league 没有带来可验证的
held-out 增益，停止该方向，不替换现有最终权重。**

同 suite / 同 seeds（validation 51×5，final 86×20）：

| 策略 | validation mean | final mean | min_norm vs RL | params |
|---|---:|---:|---:|---:|
| BC-only | 501.66 | 506.98 | 0.953 / 0.991 | 43,777 |
| RL-PPO | 505.33 | 506.49 | 1.000 | 7,617 |
| Adv-RL | 505.86 | 504.08 | 0.994 / 0.968 | 7,617 |
| scratch self-play 15k | 483.87 | 505.79 | 0.764 / 0.783 | 43,777 |

BC+red fine-tune：

- E1a（15k，KL 0.2，lr 2e-4）：validation mean 502.09（+0.43），
  min_norm 0.969（+0.016），官方 4-field 546.00 vs BC 546.84（-0.84）。
  未达到 +10 mean 或 +0.03 min_norm 的 Go 条件。
- E1b（15k，KL 0.1，lr 1e-4）：validation mean 429.21，min_norm 0.610，
  灾难性退化；训练期无 checkpoint 通过红线，回退到 iter 32。
- E1c 30k 不启动。

Red best response（E4，frozen BC blue，6k episodes）：

```text
BC blue 304.6 vs red 636.1，red win 100%
RL-PPO  307.0 vs red 648.5，red win 100%
Adv-RL  328.0 vs red 621.0，red win 100%
```

针对 frozen blue 训练的 best response 能击败所有当前策略，包括 RL/Adv；
因此“anti-exploitation”没有被任何策略解决。旧 `final_test.json` 的
held-out red（RL-BC 595|595）只是互惠型对照，不能作为验收。

完整表格、命令和决策树见 `results/selfplay_improvement/final_decision.md`；
同 seeds 对照在 `results/selfplay_improvement/compare.json`。

最终 sanity check（2026-10-26）：三个候选的 benchmark/blind 完全复现，
权重 checksum、路径和冻结建议见
`results/selfplay_improvement/sanity/README.md`。主提交建议 RL-PPO / A60 /
RL-Repair；21 维备选 SelfPlay-GRU；Adv-RL 仅在官方四分布构成匹配时使用。
