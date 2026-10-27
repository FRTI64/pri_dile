# PSRO from scratch — random init, existing models only as opponents

- **Status:** complete — **No-Go for promotion**, but **first PSRO model that
  flips the held-out red outcome** (red win 0.0).
- **Config:** `configs/psro/psro_scratch_20261027.json` (main),
  `configs/psro/psro_scratch_pilot_20261027.json` (aggressive pilot),
  `psro_scratch_v2/v3` (ablations)
- **Result:** `results/psro_scratch_20261027/`
- **Candidates:** `psro-scratch` (meta-policy), `psro-scratch-br` (final BR)
- **Design reference:** OpenSpiel PSRO v2
  ([github.com/google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel),
  `open_spiel/python/algorithms/psro_v2/`): random initial policies,
  an oracle that trains best responses, meta-strategy weights, and strategy
  selection.  We did not add OpenSpiel as a runtime dependency because the IPD
  policy is a custom 85-dim 100-day history model and the project requires
  zero-dependency inference.

## 1. 约束

用户要求：

- PSRO 模型**从零训练**；
- 已有模型/策略只能做**对手参考**，不能作为 blue population、初始化或
  fallback。

因此 blue 侧只允许：

1. `Scratch-Init`：随机 85 维 MLP（只继承 feature mean/std；没有 policy
   weights）；
2. `Scratch-Base`：从随机初始化训练 30 iter 的官方 PPO base；
3. `PSRO-BR-*`：从 scratch base 初始化、从零 PPO fine-tune 的 best
   response。

Red 侧包含已有模型/规则（RL-Repair、A60、RL-PPO、Adv-RL、Final、
Triple、SelfPlay、Blue-Meta-v2、Red-v1/v3/v4）；这些只作为对手，不进入
blue 或 oracle 初始化。

## 2. Scratch base

```bash
python3 -m ipd train rl --device auto --from-scratch --iterations 30 \
  --fields-per-dist 5 --n 80 --dist-weights 1,1,1,2 \
  --ent-coef 0.001 --lr 3e-4 --center-advantages --select-metric avg \
  --seed 20262027 \
  --out weights/psro_scratch/base_policy.json \
  --checkpoint-out weights/psro_scratch/base_checkpoint.json \
  --history-out results/psro_scratch_20261027/base_history.json
```

结果：官方四分布 validation avg **553.67**；Protocol v4 `dev_param`
(4×512) **528.99**。它是从零 policy，不是已有模型。

## 3. PSRO 主配置

- blue：`Scratch-Init` + `Scratch-Base` + BR；
- red：已有模型/规则 11 个；
- cross-play：8 seeds，100 rounds；
- meta：blue maximin / red 对偶 maximin，uniform_mix 0.15；
- oracle：从 scratch base checkpoint 初始化，`bc_kl=0.05`，
  `opponent_fraction=0.4`，`opponent_penalty=1.0`，20 iter/round，
  `--no-value-load`，`value_bias=0`；
- 2 rounds。

### 元博弈

初始 blue maximin：`Scratch-Init` 权重 1.0（随机策略更难被 red 剥削）；
初始 exploitability 62.81。两轮后：

| Blue | Meta weight | Expected vs red |
|---|---:|---:|
| PSRO-BR-br_02 | 0.9984 | +0.01 |
| PSRO-BR-br_01 | 0.0016 | +0.01 |
| Scratch-Init | 0.0000 | 0.00 |
| Scratch-Base | 0.0000 | -0.02 |

最终 exploitability **0.0**。BR 的 cross-play 对 Red-v1/v3/v4 的
score difference 为 +78.8/+195.9/+210.6，说明它不只是记住了训练 red 的
动作，还在 population 内取得了正收益。

## 4. 评估

诊断规模：4 dev fields × 512；red 10 seeds。不是完整 v4 baseline。

| Candidate | dev_param | min_norm | dev_red red win | Red-v5/v6 red win |
|---|---:|---:|---:|---:|
| **PSRO-Scratch** | 480.43 | 0.5501 | **0.0** | **0.0** |
| **PSRO-Scratch-BR** | 481.11 | 0.5437 | **0.0** | **0.0** |
| Scratch-Base | 528.99 | - | - | 1.0 |
| RL-Repair | **543.35** | 1.0000 | 1.0 | 1.0 |
| Final-tuned | 525.73 | 0.9159 | - | 1.0 |
| Triple-majority | 530.61 | 0.9375 | - | 1.0 |

Red 细节：

```text
dev_red (protocol red, blue-red convention):
  PSRO-Scratch     blue 436.2 / red 210.2 / red_win 0.0
  PSRO-Scratch-BR  blue 427.2 / red 195.2 / red_win 0.0
  RL-Repair        blue 379.2 / red 715.2 / red_win 1.0

held-out Red-v5/v6 (league, red-blue convention):
  PSRO-Scratch     blue 491.4 / red 197.4 / diff -294.0 / red_win 0.0
  PSRO-Scratch-BR  blue 508.4 / red 196.3 / diff -312.0 / red_win 0.0
  Scratch-Base     blue 351.8 / red 711.8 / diff +360.0 / red_win 1.0
  RL-Repair        blue 347.4 / red 730.0 / diff +382.5 / red_win 1.0
```

这是项目中**第一次**把 Red-v5/v6 的 red win 从 100% 翻到 0%。但代价很大：

- `dev_param` 480.43 vs RL-Repair 543.35（**-62.9**）；
- `min_norm` 0.5501；
- worst family `noisy` 249.1（RL-Repair 452.8）；
- `adaptive` / `extort` / `axelrod` 也明显下降。

结论：模型学会了针对 red 的惩罚/防御策略，但把官方分布上的合作收益
牺牲掉了。**Gate D / promotion 不通过**，主权重不变。

## 5. Ablations

| Run | Oracle | dev_param | Red-v5/v6 red win |
|---|---|---:|---:|
| pilot (random+BR only) | 40% red, 30 iter, score-diff | 473.27 | 0.0 |
| main (random+base+BR) | 40% red, 20 iter, bc_kl 0.05 | 480.43 | 0.0 |
| v2 (red exploiters only) | 20 iter, bc_kl 0.1 | 477.74 | 0.0 |
| v3 (anchored) | 20% red, own-score, bc_kl 0.5 | 532.13 (BR) / 476.65 (deployed random) | 1.0 (BR) / 0.2 (deployed random) |
| scratch base | no PSRO | 528.99 | 1.0 |

v3 说明：强 KL anchor 能保住 official（532），但 anchored BR 没有进入
maximin support；最终部署的是随机 init（476.65），并不抗 red。当前
trade-off 不是靠调一个系数就能同时解决的。

## 6. 决策

- 不 promotion；`weights/rl_policy_weights.json` 未改；
- `psro-scratch` / `psro-scratch-br` 注册为实验候选；
- 如果目标是 anti-exploitation，`psro-scratch` 是当前最强 held-out red
  结果（red win 0.0）；
- 如果目标是 official 提交，`Scratch-Base`（528.99）或 RL-Repair 更合适；
- 下一步应使用**约束/多目标 oracle**（official family floor + red
  objective），或在新的 held-out red 上复核；不要重复同一 fine-tune。

## 7. 复现

```bash
# scratch base（约 6 分钟，GPU）
python3 -m ipd train rl --device auto --from-scratch --iterations 30 \
  --fields-per-dist 5 --n 80 --dist-weights 1,1,1,2 \
  --ent-coef 0.001 --lr 3e-4 --center-advantages --select-metric avg \
  --seed 20262027 \
  --out weights/psro_scratch/base_policy.json \
  --checkpoint-out weights/psro_scratch/base_checkpoint.json \
  --history-out results/psro_scratch_20261027/base_history.json

# PSRO（约 8 分钟，2 rounds × 20 iter）
python3 -m ipd train psro \
  --config configs/psro/psro_scratch_20261027.json --workers 20

# diagnostics
bash results/psro_scratch_20261027/run_eval.sh
python3 -m ipd protocol red \
  --candidates psro-scratch,psro-scratch-br,rl-repair \
  --tier dev_red --seeds 10 \
  --out results/psro_scratch_20261027/eval_dev_red.json
```

Checksums: `results/psro_scratch_20261027/checksums.txt`。
