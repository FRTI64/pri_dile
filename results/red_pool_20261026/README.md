# P3 — red/exploiter pool + league-lite（2026-10-26）

## 0. 结论

**Red pool: Go（第一里程碑）。** 新训练的 85 维 `Red-RL-v1` 在全部
held-out blue 上都比 E4 red 更强，说明当前 blue 候选比 E4 结果更易被针对。
这不等于 anti-exploitation 已解决；它只是给出了更强的 red 压力测试。

**League-lite blue response: No-Go（Gate D）。** competitive/mixed blue
response 把 held-out E4 red 的 exploitability（red - blue）从 **407.0
降到 355.0（-12.8%）**，blue mean 296.8→311.5，official 554.74（baseline
554.76，-0.02）。方向正确，但未达到 Gate D 的 **-20%**；因此不 promotion，
保留 RL-Repair，`heldout_v4` 仍未使用。

Own-score blue response 也为 **No-Go**：blue-vs-red 分数 312→343，但
official validation 555.26→552.39，best checkpoint 仍是初始 RL-Repair。

## 1. Red-RL-v1

- 85 维 RL-PPO，从 scratch 初始化（仅复用 feature scaler）。
- `safe=False`（`--allow-probe`），可以主动 defect。
- 训练池：RL-PPO / A60 / RL-Repair 三个 frozen blue。
- 20 iter × 20 fields × 80 opponents = 32k episodes。
- 训练 reward（red 自身分数）146.9 → 715.5。

评估（E4 vs Red-RL-v1，20 seeds，held-out blue）：

| Blue | E4 red mean | Red-RL-v1 mean | E4 blue mean | Red-RL blue mean |
|---|---:|---:|---:|---:|
| RL-Repair | 703.8 | **727.6** | 296.8 | 312.1 |
| RL-PPO | 642.3 | **705.7** | 309.3 | 325.2 |
| RL-PPO-Hard60 | 662.3 | **716.7** | 309.8 | 332.2 |
| Adv-RL | 611.8 | **666.6** | 337.3 | 328.6 |
| SelfPlay-GRU | 656.6 | **718.5** | 306.1 | 322.0 |
| Final-tuned | 607.6 | **634.3** | 334.1 | 347.8 |
| Triple-majority | 577.9 | **620.0** | 335.9 | 331.0 |

Red-RL-v1 的 red win rate 对全部 blue 为 100%；E4 为 95–100%。
**结论：red pool 第一里程碑 Go；当前 blue 全部仍可被 100% 击败。**

## 2. Blue response（own-score，No-Go）

从 RL-Repair 初始化，对 Red-RL-v1 训练 20 iter（提前停在 iter 15）：

```text
blue-vs-red reward: 312 -> 343
official validation: 555.26 -> 552.39
best checkpoint: iter 0 (RL-Repair)
```

own-score 目标只提高 blue 自身分数，不直接惩罚 red；official 下降时
validation 选择回退到初始 RL-Repair。**No-Go。**

## 3. League-lite competitive blue response

新训练配置：

```text
50% official/repair fields + 50% Red-RL-v1 fields
reward = own - 1.0 * opponent
initial RL-Repair, safe=True, KL 0.05, 20 iter
```

Held-out exploitability test 用 E4 red（未参与 blue 训练）：

| Blue | E4 red blue mean | E4 red mean | diff | blue-vs-red（训练 red） |
|---|---:|---:|---:|---:|
| RL-Repair | 296.80 | 703.80 | 407.00 | 312.10 / 727.60 |
| **Blue-vs-Red-Comp** | **311.45** | **666.45** | **355.00** | 323.50 / 707.50 |

- E4 exploitability：407.0 → 355.0，**-12.8%**（Gate 要求 -20%）。
- E4 red blue mean：296.8 → 311.5（+14.7）。
- Official 20-field：554.74（RL-Repair 554.76，-0.02）。
- Blind 20-field：563.80（RL-Repair 564.61，-0.81）。
- E4 red 仍 100% 击败 Blue-vs-Red-Comp；anti-exploitation 仍未解决。

**Gate D：No-Go。** 方向正确但幅度不足；league-lite 需要更强的 red 多样性
（Adv-RL/SelfPlay exploiters）、更多 red checkpoint、PSRO/meta-Nash 权重，
而不是只对单个 Red-RL-v1 做响应。

## 4. 复现

```bash
# red best response
python3 -m ipd train rl --device auto --seed 20261036 --from-scratch \
  --iterations 20 --fields-per-dist 20 --n 80 --workers 20 \
  --opponent-weights weights/rl_policy_weights.json,weights/rl_policy_weights_hard60.json,weights/rl_policy_weights_repair.json \
  --allow-probe --no-validation \
  --lr 2e-4 --ent-coef 0.01 --target-kl 0.03 \
  --out weights/red_pool/red_rl_v1.json \
  --checkpoint-out weights/red_pool/red_rl_checkpoint_v1.json \
  --history-out results/red_pool_20261026/red_train.json

# red eval
python3 -m ipd league --config configs/eval/red_pool_eval.json \
  --skip-suites --skip-cross-play --workers 20 \
  --out results/red_pool_20261026/red_eval.json

# league-lite blue response
python3 -m ipd train rl --device auto --seed 20261038 \
  --iterations 20 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,repair \
  --dist-weights 1,1,1,2,2 \
  --opponent-weights weights/red_pool/red_rl_v1.json \
  --opponent-fraction 0.5 --opponent-penalty 1.0 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --out weights/red_pool/blue_vs_red_comp_v1.json \
  --checkpoint-out weights/red_pool/blue_vs_red_comp_checkpoint_v1.json \
  --history-out results/red_pool_20261026/blue_vs_red_comp_train.json
```

## 5. 产物

```text
results/red_pool_20261026/
  README.md
  red_train.log / red_train.json
  red_eval.json / red_eval.md
  blue_vs_red_train.log / blue_vs_red_train.json   # own-score No-Go
  blue_vs_red_comp_train.log / blue_vs_red_comp_train.json
  blue_vs_red_comp_eval.json / .md / .log           # held-out red eval
  blue_vs_red_comp_benchmark.json / .log
  blue_vs_red_comp_blind.json / .log
  checksums.txt
weights/red_pool/
  red_rl_v1.json / red_rl_checkpoint_v1.json
  blue_vs_red_comp_v1.json / blue_vs_red_comp_checkpoint_v1.json
```

主权重未改动；RL-Repair 仍是主候选。heldout_v4 仍未使用。
