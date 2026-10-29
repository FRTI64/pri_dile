# RL-Wide160：160/128 是当前最均衡宽度

> **状态（2026-10-31）：实验候选，未 promotion。**
> 从 RL-Wide 128/128 扩展到 160/128，再做 2 iter balance PPO。
> dev_param 548.26、dev_complex 454.31、dev_ood 521.37；
> complex / OOD 的 `min_norm` 都提升到 **0.997** 左右。
> test_v7 provisional 485.62；`dev_red` 仍 100% 被红方击败。

## 1. 动机

RL-Wide 128/128 已经比 RL-Pattern 更均衡，但 complex / OOD 的
`min_norm` 仍只有 0.973 / 0.960。192/128 继续提高 complex 均值，却
让 dev_param 和 test_v7 回退（见 `docs/research/rl-wide.md`）。

因此尝试中间宽度 **160/128**：只扩第一层 128→160，保持第二层
128，再做很短（2 iter）的 balance fine-tune，避免大容量过拟合。

## 2. 训练

```bash
# 1) 128/128 -> 160/128
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_wide.json \
  --hidden 160 128 \
  --out results/rl_v4_20261032/wide160_ac.json \
  --seed 20261033

# 2) 2 iter balance；最终取 iter_001
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 2 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261040 \
  --init-weights results/rl_v4_20261032/wide160_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261032/wide160_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261032/wide160_weights_best.json \
  --checkpoint-out results/rl_v4_20261032/wide160_ac_best.json \
  --history-out results/rl_v4_20261032/wide160_history.json
```

关键点：

- 起点是已经平衡过的 128/128，而不是 RL-Pattern 64/32；
- 只扩第一层，第二层保持 128；
- 只训 2 iter，lr 3e-5，KL 0.1，PB 权重 1/11；
- 最终取 `iter_001`，不是 `--out` 的 iter2。

## 3. Protocol v4 dev（30 fields × 2048，sample，v5 GPU）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| RL-PPO | 540.35 | 582.53 | 440.31 | 574.64 | 513.48 |
| RL-Repair | 547.96 | 582.76 | 434.28 | 582.46 | 515.19 |
| RL-Pattern | **548.45** | 583.07 | 447.64 | 582.22 | 519.46 |
| RL-Wide | 548.24 | **583.08** | 452.79 | 582.30 | 520.71 |
| **RL-Wide160** | 548.26 | **583.08** | **454.31** | **582.38** | **521.37** |

`min_norm`（vs RL-PPO）：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| RL-Pattern | 0.9970 | 1.0000 | 0.8448 | 1.0072 | 0.8424 |
| RL-Wide | 0.9968 | 1.0000 | 0.9726 | 1.0072 | 0.9596 |
| **RL-Wide160** | 0.9970 | 1.0000 | **0.9974** | **1.0073** | **0.9966** |

- 相对 RL-Wide：dev_param +0.02、dev_complex **+1.52**、dev_ood
  **+0.66**、dev_axelrod +0.08；
- 相对 RL-Pattern：dev_param -0.19、dev_complex **+6.67**、
  dev_ood +1.91、dev_axelrod +0.16；
- `pattern_breaker` 420.3 → **511.6**；`lookahead` 355.2 → 379.5；
- 没有新的最差 family，complex/OOD 的 min_norm 基本回到 1.0。

### 3.1 独立 seed（20261029）

| 候选 | dev_param | dev_complex | dev_ood | complex min_norm | ood min_norm |
|---|---:|---:|---:|---:|---:|
| RL-Wide | 546.60 | 452.05 | 522.14 | 0.9744 | 0.9453 |
| **RL-Wide160** | 546.39 | **454.02** | **522.99** | **0.9976** | **0.9971** |

dev_param 与 RL-Wide 基本持平（-0.21），complex/OOD 收益和 min_norm
改善均可复现。

## 4. test_v7（50 fields × 2048，sample，provisional）

| 候选 | test_v7 | min field | p05 | worst family |
|---|---:|---:|---:|---:|
| RL-PPO | 477.26 | 470.50 | 471.20 | 138.7 |
| RL-Repair | **486.31** | **480.55** | **480.79** | 121.1 |
| RL-Pattern | 485.43 | 478.69 | 479.54 | 129.2 |
| RL-Wide | 485.50 | 479.38 | 480.11 | 128.4 |
| **RL-Wide160** | 485.62 | 478.39 | 479.29 | 129.4 |

相对 RL-Repair -0.69；相对 RL-Wide +0.12；相对 RL-Pattern +0.19。
test-only 的 `hyper_extort` / `axelrod_extra` 仍未恢复到 RL-Repair
水平，不能 promotion。

### 4.5 分级诊断（30×2048，sample）

| tier | RL-Wide | **RL-Wide160** |
|---|---:|---:|
| dev_level0 | 628.26 | 627.87 |
| dev_level1 | 550.65 | 550.59 |
| dev_level2 | 550.14 | 549.95 |
| dev_level3 | 449.28 | **451.52** |
| dev_level4 | 447.43 | **447.54** |
| dev_complexity0 | 628.26 | 627.87 |
| dev_complexity1 | 543.83 | 543.77 |
| dev_complexity2 | 534.85 | 534.07 |
| dev_complexity3 | 523.47 | 523.43 |
| dev_complexity4 | 396.30 | **401.27** |

- 低难度 tier 有 0.1–0.8 的小幅回退；
- 最高思考等级 `dev_level3/4` 和最高复杂度 `dev_complexity4`
  分别 +2.24/+0.11/+4.97（相对 RL-Wide）；
- 与 dev_complex/OOD 的收益方向一致，没有新的低级 tier 崩塌。

## 5. dev_red / test_red（30 seeds）

| 候选 | blue | red | diff | red_win |
|---|---:|---:|---:|---:|
| RL-Wide160 | 378.40 | 711.73 | -333.33 | 100.0% |
| RL-Wide | 378.67 | 711.33 | -332.67 | 100.0% |
| RL-Repair | 379.17 | 713.17 | -334.00 | 100.0% |

`test_red`（provisional，未消费）：

| 候选 | blue | red | diff | red_win |
|---|---:|---:|---:|---:|
| RL-Wide160 | 319.43 | 742.43 | -423.00 | 100.0% |
| RL-Wide | 318.60 | 741.60 | -423.00 | 100.0% |
| RL-Repair | 312.80 | 747.47 | -434.67 | 100.0% |

anti-exploitation 没有改善；wide160 的 blue 比 RL-Repair 高 6.63，
但 red 仍 100% 获胜。不要用 complex 均值提升宣称 red 防御。

legacy 20-field：四分布 avg **554.77**、blind **564.75**；C++ 导出
验证通过（`N_HIDDEN0=160` / `N_HIDDEN1=128`）。

## 6. 宽度对照

| 宽度 | dev_param | dev_complex | complex min_norm | test_v7 |
|---|---:|---:|---:|---:|
| 64/32 RL-Pattern | 548.45 | 447.64 | 0.845 | 485.43 |
| 128/128 RL-Wide | 548.24 | 452.79 | 0.973 | 485.50 |
| **160/128 RL-Wide160** | **548.26** | **454.31** | **0.997** | **485.62** |
| 192/128 balance | 547.44 | 459.57 | 0.990 | 484.90 |

160/128 是当前综合最均衡的宽度：dev_param 与 128/128 持平，
complex/OOD 更高，min_norm 接近 1，test_v7 也略高。192/128 继续
提高 complex 均值，但主排序和 test 回退，不注册。补充的 192/128
gentle / balance2 可以把 dev_param 推到 548.8、test 推到 486.7，但
complex min_norm 只有 0.73–0.75（`results/rl_v4_20261031/`），
因此同样不作为候选。

### 6.1 后续消融

| 变体 | dev_param | dev_complex | dev_ood | test_v7 | 判定 |
|---|---:|---:|---:|---:|---|
| **RL-Wide160** | **548.26** | 454.31 | **521.37** | 485.62 | 保留 |
| + train_hard 2 iter | 548.31 | 454.12 | 521.19 | **485.76** | 噪声级，未注册 |
| 160/160 2 iter | 547.92 | **454.79** | 521.24 | 485.19 | 回退，未注册 |
| 192/128 gentle | 548.86 | 446.00 | 518.95 | 486.76 | min_norm 0.73，未注册 |
| 192/128 balance2 | 548.82 | 446.66 | 519.49 | 486.51 | min_norm 0.75，未注册 |

结论：160/128 是综合最均衡点；继续加第二层、加 hard 数据或加宽到
192 都不能同时改善 dev_param、complex 和 min_norm。

## 7. 判定与下一步

**判定：注册 `rl-wide160` 为实验候选，不 promotion。**

- 它是当前 Protocol v4 dev 综合最均衡的候选；
- test_v7 provisional 仍比 RL-Repair 低 0.69；
- 主权重不变；RL-Wide / RL-Pattern / RL-Repair 仍为 fallback。

下一步：

1. 独立 seed 再复现一次，确认 dev_param 不低于 RL-Wide；
2. 跑 `dev_level0..4` / `dev_complexity0..4` 分级诊断；
3. 跑 `test_red` provisional 和外部比赛；
4. 若继续加宽，160/160 或 192/160 可以试，但必须保持 min_norm 和
   test_v7 不退化。

## 8. 复现与权重

```bash
python3 -m ipd self-test --candidates rl-wide160
python3 -m ipd protocol eval \
  --candidates RL-PPO,RL-Repair,RL-Pattern,RL-Wide,RL-Wide160 \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-PPO \
  --out results/rl_v4_20261032/protocol_v4_dev_rl_wide160.json
```

权重：

- `weights/rl_policy_weights_wide160.json`
  sha256 `dca9006c9fce5da0a7a152ed1935ad1ebb8176c56a485772d5e77fccc0eb682d`
- `weights/rl_policy_checkpoint_wide160.json`
  sha256 `6e93b60f1ad6d13ad84e455c58a18641b07e9ea54c10ea6a5e58f908eea133ab`
