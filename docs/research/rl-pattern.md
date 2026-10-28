# RL-Pattern：针对 pattern_breaker 的 v4 定向 fine-tune

> **状态（2026-10-29）：实验候选，未 promotion。**
> `rl-pattern` 在 Protocol v4 全量 dev 上超过 RL-Repair；`test_v7`
> provisional 低约 0.9 分，仍高于 RL-PPO。主权重未替换。

## 1. 动机

Protocol v4 的 `dev_complex` / `dev_ood` 里，`pattern_breaker` 是当前
RL-Repair 的最大单 family 漏洞：

| 候选 | dev_complex `pattern_breaker`（argmax 诊断） |
|---|---:|
| RL-Repair | 233.0 |
| RL-PPO | 414.0 |
| Triple-majority | 378.8 |
| PSRO-BR | 786.0 |

`PatternBreaker` 检查对手最近 `window` 步里的 2/3/4 周期；检测到周期后
出反周期动作。RL-Repair 的采样策略在高置信区间接近确定性，容易形成短周期，
因此被稳定预测。安全规则（第 1 天 C、对手未 D 时 C）不能解决这个问题：
对手一旦 D 过，策略仍有大量确定性反应。

结论：需要让策略在“对手已 D”之后学到更不可预测、且仍能保持合作的反应，
而不是继续调全局 temperature/epsilon。直接调 temperature 会同时改变所有
family，dev_param 掉得比 complex 收益更快（见 5. 消融）。

## 2. 新训练 profile

`ipd/evaluation/protocol.py` 新增两个只使用训练 tier 的 profile：

| profile | 内容 |
|---|---|
| `train_pattern_breaker` | 仅 `pattern_breaker` 的 6 个 `train_complex` 参数点 |
| `train_weak_complex` | `pattern_breaker` / `adaptive_mix` / `probe_exploit` / `conditional_extort` / `lookahead` |

两者只从 `train` / `train_complex` 抽样，dev/test 参数保持 held-out。
`tests/test_protocol.py` 增加了 profile 的 tier 泄漏检查。

## 3. 训练配方

### 3.1 warm start：entropy-regularized v4 fine-tune

从 RL-Repair actor-critic checkpoint 开始，20 iter 学习
`train_complex` + `train_axelrod`，并把 entropy 从 1e-4 提到 1e-2：

```bash
python3 -m ipd train rl --device cuda --iterations 20 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261029 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --train-dists original,late,grim30,wide,train_complex,train_axelrod \
  --dist-weights 1,1,1,2,2,1 \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --no-validation \
  --out results/rl_v4_20261028/ent01_proper_weights.json \
  --checkpoint-out results/rl_v4_20261028/ent01_proper_ac.json \
  --history-out results/rl_v4_20261028/ent01_proper_history.json
```

这一步只带来约 +1 dev_param、+1.4 dev_complex；单独作为候选意义不大，
但它的 value head 让后续定向训练稳定。

### 3.2 定向训练：train_v4 + pattern_breaker

```bash
python3 -m ipd train rl --device cuda --iterations 8 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261032 \
  --init-weights results/rl_v4_20261028/ent01_proper_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,4 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261028/pb_v4_ckpts \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261028/pb_v4_weights.json \
  --checkpoint-out results/rl_v4_20261028/pb_v4_ac.json \
  --history-out results/rl_v4_20261028/pb_v4_history.json
```

- `train_v4` 保留 extort / noisy / Axelrod / policy 等全部 v4 family；
- `train_pattern_breaker` 权重 4/11，使 PB 成为显著但非唯一的训练信号；
- `--save-all-checkpoints` 保存 iter2/4/6/8。**最终候选不是 `--out`
  的 iter8，而是离线选出的 `pb_v4_ckpts/iter_006.json`**；
- `weights/rl_policy_weights_pattern.json` 由 iter6 actor-critic 导出为
  `ipd-neural-policy-v1`；`weights/rl_policy_checkpoint_pattern.json`
  保存 actor-critic checkpoint。

只训 PB（权重 8/13）4 iter 也能把 PB 拉到 488（argmax），但 extort /
adaptive 等开始回退。`pb_mix`（`train_complex` + PB 4/11）dev_complex
只有 440.3，也不如 `train_v4` 版本。因此最终采用 `train_v4` 混合。

## 4. 结果

### 4.1 Protocol v4 dev（30 fields × 2048，sample，v5 GPU）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| RL-PPO | 540.35 | 582.53 | 440.31 | 574.64 | 513.48 |
| RL-Repair | 547.96 | 582.76 | 434.28 | 582.46 | 515.19 |
| **RL-Pattern** | **548.45** | **583.07** | **447.64** | 582.22 | **519.46** |

- 相对 RL-Repair：dev_param **+0.49**、dev_family **+0.31**、
  dev_complex **+13.36**、dev_ood **+4.27**、dev_axelrod **-0.24**；
- `min_norm`（vs RL-PPO）：dev_param 0.9970、dev_complex 0.8448、
  dev_ood 0.8424、dev_axelrod 1.0072、dev_family 1.0000；
  RL-Repair 对应为 0.9909 / 0.4760 / 0.4730 / 0.9975 / 1.0000；
- `pattern_breaker`：236.8 → **420.3**；`lookahead` 334.9 → 355.2；
  `adaptive_mix` 284.9 → 292.9；`phase_switch` 485.7 → 491.8；
- 代价：`extort` 496.3 → 490.7、`axelrod` 563.9 → 561.4、
  `meta_mix` 499.7 → 496.1。

### 4.2 test_v7（50 fields × 2048，sample，provisional，未消费）

| 候选 | test_v7 | min field | p05 | worst family |
|---|---:|---:|---:|---:|
| RL-PPO | 477.26 | 470.50 | 471.20 | 138.7 |
| RL-Repair | **486.31** | **480.55** | **480.79** | 121.1 |
| RL-Pattern | 485.43 | 478.69 | 479.54 | 129.2 |

- 相对 RL-Repair **-0.88**；相对 RL-PPO **+8.17**；
- `pattern_breaker` 266.0 → **407.6**，`lookahead` +16.6、
  `portfolio` +14.0、`early_harvest` +8.2；
- test-only / 未训练 family 有回退：`hyper_extort` -30.2、
  `axelrod_extra` -21.8、`defect` -9.7、`extort` -8.3；
- 因此 `test_v7` 不用于选模，也不能据此 promotion；它说明定向
  fine-tune 的收益主要集中在 dev 的 pattern/complex 结构上，仍有
  过拟合训练 catalog 的迹象。

### 4.3 legacy 20-field 兼容性

| 候选 | 四分布 avg | blind 20-field |
|---|---:|---:|
| RL-Pattern | **554.93** | **564.91** |
| RL-Repair（历史） | 554.72 | 564.68 |

legacy 结果只作兼容性检查，不参与 v4 选模。

## 5. 消融

| 变体 | dev_param（argmax 诊断） | dev_complex | pattern_breaker |
|---|---:|---:|---:|
| RL-Repair | 551.77 | 435.11 | 233.0 |
| temperature 2 | 551.26 | 441.70 | 251.2 |
| temperature 3 | 543.79 | 450.98 | 276.9 |
| epsilon 0.10 | 541.13 | 454.44 | 256.4 |
| PB-heavy iter4（只混 legacy + PB） | 551.24 | 451.80 | 488.0 |
| **PB v4 iter6（最终；argmax 诊断）** | 551.05 | 451.59 | 488.0 |
| **PB v4 iter6（最终；sample 全量 dev）** | **548.45** | **447.64** | **420.3** |

- 纯推理平滑能抬 complex，但 dev_param 掉得更多，且 PB 仍只有
  250–280；
- PB-heavy 的 complex 与最终版接近，但 test_v7 只有 484.93，
  最终 `train_v4` 版为 485.43；
- 详细中间 checkpoint 与 eval JSON 在
  `results/rl_v4_20261028/pb_v4_ckpts/`。

## 6. 判定与下一步

**判定：注册 `rl-pattern` 为实验候选，不 promotion。**

- 它现在是 Protocol v4 dev 上最好的注册候选，但不是 final：
  test_v7 provisional 仍略低于 RL-Repair；
- 主权重 `weights/rl_policy_weights.json` 不变；RL-Repair / A60 /
  RL-PPO 继续作为 fallback；
- 下一步若继续这个方向，应优先：
  1. 在 `train_v4` 中增加 test-only 风格的近似 family（不碰 test_v7），
     观察 hyper_extort / axelrod_extra 回退是否可修；
  2. 用 `--save-all-checkpoints` 做更多 checkpoint 选择，而不是继续
     加大 PB 权重；
  3. 固定独立 seed 复现 dev 收益，再决定是否进入 promotion 流程。

## 7. 复现

```bash
python3 -m ipd list-policies | grep rl-pattern
python3 -m ipd self-test --candidates rl-pattern

# dev（与本文表一致）
python3 -m ipd protocol eval --candidates RL-PPO,RL-Repair,RL-Pattern \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-PPO \
  --out results/rl_v4_20261028/protocol_v4_dev_rl_pattern.json

# test_v7（provisional；不会消费）
python3 -m ipd protocol eval --candidates RL-PPO,RL-Repair,RL-Pattern \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 --candidate-chunk 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --no-consume \
  --out results/rl_v4_20261028/protocol_v4_test_v7_rl_pattern_provisional.json
```

权重：

- `weights/rl_policy_weights_pattern.json`
  sha256 `5c50db0ed1c560e2d05cbc52a4006dc0b3e7fa55cf102f0ba932c1b7433619ae`
- `weights/rl_policy_checkpoint_pattern.json`
  sha256 `13f59b869e0cdc5380ba71b4ae97ca13fcd1bdca8b1553883ef4bb46fc413f42`
