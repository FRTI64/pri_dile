# RL-Wide（2026-10-29）

> 128/128 宽度实验：从 RL-Pattern 64/32 用 net2wider 扩展，再做
> train_v4 + 小权重 pattern_breaker fine-tune。**实验候选，未 promotion。**

## 结论

| 集 | RL-PPO | RL-Repair | RL-Pattern | **RL-Wide** |
|---|---:|---:|---:|---:|
| dev_param | 540.35 | 547.96 | 548.45 | 548.24 |
| dev_family | 582.53 | 582.76 | 583.07 | **583.08** |
| dev_complex | 440.31 | 434.28 | 447.64 | **452.79** |
| dev_axelrod | 574.64 | 582.46 | 582.22 | **582.30** |
| dev_ood | 513.48 | 515.19 | 519.46 | **520.71** |
| test_v7（provisional） | 477.26 | **486.31** | 485.43 | 485.50 |

`min_norm`（vs RL-PPO）：

| 候选 | dev_param | dev_complex | dev_ood |
|---|---:|---:|---:|
| RL-Pattern | 0.9970 | 0.8448 | 0.8424 |
| **RL-Wide** | 0.9968 | **0.9726** | **0.9596** |

- 相对 RL-Pattern：dev_param -0.21、dev_family +0.01、dev_complex
  **+5.15**、dev_axelrod +0.08、dev_ood **+1.25**；
- `pattern_breaker` 420.3 → **483.8**，`lookahead` 355.2 → **379.6**；
- test_v7 相对 RL-Repair -0.81，但相对 RL-Pattern +0.07；
- legacy 20-field：四分布 avg **554.96**、blind **564.72**，与
  RL-Pattern（554.93 / 564.91）基本持平；
- 主权重未替换。

## 文件

| 文件 | 内容 |
|---|---|
| `protocol_v4_dev_rl_wide.json` | **官方 dev 结果**（4 候选） |
| `protocol_v4_test_v7_rl_wide_provisional.json` | **官方 provisional test 结果** |
| `wide_balance_ckpts/iter_004.json` | 最终 actor-critic checkpoint |
| `wide_balance_history.json` | 最终 4 iter 平衡训练曲线 |
| `wide128_ckpts/iter_004.json` | 128/128 gentle 源 checkpoint |
| `wide_gentle_ckpts/iter_008.json` | 128/64 gentle 源 checkpoint |
| `wide_ckpts/iter_004.json` | 128/64 PB-heavy 消融 |
| `wide*_iter*_dev_*.json` | 固定 seed 的逐 tier 诊断 |
| `wide*_iter*_test_v7.json` | 直接 test_v7 诊断 |
| `eval_ckpt_json.py` / `eval_tier_gpu.py` | 诊断脚本（argmax / GPU sample） |

最终部署权重：

- `weights/rl_policy_weights_wide.json`
  sha256 `176787bdb6048d7a5cb0e5adcda9afef12fb77ed1d676b4f2c6ce67b1669d00b`
- `weights/rl_policy_checkpoint_wide.json`
  sha256 `3a2ba29e34e07aded66b531cc564f26ee444990ac7bd073fd76c2e23d5a7fde4`

## 复现

```bash
# 1) 64/32 -> 128/64
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_pattern.json \
  --hidden 128 64 \
  --out results/rl_v4_20261029/wide_ac.json \
  --weights-out results/rl_v4_20261029/wide_weights.json

# 2) 128/64 gentle（8 iter，无额外 PB）
python3 -m ipd train rl --device cuda --hidden 128 64 \
  --iterations 8 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261034 \
  --init-weights results/rl_v4_20261029/wide_ac.json \
  --train-dists original,late,grim30,wide,train_v4 \
  --dist-weights 1,1,1,2,2 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261029/wide_gentle_ckpts \
  --bc-kl-coef 0.1 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261029/wide_gentle_weights_best.json \
  --checkpoint-out results/rl_v4_20261029/wide_gentle_ac_best.json \
  --history-out results/rl_v4_20261029/wide_gentle_history.json

# 3) 128/64 -> 128/128
python3 -m ipd train expand \
  --input results/rl_v4_20261029/wide_gentle_ckpts/iter_008.json \
  --hidden 128 128 \
  --out results/rl_v4_20261029/wide128_ac.json

# 4) 128/128 gentle（8 iter，无额外 PB）
python3 -m ipd train rl --device cuda --hidden 128 128 \
  --iterations 8 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261035 \
  --init-weights results/rl_v4_20261029/wide128_ac.json \
  --train-dists original,late,grim30,wide,train_v4 \
  --dist-weights 1,1,1,2,2 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261029/wide128_ckpts \
  --bc-kl-coef 0.1 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261029/wide128_weights_best.json \
  --checkpoint-out results/rl_v4_20261029/wide128_ac_best.json \
  --history-out results/rl_v4_20261029/wide128_history.json

# 5) 128/128 平衡版（4 iter，PB 权重 1/11）；最终取 iter_004
python3 -m ipd train rl --device cuda --hidden 128 128 \
  --iterations 4 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261036 \
  --init-weights results/rl_v4_20261029/wide128_ckpts/iter_004.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261029/wide_balance_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261029/wide_balance_weights_best.json \
  --checkpoint-out results/rl_v4_20261029/wide_balance_ac_best.json \
  --history-out results/rl_v4_20261029/wide_balance_history.json

# 6) dev / provisional test
python3 -m ipd protocol eval --candidates RL-PPO,RL-Repair,RL-Pattern,RL-Wide \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-PPO \
  --out results/rl_v4_20261029/protocol_v4_dev_rl_wide.json

python3 -m ipd protocol eval --candidates RL-PPO,RL-Repair,RL-Pattern,RL-Wide \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 --candidate-chunk 1 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 2 --seed 20261027 --no-consume \
  --out results/rl_v4_20261029/protocol_v4_test_v7_rl_wide_provisional.json
```
