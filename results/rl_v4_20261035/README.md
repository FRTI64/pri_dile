# 输入特征 v2（85 → 97）消融（2026-10-29）

> 在 RL-Wide160 上加入 12 个周期/响应特征，验证 Python / GPU / C++
> 三条路径的 v2 契约。**未注册候选**：dev_complex 略升，但
> dev_param / test_v7 小幅回退，收益不足以 promotion。

## 结果

| 集 | RL-Wide160 85 维 | v2 iter2 | v2 iter4 |
|---|---:|---:|---:|
| dev_param | **548.26** | 547.90 | 547.59 |
| dev_family | **583.08** | 583.05 | 583.03 |
| dev_complex | 454.31 | 456.43 | **458.72** |
| dev_axelrod | **582.38** | 582.09 | 581.52 |
| dev_ood | 521.37 | 521.62 | **522.04** |
| test_v7（provisional） | **485.62** | 485.27 | 485.25 |

`min_norm`：

| 变体 | dev_param | dev_complex | dev_ood |
|---|---:|---:|---:|
| v2 iter2 | 0.9980 | 0.9983 | 0.9942 |
| v2 iter4 | 0.9989 | 0.9987 | 0.9922 |

- v2 把 `pattern_breaker` 的 test 值从约 461 拉到 519–577；
- 但 `hyper_extort` / `axelrod_extra` 仍未恢复，test 均值反而 -0.35；
- dev_param 的 0.36–0.67 回退大于 complex 的收益，因此不注册。

## 文件

| 文件 | 内容 |
|---|---|
| `wide160_v2_history.json` | 8 iter v2 训练曲线 |
| `wide160_v2_iter{2,4}_dev_*.json` | 固定 seed 逐 tier 诊断 |
| `wide160_v2_iter{2,4}_test_v7.json` | test_v7 provisional 诊断 |

大 checkpoint 未保留；训练用 `train expand --features v2` 从
`weights/rl_policy_checkpoint_wide160.json` 生成 97 维初始化，再用
`train rl --features v2 --hidden 160 128` fine-tune。

## 复现

```bash
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_wide160.json \
  --features v2 \
  --out results/rl_v4_20261035/wide160_v2_ac.json \
  --weights-out results/rl_v4_20261035/wide160_v2_weights.json \
  --seed 20261035

python3 -m ipd train rl --device cuda --features v2 --hidden 160 128 \
  --iterations 8 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261051 \
  --init-weights results/rl_v4_20261035/wide160_v2_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261035/wide160_v2_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261035/wide160_v2_weights_best.json \
  --checkpoint-out results/rl_v4_20261035/wide160_v2_ac_best.json \
  --history-out results/rl_v4_20261035/wide160_v2_history.json
```
