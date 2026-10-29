# 192/128 宽度消融（2026-10-30）

> 从 RL-Wide 128/128 扩到 192/128，再做 6 iter balance fine-tune。
> **未注册候选**：dev_complex 更高，但 dev_param / test_v7 回退，
> 128/128 仍是更均衡的宽度。

## 结果

| 集 | RL-Wide 128/128 | 192/128 iter6 |
|---|---:|---:|
| dev_param | **548.24** | 547.44 |
| dev_family | 583.08 | **583.10** |
| dev_complex | 452.79 | **459.57** |
| dev_axelrod | **582.30** | 581.25 |
| dev_ood | 520.71 | **522.32** |
| test_v7（provisional） | **485.50** | 484.90 |

- dev_complex +6.78、dev_ood +1.61；
- dev_param -0.80、dev_axelrod -1.05、test_v7 -0.60；
- 说明继续加宽主要放大 complex 收益，不能改善主排序/测试泛化；
  暂不注册 `rl-wide192`。

## 文件

| 文件 | 内容 |
|---|---|
| `wide192_ckpts/iter_006.json` | 192/128 actor-critic checkpoint |
| `wide192_history.json` | 6 iter 训练曲线 |
| `wide192_iter6_dev_*.json` | 固定 seed 逐 tier 诊断 |
| `wide192_iter6_test_v7.json` | test_v7 provisional 诊断 |

## 复现

```bash
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_wide.json \
  --hidden 192 128 \
  --out results/rl_v4_20261030/wide192_ac.json \
  --seed 20261031

python3 -m ipd train rl --device cuda --hidden 192 128 \
  --iterations 6 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261037 \
  --init-weights results/rl_v4_20261030/wide192_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261030/wide192_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261030/wide192_weights_best.json \
  --checkpoint-out results/rl_v4_20261030/wide192_ac_best.json \
  --history-out results/rl_v4_20261030/wide192_history.json
```
