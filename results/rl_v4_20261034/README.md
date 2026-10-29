# 160/160 宽度消融（2026-10-31）

> 从 RL-Wide160 160/128 扩展第二层到 160，再做 2 iter balance。
> **未注册**：dev_param / dev_ood / test_v7 均小幅回退，160/128 仍
> 是更均衡的宽度。

## 结果

| 集 | RL-Wide160 160/128 | 160/160 iter1 |
|---|---:|---:|
| dev_param | **548.26** | 547.92 |
| dev_family | **583.08** | **583.08** |
| dev_complex | 454.31 | **454.79** |
| dev_axelrod | **582.38** | 581.91 |
| dev_ood | **521.37** | 521.24 |
| test_v7（provisional） | **485.62** | 485.19 |

complex `min_norm` 0.991、ood 0.992，和 160/128 接近；但主排序和
test 回退，收益不足以注册新候选。

## 文件

- `wide160x160_history.json`；
- `wide160x160_iter1_dev_*.json` / `wide160x160_iter1_test_v7.json`；
- 大 checkpoint 未保留。

## 复现

```bash
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_wide160.json \
  --hidden 160 160 \
  --out results/rl_v4_20261034/wide160x160_ac.json \
  --seed 20261034

python3 -m ipd train rl --device cuda --hidden 160 160 \
  --iterations 2 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261042 \
  --init-weights results/rl_v4_20261034/wide160x160_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261034/wide160x160_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261034/wide160x160_weights_best.json \
  --checkpoint-out results/rl_v4_20261034/wide160x160_ac_best.json \
  --history-out results/rl_v4_20261034/wide160x160_history.json
```
