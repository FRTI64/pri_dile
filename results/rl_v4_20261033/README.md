# RL-Wide160 hard fine-tune（2026-10-31）

> 在 RL-Wide160 上做 2 iter `train_hard` fine-tune（lr 2e-5，KL 0.1）。
> **未注册**：dev/test 只有 +0.05 / +0.14，属于噪声级变化。

## 结果

| 集 | RL-Wide160 | hard iter1 |
|---|---:|---:|
| dev_param | 548.26 | **548.31** |
| dev_family | 583.08 | 583.06 |
| dev_complex | **454.31** | 454.12 |
| dev_axelrod | 582.38 | **582.55** |
| dev_ood | **521.37** | 521.19 |
| test_v7（provisional） | 485.62 | **485.76** |

`min_norm` 与 RL-Wide160 基本一致（0.996–0.997）。没有足够收益，
不注册新候选；保留 `train_hard` 作为后续 hard-mining 的参考。

## 文件

- `wide160_hard_history.json`；
- `wide160_hard_iter1_dev_*.json` / `wide160_hard_iter1_test_v7.json`；
- 大 checkpoint 未保留。

## 复现

```bash
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 2 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261041 \
  --init-weights weights/rl_policy_checkpoint_wide160.json \
  --train-dists original,late,grim30,wide,train_v4,train_hard \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261033/wide160_hard_ckpts \
  --bc-kl-coef 0.1 --lr 2e-5 --ent-coef 0.005 --target-kl 0.02 \
  --out results/rl_v4_20261033/wide160_hard_weights_best.json \
  --checkpoint-out results/rl_v4_20261033/wide160_hard_ac_best.json \
  --history-out results/rl_v4_20261033/wide160_hard_history.json
```
