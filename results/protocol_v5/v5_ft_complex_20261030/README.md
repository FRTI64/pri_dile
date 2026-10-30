# V5-FT-Complex：第二段 `train_complex` fine-tune（2026-10-30）

> **实验基座，未注册为提交候选。** 从 RL-V5-FT 出发，用
> `train_v5 + train_complex`（1:1）做 10 iter PPO fine-tune，选 iter6。
> dev_complex / dev_ood / dev_axelrod 高于 V5-FT，dev_param 基本持平；
> test_v8 provisional 496.45 / 496.35 vs V5-FT 496.37 / 496.22。
> `probe_exploit` / `late` 仍使 min_norm 低于 0.99，因此只作为
> `rl-full-guard-c` 的基座。

## 训练

```bash
python3 -m ipd train rl --device cuda \
  --iterations 10 --fields-per-dist 3 --n 160 --workers 20 \
  --seed 20261063 --hidden 160 128 \
  --init-weights weights/rl_policy_checkpoint_v5_ft.json \
  --train-dists train_v5,train_complex --dist-weights 1,1 \
  --dists dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --val-fields-per-dist 3 --val-n 160 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/protocol_v5/v5_ft_complex_20261030/ckpts \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/protocol_v5/v5_ft_complex_20261030/weights.json \
  --checkpoint-out results/protocol_v5/v5_ft_complex_20261030/checkpoint.json \
  --history-out results/protocol_v5/v5_ft_complex_20261030/history.json
```

选择 iter6（`ckpts/iter_006.json`）；权重已复制到
`weights/rl_policy_weights_v5_ft_complex.json` / checkpoint
`weights/rl_policy_checkpoint_v5_ft_complex.json`。

## Dev（30 fields × 2048）

| seed | 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---|---|---:|---:|---:|---:|---:|
| 20261027 | **V5-FT-Complex** | 532.14 | 583.15 | **485.95** | **569.28** | **524.79** |
| 20261027 | RL-V5-FT | **532.18** | 583.14 | 482.64 | 568.74 | 523.94 |
| 20261029 | **V5-FT-Complex** | 531.54 | 582.84 | **487.76** | 568.51 | **524.84** |
| 20261029 | RL-V5-FT | **531.61** | 582.82 | 484.28 | 568.02 | 524.01 |

- 相对 V5-FT：dev_param -0.04 / -0.07，dev_complex **+3.31 / +3.48**，
  dev_axelrod +0.54 / +0.49，dev_ood +0.85 / +0.83；
- min_norm 仍被 `probe_exploit`（0.934）和 `late`（0.981）卡住，
  所以单模型不能过旧 gate。

## test_v8 provisional（不消费）

| 候选 | seed 20261027 | seed 20261029 |
|---|---:|---:|
| **V5-FT-Complex** | **496.45** | **496.35** |
| RL-V5-FT | 496.37 | 496.22 |
| RL-PPO | 487.83 | 487.95 |

相对 V5-FT：+0.08 / +0.13，方向一致但幅度很小。

## `train_complex` 权重消融

| 权重（train_v5 : train_complex） | dev_param | dev_complex | test_v8 |
|---|---:|---:|---:|
| 1 : 1（本候选，iter6） | 532.14 | 485.95 | **496.45** |
| 1 : 2（iter10） | 531.48 | **490.49** | 495.85 |

1:2 的 dev_complex 更高，但 dev_param / test 回退；**1:1 是本轮最优**。
1:2 产物保留在 `results/protocol_v5/v5_ft_complex2_20261030/`，不注册。

## 结论

- `V5-FT-Complex` 是一个**更好的基座**：dev_complex / ood / axelrod
  更高，dev_param 持平，test 略高；
- 仍不能单独过 dev gate；交给 `rl-full-guard-c` 修复 probe / late；
- 实验产物，未 promotion、`test_v8` 未消费。
