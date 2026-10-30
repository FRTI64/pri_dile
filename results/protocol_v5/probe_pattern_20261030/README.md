# ProbePattern v1：probe_exploit / pattern_breaker 定向 fine-tune

> **第一轮实验，未注册、未 promotion。** 从 RL-Wide160 的 actor-critic
> checkpoint 出发，用 `train_v5 + train_probe_pattern`（权重 2:1）
> 做 4 iter fine-tune。

## 训练

```bash
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 4 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261050 \
  --init-weights weights/rl_policy_checkpoint_wide160.json \
  --train-dists train_v5,train_probe_pattern \
  --dist-weights 2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/protocol_v5/probe_pattern_20261030/ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/protocol_v5/probe_pattern_20261030/probe_pattern_weights.json \
  --checkpoint-out results/protocol_v5/probe_pattern_20261030/probe_pattern_ac.json \
  --history-out results/protocol_v5/probe_pattern_20261030/probe_pattern_history.json
```

## 结果（30 fields × 2048，默认 sample RNG）

| 候选 | dev_param | dev_complex | dev_axelrod | dev_ood | probe_exploit | pattern_breaker |
|---|---:|---:|---:|---:|---:|---:|
| RL-Wide160（baseline） | 531.66 | 477.77 | 567.77 | 522.24 | 115.8 | 632.1 |
| ProbePattern-iter3 | 531.25 | 479.38 | 567.65 | 522.57 | 116.8 | 674.6 |

- `dev_complex` +1.61、`dev_ood` +0.33、`dev_param` -0.41；
- `pattern_breaker` +42.5，`probe_exploit` 只 +1.0；
- iter3 被第二轮继续 fine-tune；见
  `results/protocol_v5/probe_pattern_v2_20261030/README.md`。

## 文件

| 文件 | 内容 |
|---|---|
| `probe_pattern_dev.json` | iter3 的完整 5-tier dev |
| `screen_dev_complex_ood.json` | iter1–4 的 complex/ood 筛选 |
| `probe_pattern_weights.json` | 训练导出的 best 权重 |
| `probe_pattern_ac.json` | best actor-critic checkpoint |
| `ckpts/` | iter1–4 checkpoints |
