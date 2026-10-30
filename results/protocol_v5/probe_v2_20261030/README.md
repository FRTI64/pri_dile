# Probe specialist + 特征 v2（2026-10-30）

> **实验，未注册、未 promotion。** 在 85→97 维 v2 特征上训练
> `train_probe_only` specialist，验证 `opp_exploit` / 周期特征是否能
> 帮助 probe_exploit。

## 训练

1. 从 `weights/rl_policy_checkpoint_wide160.json` 扩展到 97 维：

```bash
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_wide160.json \
  --features v2 \
  --out results/protocol_v5/probe_v2_20261030/wide160_v2_ac.json \
  --weights-out results/protocol_v5/probe_v2_20261030/wide160_v2_weights.json \
  --seed 20261053
```

2. `train_probe_only`，8 iter，lr 1e-4，bc-kl 0.02。

## 结果（dev_complex，30 fields × 2048）

| 候选 | dev_complex | probe_exploit | pattern_breaker |
|---|---:|---:|---:|
| ProbeV2-iter0（v2 扩展起点） | 477.80 | 115.7 | 634.4 |
| ProbeV2-iter1 | 476.51 | 114.7 | 609.7 |
| ProbeV2-iter4 | 473.85 | 112.2 | 570.1 |
| ProbeV2-iter8 | 471.40 | 110.0 | 539.3 |

结论：**v2 的 12 个周期/响应特征没有解决 probe_exploit**，训练后反而
回退。`opp_exploit` 是被动观测率，不能直接告诉策略“现在应该切换到
反 probe 行为”。

## 文件

| 文件 | 内容 |
|---|---|
| `screen_dev_complex.json` | iter0/1/4/8 的 dev_complex 筛选 |
| `probe_v2_history.json` | 训练历史 |
| `probe_v2_weights.json` | 导出权重 |
| `ckpts/` | iter1–8 |
