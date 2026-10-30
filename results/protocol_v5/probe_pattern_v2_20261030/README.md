# ProbePattern v2：probe_exploit / pattern_breaker 定向 fine-tune

> **实验候选，未注册、未 promotion。** Protocol v5 dev 诊断显示
> `probe_exploit` 是头部学习策略的共同瓶颈；本实验从第一轮
> `ProbePattern-iter3` 继续，提高 `train_probe_pattern` 权重，验证
> probe 是否能靠训练信号解决。

## 训练

- 起点：`results/protocol_v5/probe_pattern_20261030/ckpts/iter_003.json`
- 训练分布：`train_v5` + `train_probe_pattern`，权重 **1:2**
- 6 iter，lr 5e-5，bc-kl 0.05，`--select-metric balanced`
- 校验：`dev_param,dev_complex,dev_ood`，2 fields × 256
- 全量选择：iter4（30-field dev_complex / dev_ood 最优折中）

```bash
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 6 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261051 \
  --init-weights results/protocol_v5/probe_pattern_20261030/ckpts/iter_003.json \
  --train-dists train_v5,train_probe_pattern \
  --dist-weights 1,2 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/protocol_v5/probe_pattern_v2_20261030/ckpts \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/protocol_v5/probe_pattern_v2_20261030/probe_pattern_v2_weights.json \
  --checkpoint-out results/protocol_v5/probe_pattern_v2_20261030/probe_pattern_v2_ac.json \
  --history-out results/protocol_v5/probe_pattern_v2_20261030/probe_pattern_v2_history.json
```

## 结果（30 fields × 2048，默认 sample RNG）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | probe_exploit | pattern_breaker |
|---|---:|---:|---:|---:|---:|---:|---:|
| RL-Wide160（baseline） | 531.66 | 583.08 | 477.77 | 567.77 | 522.24 | 115.8 | 632.1 |
| ProbePattern-v2-iter4 | 531.11 | 583.11 | **481.81** | 567.60 | **523.14** | 118.1 | **762.0** |

相对 RL-Wide160：

- `dev_complex` **+4.04**、`dev_ood` **+0.90**；
- `dev_param` -0.55、`dev_axelrod` -0.17；
- `pattern_breaker` **+129.9**；
- `probe_exploit` 只 **+2.3**，仍低于 RL-PPO 的 124.0；
- `probe_exploit` min_norm 约 **0.951**（旧 gate 0.99 未过）；
- `adaptive_mix` / `conditional_extort` 回退约 -20 / -9。

## 结论

- `pattern_breaker` 可以通过 `train_probe_pattern` 定向训练显著改善。
- `probe_exploit` **不是简单增加训练权重能解决**：两轮训练后仍约
  118，远低于 gate；下一步应做 probe 检测特征、显式反 probe 规则，
  或 specialist + gate，而不是继续扫同一训练权重。
- 本候选只保留为实验产物，不进入注册表、不 promotion；`test_v8`
  未消费。

## 文件

| 文件 | 内容 |
|---|---|
| `probe_pattern_v2_dev.json` | iter4 的完整 5-tier dev |
| `screen_dev_complex_ood.json` | iter0/4/5/6 的 complex/ood 筛选 |
| `probe_pattern_v2_history.json` | 训练历史与 best iteration |
| `probe_pattern_v2_weights.json` | iter4 导出的可部署权重 |
| `probe_pattern_v2_ac.json` | best iter4 actor-critic checkpoint |
| `ckpts/` | iter1–6 checkpoints |
