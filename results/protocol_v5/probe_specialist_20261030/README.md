# Probe-only specialist（2026-10-30）

> **实验，未注册、未 promotion。** 只训 `probe_exploit` 一个 family，
> 验证“专门的反 probe 专家”是否可行。

## 训练

- 起点：`weights/rl_policy_checkpoint_wide160.json`
- 训练分布：`train_probe_only`（只含 `train_complex` 的 probe_exploit
  参数点 + anchors）
- 8 iter，lr 1e-4，bc-kl 0.02，`--select-metric avg`
- 产物：`ckpts/iter_001..008.json`

## 结果（30 fields × 2048）

| 候选 | dev_complex | probe_exploit | pattern_breaker |
|---|---:|---:|---:|
| RL-Wide160（baseline） | 477.77 | 115.7 | 632.1 |
| ProbeSpec-iter1 | 477.29 | 114.7 | 619.9 |
| ProbeSpec-iter2 | 476.54 | 114.0 | 605.8 |
| ProbeSpec-iter4 | 473.67 | 113.1 | 572.8 |
| ProbeSpec-iter8 | 472.24 | 111.2 | 560.4 |

结论：**probe-only specialist 没有超过 base，训练越久越差**。训练 reward
约 218–234，但 dev probe 不升；说明不是“训练量不够”，而是训练参数覆盖
与特征表达能力不足。

## 关键发现：参数覆盖缺口

`dev_complex` 的 probe_exploit 只有：

```text
mode=alld, probe=6
mode=alld, probe=8
```

而 `train_complex` 的 probe_exploit 是：

```text
mode=alld,   probe=2,4
mode=extort, probe=2,4,8
mode=harvest,probe=2,4,8
```

**dev/test 的 `alld` probe=6/8 在 train 中没有对应参数点。**
因此 specialist 无法从 train 学会这两类 dev 配置；这解释了为什么
`train_probe_only` 训练 reward 在升、dev probe 却不动。

## 文件

| 文件 | 内容 |
|---|---|
| `screen_dev_complex.json` | iter1/2/4/8 的 dev_complex 筛选 |
| `probe_specialist_history.json` | 训练历史 |
| `probe_specialist_weights.json` | best 导出权重（iter0/best） |
| `probe_specialist_ac.json` | best actor-critic |
| `ckpts/` | iter1–8 |
