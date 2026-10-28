# Plan 5：GPU sample（per-match RNG，opt-in）

- **状态：** 实现完成；opt-in，默认仍为 Protocol v4 sample（CPU）
- **代码：** `ipd/evaluation/protocol.py` 的 `protocol_match_seed` /
  `sample_uniforms`；`ipd/evaluation/gpu/{policies,blue,simulator}.py`
  的 sample kernel；`protocol eval --sample-rng per-match`
- **测试：** `tests/test_gpu_parity.py` 的 `SampleV5ParityTests` /
  `test_sample_v5_enables_gpu`
- **结果：** `results/protocol_v4/protocol_v4_test_v7_all24_gpu_sample_v5.json`

## 1. 为什么要做

Plan 2 只让 `argmax` 上 GPU，因为 Protocol v4 的 sample 语义是：

- 每个候选、每个 field 一条**顺序 Mersenne Twister 流**；
- 它跨 2048 场对手顺序推进，GPU 的 lock-step batch 无法复现。

直接改用 argmax 会把 sample 训练的策略变成确定性策略。实测
`test_v7`（50 fields × 2048）中 RL-PPO-Hard60 下降 16.66、GRU 下降
13.40、Adv-GRU 下降 12.51、RL-PPO 下降 10.31；规则候选基本不变。
这是 mode mismatch，不是 GPU 实现错误。

## 2. v5 契约

`--sample-rng per-match` 显式选择新契约：

```text
seed(base_seed, field_seed, candidate, match_index)
  = crc32(f"{base_seed}|{field_seed}|{candidate}|{match_index}") & 0x7fffffff
```

每场独立一条 `random.Random(seed)` 流；CPU 参考实现和 GPU 使用
**同一组逐轮 uniform**，因此可以逐动作 parity。

- 默认 `--sample-rng v4` 完全不变；v5 必须显式开启。
- v5 结果不是 v4 的逐位等价，不能与 v4 sample 表混用。
- 规则/Axelrod 仍在 CPU；v5 只改学习型蓝方和（GPU 支持的）学习型
  policy 对手的采样。
- policy 对手在 v5 中走 CPU，保留其原始 RNG 流。
- `Adv-*` 包装和 PSRO meta-policy 暂不上 GPU sample：前者会在惩罚轮跳过
  base actor 的采样，后者每场要抽 population 成员；两者先留在 CPU。

## 3. 命令

```bash
python3 -m ipd protocol eval --candidates all --tiers test_v7 \
  --fields 50 --n 2048 --workers 20 --gpu-workers 8 \
  --candidate-chunk 4 --field-reuse on \
  --mode sample --sample-rng per-match --gpu on \
  --gpu-dtype float32 --gpu-field-batch 2 --gpu-max-batch 32768 \
  --seed 20261027 --no-consume \
  --out results/protocol_v4/protocol_v4_test_v7_all24_gpu_sample_v5.json
```

`--gpu on` 必须显式给出；`--gpu off` / 自动 batch 不足会报错，避免把
v5 请求静默跑成 v4。结果会带 `sample_rng=per-match`、`gpu_sample=true`、
`protocol_version=5`。

## 4. Parity 与分布

`SampleV5ParityTests` 对 `rl-repair`、`gru`、`selfplay`、
`psro-scratch-br` 做逐场 CPU/GPU 对比（n=8，float64），
`mine` / `theirs` / `score` 全部一致。

5 fields × n=256 的分布对比（v4 CPU sample vs v5 GPU sample）：

| 候选 | v4 sample | v5 sample | Δ |
|---|---:|---:|---:|
| RL-Repair | 494.11 | 494.68 | +0.57 |
| RL-PPO | 482.47 | 484.36 | +1.90 |
| RL-PPO-Hard60 | 485.75 | 486.04 | +0.30 |
| GRU | 474.50 | 475.11 | +0.60 |
| SelfPlay-GRU | 472.57 | 472.50 | -0.07 |
| PSRO-BR | 477.77 | 475.71 | -2.06 |
| 规则/Adv/PSRO-Meta | — | — | 0.00 |

差异来自 per-match RNG 与 v4 顺序流的抽样差异，量级 0–2 分；argmax
是 10–17 分。v5 恢复了 sample 的随机合作/修复行为。

## 5. 全量 test_v7

50 fields × 2048、24 candidates、8 GPU + 16 CPU：

| 项 | 结果 |
|---|---:|
| wall time | **1365.63s（22:45）** |
| 旧 v4 CPU sample | 9078.47s（2.52h）→ **6.65×** |
| Plan 1 CPU sample | 3052.42s（0.85h）→ **2.23×** |
| GPU argmax | 709.86s（11:50）→ v5 慢 1.92×，但保留 sample 语义 |

GPU v5 支持候选与 v4 sample 的差异：

| 候选 | v4 sample | v5 sample | Δ | argmax Δ |
|---|---:|---:|---:|---:|
| RL-Repair | 486.16 | 486.31 | +0.16 | -1.47 |
| RL-PPO-Hard60 | 479.95 | 479.92 | -0.03 | -16.66 |
| RL-PPO | 476.98 | 477.26 | +0.28 | -10.31 |
| GRU | 470.82 | 470.74 | -0.07 | -13.40 |
| Neural | 471.20 | 470.95 | -0.25 | -8.01 |
| SelfPlay-GRU | 466.05 | 465.92 | -0.13 | -0.02 |
| PSRO-BR | 470.48 | 470.41 | -0.07 | -3.06 |

规则/Axelrod/Adv/PSRO meta 仍走 CPU v4，分数与 v4 表基本一致。

## 6. 限制

- v5 **不是** v4 的 bitwise parity；它是新的 RNG 契约，需要单独基线。
- `Adv-MLP` / `Adv-GRU` / `Adv-RL`、`psro`、`psro-scratch` 仍走 CPU。
- 规则/Axelrod 仍是 CPU；GPU 只加速学习型蓝方。
- 测试集仍永久锁定；`test_v7` v5 结果也只是 provisional。
- 需要 CUDA；无 CUDA 时 `--sample-rng per-match` 会报错，而不是静默回退。
- Plan 6 已把 sample policy 对手接入 v5 GPU；本文的“policy 对手仍走
  CPU”是 Plan 5 当时的状态。`Adv-*` / PSRO meta 仍留在 CPU，以保持
  顺序 RNG 和分数一致。见
  [plan6-python-bottleneck.md](plan6-python-bottleneck.md)。
