> **历史实验：基于 980-config 临时 catalog，已回滚；当前 v5 为 977 configs。**

# probe 参数覆盖修复后的 specialist 验证（2026-10-30）

> **实验，未注册、未 promotion。** 在 v5 train 增加
> `probe_exploit:alld probe=5/7/9` 后，重新训练 safe specialist。

## 覆盖修复

- catalog：977 → **980 configs**
- train-only 新增：`probe_exploit:alld probe=5/7/9`
- dev/test 保持：`alld probe=6/8`、`extort/harvest probe=6`
- manifest：`configs/splits/protocol_v5/manifest.json`
  （sha256 `3747e24f...`）

## safe specialist 结果

| 实验 | 训练分布 | dev_complex | probe_exploit |
|---|---|---:|---:|
| ProbeCover | `train_probe_only`（11 probe + anchors） | 478.85 | 114.2 |
| ProbePure | `train_probe_pure`（probe only，无 anchors） | 476.63 | 114.5 |
| ProbeAlld | `train_probe_alld`（alld only，无 anchors） | 475.66 | 111–114 |

结论：**增加参数覆盖仍不能让 safe specialist 超过 base（115.7）**。
safe 规则禁止在对手 D 之前主动 D，而 probe alld 的收益正来自提前 D。

## 相关目录

- `../probe_pure_20261030/`
- `../probe_alld_20261030/`
- `../probe_alld_unsafe_20261030/`：已拒绝；关闭 safety 的方案不可采用。
