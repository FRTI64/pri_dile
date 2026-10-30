# 精确反 probe 规则 overlay（2026-10-30）

> **实验，未注册、未 promotion。** 规则：若对手首次 D 发生在 t=5
> 或 t=7、此前全 C，且首次 D 后至少再 D 两次，则切换到 AllD。
> 目标是对 `ProbeExploit:alld` 的 probe=6/8，避免误伤早期合作的
> `conditional_extort`。

## 结果

### 以 RL-Wide160 为 base（30 fields × 2048）

| 变体 | dev_complex | dev_ood | probe（complex/ood） |
|---|---:|---:|---:|
| base | **477.77** | **522.24** | 115.67 / 115.98 |
| first D=5 | 476.90 | 520.86 | 122.08 / 121.96 |
| first D=7 | 476.07 | 520.58 | 122.49 / 123.26 |
| first D=5/7 | 474.62 | 519.53 | **128.90 / 129.23** |

### 以 ProbePattern-v2-iter4 为 base（全 5-tier）

| 变体 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | probe（complex/ood） |
|---|---:|---:|---:|---:|---:|---:|
| base | **531.11** | 583.11 | **481.81** | **567.60** | **523.14** | 117.80 / 118.18 |
| first D=5 | 528.48 | 583.13 | 480.71 | 565.91 | 521.93 | 123.35 / 123.27 |
| first D=7 | 529.25 | 583.11 | 479.58 | 566.46 | 521.55 | 123.32 / 124.04 |
| first D=5/7 | 526.61 | 583.13 | 478.69 | 564.88 | 520.62 | **128.90 / 129.23** |

结论：

- 精确规则确实能把 `probe_exploit` 从约 118 拉到 123–129；
- 但会触发少量非 probe 对手，`dev_param` 下降 1.9–4.5、
  `dev_axelrod` 下降 1.1–2.7、`dev_complex/ood` 也回退；
- 只有 first D=5 变体在 v2 base 上让 probe min_norm 接近 0.99
  （123.3 / 124.0 = 0.994），但整体仍低于 base；
- **规则 overlay 不满足净收益**，不注册。

## 文件

| 文件 | 内容 |
|---|---|
| `probe_rule_dev.json` | RL-Wide160 base 的 complex/ood 规则扫描 |
| `probe_rule_v2dev.json` | ProbePattern-v2 base 的 complex/ood |
| `probe_rule_v2dev_full.json` | ProbePattern-v2 base 的完整 5-tier |
| `eval_rule.py` | 规则 overlay 评估脚本 |
