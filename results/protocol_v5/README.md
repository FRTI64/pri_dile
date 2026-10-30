# Protocol v5（2026-10-30）

> **当前主口径；dev baseline 已运行。** v5 保留 977-config catalog
> 并重新划分 train/dev/test；新 final 为 `test_v8`。v4 的 `test_v7`
> 与全部 v4 结果已冻结为 legacy。

- catalog：977 configs，sha256 `c7aab4da...`
- manifest：`configs/splits/protocol_v5/manifest.json`
- test policy：`configs/splits/protocol_v5/test_policy.json`
  - `test_v8=false`、`test_red=false`（初始锁定）
- 规模：2048 人 / dev 30 fields / test 50 fields
- 变更：405/977 个 config 的 tier 与 v4 不同
- 训练 profile：`train_v5`（`train_v4` / `train_v3` 为 legacy 别名）
- usage ledger：`results/protocol_v5/usage.json`（首次 final 后创建）

## 状态

| 项 | 状态 |
|---|---|
| v5 dev baseline | **已运行**（33 candidates，5 tiers × 30 fields × 2048） |
| v5 test_v8 | 锁定；未消费 |
| v4 对照 | legacy，不能与 v5 混表 |

## v5 dev baseline（2026-10-30）

命令使用默认顺序 sample RNG（`sample_rng=v4`，CPU），不是 per-match GPU
表。完整结果：`protocol_v5_dev.json`；profile：
`profile_protocol_v5_dev.json`。

- 33 candidates，5 tiers × 30 fields × 2048，20 workers
- 本 baseline 在 977-config catalog 上运行，与当前 manifest 一致。
- `dev_axelrod` 单独跑：401.67s；其余 4 tiers：1260.48s
- **总 wall：1662.15s**
- 两个文件已合并；`protocol_v5_dev_rest.json` 保留中间产物

按 `dev_param` 排序（前 8）：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | 最差 family norm |
|---|---:|---:|---:|---:|---:|---:|
| RL-Gate | **532.82** | 582.95 | 477.00 | 567.78 | 522.11 | 0.899（probe_exploit） |
| RL-Repair | 532.53 | 582.78 | 456.41 | 565.46 | 515.80 | 0.790（pattern_breaker） |
| RL-Switch | 532.37 | 582.91 | 477.57 | 566.53 | 521.73 | 0.934（probe_exploit） |
| RL-Pattern | 532.30 | 583.09 | 476.06 | 568.00 | 521.94 | 0.932（probe_exploit） |
| GRU-DAgger | 532.30 | 582.93 | 477.29 | 567.17 | 522.09 | 0.934（probe_exploit） |
| RL-Wide | 531.89 | 583.09 | 476.94 | 568.01 | 522.07 | 0.932（probe_exploit） |
| RL-Wide160 | 531.66 | 583.08 | 477.77 | 567.77 | 522.24 | 0.934（probe_exploit） |
| RL-PPO-Hard60 | 528.55 | 582.72 | 457.81 | 564.79 | 514.68 | 0.874（pattern_breaker） |
| RL-PPO（baseline） | 525.99 | 582.55 | 457.26 | 563.43 | 513.30 | 1.000 |

结论：

- `dev_axelrod` 全候选正常完成，Axelrod 子族分数无异常；矩阵迁移后的
  6/0/10/1 行为在 v5 split 上可运行。
- **v5 的共同弱项是 `probe_exploit`**：RL-Switch / GRU-DAgger /
  RL-Wide / RL-Wide160 的 `dev_ood` min_norm 约 0.932–0.934；RL-Gate
  更低（0.899）。按 v4 常用的 `min_norm ≥ 0.99` 门槛，当前没有任何
  top candidate 通过。
- 因此本 baseline **只作诊断，不 promotion**；需要先决定 v5 的
  family gate 是否调整，或针对 `probe_exploit` / `pattern_breaker`
  做定向训练。

## v5 分级诊断（2026-10-30）

命令：9 candidates（RL-PPO baseline + 8 个头部），
`dev_level0..4` + `dev_complexity0..4`，30 fields × 2048，
默认顺序 sample RNG，20 workers；总 wall **1605.51s**。

结果：`protocol_v5_dev_levels.json`；profile：
`profile_protocol_v5_dev_levels.json`。

关键 family：

| family | RL-PPO | RL-Wide160 | RL-Switch | GRU-DAgger | Triple-v4 | RL-Repair |
|---|---:|---:|---:|---:|---:|---:|
| `probe_exploit`（level3 / complexity3） | 124.0 | 115.8 | 115.8 | 115.8 | 120.9 | 105.9 |
| `pattern_breaker`（level3 / complexity4） | 452/449 | 637/632 | 638/635 | 639/633 | 844/844 | 357/357 |
| level3 mean | 422.44 | 449.67 | 449.26 | 449.30 | **466.27** | 422.74 |
| complexity4 mean | 451.85 | 500.14 | 499.29 | 499.24 | **529.79** | 446.69 |

结论：

- **`probe_exploit` 是共同瓶颈**：头部学习策略都只有约 116 分，低于
  RL-PPO 的 124；`pattern_breaker` 对 RL-Wide160 / RL-Switch /
  GRU-DAgger 已经明显高于 RL-PPO（632–639 vs 449–452）。
- `RL-Repair` 的 `pattern_breaker` 只有约 357，是它 `dev_complex` /
  `dev_ood` min_norm 低的主因；不要把 RL-Repair 的 PB 行为当成头部
  学习策略的通用问题。
- `Triple-v4` 在 `pattern_breaker` 和 complexity4 最强，但
  `probe_exploit` 仍只有 120.9，且学习型候选的 dev_param 更高。

因此定向训练以 **`probe_exploit` 为主、`pattern_breaker` 作为防回退
锚点**，新增 `train_probe_pattern` profile，以 RL-Wide160 为起点。

## 定向 fine-tune（2026-10-30）

### 第一轮：`train_v5 + train_probe_pattern`（权重 2:1）

- 起点：`weights/rl_policy_checkpoint_wide160.json`
- 4 iter，lr 3e-5，bc-kl 0.1，10 GPU/20 CPU
- 产物：`results/protocol_v5/probe_pattern_20261030/`
- 选择：iter3（全量 30-field dev_complex / dev_ood 优于 iter1）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | probe_exploit | pattern_breaker |
|---|---:|---:|---:|---:|---:|---:|---:|
| RL-Wide160（baseline） | 531.66 | 583.08 | 477.77 | 567.77 | 522.24 | 115.8 | 632.1 |
| ProbePattern-iter3 | 531.25 | 583.10 | 479.38 | 567.65 | 522.57 | 116.8 | 674.6 |

- `dev_complex` +1.61、`dev_ood` +0.33，`dev_param` -0.41；
- `pattern_breaker` +42.5，但 `probe_exploit` 只 +1.0；
- `probe_exploit` min_norm 仍约 0.943（vs RL-PPO），未过 0.99 gate。

### 第二轮：更强 probe 权重（`train_v5:train_probe_pattern = 1:2`）

- 起点：第一轮 iter3；6 iter，lr 5e-5，bc-kl 0.05
- 产物：`results/protocol_v5/probe_pattern_v2_20261030/`
- 选择：iter4（全量 dev 最优的折中点）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | probe_exploit | pattern_breaker |
|---|---:|---:|---:|---:|---:|---:|---:|
| ProbePattern-v2-iter4 | 531.11 | 583.11 | **481.81** | 567.60 | **523.14** | 118.1 | 762.0 |

- 相对 RL-Wide160：`dev_complex` **+4.04**、`dev_ood` **+0.90**，
  `dev_param` -0.55、`dev_axelrod` -0.17；
- `pattern_breaker` 632 → **762**，但 `adaptive_mix` / `conditional_extort`
  回退（约 -20 / -9）；
- `probe_exploit` 仍只有 **118.1**（RL-PPO 124.0），min_norm 约 0.951；
- 结论：**当前特征/安全规则下，probe_exploit 不是简单加训练权重能解决的**。
  `pattern_breaker` 已可显著改善；`probe_exploit` 需要新的特征、显式反
  probe 规则，或 specialist + gate。

> 两个 ProbePattern 候选都只是实验产物，未注册、未 promotion；`test_v8`
> 仍未消费。

## 3 → 1 → 2 的 probe 专项结论

按 specialist+gate → 特征 → 显式规则的顺序做了补充实验：

1. **Probe-only specialist**（`probe_specialist_20261030`）：
   `train_probe_only` 训练 8 iter，dev probe 从 115.7 降到
   111–115，**没有增益**。
2. **特征 v2**（`probe_v2_20261030`）：97 维 `opp_exploit` /
   周期特征同样没有帮助，训练后 probe 降到 110–115。
3. **精确规则 overlay**（`probe_rule_20261030`）：首次 D=5/7 后
   切 AllD，可把 probe 拉到 123–129，但会误伤少量标准/复杂对手；
   最佳变体仍让 `dev_param` -1.9～-4.5、`dev_axelrod` -1.1～-2.7，
   净收益为负。

**probe 专项暂停：** 不采用 unsafe specialist / `--allow-probe` /
预判 gate，也不为 probe 修改 protocol、catalog 或 safety 规则。
v5 保持 **977 configs** 和原 manifest；此前的 980-config 覆盖实验
已回滚，相关目录只作历史记录，不能用于当前候选。

> **跨版本泄漏提醒：** v5 与 v4 共用 catalog。v5 `test_v8` 只对在 v5
> train 上训练/选模的候选构成 held-out；旧 v4 候选的 v5 test 分数不能
> 当作独立 test 证据。

## 命令

```bash
python3 -m ipd protocol audit
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v5/protocol_v5_dev.json
```

final 前先在 test policy 中解禁 `test_v8`，再传
`--acknowledge-once`；provisional 诊断使用 `--no-consume`。
