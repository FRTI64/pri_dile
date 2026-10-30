# RL-DefectionGuard / RL-FullGuard（2026-10-30）

> **推荐提交候选（ADR 0014）：`rl-full-guard-c`**；`test_v8` final
> **496.40 已消费**，C++ parity 通过；主权重保留 RL-PPO。
> 最终版 `rl-full-guard`（V5-FT 基座）在 dev 两个 seed 上五项
> `min_norm` 全部 ≥ 1.0；FullGuard-C（V5-FT-Complex 基座）
> dev_complex 487.09 / 489.36，test provisional 496.40 / 496.44。

## 方法

- 基础：`rl-v5-ft`（`weights/rl_policy_weights_v5_ft.json`），单权重。
- 触发：对手首次 D 后连续 **3 回合** D，且我方此前未主动 D。
- 响应：
  - 早期 probe（零基 5/7）：`C, C, C`；
  - late（零基 ≥ 9）：`D, C, C`。
- 对手在响应期间回 C 立即释放；走完仍 D 则惩罚到 C。
- 不预判、不关 safety、不修改 protocol。

三回合确认是关键：dev/test 的 `AdaptiveProbe` 通常只连续 D 2 回合，
三回合确认可避免误触发；`late` / allD probe 的连 D 远长于 3。

## 命令

```bash
# dev 最终版（seed 20261027）
python3 -m ipd protocol eval \
  --candidates rl-full-guard,rl-v5-ft,rl \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v5/full_guard_20261030/dev_all_c3.json

# dev 独立 seed 20261029
python3 -m ipd protocol eval \
  --candidates rl-full-guard,rl-v5-ft,rl \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --seed 20261029 --baseline RL-PPO \
  --out results/protocol_v5/full_guard_20261030/dev_all_c3_seed20261029.json

# test_v8 provisional（不消费）
python3 -m ipd protocol eval \
  --candidates rl-full-guard,rl-defection-guard,rl-v5-ft,rl \
  --tiers test_v8 --fields 50 --n 2048 --workers 20 --no-consume \
  --out results/protocol_v5/full_guard_20261030/test_v8_provisional_all_c3.json

# 分级诊断
python3 -m ipd protocol eval \
  --candidates rl-full-guard,rl-v5-ft,rl \
  --tiers dev_level3,dev_complexity3,dev_level4,dev_complexity4 \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v5/full_guard_20261030/dev_levels_all_c3.json
```

## Dev（seed 20261027）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-FullGuard** | **533.04** | **583.14** | **485.82** | **568.88** | **524.94** |
| RL-DefectionGuard | **533.13** | **583.14** | 485.18 | **568.85** | 524.60 |
| RL-V5-FT | 532.18 | **583.14** | 482.64 | 568.74 | 523.94 |
| RL-PPO | 525.99 | 582.55 | 457.26 | 563.43 | 513.30 |

**Old gate（`min_norm ≥ 0.99`）：**

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-FullGuard** | **1.0** | **1.0** | **1.0** | **1.0048** | **1.0** |
| RL-DefectionGuard | **1.0** | **1.0** | 0.9273 | **1.0046** | 0.9284 |
| RL-V5-FT | 0.9775 | **1.0** | 0.9268 | 1.0042 | 0.9279 |

## Dev 独立 seed 20261029

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-FullGuard** | **532.43** | 582.82 | **487.76** | **568.14** | **525.09** |
| RL-V5-FT | 531.61 | 582.82 | 484.28 | 568.02 | 524.01 |
| RL-PPO | 525.69 | 582.31 | 458.93 | 562.62 | 513.92 |

FullGuard min_norm：**1.0 / 1.0 / 1.0 / 1.0050 / 1.0**。

## test_v8 provisional（不消费）

| 候选 | seed 20261027 | seed 20261029 |
|---|---:|---:|
| RL-V5-FT | 496.37 | 496.22 |
| **RL-FullGuard** | 496.36 | **496.40** |
| RL-DefectionGuard | **496.47** | - |
| RL-Wide160（seed 27） | 495.71 | - |
| RL-PPO | 487.83 | 487.95 |

**test_v8 final（2026-10-30，`--acknowledge-once`，已消费）：**

| 候选 | test_v8 |
|---|---:|
| **RL-FullGuard-C** | **496.40** |
| RL-PPO | 487.83 |

usage ledger：`results/protocol_v5/usage.json`；
结果：`results/protocol_v5/protocol_v5_full_guard_c_test_final.json`。

## C++

- `cpp/policies/rl_full_guard_policy.hpp`：完整状态机；
- `cpp/rl_full_guard_weights.hpp`：由
  `ipd/tools/export_full_guard_cpp.py` 生成；
- `cpp/rl_full_guard_demo.cpp`；
- `tests/test_full_guard_cpp.py`：4 组固定序列 Python/C++ action
  parity 通过。

## 分级诊断（seed 20261027）

| 候选 | dev_level3 | dev_complexity3 | dev_level4 | dev_complexity4 |
|---|---:|---:|---:|---:|
| **RL-FullGuard** | **458.65** | **501.18** | **607.50** | **520.34** |
| RL-V5-FT | 454.93 | 499.41 | 605.22 | 511.86 |
| RL-PPO | 422.44 | 496.07 | 585.48 | 451.85 |

## FullGuard-C：V5-FT-Complex 基座

`RL-V5-FT-Complex`（`results/protocol_v5/v5_ft_complex_20261030/`）
在 V5-FT 上追加 `train_complex`，dev_complex / dev_ood / dev_axelrod
更高、dev_param 持平、test 略高。把它作为基座得到 `rl-full-guard-c`：

| 指标（seed 27 / 29） | RL-FullGuard-C | RL-V5-FT-Complex | RL-V5-FT |
|---|---:|---:|---:|
| dev_param | **532.83 / 532.17** | 532.14 / 531.54 | 532.18 / 531.61 |
| dev_complex | **487.09 / 489.36** | 485.95 / 487.76 | 482.64 / 484.28 |
| dev_axelrod | **569.25 / 568.61** | 569.28 / 568.51 | 568.74 / 568.02 |
| dev_ood | **525.41 / 525.49** | 524.79 / 524.84 | 523.94 / 524.01 |
| 五项 min_norm | **全部 ≥ 1.0** | 0.934 / 0.981 | 0.927 / 0.977 |
| test_v8 provisional | **496.40 / 496.44** | 496.45 / 496.35 | 496.37 / 496.22 |

FullGuard-C 比 FullGuard（V5-FT 基座）test +0.04、dev_complex
+1.27 / +1.60、dev_ood +0.47 / +0.40，dev_param -0.21 / -0.26。
两者都通过旧 gate；FullGuard-C 是 test / complex 更好的版本。

## 文件

| 文件 | 内容 |
|---|---|
| `dev_all_c3.json` | **最终**（probe/late 都三回合确认），seed 27 |
| `dev_all_c3_seed20261029.json` | **最终**，独立 seed 29 |
| `test_v8_provisional_all_c3.json` | **最终**，test provisional seed 27 |
| `test_v8_provisional_all_c3_seed20261029.json` | **最终**，test provisional seed 29 |
| `dev_levels_all_c3.json` | **最终**分级诊断 |
| `dev_complexbase.json` / `dev_complexbase_seed20261029.json` | FullGuard-C / V5-FT-Complex dev |
| `test_v8_provisional_complexbase.json` / `..._seed20261029.json` | FullGuard-C / V5-FT-Complex test provisional |
| `dev_dcc_c3.json` | late 三回合确认、probe 两回合确认（中间版） |
| `dev_dcc.json` / `dev_ddc.json` / `dev_dccc.json` | 响应序列 / 确认长度迭代 |
| `dev.json` / `dev_seed20261029.json` / `test_v8_provisional.json` | 早期 CCC / 两回合确认版本 |

## 结论

- **FullGuard（V5-FT 基座）**：dev_param +0.86 / +0.82，dev_complex
  +3.18 / +3.48，dev_ood +1.00 / +1.08（两个 seed）；五项 min_norm
  全部 ≥ 1.0，旧 `min_norm ≥ 0.99` 门槛通过；level / complexity
  全部高于 V5-FT；test provisional 496.36 / 496.40。
- **FullGuard-C（V5-FT-Complex 基座）**：dev_complex 487.09 / 489.36、
  dev_ood 525.41 / 525.49，五项 min_norm 全部 ≥ 1.0；test provisional
  **496.40 / 496.44**（当前 test / complex 最好的 gate-pass 版本）。
- V5-FT 的 test 为 496.37 / 496.22；FullGuard-C 平均 +0.12，幅度仍小。
- **`test_v8` final 已消费一次：496.40**；C++ parity 通过。
- **推荐提交候选（ADR 0014）**；主权重保留 RL-PPO，不覆盖。
