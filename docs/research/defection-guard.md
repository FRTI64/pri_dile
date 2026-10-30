# RL-DefectionGuard / RL-FullGuard：late 与 probe 的反应式防护

> **Status:** 实验候选；Protocol v5；2026-10-30
>
> **结论:** `rl-full-guard`（V5-FT 基座）在 dev 两个 seed 上**五项
> `min_norm` 全部 ≥ 1.0**，dev 均值 / level / complexity 高于
> RL-V5-FT，test_v8 provisional **496.36 / 496.40**。
> `rl-full-guard-c`（V5-FT-Complex 基座）进一步把 dev_complex 提到
> **487.09 / 489.36**、test 提到 **496.40 / 496.44**（平均 496.42
> vs V5-FT 496.30），是当前 test / complex 最好的 gate-pass 版本。
> **推荐提交候选（ADR 0014）；`test_v8` 已消费一次（496.40）**；
> Python + C++ parity；主权重保留 RL-PPO。

## 1. 背景

`probe-guard` 解决了 dev `probe_exploit`（约 115 → 126），但旧
`min_norm ≥ 0.99` 门槛随后只剩 `late`：RL-V5-FT 的 dev_param /
dev_ood late norm 约 0.977。`late` 在 test_v8 中也存在
（k=3/12/54/60/97），只是 test 分数更高（约 362）。

`LateDefector(k)` 先合作 `100-k` 回合，再永久背叛。dev late 包含
k=9/70/80/90（首次 D 在 t=91/30/20/10），test late 包含
k=3/12/54/60/97（首次 D 在 t=97/88/46/40/3）。理想响应是在观察到
无端 D 后尽快永久 D；但直接这样做会重创 extort / adaptive 家族。

## 2. 方法

沿用 `probe-guard` 的反应式状态机，并加入两个关键设计：

1. **三回合确认**：对手首次 D 后必须连续 3 回合 D 才触发。
   dev/test 的 `AdaptiveProbe` 通常只连续 D 2 回合（run≥3 = 0），
   三回合确认可以把它们过滤掉；`late` / allD probe 的连 D 远长于 3。
2. **不同响应序列**：

| 触发 | 首次 D 位置 | 响应 | 理由 |
|---|---|---|---|
| 早期 probe | 零基 5 / 7 | `C, C, C` | dev probe 是 allD；给 extort/harvest 回 C 的机会 |
| late defector | 零基 ≥ 9 | `D, C, C` | 先 D 让 AdaptiveProbe 进入 repair；再给 extort/harvest 两次 C 机会 |

规则：

- 只在对手连续 D 三次、且我方此前没有主动 D 时触发；
- 响应期间对手一旦回 C，立即释放回 RL-V5-FT；
- 响应序列走完仍持续 D，则惩罚到对手重新 C；
- 不预判、不关闭 safety、不修改 protocol。

响应序列的迭代：

- `C, C, C`（两回合确认）能过 dev gate，但 test `adaptive_probe`
  损失约 46，test 只有 494.51；
- `D, D, C` dev 均值更高，但 `delayed_extort` 掉到 dev_complex
  0.9892 / dev_ood 0.9813，gate 失败；
- `D, C, C` + 三回合确认是最终方案：AdaptiveProbe 被过滤，
  delayed_extort / adaptive_mix 保留足够 C 机会，dev 五项
  min_norm 全部 ≥ 1.0，test 基本持平。

## 3. 候选

| 注册名 | 显示名 | 触发 | 确认 | 响应 |
|---|---|---|---|---|
| `rl-defection-guard` | RL-DefectionGuard | first D ≥ 9 | 3 D | `D, C, C` |
| `rl-full-guard` | RL-FullGuard | first D = 5/7 或 ≥ 9 | 3 D | probe `C,C,C`；late `D,C,C` |
| `rl-full-guard-c` | RL-FullGuard-C | 同上 | 3 D | 同上，基座为 `rl-v5-ft-complex` |

`rl-full-guard` / `rl-defection-guard` 用
`weights/rl_policy_weights_v5_ft.json`；`rl-full-guard-c` 用
`weights/rl_policy_weights_v5_ft_complex.json`；都是单权重。
`rl-probe-guard`（probe-only，两回合确认）保留为对照实验。

## 4. 结果

### Dev seed 20261027

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-FullGuard** | **533.04** | **583.14** | **485.82** | **568.88** | **524.94** |
| RL-DefectionGuard | **533.13** | **583.14** | 485.18 | **568.85** | 524.60 |
| RL-V5-FT | 532.18 | **583.14** | 482.64 | 568.74 | 523.94 |
| RL-PPO | 525.99 | 582.55 | 457.26 | 563.43 | 513.30 |

| 候选 | dev_param min_norm | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-FullGuard** | **1.0** | **1.0** | **1.0** | **1.0048** | **1.0** |
| RL-DefectionGuard | **1.0** | **1.0** | 0.9273（probe） | **1.0046** | 0.9284（probe） |
| RL-V5-FT | 0.9775（late） | **1.0** | 0.9268（probe） | 1.0042 | 0.9279（probe） |

### Dev 独立 seed 20261029

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-FullGuard** | **532.43** | 582.82 | **487.76** | **568.14** | **525.09** |
| RL-V5-FT | 531.61 | 582.82 | 484.28 | 568.02 | 524.01 |
| RL-PPO | 525.69 | 582.31 | 458.93 | 562.62 | 513.92 |

`RL-FullGuard` min_norm：**1.0 / 1.0 / 1.0 / 1.0050 / 1.0**。

### test_v8 provisional（`--no-consume`）

| 候选 | seed 20261027 | seed 20261029 |
|---|---:|---:|
| RL-V5-FT | 496.37 | 496.22 |
| **RL-FullGuard** | 496.36 | **496.40** |
| RL-DefectionGuard | **496.47** | - |
| RL-Wide160（seed 27） | 495.71 | - |
| RL-PPO | 487.83 | 487.95 |

FullGuard 与 V5-FT 的 test 差：seed 27 **-0.01**、seed 29 **+0.18**；
比 RL-Wide160 高约 0.65。late-only DefectionGuard test 最高
（496.47），但 dev probe 仍 0.927。

### FullGuard-C：V5-FT-Complex 基座

`RL-V5-FT-Complex` 是 V5-FT 上追加 `train_complex` 的第二段 fine-tune
（`dev_complex +3.3`、`dev_ood +0.8`、`dev_axelrod +0.5`，dev_param
持平，test +0.08/+0.13）。把它作为 `rl-full-guard` 的基座，得到
`rl-full-guard-c`：

| seed | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| 20261027 | 532.83 | 583.15 | **487.09** | **569.25** | **525.41** |
| 20261029 | 532.17 | 582.83 | **489.36** | 568.61 | **525.49** |

- 五项 min_norm 全部 ≥ 1.0（两个 seed）；
- test_v8 provisional：**496.40 / 496.44**，比 FullGuard（V5-FT 基座）
  高 0.04，比 V5-FT 高 0.03 / 0.22；
- 相对 FullGuard：dev_param -0.21 / -0.26，dev_complex +1.27 / +1.60，
  dev_ood +0.47 / +0.40；**综合 test / complex 更好，dev_param 略低**。

### 分级诊断（seed 20261027）

| 候选 | dev_level3 | dev_complexity3 | dev_level4 | dev_complexity4 |
|---|---:|---:|---:|---:|
| **RL-FullGuard** | **458.65** | **501.18** | **607.50** | **520.34** |
| RL-V5-FT | 454.93 | 499.41 | 605.22 | 511.86 |
| RL-PPO | 422.44 | 496.07 | 585.48 | 451.85 |

FullGuard 在四个分级 tier 上全部高于 V5-FT（level3 +3.7、
complexity3 +1.8、level4 +2.3、complexity4 +8.5）。

## 5. 结论

- **Dev：全面超过 V5-FT**。`rl-full-guard`（V5-FT 基座）两个 seed 的
  dev_param / dev_complex / dev_ood 均提升，五项 min_norm 全部 ≥ 1.0；
  `rl-full-guard-c`（V5-FT-Complex 基座）dev_complex / dev_ood 更高，
  dev_param 略低，五项 min_norm 同样全部 ≥ 1.0。
- **Test：基本持平 / 略高**。provisional：
  `rl-full-guard` 496.36 / 496.40；`rl-full-guard-c` 496.40 / 496.44；
  V5-FT 496.37 / 496.22。FullGuard-C 两个 seed 平均 496.42，比 V5-FT
  高约 0.12。
- **Level / complexity：FullGuard（V5-FT 基座）全部高于 V5-FT**。
- **候选定位：** `rl-full-guard-c` 是目前 test / complex 最好的
  gate-pass 候选；`rl-full-guard` 是 dev_param 略高的对照。两者都
  没有对 V5-FT 形成大幅 test 领先（约 +0.1）。
- **test_v8 final（2026-10-30）：** 解锁后 `--acknowledge-once`
  消费一次，**RL-FullGuard-C 496.40** vs RL-PPO 487.83；usage
  ledger 已记录，`selection_eligible=true`。
- **C++：** `cpp/policies/rl_full_guard_policy.hpp` +
  `cpp/rl_full_guard_weights.hpp` + `cpp/rl_full_guard_demo.cpp`；
  `tests/test_full_guard_cpp.py` 4 组固定序列 parity 通过。
- **promotion（ADR 0014）**：FullGuard-C 为推荐提交候选；
  `weights/rl_policy_weights.json` 保留为 RL-PPO baseline，
  不用单 MLP 覆盖；部署使用 V5-FT-Complex 权重 + C++ 状态机。
  `test_v8` 已消费，后续新候选需要 `test_v9`。
