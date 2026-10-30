# RL-ProbeGuard：反应式早期 probe 防护

> **Status:** 实验候选；Protocol v5；2026-10-30
>
> **结论:** dev `probe_exploit` 从约 115 提到约 126，`dev_complex`
> min_norm 0.927 → 0.996；dev 两个 seed 复现。test_v8 provisional
> 495.92 vs `rl-v5-ft` 496.37（-0.45，噪声内）。**No-Go for promotion**；
> 剩余 gate 瓶颈是 `late`。`test_v8` 未消费。

## 1. 背景

Protocol v5 dev baseline 中，头部学习策略的共同弱项是
`probe_exploit`：RL-Switch / GRU-DAgger / RL-Wide / RL-Wide160 约
115–116，RL-PPO 约 124，导致 `dev_complex` / `dev_ood` 的
`min_norm` 只有约 0.934。此前的 probe 专项（probe-only specialist、
97 维特征、精确规则 overlay）都没有取得净收益，probe 专项一度暂停。

`ProbeExploit` 的机制是：先合作若干回合，在 probe 回合 `D` 一次；
如果对手在同一回合也 `D`（`retaliated=True`），之后回到 TFT；
否则进入 extort / harvest / allD 模式。由于双方同时出招，安全策略
在 probe 回合前看不到这次 `D`，无法“预判”并同时惩罚；这是之前
精确规则和 `--allow-probe` 都不理想的根本原因。

## 2. 方法

不预判，也不关闭 safety。只利用已经观察到的对手 `D`：

1. **触发条件**
   - 对手首次 `D` 出现在零基回合 5 或 7（第 6 / 8 回合）；
   - 下一回合对手继续 `D`；
   - 我方在对手首次 `D` 之前没有主动 `D`。
2. **观察**
   - 连续让 3 回合 `C`；
   - 对手一旦回 `C`，立即回到基础策略（`rl-v5-ft`）。
3. **惩罚**
   - 对手持续 `D`：切到永久 `D`，直到对手重新 `C`。

这个设计区分了两类对手：

- **allD / 永久剥削**：不会回 `C`，3 回合后进入惩罚；
- **extort / harvest**：通常在 probe 后 1–3 回合内会回 `C`，观察期
  结束前就释放，避免把可恢复的 ZD 对手当成永久 defector。

3 回合是刻意选择的折中：allD 的理论上限约 125–133（probe 6/8），
观察 3 回合后仍可拿到约 121–131，与 RL-PPO 的 124 基本持平；
更长的观察会跌破 `min_norm ≥ 0.99`，更短则更容易误伤 extort。

## 3. 实现

- `ipd/policies/meta.py`：`ProbeGuardPolicy` 状态机
  `base → observe → punish`；
- `ipd/policies/candidates/rl_probe_guard.py`：注册
  `rl-probe-guard` / `RL-ProbeGuard`；
- 基础权重：`weights/rl_policy_weights_v5_ft.json`（单权重）；
- 测试：`tests/test_meta.py::ProbeGuardTests` 覆盖触发、晚触发、
  主动 D 排除、合作释放和持续 D 惩罚。

## 4. 结果

### Dev（seed 20261027）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-ProbeGuard** | 532.07 | 583.13 | **482.79** | 568.58 | 523.77 |
| RL-V5-FT | **532.18** | **583.14** | 482.64 | **568.74** | **523.94** |
| RL-PPO | 525.99 | 582.55 | 457.26 | 563.43 | 513.30 |

- `probe_exploit`：**125.89 / 126.20**（V5-FT 114.82 / 115.18）；
- probe family norm：**1.016**（V5-FT 0.927）；
- `dev_complex` min_norm：**0.9959**（V5-FT 0.9268）；
- `dev_ood` min_norm：0.9776；`dev_param` min_norm：0.9776。

### Dev（独立 seed 20261029）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-ProbeGuard** | **531.66** | 582.82 | **485.02** | 568.00 | 523.93 |
| RL-V5-FT | 531.61 | 582.82 | 484.28 | **568.02** | **524.01** |
| RL-PPO | 525.69 | 582.31 | 458.93 | 562.62 | 513.92 |

probe **126.03 / 125.95**，`dev_complex` min_norm **0.9960**。

### test_v8 provisional（50 fields，`--no-consume`）

| 候选 | test_v8 |
|---|---:|
| RL-V5-FT | **496.37** |
| RL-ProbeGuard | 495.92 |
| RL-Wide160 | 495.71 |
| RL-PPO | 487.83 |

test probe 260.53（guard）vs 267.39（V5-FT）vs 233.98（RL-PPO）。
guard 在 test 的 extort/harvest probe 上损失约 6.9，整体 -0.45，
属于噪声范围；test late 362.48 vs 362.49，基本不变。

## 5. 结论与下一步

- **Go for dev probe gate**：两个 seed 都把 probe 从约 115 提到约
  126，`dev_complex` min_norm 从 0.927 提到约 0.996。
- **No-Go for promotion**：test_v8 provisional 没有超过 `rl-v5-ft`，
  规则本身是为了 dev gate / 安全性，而不是 test 分数。
- 旧 `min_norm ≥ 0.99` gate 现在只剩 `late`（dev_param / dev_ood
  约 0.977–0.978）。这比 probe 的 0.934 缺口小，但 `late` 在 test
  中也存在（test late 约 362 vs RL-PPO 365），是下一步更值得做的
  真实弱点；不要再盲目扩大 probe 权重或加全局主动 D。
- C++ 移植待做；当前只注册为实验候选，主权重和 `rl-v5-ft` 权重
  均未替换。
- 后续 [defection-guard.md](defection-guard.md) 在同一个状态机上加入
  late 窗口（首次 D ≥ 9，响应 `D,C,C`，三回合确认），得到
  `rl-full-guard`：dev 五项 min_norm 全部 ≥ 1.0（两个 seed），
  test provisional 496.36 / 496.40。probe 仍用 `C,C,C`；
  late 用 `D,C,C`。
