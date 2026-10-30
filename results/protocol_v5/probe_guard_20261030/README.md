# RL-ProbeGuard（2026-10-30）

> **实验候选，未 promotion；`test_v8` 未消费。** 基于 `rl-v5-ft` 的
> 单权重反应式早期 probe 防护。dev 两个 seed 把 `probe_exploit`
> 从约 115 提到约 126，`dev_complex` min_norm 0.927 → 0.996；
> `dev_param` 与 `rl-v5-ft` 基本持平。test_v8 provisional
> 495.92 vs `rl-v5-ft` 496.37（-0.45，噪声内），比 RL-Wide160
> 495.71 高 0.21。

## 方法

- 基础策略：`rl-v5-ft`（`weights/rl_policy_weights_v5_ft.json`）。
- 触发：对手首次 `D` 出现在零基回合 5 或 7（即第 6 / 8 回合），
  下一回合继续 `D`，且我方在对手首次 `D` 之前没有主动 `D`。
- 响应：连续让 3 回合 `C`。
  - 对手在这 3 回合内出现 `C`：立即回到基础策略；
  - 对手持续 `D`：切到永久 `D`，直到对手重新 `C`。
- 不修改 safety 规则，不在对手 `D` 之前主动 `D`；不是预判 gate。

## 命令

```bash
# dev（seed 20261027 / 20261029）
python3 -m ipd protocol eval \
  --candidates rl-probe-guard,rl-v5-ft,rl \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v5/probe_guard_20261030/dev_v5ft.json

python3 -m ipd protocol eval \
  --candidates rl-probe-guard,rl-v5-ft,rl \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --seed 20261029 --baseline RL-PPO \
  --out results/protocol_v5/probe_guard_20261030/dev_v5ft_seed20261029.json

# test_v8 provisional（不消费）
python3 -m ipd protocol eval \
  --candidates rl-probe-guard,rl-v5-ft,rl \
  --tiers test_v8 --fields 50 --n 2048 --workers 20 --no-consume \
  --out results/protocol_v5/probe_guard_20261030/test_v8_provisional_v5ft.json
```

## Dev（seed 20261027）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-ProbeGuard** | 532.07 | 583.13 | **482.79** | 568.58 | 523.77 |
| RL-V5-FT | **532.18** | **583.14** | 482.64 | **568.74** | **523.94** |
| RL-PPO | 525.99 | 582.55 | 457.26 | 563.43 | 513.30 |

| 指标 | RL-ProbeGuard | RL-V5-FT |
|---|---:|---:|
| `probe_exploit`（dev_complex / dev_ood） | **125.89 / 126.20** | 114.82 / 115.18 |
| probe family norm（vs RL-PPO） | **1.016** | 0.927 |
| dev_complex min_norm | **0.9959** | 0.9268 |
| dev_ood min_norm | 0.9776 | 0.9279 |
| dev_param min_norm | 0.9776 | 0.9775 |

## Dev（独立 seed 20261029）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| **RL-ProbeGuard** | **531.66** | 582.82 | **485.02** | 568.00 | 523.93 |
| RL-V5-FT | 531.61 | 582.82 | 484.28 | **568.02** | **524.01** |
| RL-PPO | 525.69 | 582.31 | 458.93 | 562.62 | 513.92 |

- probe：**126.03 / 125.95**（norm 1.016）；
- dev_complex min_norm **0.9960**，dev_ood min_norm 0.9779。

## test_v8 provisional（50 fields，不消费）

| 候选 | test_v8 |
|---|---:|
| RL-V5-FT | **496.37** |
| RL-ProbeGuard | 495.92 |
| RL-Wide160 | 495.71 |
| RL-PPO | 487.83 |

- test probe：260.53（guard）vs 267.39（V5-FT）vs 233.98（RL-PPO）；
  guard 在 test 的 extort/harvest probe 上损失约 6.9；
- test late：362.48 vs 362.49（V5-FT），基本不变。

## 文件

| 文件 | 内容 |
|---|---|
| `dev_v5ft.json` | 最终 V5-FT base，seed 20261027 |
| `dev_v5ft_seed20261029.json` | 最终 V5-FT base，独立 seed 20261029 |
| `test_v8_provisional_v5ft.json` | 最终 V5-FT base，test_v8 provisional（未消费） |
| `dev.json` / `dev_v2..v4.json` | 早期 Wide160 / RL-PPO / 触发窗口迭代 |
| `dev_seed20261029.json` | 早期 Wide160 base，seed 20261029 |
| `test_v8_provisional.json` | 早期 Wide160 base，test_v8 provisional |

## 结论

- **Go for dev probe gate**：两个 seed 都把 `dev_complex` 的
  `probe_exploit` 从约 115 提到约 126，min_norm 0.927 → 0.996；
  `dev_param` 与 V5-FT 持平。
- **No-Go for promotion**：test_v8 provisional 比 V5-FT 低 0.45，
  没有真实 test 增益；剩余 gate 瓶颈是 `late`（dev_param / dev_ood
  min_norm 约 0.977–0.978）。
- 单权重、可在 Python 中直接部署；C++ 移植待做。`test_v8` 未消费。
