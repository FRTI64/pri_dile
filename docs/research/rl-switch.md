# RL-Switch：窄 meta 模型（RL-Wide160 → RL-Repair）

> **状态（2026-10-29）：当前最佳 v4 dev/test 候选，未 promotion。**
> 默认走 RL-Wide160；仅当第 6 回合前双方都至少 D 过 2 次、且对手
> 最新一手为 C 时，切换到 RL-Repair。dev 基本不变，test_v7
> provisional +2.6～2.9，Protocol catalog suite +0.75。
> **Python-only，尚未移植 C++。**

## 1. 动机

全量扫描显示 RL-Wide160 在 extortion / noisy / Joss / Axelrod
尾部偏弱，而 RL-Repair 在这些结构上更稳。直接把两个模型的概率
平均会破坏安全规则：在 Grim 上从 604 掉到 103；简单 prefix
切换也会损失 dev_param。

因此只加一个很窄的早期开关：

```text
默认：RL-Wide160
若 t >= 6 且
  对手前 6 手至少有 2 个 D
  自己前 6 手至少有 2 个 D
  对手最新一手是 C
则本场剩余时间：RL-Repair
```

它只在“已经互相 D 过、对手又开始合作”的结构触发；TFT / Grim /
AllC / 正常合作路径不会触发。触发后不回头，避免来回抖动。

## 2. 实现

- `ipd/policies/meta.py::RLSwitchPolicy`；
- `ipd/policies/candidates/rl_switch.py`（`rl-switch` / `RL-Switch`）；
- 两个子策略仍按 `ipd-rl-policy-v1` / `ipd-neural-policy-v1` 加载；
- 每场 `reset()` 清除 `switched`；
- 目前只在 Python 路径；GPU 评估会把它路由到 CPU；
- C++ 版本需要同时嵌入两份权重和开关逻辑，尚未实现。

## 3. Protocol v4 dev（seed 20261027，30×2048）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| RL-Wide160 | 548.26 | **583.08** | **454.31** | 582.38 | 521.37 |
| **RL-Switch** | **548.58** | 582.91 | 454.24 | **582.86** | **521.42** |

- dev_param +0.32、dev_axelrod +0.48、dev_ood +0.05；
- dev_family -0.17、dev_complex -0.07；
- `min_norm`（vs RL-Wide160）：param 0.9951、family 0.9976、
  complex 0.9944、axelrod 0.9972、ood 0.9936；没有新的最差 family。

## 4. 独立 seed 复现（20261029）

| 集 | RL-Wide160 | RL-Switch | Δ |
|---|---:|---:|---:|
| dev_param | 546.39 | **546.80** | +0.41 |
| dev_family | **582.77** | 582.62 | -0.15 |
| dev_complex | 454.02 | **454.27** | +0.25 |
| dev_axelrod | 581.67 | **582.13** | +0.46 |
| dev_ood | **522.99** | 522.84 | -0.15 |
| test_v7（provisional） | 484.61 | **487.51** | +2.90 |

seed 20261027 的 test_v7 为 485.62 → **488.26（+2.64）**；两次
独立 seed 的 test 增益一致。

## 5. 全量对手

### Protocol catalog（8 reps）

| 指标 | RL-Wide160 | RL-Switch |
|---|---:|---:|
| nine-suite avg | 520.43 | **521.18** |
| worst suite | 484.55 | **485.43** |

### 158 类 bank（16 reps）

| 指标 | RL-Wide160 | RL-Switch |
|---|---:|---:|
| nine-suite avg | **554.45** | 554.37 |
| worst suite | **530.23** | 529.56 |

158 bank 基本持平；protocol catalog 全 suite 小幅提升。

### 主要改善对手

| 对手 | RL-Wide160 | RL-Switch |
|---|---:|---:|
| Extort3 | 274.8 | **295.9** |
| Extort.F3.0 | 282.9 | **298.1** |
| NoisyTFT.20 | 463.2 | **476.8** |
| Random.70 | 663.8 | **676.5** |
| RemorsefulProber | 573.0 | **585.0** |
| EarlyProbe | 606.0 | **616.5** |

主要回退：FSM.F2/F3、Random.50/20/30、Prober、Tester；幅度
-5～-14，属于可接受权衡。

## 6. red

| tier | RL-Switch | RL-Wide160 |
|---|---:|---:|
| dev_red blue | 378.40 | 378.40 |
| test_red blue | 319.37 | 319.43 |
| red_win | 100% | 100% |

anti-exploitation 没有改善；RL-Switch 不改变 red 结论。

## 7. 判定与下一步

**注册 `rl-switch`，作为当前最佳 v4 dev/test 候选，但不 promotion。**

- dev 几乎不变，test_v7 和 protocol catalog 稳定提升；
- 158 bank 持平，没有明显 overall 回退；
- red 仍 100%，C++ 还没有；
- 下一步：
  1. 移植 C++ 双权重 + 开关逻辑；
  2. 跑 `dev_level0..4` / `dev_complexity0..4` 和 `test_red`；
  3. 独立 seed 再复现一次 test_v7；
  4. 如果用户确认，再讨论 promotion / 主权重替换。

## 8. 复现

```bash
python3 -m ipd self-test --candidates rl-switch

python3 -m ipd protocol eval --candidates RL-Wide160,RL-Switch \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 1 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-Wide160 \
  --out results/meta_20261039/protocol_v4_dev_rl_switch.json

python3 -m ipd protocol eval --candidates RL-Wide160,RL-Switch \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 --candidate-chunk 1 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 2 --seed 20261027 --no-consume \
  --out results/meta_20261039/protocol_v4_test_v7_rl_switch_provisional.json

python3 -m ipd matchup --candidates rl-switch,rl-wide160 \
  --opponent-set protocol --reps 8 --workers 20 --seed 20261036 \
  --out results/meta_20261039/matchup_protocol_switch.csv
```
