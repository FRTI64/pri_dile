# Extort-hard 训练：高 chi ZD 定向 fine-tune（No-Go）

> **状态（2026-10-29）：基础设施完成，模型实验 No-Go。**
> 新增 `extort_hard` 训练分布，覆盖连续 chi 1.5–7.0；在 RL-Wide160
> 上 fine-tune 只能把 test extort/hyper_extort 提高约 +2/+3（gentle）
> 到 +6/+13（strong），同时 PatternBreaker 回退 21–58，min_norm 下降。
> **不注册候选。**

## 1. 动机

全量扫描显示 extortion / ZD 是最大薄弱点：

- test `hyper_extort` 176、`extort` 402、`conditional_extort` 324；
- Axelrod `ZDExtort2/3/4`、`ZDExtortion`、`ZDMischief` 的 diff
  -100～-364。

因此新增一个只用于训练的 `extort_hard` 分布，直接采样高 chi ZD。

## 2. 训练分布

`ipd/evaluation/extort.py`：

| 组件 | 权重 | 参数 |
|---|---:|---|
| ExtortZD strong | 6.0 | chi 4.5–7.0 |
| ExtortZD mid | 4.0 | chi 2.5–4.5 |
| ExtortZD mild | 2.0 | chi 1.5–2.5 |
| GenerousZD | 1.0 | kappa 2.0–5.0 |
| RandomMoves | 1.0 | p 0.05–0.30 |
| TFT / Grim / AllC | 1.5 / 1.0 / 1.0 | anchors |
| AllD / Bully | 0.5 / 0.5 | anchors |

注册为 `TRAIN_DISTS` 的 `extort_hard`，不进入 Protocol catalog；
`tests/test_robust.py` 覆盖确定性和强 extortion 覆盖。

## 3. 两次 fine-tune

起点均为 `weights/rl_policy_checkpoint_wide160.json`（160/128）。

### 3.1 strong：extort_hard 权重 2/9，6 iter

`original,late,grim30,wide,train_v4,extort_hard` 权重
`1,1,1,2,2,2`，lr 3e-5，KL 0.1。

| 集 | RL-Wide160 | strong iter4 |
|---|---:|---:|
| dev_param | 548.26 | **548.77** |
| dev_family | **583.08** | 583.02 |
| dev_complex | **454.31** | 450.25 |
| dev_axelrod | 582.38 | **583.33** |
| dev_ood | **521.37** | 520.39 |
| test_v7（provisional） | 485.62 | **486.44** |

test family 变化：

| family | 变化 |
|---|---:|
| hyper_extort | 176.2 → 189.3（+13.2） |
| extort | 435.2 → 440.7（+5.5） |
| delayed_extort | 399.8 → 405.0（+5.2） |
| probe_exploit | 262.8 → 267.2（+4.4） |
| pattern_breaker | 460.7 → **402.5（-58.2）** |

`min_norm`：complex 0.850、ood 0.847（RL-Wide160 0.997/0.996）。

### 3.2 gentle：extort_hard 权重 1/10 + train_complex，3 iter

`original,late,grim30,wide,train_v4,train_complex,extort_hard` 权重
`1,1,1,2,2,1,1`，lr 2e-5，KL 0.1。

| 集 | RL-Wide160 | gentle iter1 |
|---|---:|---:|
| dev_param | 548.26 | **548.42** |
| dev_family | **583.08** | 583.06 |
| dev_complex | **454.31** | 453.93 |
| dev_axelrod | 582.38 | **582.68** |
| dev_ood | 521.37 | **521.41** |
| test_v7（provisional） | 485.62 | 485.65 |

test family：hyper_extort +3.0、extort +1.7、pattern_breaker -21.4。
`min_norm` 基本保持（0.994–0.996），但 extort 收益也很小。

## 4. 解释

- PPO 对高 chi ZD 的响应不是“更激进地背叛”：
  对 ExtortZD 的有限期最优期望分数随 chi 增大反而下降
  （chi 6.5 约 224，chi 3.75 约 295）；盲目锁死 D 会更差；
- 高 chi 训练会把策略推向更保守/更易被 PatternBreaker 利用的区域，
  因此 PB 回退；
- `extort_hard` 的 320–350 训练 reward 说明模型确实在适应，但收益
  不足以抵消 complex/PB 的损失；
- 这不是“多训几轮就好”的问题：strong 6 iter 已经过拟合到 extort
  方向，gentle 3 iter 则几乎不动。

## 5. 判定与下一步

**No-Go for registration / promotion。** 保留 `extort_hard` 分布和
测试，供后续 expert/portfolio 实验使用；不注册 `rl-wide160-extort`。

更可能有效的下一步：

1. **显式 ZD 识别 + 专家响应**：用 `p_c` / `p_d` / 转移概率估计
   对手是否接近 ZD，再切换到一个小型 expert（可由 MDP 求最优动作
   蒸馏得到），而不是让全局 PPO 学；
2. **Portfolio**：Triple-majority 在部分 ZD/Gap/Kamikaze 上更强；
   若能低成本识别，可用 portfolio；
3. **不要继续加大 extort_hard 权重**；会继续伤害 PB/complex。

## 6. 复现

```bash
# strong
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 6 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261052 \
  --init-weights weights/rl_policy_checkpoint_wide160.json \
  --train-dists original,late,grim30,wide,train_v4,extort_hard \
  --dist-weights 1,1,1,2,2,2 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261037/wide160_extort_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261037/wide160_extort_weights_best.json \
  --checkpoint-out results/rl_v4_20261037/wide160_extort_ac_best.json \
  --history-out results/rl_v4_20261037/wide160_extort_history.json

# gentle
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 3 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261053 \
  --init-weights weights/rl_policy_checkpoint_wide160.json \
  --train-dists original,late,grim30,wide,train_v4,train_complex,extort_hard \
  --dist-weights 1,1,1,2,2,1,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261038/wide160_extort_gentle_ckpts \
  --bc-kl-coef 0.1 --lr 2e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261038/wide160_extort_gentle_weights_best.json \
  --checkpoint-out results/rl_v4_20261038/wide160_extort_gentle_ac_best.json \
  --history-out results/rl_v4_20261038/wide160_extort_gentle_history.json
```

数据：`results/rl_v4_20261037/`、`results/rl_v4_20261038/`。
