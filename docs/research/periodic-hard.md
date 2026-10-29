# Periodic-hard 训练：Kamikaze / Periodic / DynamicWindow（No-Go）

> **状态（2026-10-29）：基础设施完成，模型实验 No-Go。**
> 新增 `periodic_hard` 训练分布；RL-Wide160 上 4 iter fine-tune
> test_v7 486.39（+0.77），但 dev_complex -2.90、PatternBreaker
> -40.7、min_norm 0.89–0.91。**不注册候选。**

## 1. 动机

全量扫描中 `frequent_kamikaze` 169、`temporal` 479；
`Kamikaze period=4..9` 只有 129–208，而 Triple 在部分 period 上
能到 571。因此新增周期/Kamikaze 训练分布。

## 2. 训练分布

`ipd/evaluation/periodic.py`：

| 组件 | 权重 | 参数 |
|---|---:|---|
| Kamikaze | 6.0 | period 4–12，p_steal 0.75/1.0 |
| Periodic | 4.0 | 7 种短周期 pattern，start 0–3 |
| DynamicWindow | 2.0 | window 5–15，p_defect .2–.5，retaliate .7–1.0 |
| RandomFSM | 1.0 | k 3–5 |
| TFT / Grim / AllC / AllD / Bully | anchors | — |

注册为 `TRAIN_DISTS` 的 `periodic_hard`；不进入 Protocol catalog。
`tests/test_robust.py` 覆盖确定性和关键类型。

## 3. 训练

```bash
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 4 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261054 \
  --init-weights weights/rl_policy_checkpoint_wide160.json \
  --train-dists original,late,grim30,wide,train_v4,train_complex,periodic_hard \
  --dist-weights 1,1,1,2,2,1,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261041/wide160_periodic_ckpts \
  --bc-kl-coef 0.1 --lr 2e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261041/wide160_periodic_weights_best.json \
  --checkpoint-out results/rl_v4_20261041/wide160_periodic_ac_best.json \
  --history-out results/rl_v4_20261041/wide160_periodic_history.json
```

## 4. 结果

| 集 | RL-Wide160 | periodic iter4 |
|---|---:|---:|
| dev_param | 548.26 | **548.54** |
| dev_family | **583.08** | 583.01 |
| dev_complex | **454.31** | 451.41 |
| dev_axelrod | 582.38 | **582.99** |
| dev_ood | **521.37** | 520.43 |
| test_v7（provisional） | 485.62 | **486.39** |

test family：

- `frequent_kamikaze` 约 172 → 180（+8）；
- `temporal` 约 504 → 506（+2）；
- `pattern_breaker` 461 → 420（-41）；
- `hyper_extort` 176 → 185（+9，但不是目标 family）。

`min_norm`：complex 0.909、ood 0.893、param 0.995。

## 5. 解释与判定

- 周期训练确实提高了 Kamikaze/temporal 的少量分数，但不足以抵消
  PatternBreaker 和 complex 的回退；
- 与 extort-hard 类似，PPO 在全局策略上做定向 fine-tune 会带来
  family 间 trade-off；
- **No-Go for registration / promotion**；保留 `periodic_hard`
  供后续 expert / portfolio 实验。

如果继续 periodic 方向，更合理的是：

1. 训练一个 **periodic specialist**，再用 meta gate 选择；
2. 或在 RL-Switch 的 gate 中增加“周期检测”触发条件；
3. 不要继续加大 periodic_hard 权重或迭代数。

## 6. 复现

见上方训练命令；结果在 `results/rl_v4_20261041/`。
