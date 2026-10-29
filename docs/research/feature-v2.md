# 输入特征 v2：85 → 97

> **状态（2026-10-29）：基础设施完成，模型实验 No-Go。**
> v2 的 12 个周期/响应特征可以通过 Python、GPU、C++ 三条路径逐位
> 对齐；但在 RL-Wide160 上 fine-tune 只换来 +2～4 的 dev_complex，
> dev_param / test_v7 小幅回退，因此**不注册新候选**。

## 1. 动机

RL-Wide160 的 85 维特征已经覆盖计数、条件响应、最近动作和
defection timing，但没有显式的短周期信号。`pattern_breaker` 仍然
是 dev/test 里最敏感的 family，因此尝试加入 12 个周期/响应特征：

| # | 名称 | 含义 |
|---:|---|---|
| 1–4 | `opp_period2..5` | 对手最近 8 步中 2/3/4/5 周期匹配率 |
| 5–8 | `self_period2..5` | 自己最近 8 步的周期匹配率 |
| 9 | `opp_mirror` | 最近 10 步对手是否复制我上一步 |
| 10 | `self_mirror` | 最近 10 步我是否复制对手上一步 |
| 11 | `opp_exploit` | 最近 10 步“我 C、对手 D”的比例 |
| 12 | `opp_forgive` | 最近 10 步“我 D、对手 C”的比例 |

全部在 `[0, 1]`，v2 总共 **97 维**。v1 的 85 维保持不变，
旧权重继续按 `feature_version` 使用 v1。

## 2. 三端同步

| 端 | 实现 | 验证 |
|---|---|---|
| Python | `ipd/policies/neural.py::extract_features_v2` | `tests/test_features_v2.py` |
| GPU | `ipd/evaluation/gpu/features.py::features97` / `FeatureState97` | 与 Python 逐位 parity（1e-12） |
| C++ | `cpp/ipd_features.hpp::extract_features_v2` | g++ 编译后与 Python 逐位 parity（1e-12） |

- `NeuralPolicy` / `RLPolicy` 根据 `feature_version` 选择 v1/v2 提取器；
- `BatchMLP` / `_PlainActor` / policy opponents 根据 `n_features == 97`
  选择 `FeatureState97`；
- `export_mlp` 在 97 维权重头里写 `#define IPD_FEATURES_V2 1`，
  `cpp/policies/neural_policy.hpp` 据此调用 v2 特征；
- 修复了 C++ 导出器对 `0.0` 写成 `0f` 的非法字面量问题；
- 全量测试 **320 tests OK**。

## 3. 初始化

从 RL-Wide160 的 85→160/128 checkpoint 扩展输入：

```bash
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_wide160.json \
  --features v2 \
  --out results/rl_v4_20261035/wide160_v2_ac.json \
  --weights-out results/rl_v4_20261035/wide160_v2_weights.json \
  --seed 20261035
```

新输入列用 1% 相对噪声初始化；新特征的 mean/std 初始为 0/1。
扩展后的 logit 与原模型的差约 8e-4，因此 PPO 从接近
RL-Wide160 的策略开始。

## 4. 训练

```bash
python3 -m ipd train rl --device cuda --features v2 --hidden 160 128 \
  --iterations 8 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261051 \
  --init-weights results/rl_v4_20261035/wide160_v2_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261035/wide160_v2_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261035/wide160_v2_weights_best.json \
  --checkpoint-out results/rl_v4_20261035/wide160_v2_ac_best.json \
  --history-out results/rl_v4_20261035/wide160_v2_history.json
```

## 5. 结果

| 集 | RL-Wide160 85 维 | v2 iter2 | v2 iter4 |
|---|---:|---:|---:|
| dev_param | **548.26** | 547.90 | 547.59 |
| dev_family | **583.08** | 583.05 | 583.03 |
| dev_complex | 454.31 | 456.43 | **458.72** |
| dev_axelrod | **582.38** | 582.09 | 581.52 |
| dev_ood | 521.37 | 521.62 | **522.04** |
| test_v7（provisional） | **485.62** | 485.27 | 485.25 |

`min_norm`：v2 iter2 / iter4 的 dev_complex 为 0.9983 / 0.9987，
dev_ood 为 0.9942 / 0.9922；没有新的最差 family。

- v2 对 `pattern_breaker` 的 test 值有明显帮助（约 461 → 519/577）；
- 但对 `hyper_extort` / `axelrod_extra` 没有恢复，test 均值反而
  -0.35；
- dev_param 的 0.36–0.67 回退大于 complex 的收益，综合不如
  RL-Wide160。

## 6. 判定

**No-Go for promotion / registration。**

- 保留 v2 特征契约和 `--features v2` 训练入口，供后续实验；
- 不注册 `rl-v2` 候选，不替换 `rl-wide160`；
- v2 的主要问题是新增特征偏向 pattern 周期，对 extort / hyper_extort
  没有帮助；下一步如果继续特征方向，应优先设计 **extortion /
  payoff-differential / opponent-type** 特征，而不是继续增加周期族。

## 7. 复现

```bash
python3 -m unittest tests.test_features_v2 tests.test_expand -v
python3 -m unittest tests.test_gpu_features -v
```

报告与逐 tier 结果：`results/rl_v4_20261035/`。
