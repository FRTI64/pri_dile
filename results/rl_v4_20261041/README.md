# Periodic-hard fine-tune（2026-10-29）

> `periodic_hard` 权重 1/10 + `train_complex`，4 iter。**未注册**：
> test +0.77，但 dev_complex -2.90、PatternBreaker -40.7、
> min_norm 0.89–0.91。

## 结果

| 集 | RL-Wide160 | periodic iter4 |
|---|---:|---:|
| dev_param | 548.26 | **548.54** |
| dev_family | **583.08** | 583.01 |
| dev_complex | **454.31** | 451.41 |
| dev_axelrod | 582.38 | **582.99** |
| dev_ood | **521.37** | 520.43 |
| test_v7 | 485.62 | **486.39** |

test family：frequent_kamikaze +8、temporal +2、pattern_breaker -41。

## 文件

- `wide160_periodic_history.json`；
- `wide160_periodic_iter4_dev_*.json` / `_test_v7.json`；
- 大 checkpoint 未保留。

复现命令见 `docs/research/periodic-hard.md`。
