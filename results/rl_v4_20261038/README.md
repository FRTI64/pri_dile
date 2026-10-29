# Extort-hard gentle fine-tune（2026-10-29）

> `extort_hard` 权重 1/10 + `train_complex`，3 iter。**未注册**：
> 几乎不动，extort 只有 +1.7/+3.0。

## 结果

| 集 | RL-Wide160 | gentle iter1 |
|---|---:|---:|
| dev_param | 548.26 | **548.42** |
| dev_family | **583.08** | 583.06 |
| dev_complex | **454.31** | 453.93 |
| dev_axelrod | 582.38 | **582.68** |
| dev_ood | 521.37 | **521.41** |
| test_v7 | **485.62** | 485.65 |

test family：hyper_extort +3.0、extort +1.7、pattern_breaker -21.4。

## 文件

- `wide160_extort_gentle_history.json`；
- `wide160_extort_gentle_iter1_dev_*.json` / `_test_v7.json`；
- 大 checkpoint 未保留。

复现命令见 `docs/research/extort-hard.md`。
