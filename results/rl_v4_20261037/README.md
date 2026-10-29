# Extort-hard strong fine-tune（2026-10-29）

> `extort_hard` 权重 2/9，6 iter。**未注册**：test extort/hyper
> +5/+13，但 PatternBreaker -58、complex min_norm 0.850。

## 结果

| 集 | RL-Wide160 | strong iter4 |
|---|---:|---:|
| dev_param | 548.26 | **548.77** |
| dev_family | **583.08** | 583.02 |
| dev_complex | **454.31** | 450.25 |
| dev_axelrod | 582.38 | **583.33** |
| dev_ood | **521.37** | 520.39 |
| test_v7 | 485.62 | **486.44** |

test family：hyper_extort 176.2→189.3、extort 435.2→440.7、
pattern_breaker 460.7→402.5。

## 文件

- `wide160_extort_history.json`；
- `wide160_extort_iter{2,4}_dev_*.json` / `wide160_extort_iter{2,4}_test_v7.json`；
- 大 checkpoint 未保留。

复现命令见 `docs/research/extort-hard.md`。
