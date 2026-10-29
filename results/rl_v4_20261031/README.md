# 192/128 的 dev_param / test 变体（2026-10-31）

> 192/128 的补充消融：gentle（无额外 PB）和 2 iter balance。
> **均未注册**：dev_param / test_v7 可以更高，但 complex / OOD 的
> `min_norm` 掉到 0.71–0.75，明显不如 160/128 的 0.997。

## 结果

| 变体 | dev_param | dev_complex | dev_ood | test_v7 | complex min_norm | ood min_norm |
|---|---:|---:|---:|---:|---:|---:|
| 192/128 gentle iter2 | **548.86** | 446.00 | 518.95 | **486.76** | 0.7284 | 0.7125 |
| 192/128 balance2 iter1 | 548.82 | 446.66 | 519.49 | 486.51 | 0.7450 | 0.7537 |
| 160/128 RL-Wide160 | 548.26 | **454.31** | **521.37** | 485.62 | **0.9974** | **0.9966** |

192/128 的 dev_param 和 test 可以略高，但 worst-family 退化明显；
160/128 是更均衡的选择。

## 文件

- `wide192_gentle_history.json` / `wide192_gentle_iter2_*.json`；
- `wide192_balance2_history.json` / `wide192_balance2_iter1_*.json`；
- 大 checkpoint 未保留；训练命令与 `results/rl_v4_20261030/README.md`
  的 192/128 配方相同，只是初始化 / 迭代数不同。
