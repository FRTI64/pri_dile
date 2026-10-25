# 新测试集与更多对手类型

## 新增对手类型

`opponent_plus.py` 新增 67 个类型：

- 8 个固定随机 memory-one；
- 8 个固定随机 reactive；
- 6 个固定随机有限状态机 FSM；
- 5 个 Extort 参数档；
- 3 个 Generous ZD；
- 7 个细 GTFT；
- 7 个细 Joss；
- 7 个细 NoisyTFT；
- 7 个细 Random；
- 以及 FinalTuned、RespTuned、Harvest21/35/50/75、Gap.45/.55/.64 等聪明变体。

本测试集的对手库共约 **158 类**。

## 测试集

每个测试集 12 个随机 300 人赛场，每场 299 个对手：

1. `original`：原始混合；
2. `late`：末期收割；
3. `grim`：高 Grim；
4. `noise`：Joss/NoisyTFT/GTFT 噪声型；
5. `complex`：MemOne/Reactive/FSM/Extort/GenZD；
6. `smart`：我方聪明策略及变体；
7. `punitive`：Grim/Grudger/Mistrust；
8. `random`：Random/周期型；
9. `wide`：综合宽分布。

`run_suite_benchmark.py` 负责生成并评估。

## 结果

| 策略 | original | late | grim | noise | complex | smart | punitive | random | wide | 平均 | 最差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final-base | 551.4 | 532.5 | 562.3 | 545.7 | 523.0 | 549.3 | 560.9 | 521.0 | 540.4 | 542.94 | 520.95 |
| **Final-tuned** | **555.0** | 533.5 | 564.2 | 546.6 | 525.8 | 545.0 | 562.6 | 523.4 | **541.6** | 544.17 | 523.40 |
| Resp3-base | 550.3 | 532.0 | 559.7 | 544.6 | 521.8 | 539.7 | 558.0 | 515.6 | 536.6 | 539.81 | 515.55 |
| Resp3-tuned | 551.2 | 530.7 | 560.6 | 544.5 | 520.8 | 544.7 | 558.3 | 517.6 | 535.4 | 540.43 | 517.60 |
| Bayes-tuned | 553.4 | 534.6 | **568.1** | 541.8 | 513.0 | 503.1 | **565.6** | 521.5 | 528.7 | 536.64 | 503.13 |
| Final+Resp3 | 552.1 | 531.9 | 562.1 | 546.9 | 522.9 | 547.2 | 559.9 | 521.9 | 539.7 | 542.73 | 521.90 |
| **Triple-majority** | 552.1 | 532.8 | 561.8 | **549.1** | **525.8** | **551.9** | 560.6 | **524.3** | 540.8 | **544.37** | **524.34** |
| Harvest45 | 547.1 | **563.7** | 521.7 | 544.3 | 518.0 | 551.2 | 511.2 | 516.0 | 534.1 | 534.14 | 511.24 |

## 结论

- **Triple-majority**：9 个测试集平均最高，最差测试集也最高；
- **Final-tuned**：非常接近 Triple，且在 original/wide 最好；
- **Bayes-tuned**：grim/punitive 最强，但 complex/smart/wide 明显弱；
- **Harvest45**：late 最强，但 grim/punitive 很差；
- **Resp3**：在新增的 complex/smart 测试集中没有超过 Final/Triple；
- 新增的复杂对手并没有推翻“Final 稳健、Triple 综合最好”的结论，但让 Bayes 的弱点更加明显。
