# 策略 vs 85 类对手的完整对阵结果

生成方式：`run_matchup_experiment.py --reps 16`。

- 85 类对手全部参与，包括复杂类型：随机 memory-one、随机 reactive、随机 FSM、Responsive、响应式探测、Extort/GenZD 等。
- 每类对手 16 个随机实例。
- 每个候选策略与每个对手单独打满 100 天。
- 表中“我方”是候选策略得分，“对手”是对手得分。
- 类型等权平均，不按实际人群配比加权。
- 明细见 `matchup_results.csv`。

## 总体等权平均

| 策略 | 等权平均 |
|---|---:|
| **Final** | **549.93** |
| BayesPlus | 549.45 |
| GapStart | 549.40 |
| Responsive3 | 546.70 |
| Harvest40 | 543.78 |
| Harvest45 | 543.60 |
| EarlyProbe | 537.49 |
| GTFT.2-lastD | 520.78 |
| TFT | 501.26 |
| AllC | 493.27 |
| AllD | 379.26 |

在“每类对手等权”的设定下，Final 仍然第一，但 BayesPlus 和 GapStart 只差不到 1 分。

## 按对手族平均

| 策略 | adaptive | defect | forgiving | late | nice | punitive | random | wild |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Final | 522.6 | 572.2 | 570.9 | 460.4 | 598.6 | 602.2 | 490.1 | 554.2 |
| Harvest45 | 501.9 | 572.2 | 570.4 | 601.5 | 604.5 | 426.7 | 490.0 | 572.5 |
| Harvest40 | 499.5 | 572.2 | 570.4 | 621.9 | 605.5 | 411.0 | 490.0 | 574.2 |
| EarlyProbe | 504.9 | 594.1 | 554.0 | 771.9 | 556.9 | 302.0 | 498.1 | 575.0 |
| GapStart | 413.0 | 648.7 | 591.0 | 780.6 | 603.5 | 334.5 | 490.9 | 561.1 |
| BayesPlus | 469.6 | 565.9 | 562.9 | 477.1 | 598.7 | 635.3 | 512.8 | 554.5 |
| Responsive3 | 527.8 | 555.1 | 571.1 | 459.8 | 598.9 | 601.6 | 474.8 | 550.6 |
| GTFT.2-lastD | 511.2 | 442.6 | 557.4 | 473.1 | 598.5 | 602.4 | 402.4 | 499.5 |
| TFT | 471.2 | 437.1 | 498.0 | 477.1 | 590.3 | 598.9 | 410.2 | 471.4 |
| AllC | 494.0 | 311.1 | 573.3 | 453.8 | 595.9 | 596.7 | 309.9 | 477.6 |
| AllD | 167.7 | 443.3 | 319.3 | 780.6 | 303.6 | 272.8 | 497.7 | 504.8 |

## 各对手族的最佳候选

| 对手族 | 最佳策略 | 分数 |
|---|---|---:|
| adaptive | Responsive3 | 527.8 |
| defect | GapStart | 648.7 |
| forgiving | GapStart | 591.0 |
| late | GapStart | 780.6 |
| nice | Harvest40 | 605.5 |
| punitive | BayesPlus | 635.3 |
| random | BayesPlus | 512.8 |
| wild | EarlyProbe | 575.0 |

没有单一策略在所有族上都最好。

## Final 的主要强项和弱项

强项：

- AllC/TFT/Grim/Pavlov/GTFT：约 604
- EarlyD10/25：约 900/770
- Random 0.7/0.8：约 672/732
- Periodic3：约 669
- FSM-rand：约 589
- Reactive-rand：约 531

弱项：

- LateD40/50：约 367/316，对手约 694/652
- Extort2/3：约 313/212
- Random 0.05/0.10/0.20：约 134/185/273
- AllD：约 95（但对手也只有约 149）
- Joss(20%)：约 471，对手约 613

## 关键 1v1 对照

| 对手 | 指标 | Final | Harvest40 | EarlyProbe | GapStart | BayesPlus |
|---|---|---:|---:|---:|---:|---:|
| AllC | 我方 / 对手 | 604 / 594 | 844 / 234 | 996 / 6 | 1000 / 0 | 604 / 594 |
| TFT | 我方 / 对手 | 604 / 594 | 597 / 587 | 602 / 592 | 597 / 587 | 604 / 594 |
| Grim | 我方 / 对手 | 604 / 594 | 296 / 366 | 107 / 117 | 107 / 117 | 604 / 594 |
| LateD10 | 我方 / 对手 | 545 / 598 | 752 / 262 | 903 / 46 | 910 / 10 | 549 / 559 |
| LateD25 | 我方 / 对手 | 456 / 650 | 617 / 277 | 765 / 83 | 775 / 25 | 474 / 484 |
| LateD50 | 我方 / 对手 | 316 / 652 | 386 / 356 | 541 / 99 | 550 / 50 | 349 / 359 |
| Joss(20%) | 我方 / 对手 | 471 / 613 | 471 / 613 | 419 / 545 | 472 / 620 | 400 / 521 |
| Random(50%) | 我方 / 对手 | 522 / 114 | 522 / 114 | 527 / 105 | 449 / 343 | 533 / 79 |
| Extort2 | 我方 / 对手 | 313 / 503 | 313 / 503 | 239 / 370 | 343 / 554 | 302 / 482 |
| Extort3 | 我方 / 对手 | 212 / 369 | 212 / 369 | 156 / 228 | 214 / 392 | 184 / 312 |
| MemOne-rand | 我方 / 对手 | 542 / 534 | 542 / 534 | 494 / 424 | 455 / 392 | 519 / 472 |
| Reactive-rand | 我方 / 对手 | 531 / 388 | 531 / 388 | 524 / 294 | 551 / 387 | 552 / 318 |
| FSM-rand | 我方 / 对手 | 589 / 363 | 649 / 273 | 707 / 228 | 677 / 264 | 592 / 316 |
| RemorsefulProber | 我方 / 对手 | 586 / 590 | 586 / 590 | 599 / 589 | 587 / 577 | 108 / 108 |

## 解释

- **Final 是综合最均衡的**：对 nice/punitive 接近满分，对复杂 random/adaptive 也没有灾难性崩溃。
- **Harvest40/45 是“late 特化”**：对 late 从 460 提升到 622，但 punitive 从 602 掉到 411。
- **EarlyProbe/GapStart 是“AllC/late 特化”**：对 AllC 接近 1000、对 late 约 780，但对 Grim 只有 107。
- **BayesPlus 是“punitive/random 特化”**：对 punitive 635、对 random 513，但对 late 只有 477，而且会被 RemorsefulProber 卡到 108。
- **Responsive3 是“adaptive 特化”**：对 adaptive 527.8 最高，但对 late 只有 459.8。
- **TFT/GTFT** 在简单互惠型上表现不错，但对复杂/非响应型明显吃亏。

## 结论

如果按“85 类对手等权”比较，Final 仍然第一，但优势非常小。没有一种策略在所有类型上占优：

- 追求综合稳定：Final
- 追求末期收割收益：GapStart / EarlyProbe
- 想兼顾一点 late 但减少 Grim 风险：Harvest40/45
- 对手以惩罚型/随机型为主：BayesPlus 或 Final
- 对手以自适应复杂策略为主：Responsive3 或 Final
- 对手包含大量 Extort/GenZD：所有候选都会明显掉分，需专门设计应对

## 排除明显自杀型对手后的比较

这里不按“被我们克制”来排除，而是按策略本身明显不具竞争力来排除：

- AllC、AllC-lastD；
- AllD；
- EarlyD1/10/25；
- Random 0.05/0.10/0.20/0.30。

保留：

- TFT、GTFT、Joss、NoisyTFT、Pavlov；
- Grim、Mistrust、Grudger；
- LateD 全部档位；
- Tester/Prober；
- Random 0.5/0.7/0.8/0.9、Alternator、Periodic；
- MemOne-rand、Reactive-rand、FSM-rand；
- Responsive、ProbeBot、Extort/GenZD。

剩余 75 类对手的等权平均：

| 策略 | 平均分 |
|---|---:|
| **Final** | **562.80** |
| Responsive3 | 560.57 |
| BayesPlus | 556.49 |
| Harvest45 | 549.77 |
| Harvest40 | 549.44 |
| GapStart | 546.67 |
| GTFT.2-lastD | 540.73 |
| EarlyProbe | 538.01 |
| TFT | 517.30 |
| AllC | 516.61 |
| AllD | 352.86 |

按族平均（保留复杂类型）：

| 策略 | adaptive | defect | forgiving | late | nice | punitive | random | wild |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Final | 522.6 | 546.7 | 570.9 | 460.4 | 598.5 | 602.2 | 600.2 | 554.2 |
| Responsive3 | 527.8 | 537.1 | 571.1 | 459.8 | 598.8 | 601.6 | 579.9 | 550.6 |
| BayesPlus | 469.6 | 396.3 | 562.9 | 477.1 | 598.6 | 635.3 | 630.1 | 554.5 |
| Harvest45 | 501.9 | 546.7 | 570.4 | 601.5 | 573.8 | 426.7 | 600.2 | 572.5 |
| Harvest40 | 499.5 | 546.7 | 570.4 | 621.9 | 572.0 | 411.0 | 600.2 | 574.2 |
| GapStart | 413.0 | 590.3 | 591.0 | 780.6 | 547.5 | 334.5 | 604.9 | 561.1 |
| EarlyProbe | 504.9 | 594.7 | 554.0 | 771.9 | 494.8 | 302.0 | 610.8 | 575.0 |

结论：

- 排除明显自杀型后，Final 的领先从约 0.5 分扩大到约 2.2 分；
- Responsive3 升到第二，BayesPlus 第三；
- GapStart/EarlyProbe 从“总体第一梯队”掉到中游；
- 它们的高均分主要来自 AllC、EarlyD 和低合作率 Random；
- 在保留复杂对手的情况下，Final 仍是最稳的综合选择。

## 排除“互害/永久报复型”对手

这里的排除不是按“送分”判断，而是按策略机制：

### A. 永久惩罚型

排除：

- Grim、GrimForgive.05/.15；
- Grudger2/3；
- Mistrust；
- OmegaTFT-lite。

保留 TFT、GTFT、Joss、LateD、Random、复杂策略等。

剩余 78 类对手：

| 策略 | 平均分 |
|---|---:|
| GapStart | **579.98** |
| EarlyProbe | 566.86 |
| Harvest40 | 560.42 |
| Harvest45 | 558.43 |
| Final | 545.07 |
| BayesPlus | 544.55 |
| Responsive3 | 541.56 |
| GTFT.2-lastD | 513.31 |
| TFT | 492.39 |

### B. 永久惩罚 + 简单 TFT/无限报复家族

再排除：

- TFT、TFT-lastD、TFT-last2D、TFT-last5D；
- TFT2、TFT2-lastD、TFT-delay2、sTFT；
- ContriteTFT、TFTForgiveOnce、TTFT、FirmButFair；
- TwoStrike2/3、ForgiveTFT1/2。

剩余 62 类：

| 策略 | 平均分 |
|---|---:|
| EarlyProbe | **573.04** |
| GapStart | 571.30 |
| Harvest40 | 551.67 |
| Harvest45 | 549.17 |
| Final | 531.69 |
| BayesPlus | 531.02 |
| Responsive3 | 527.20 |

### C. 再排除 Joss/NoisyTFT 等噪声报复型

剩余 54 类：

| 策略 | 平均分 |
|---|---:|
| EarlyProbe | **583.96** |
| GapStart | 576.27 |
| Harvest40 | 553.50 |
| Harvest45 | 550.62 |
| BayesPlus | 532.40 |
| Final | 530.53 |
| Responsive3 | 525.30 |

结论：

- 一旦去掉永久惩罚型，GapStart/EarlyProbe 立刻反超 Final；
- 再去掉 TFT/噪声报复型，EarlyProbe 优势更大；
- Final 的防御价值高度依赖 Grim、Mistrust、TFT 噪声等“报复陷阱”确实存在；
- 如果实际赛场主要是可合作、可预测、不会永久报复的策略，Final 的谨慎反而是一种得分损失。

## 把我方聪明策略加入对手池

`run_smart_opponents.py` 在原有 85 类对手之外加入：

| 聪明对手 | 权重 |
|---|---:|
| Final | 4 |
| Harvest45 | 3 |
| Harvest40 | 2 |
| EarlyProbe | 2 |
| GapStart | 3 |
| BayesPlus | 2 |
| Responsive3 | 2 |

`smart-weight` 是对这组权重的整体倍率。因为基础池权重合计约 100，所以：

- `smart-weight=1`：聪明对手约 **5%–10%**；
- `smart-weight=2`：约 **9%–19%**；
- `smart-weight=3`：约 **13%–26%**；
- `smart-weight=20`：约 **51%–70%**。

### 40 个赛场的平均分

| smart 权重 | 聪明对手占比 | Final | Harvest45 | Responsive3 | BayesPlus | EarlyProbe | GapStart |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 约 7% | **557.0** | 548.9 | 554.5 | 548.8 | 534.7 | 531.0 |
| 2 | 约 13% | **558.4** | 552.5 | 555.3 | 542.8 | 533.2 | 522.5 |
| 3 | 约 18% | **559.3** | 550.6 | 554.6 | 540.0 | 528.4 | 512.2 |
| 20 | 约 59% | **573.6** | 560.7 | 558.1 | 503.7 | 496.1 | 418.0 |

趋势非常明显：

- 加入聪明对手后，Final 反而更稳；
- EarlyProbe / GapStart 被其他聪明策略惩罚，分数持续下降；
- Harvest45 仍然比 EarlyProbe/GapStart 稳；
- BayesPlus 在聪明对手多的池子里下降明显；
- Responsive3 比较稳，但始终低于 Final。

### 5 个完整循环赛（smart-weight=2，约 13% 聪明对手）

| 策略 | 平均排名 | 前 10 | 最差 | 综合分 |
|---|---:|---:|---:|---:|
| **Final** | **6.20** | 5/5 | 9 | 558.6 |
| Responsive3 | 15.00 | 0/5 | 19 | 553.8 |
| Harvest45 | 23.80 | 0/5 | 52 | 549.6 |
| BayesPlus | 59.00 | 0/5 | 119 | 539.2 |
| EarlyProbe | 104.40 | 0/5 | 152 | 525.6 |
| GapStart | 142.80 | 0/5 | 180 | 509.5 |

结论：

- 当对手池里包含会反击的聪明策略时，Final 的“避免互害”价值进一步放大；
- EarlyProbe/GapStart 依赖大量不回应、不报复的对手；
- 如果实际参赛者中包含不少类似 Final/Harvest/Responsive 的机器，应该优先选择 Final，而不是早期探测型。
