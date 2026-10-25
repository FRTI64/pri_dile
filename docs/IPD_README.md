# 100 天囚徒困境策略

> 项目总览与快速开始见根目录 [README.md](../README.md)。
> 规范 Python 源码位于 `ipd/`，C++ 实现位于 `cpp/`。分层与接口见
> [STRUCTURE.md](STRUCTURE.md)。

## 文件

- `ipd/paths.py`：统一解析 `configs/`、`weights/`、`results/`、`cpp/` 路径。
- `ipd/core/machine.py`：稳健版最终策略，类 `Machine`；第 1 天合作，避免触发 Grim。
- `cpp/ipd_strategy.cpp`：稳健版的 C++ 实现。
- `ipd/core/early.py`：早期探测/剥削版 `EarlyProbeMachine`；第 1 天背叛。
- `ipd/core/harvest.py`：末日收割反制版 `HarvestMachine`；默认第 45 天开始探测。
- `ipd/core/ema.py`：EMA + 末段动态阈值版本。
- `ipd/core/ema_forgive.py`：EMA 动态原谅概率 + 永久交替检测实验版。
- `ipd/core/bayes.py`：无额外探测的贝叶斯类型推断版。
- `ipd/core/simulator.py`：计分、对局和常用基线策略。
- `ipd/core/opponents/basic.py`：基础对手。
- `ipd/core/opponents/reciprocal.py`：38 类基础对手中的响应/原谅/自适应变体。
- `ipd/core/opponents/bank.py`：85 类对手、9 种群先验、Dirichlet 随机配比采样器。
- `ipd/training/search/common.py`：P0 多进程批量评估器。
- `ipd/training/search/p1_parameter.py`：P1 参数搜索与留出验证。
- `ipd/training/search/p2_gp.py`：P2 遗传规划决策树搜索（均值适应度）。
- `ipd/training/search/p2_robust.py`：P2b 鲁棒多目标 GP。
- `ipd/training/search/p2_fsm.py`：P2c 有状态 FSM GP，可用 `--max-states` 解除状态上限。
- `ipd/training/search/p2_skeleton.py`：P2e 参数化 GapStart 骨架搜索。
- `ipd/training/search/p1_grim.py`：高 Grim 训练集上的参数搜索。
- `ipd/training/search/wide.py`：宽模型库、细参数、分布训练。
- `ipd/training/search/multiobj.py`：Final/Harvest/Gap/Responsive3/BayesPlus 多目标训练。
- `ipd/training/search/joint.py`：Final + Responsive3 联合训练。
- `ipd/training/search/bayes_priors.py`：BayesPlus 模型先验参数化训练。
- `ipd/policies/candidates/triple_majority.py`：Final + Responsive3 + BayesPlus 多数投票融合。
- `ipd/training/neural.py`：GPU/NPU 训练的 MLP 神经策略（PyTorch + DAgger）。
- `ipd/policies/neural.py`：MLP 策略的纯 Python 推理实现（无需 PyTorch）。
- `cpp/neural_policy.cpp`：MLP 策略的 C++17 推理实现（含 demo）。
- `ipd/training/gru.py`：GPU/NPU 训练的 GRU-Net 策略。
- `ipd/policies/gru.py`：GRU-Net 的纯 Python 推理实现。
- `cpp/gru_policy.cpp`：GRU-Net 的 C++17 推理实现（含 demo）。
- `ipd/tools/export_cpp.py`：权重 JSON → C++ 头文件。
- `cpp/neural_policy_weights.hpp` / `cpp/gru_policy_weights.hpp`：生成的 C++ 权重。
- `ipd/policies/adversarial.py`：在 MLP/GRU/RL 之上的约束剥削 wrapper。
- `ipd/training/adversarial.py`：四分布不掉分约束下的对抗调参。
- `configs/adversarial_policy_config*.json`：MLP 保守/激进剥削配置。
- `configs/adversarial_gru_policy_config*.json`：GRU 保守/激进剥削配置。
- `configs/adversarial_rl_policy_config*.json`：RL 保守/激进剥削配置。
- `ipd/training/rl.py`：完整 PPO 强化学习训练（奖励驱动，24 worker rollout）。
- `ipd/policies/rl.py`：RL actor-critic 的纯 Python 推理与部署导出。
- `cpp/rl_policy_weights.hpp`：RL 部署权重的 C++ 头文件。
- `ipd/evaluation/benchmark.py`：规则/神经/RL/对抗策略的四分布统一对比。
- `ipd/evaluation/blind.py`：规则/神经/RL/对抗策略的 300 人盲测排名。
- `docs/NEURAL_POLICY.md`：MLP 策略的方法、训练、接口、结果与风险。
- `docs/GRU_POLICY.md`：GRU-Net 的方法、训练、接口、结果与风险。
- `docs/RL_POLICY.md`：完整 PPO 强化学习的方法、训练、结果与风险。
- `docs/ADVERSARIAL_POLICY.md`：对抗性剥削规则、调参、结果与风险。
- `HANDOVER.md`：交接文档（推荐方案、参数、文件、风险、检查清单）。
- `docs/FINAL_COMPARISON.md`：各测试集完整对照表。
- `ipd/core/opponents/plus.py`：额外 67 类复杂/细粒度对手。
- `ipd/evaluation/suite.py`：9 个测试集的统一对比。
- `docs/SUITE_RESULTS.md`：新测试集结果。
- `cpp/policies/bayes_plus_policy.hpp` / `cpp/policies/responsive3_policy.hpp`：BayesPlus / Responsive3 的无依赖 C++17 实现。
- `docs/BAYES_RESPONSIVE_FLOW.md`：两者的 Mermaid 流程图和参数说明。
- `docs/ADVANCED_RESULTS.md`：P0/P1/P2 的完整结果。
- `tests/`：目录、核心、策略、评估、训练分层测试。
- 早期 `run_*` 研究实验脚本已移除；历史结果仍保存在 `results/` 和 `docs/`。

## 运行

```bash
python3 -m ipd self-test --candidates core
g++ -std=c++17 -O2 -DIPD_STRATEGY_DEMO cpp/ipd_strategy.cpp -o ipd_demo
./ipd_demo

# 神经策略：训练需要 PyTorch，推理不需要
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-train.txt
python3 -m ipd train neural --device auto
python3 -m ipd train gru --device auto
python3 -m ipd train adversarial --arch mlp --seed 20261026
python3 -m ipd benchmark --fields 20 --workers 20
python3 -m ipd benchmark --arch gru --fields 20 --workers 20
python3 -m ipd benchmark --arch adversarial --config configs/adversarial_policy_config_aggressive.json --fields 20 --workers 20
python3 -m ipd blind --fields 10 --workers 12
python3 -m ipd blind --arch gru --fields 10 --workers 12
python3 -m ipd blind --arch adversarial --config configs/adversarial_policy_config_aggressive.json --fields 10 --workers 12
python3 -m ipd export --arch mlp
python3 -m ipd export --arch gru
g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo
g++ -std=c++17 -O2 -DGRU_POLICY_DEMO cpp/gru_policy.cpp -o gru_demo
./neural_demo
./gru_demo
```

动作约定：`0=C`，`1=D`；`day` 从 1 到 100；每场开始前调用 `reset()`。

## 三个版本

| 版本 | 第 1 天 | 适用场景 | 主要风险 |
|---|---|---|---|
| `Machine` | 合作 | 不知道对手构成，Grim/永久惩罚型较多 | 对大量 AllC / 末期收割型少赚很多 |
| `HarvestMachine` | 合作 | 有相当多末期收割型，但 Grim 不算少 | 第 45 天探测会触发 Grim |
| `EarlyProbeMachine` | 背叛 | 末期收割型/AllC/TFT2 很多，Grim 较少 | 遇 Grim 单场损失约 500 分 |

## 稳健版核心规则

1. 第 1 天合作；第 100 天一定背叛。
2. `pC = P(对手背叛 | 我上回合合作)`，`pD = P(对手背叛 | 我上回合背叛)`。
3. `pD - pC` 大：对手会报复也会奖励合作，继续合作并原谅偶发背叛。
4. `pC`、`pD` 接近且背叛率高：对手不响应我，转入剥削。
5. 惩罚阶段最多做两次和平测试。
6. 长时间严格交替的对手直接背叛。
7. 未确认对手类型时，对无端背叛以 25% 概率原谅；确认响应型后以 95% 概率原谅。

## 如果“前期合作、末期收割”很多

这类策略最大的弱点是：**它不响应你的动作，只按日程行动**。因此在它收割之前先背叛，可以在它仍合作时连续拿 10 分。两个反制版本：

### `HarvestMachine`：第 45 天开始收割反制

- 若对手到第 45 天从未背叛，先背叛一次。
- 对手仍合作：继续背叛，直到它报复。
- 对手报复：连续合作两回合修复；修复成功则恢复互惠，失败则转入惩罚。
- 对 TFT 的代价只有约 2 分；对 Grim 则会永久损失。

### `EarlyProbeMachine`：第 1 天就探测

- 第 1 天背叛，测试对手是否真的会报复。
- 若对手一直合作，则从第 1 天开始持续剥削。
- 若对手报复，则尝试修复；对普通 TFT 的代价约 2 分。
- 对 Grim/Mistrust 等永久惩罚型，单场会从约 600 掉到约 100。

## 大样本一般混合赛场

85 类对手，500 个赛场，每场 300 人：

| 策略 | 每对手平均 | 中位数 | 5% 分位 | 95% 分位 | 最差 |
|---|---:|---:|---:|---:|---:|
| EarlyProbe | **553.49** | 546.74 | 480.86 | 638.75 | 429.72 |
| Harvest45 | 551.69 | 549.50 | 504.24 | 602.52 | 474.02 |
| 稳健 Final | 550.14 | 552.52 | 515.01 | 578.81 | 486.48 |
| Responsive3 | 549.20 | 552.19 | 514.05 | 578.97 | 481.29 |
| GTFT(0.2)+lastD | 533.21 | 538.56 | 480.54 | 572.44 | 434.26 |
| TFT2+lastD | 530.63 | 536.17 | 475.43 | 572.45 | 426.09 |
| TFT | 517.06 | 519.35 | 471.63 | 554.44 | 418.16 |
| AllC | 514.57 | 521.50 | 444.59 | 564.68 | 402.12 |
| AllD | 351.38 | 341.66 | 281.28 | 451.33 | 217.84 |

注：表中 `EarlyProbe` 与 `ipd/core/early.py` 的 `EarlyProbeMachine` 逻辑相同，只是随机种子不同；对均值影响可忽略。

一般混合赛场中 `EarlyProbe` 均值最高，但方差也最大。是否使用它，取决于 Grim 的比例。

## 末期收割型很多的针对性结果

### 场景 A：42% 末期收割 + 20% AllC + 20% TFT，Grim 3%

| 策略 | 每对手平均 | 40 场完整循环赛平均排名 | 第 1 名 | 前 10 名 | 最差 |
|---|---:|---:|---:|---:|---:|
| 稳健 Final | 533.7 | 59.33 | 2/40 | 6/40 | 137 |
| Harvest45 | 605.9 | 2.27 | 22/40 | 40/40 | 8 |
| **EarlyProbe** | **688.2** | **1.25** | **35/40** | **40/40** | **4** |

在这个场景下，早期探测几乎完全压制了“末期收割”策略：在它们仍合作的前几十天，`EarlyProbe` 每回合拿 10 分，而它们只能拿 0 分。

### 场景 B：约 36%–42% 末期收割 + 10% AllC，改变 Grim 比例

| Grim 比例 | 稳健 Final | Harvest45 | EarlyProbe |
|---:|---:|---:|---:|
| 10% | 533.0 | 573.9 | **630.8** |
| 15% | 536.3 | 563.0 | **607.9** |
| 20% | 538.8 | 553.3 | **588.0** |
| 25% | 541.6 | 544.4 | **569.1** |

即使 Grim 达到 25%，只要末期收割型约占 36% 以上，早期探测的均值仍然最高。但它遇到高 Grim 时尾部风险明显更大。

### 反例：末期收割少、Grim 多

| 场景 | 稳健 Final | Harvest45 | EarlyProbe |
|---|---:|---:|---:|
| 末期收割 22%，Grim 20% | **536.0** | 487.4 | 466.2 |
| 末期收割 11%，Grim 30% | **550.5** | 460.0 | 400.6 |

这时不应早期探测；稳健版明显更好。

## 选择建议

根据模拟，粗略决策规则：

- **末期收割 + AllC + TFT2 合计 > 约 35%，且 Grim < 约 20%**：  
  用 `EarlyProbeMachine`，最大化利用它们的“不响应”。
- **末期收割较多，但 Grim 约 15%–25%**：  
  用 `Harvest45`，比早期探测少损失，仍能反制收割。
- **末期收割不多，或 Grim/永久惩罚型 > 约 25%–30%**：  
  用稳健版 `Machine`，不要为了贪 AllC 而触发 Grim。

如果只能根据“大部分人不聪明，末期收割很多”选一个版本，我会选 **`EarlyProbeMachine`**；如果担心 Grim，则选 **`Harvest45`**。两者都比原来的稳健版更适合这个前提。

## 单纯 EMA + 终局动态阈值的效果

`ipd/core/ema.py` 实现了只依赖统计反应、不做提前探测的版本：

- 对历史观测做半衰期 12 天的时间衰减；
- 末段自动降低“非响应型”的背叛率阈值（0.18 → 0.14 → 0.10 → 0.07）；
- 末段降低所需有效样本数；
- 不主动在第 1 天或第 45 天背叛。

在“42% 末期收割 + 20% AllC + 20% TFT，Grim 3%”的 60 个完整循环赛中：

| 策略 | 每对手平均 | 平均排名 | 前 10 名 | 最差 |
|---|---:|---:|---:|---:|
| 稳健 Final | 534.2 | 57.25 | 7/60 | 180 |
| EMA + 动态阈值 | 537.1 | 31.08 | 14/60 | 114 |
| Harvest45 | 607.3 | 2.28 | 59/60 | 14 |
| EarlyProbe | 690.5 | 1.28 | 60/60 | 4 |

对单个末期收割对手：

| 对手 | 稳健 Final | EMA | Harvest45 | EarlyProbe |
|---|---:|---:|---:|---:|
| LateD5 | 573 | 573 | 777 | 950 |
| LateD10 | 545 | 543 | 732 | 904 |
| LateD25 | 455 | 463 | 597 | 767 |
| LateD50 | 314 | 337 | 366 | 542 |

结论：EMA 能把末期切换后的反应加快，最多挽回约 20–25 分/场，但它**无法在对手第一次背叛之前识别末期收割者**。AllC、TFT、Grim 和“一直合作到第 80 天再收割”的对手，在第 80 天之前的历史完全相同。要获得前期剥削收益，必须承担一次主动探测的风险，或者使用基于种群先验的定时阈值；后者本质上就是 `HarvestMachine`。


## 更多因子 + 贝叶斯类型推断（不增加主动探测）

`ipd/core/bayes.py` 维护一个对手模型库：

- AllC / AllD / TFT / sTFT / Grim / TFT2；
- GTFT 两档 / Joss 两档 / Random 三档；
- Pavlov / Alternator；
- LateD5、10、15、20、25、30、40、50。

每天用观察到的历史计算各模型的似然和后验。因为不增加主动探测，所以“对手从未背叛”时仍然只能合作到第 100 天；推断的价值只出现在第一次背叛之后。

在“42% 末期收割 + 20% AllC + 20% TFT，Grim 3%”的 40 个完整循环赛中：

| 策略 | 每对手平均 | 平均排名 | 前 10 名 | 最差 |
|---|---:|---:|---:|---:|
| 稳健 Final | 533.7 | 59.33 | 6/40 | 137 |
| 贝叶斯类型推断 | 539.0 | 22.40 | 15/40 | 126 |
| Harvest45 | 605.9 | 2.27 | 40/40 | 8 |
| EarlyProbe | 688.2 | 1.25 | 40/40 | 4 |

对单个末期收割对手：

| 对手 | 稳健 Final | 贝叶斯推断 | Harvest45 | EarlyProbe |
|---|---:|---:|---:|---:|
| LateD10 | 545 | 549 | 732 | 904 |
| LateD25 | 455 | 474 | 597 | 767 |
| LateD50 | 314 | 349 | 366 | 542 |

结论：更多因子和贝叶斯推断能更快识别“它已经切换”，把晚期收割的损失减少约 20–35 分；但它仍然无法在第一次 D 之前区分 AllC、TFT、Grim 和 LateD。要提前拿到那部分分数，只能引入基于先验的定时探测/收割。

### 进一步增加因子：`ipd/core/bayes_plus.py`

在贝叶斯模型库上继续加入：

- LateD 的**所有可能切换日** k=1..99；
- TFT-delay 1/2/3、TFT-lastD 多档；
- 更多 GTFT / NoisyTFT 噪声档位；
- Grudger1/2/3、Mistrust；
- Periodic3/4、AlternatorD；
- Tester、Prober、RemorsefulProber；
- AllC-lastD、EarlyD10/25。

同一晚期收割场景（40 个完整循环赛）：

| 策略 | 每对手平均 | 平均排名 | 前 10 名 |
|---|---:|---:|---:|
| 稳健 Final | 533.7 | 59.33 | 6/40 |
| Bayes | 539.0 | 22.40 | 15/40 |
| BayesPlus | 539.5 | 20.65 | 17/40 |
| Harvest45 | 605.9 | 2.27 | 40/40 |
| EarlyProbe | 688.2 | 1.25 | 40/40 |

增加全部切换日模型只多拿约 0.5 分/对手，排名从 22.4 提升到 20.7。  
原因仍然是：这些模型只有在对手第一次 D 出现后才能发挥作用。  
此外，模型库不是越大越好：在 Joss(15%) 上，过多噪声/非响应模型会稀释后验，反而比简化 Bayes 更差。因此实际使用时应按先验裁剪模型库，而不是无脑堆模型。

## 回到原始混合样本（不强调末期收割）

早期实验使用 85 类对手，但去掉 `late_harvest` 强先验，恢复原来的 8 种混合先验。300 个赛场结果：

| 策略 | 每对手平均 | 中位数 | 5% 分位 | 95% 分位 | 最差 |
|---|---:|---:|---:|---:|---:|
| BayesPlus | 551.21 | 554.13 | 512.38 | 584.23 | 469.83 |
| 稳健 Final | 550.89 | 553.38 | 514.38 | 582.97 | 477.35 |
| EMAMachine | 550.89 | 553.46 | 514.66 | 581.97 | 471.31 |
| Bayes | 549.69 | 551.78 | 511.55 | 582.27 | 473.64 |
| Harvest45 | 547.02 | 546.26 | 501.41 | 594.46 | 460.65 |
| ProbeBot | 542.97 | 544.76 | 474.57 | 604.88 | 396.27 |
| EarlyProbe | 542.90 | 544.24 | 473.82 | 604.74 | 393.99 |
| TFT | 515.38 | 517.55 | 472.09 | 554.87 | 436.48 |
| AllD | 338.19 | 333.44 | 273.42 | 412.13 | 237.80 |

30 个赛场的完整循环赛：

| 策略 | 平均排名 | 第 1 名 | 前 3 名 | 前 10 名 |
|---|---:|---:|---:|---:|
| 稳健 Final | **2.63** | 13/30 | 21/30 | **30/30** |
| Harvest45 | 11.13 | 6/30 | 12/30 | 23/30 |
| BayesPlus | 14.87 | 13/30 | 19/30 | 24/30 |
| EarlyProbe | 28.97 | 6/30 | 11/30 | 20/30 |

结论：回到不以末期收割为主的原始样本后，**稳健 Final 仍是平均排名最好、进入前 10 最稳定的策略**。BayesPlus 的平均分略高约 0.3 分/对手，但排名波动明显更大；早期探测和 Harvest45 的平均分反而下降。

因此在原始混合样本下，不建议使用末期收割特化或早期探测版本；只有确认末期收割比例显著上升时，才切换到 `Harvest45` 或 `EarlyProbe`。
