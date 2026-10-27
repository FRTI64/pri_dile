# BayesPlus / Responsive3 C++ 实现与流程图

代码文件：`cpp/bayesplus_responsive3.cpp`

动作约定：

- `0 = C = 合作`
- `1 = D = 背叛`
- `day` 从 1 到 100
- 每场比赛开始前调用一次 `reset()`

编译：

```bash
g++ -std=c++17 -O2 -DRESP_BAYES_DEMO cpp/bayesplus_responsive3.cpp -o demo
./demo
```

---

## 一、Responsive3 参数

| 参数 | 数值 |
|---|---:|
| `G_UNPROVOKED` | 0.55 |
| `G_MUTUAL` | 0.40 |
| `G_RESPONSIVE` | 0.95 |
| `MIN_C` | 5 |
| `MIN_D` | 3 |
| `RESP_GAP` | 0.38 |
| `DEFECT_RATE` | 0.18 |
| `MAX_TESTS` | 2 |
| `COOLDOWN` | 8 |
| 交替检测长度 | 10 |
| 最后一天 | 100，固定 D |

### Responsive3 流程图

```mermaid
flowchart TD
    A[开始新的一天] --> B{day == 1?}
    B -- 是 --> C[合作 C]
    B -- 否 --> D{day >= 100?}
    D -- 是 --> E[背叛 D]
    D -- 否 --> F[统计 nC,nD,dC,dD,dRate]
    F --> G{最近10步严格交替?}
    G -- 是 --> E
    G -- 否 --> H{当前是 peace test 状态?}
    H -- test1 --> I[转 test2，合作 C]
    H -- test2 --> J[转 test3，合作 C]
    H -- test3 --> K{对手上一步 C?}
    K -- 是 --> L[回 nice，合作 C]
    K -- 否 --> M[转 punish，背叛 D]
    H -- 否 --> N{当前是 punish 模式?}
    N -- 是 --> O{满足和平测试条件?}
    O -- 是 --> P[计一次测试，转 test1，合作 C]
    O -- 否 --> E
    N -- 否 --> Q{证据足够且非响应型?}
    Q -- 是 --> R[转 punish，背叛 D]
    Q -- 否 --> S{对手上一步 C?}
    S -- 是 --> C
    S -- 否 --> T{响应型?}
    T -- 是 --> U[g = 0.95]
    T -- 否 --> V{我上一步 C?}
    V -- 是 --> W[g = 0.55]
    V -- 否 --> X[g = 0.40]
    U --> Y{random < g?}
    W --> Y
    X --> Y
    Y -- 是 --> C
    Y -- 否 --> E
```

### 判定条件

- `pC = dC / nC`
- `pD = dD / nD`
- `enough = nC >= 5 && nD >= 3`
- `responsive = enough && pC <= 0.35 && pD >= 0.65 && (pD-pC) >= 0.38`
- `nonresponsive = enough && (pD-pC) < 0.35 && dRate >= 0.18`
- 和平测试：`tests < 2 && day-last_test >= 8 && nC >=5 && nD >=2 && pC<=0.35 && pD>=0.65 && (pD-pC)>=0.40`

---

## 二、BayesPlus 参数与模型库

BayesPlus 不做主动探测。第 1 天合作；第 100 天固定背叛；在第一次观察到对手 D 之前一直合作。

核心阈值：

| 参数 | 数值 |
|---|---:|
| `NONRESP_THRESHOLD` | 0.40 |
| `PUNITIVE_THRESHOLD` | 0.55 |
| 响应型后验门槛 | 0.50 |
| 最后一天 | 100，固定 D |

### BayesPlus 流程图

```mermaid
flowchart TD
    A[开始新的一天] --> B{day == 1?}
    B -- 是 --> C[合作 C]
    B -- 否 --> D{day >= 100?}
    D -- 是 --> E[背叛 D]
    D -- 否 --> F{对手历史中有 D?}
    F -- 否 --> C
    F -- 是 --> G[用新观测更新模型后验]
    G --> H[按模型分组求和]
    H --> I{非响应型后验 >= 0.40?}
    I -- 是 --> E
    I -- 否 --> J{惩罚型后验 >= 0.55?}
    J -- 是 --> E
    J -- 否 --> K{响应型后验 >= 0.50?}
    K -- 是 --> C
    K -- 否 --> L{最近两次对手都是 D?}
    L -- 是 --> E
    L -- 否 --> C
```

### 模型分组

响应型：

- `TFT*`
- `sTFT`
- `GTFT*`
- `Joss*`
- `NoisyTFT*`
- `Pavlov`

惩罚型：

- `Grim*`
- `Grudger*`
- `Mistrust`

其余进入非响应型。

### 主要先验

| 模型 | 先验 |
|---|---:|
| AllC | 8 |
| AllD | 4 |
| TFT | 18 |
| sTFT | 3 |
| Grim | 5 |
| TFT2 | 8 |
| GTFT(.10) | 8 |
| GTFT(.25) | 3 |
| Joss(.05) | 4 |
| Joss(.15) | 2 |
| Random(.2/.5/.8) | 3/3/1 |
| Pavlov | 4 |
| LateD5/10/15/20/25/30/40/50 | 各 3 |
| LateDx k=1..99 | 各 0.16 |
| GTFTx 5/10/20/33/50 | 各 1.5 |
| NoisyTFTx .02/.10/.20 | 各 1.5 |
| TFTdelay1/2/3 | 各 1.5 |
| TFTlastD1/2/3/5/10 | 各 1.0 |
| Grudger1/2/3 | 各 2.0 |
| Mistrust | 2.0 |
| Periodic3/4 | 各 1.0 |
| AlternatorD | 1.0 |
| Tester/Prober | 各 2.0 |
| RemorsefulProber | 1.5 |
| AllC-lastD/EarlyD10/EarlyD25 | 1.0/1.0/0.5 |

---

## 三、接口

```cpp
ipd::Responsive3 r3(12345);
r3.reset();
int action = r3.move(day, my_history, opponent_history);

ipd::BayesPlus bp;
bp.reset();
int action2 = bp.move(day, my_history, opponent_history);
```

`my_history` 和 `opponent_history` 长度均为 `day-1`。
