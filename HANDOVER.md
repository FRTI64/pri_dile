# 100 天囚徒困境项目交接文档

## 1. 结论与推荐

当前有三套候选提交方案：

| 方案 | 复杂度 | 盲测均分 | 盲测平均排名 | 适用 |
|---|---:|---:|---:|---|
| **Triple-majority** | 高 | **554.99** | **8.30** | 追求最高盲测表现 |
| **Final-tuned** | 低 | 549.39 | 10.40 | 实现最简单、综合稳定 |
| **Final+Resp3** | 中 | 551.75 | 8.90 | 简单联合的折中 |

**建议：**

- 如果比赛接口允许每天运行多个子策略：提交 **Triple-majority**；
- 如果只允许提交一个简单策略：提交 **Final-tuned**；
- 如果不希望三重策略复杂，但想要联合收益：用 **Final+Resp3**。

不要直接使用 `ipd_strategy.py` 中的默认参数当作 Final-tuned；默认是早期版本。Final-tuned 的参数在下文和 JSON 中。

---

## 2. 核心测试结果

### 2.1 统一四分布测试

| 策略 | 原始混合 | 末期收割 | 高 Grim | 宽+聪明 | 四分布均值 |
|---|---:|---:|---:|---:|---:|
| Final-tuned | 549.30 | 542.83 | 549.69 | 543.56 | 546.35 |
| Final+Resp3 | 548.33 | 542.56 | 550.24 | **547.68** | **547.20** |
| Resp3-tuned | 547.69 | 542.49 | 549.84 | 543.77 | 545.95 |
| Bayes-tuned | **550.19** | **546.23** | **554.16** | 526.24 | 544.21 |

### 2.2 盲测（10 个未见过的 300 人赛场）

| 策略 | 均分 | 平均排名 | 第 1 名 | 前 10 |
|---|---:|---:|---:|---:|
| **Triple-majority** | **554.99** | **8.30** | 7/10 | 7/10 |
| Final+Resp3 | 551.75 | 8.90 | 7/10 | 7/10 |
| Final-tuned | 549.39 | 10.40 | 5/10 | 6/10 |
| Resp3-tuned | 547.84 | 11.40 | 4/10 | 7/10 |
| Bayes-tuned | 527.67 | 64.40 | 0/10 | 0/10 |

完整结果：

- `FINAL_COMPARISON.md`
- `final_benchmark.json`
- `SUITE_RESULTS.md`
- `OPPONENT_PERFORMANCE.md`
- `ADVANCED_RESULTS.md`

---

## 3. 策略定义

### 3.1 Final-tuned（推荐单策略）

机制：`ipd_strategy.py` 中的 `Machine`，但使用以下参数：

```text
P_UNPROVOKED = 0.2395699546614404
P_MUTUAL = 0.6399155368157192
P_RESPONSIVE = 0.9544467968114831
MIN_C = 5
MIN_D = 2
RESP_GAP = 0.3871088255996472
DEFECT_RATE = 0.1170465108152342
PEACE_TESTS = 1
PEACE_COOLDOWN = 9
```

运行规则：

- 第 1 天合作；
- 第 100 天背叛；
- 用 `pC/pD` 判断响应型/非响应型；
- 非响应型转入惩罚；
- 响应型高概率原谅；
- 未确认时按 0.2396 / 0.6399 的概率原谅；
- 最近 10 步严格交替时背叛；
- 和平测试最多 1 次，间隔至少 9 天。

### 3.2 Responsive3-tuned

参数：

```text
g_unprov = 0.31711381663242244
g_mutual = 0.26173130084075913
g_resp = 0.9804446329081643
min_c = 5
min_d = 2
resp = 0.46788262082576104
rate = 0.197064176521839
max_tests = 3
cooldown = 14
```

实现：`ipd_field.py` 的 `Responsive3`，通过 `train_multiobj.ParamResp` 设置参数。

### 3.3 BayesPlus-tuned

使用 `ipd_bayes_plus.py` 的模型库，参数先验倍率：

```text
coop_scale = 1.8369823500713045
punitive_scale = 2.4705958305576154
late_scale = 0.5216017158776368
noise_scale = 0.5817914534720606
probe_scale = 3.315461421931959
NONRESP_THRESHOLD = 0.27974873408266854
PUNITIVE_THRESHOLD = 0.7531347429866336
```

实现：`train_bayes_priors.ParamBayesPrior`。

**注意：** Bayes 对 Grim/punitive 很强，但在宽+聪明和盲测中明显掉队；不适合单独提交。

### 3.4 Triple-majority

实现：`triple_hybrid.py`。

三个子策略：

- Final-tuned；
- Responsive3-tuned；
- BayesPlus-tuned。

每天同时运行三者，按多数票：

```text
3 个 C → C
2 个 C、1 个 D → C
1 个 C、2 个 D → D
3 个 D → D
```

第 100 天三家都会 D。

### 3.5 Final+Resp3

实现：`train_joint_final_resp.Hybrid`。

最佳配置：

- Final-tuned；
- **Responsive3 使用默认参数**，不是 tuned；
- 两者动作一致时采用该动作；
- 不一致时选择背叛（`p=0`）。

---

## 4. 重要文件

### 核心策略

| 文件 | 内容 |
|---|---|
| `ipd_strategy.py` | Final 机制（默认参数不是 tuned） |
| `ipd_strategy.cpp` | Final 的 C++ 版本（默认参数不是 tuned） |
| `ipd_field.py` | 基础对手、Responsive3、GTFT 等 |
| `ipd_extended.py` | 85 类扩展对手、采样器 |
| `ipd_bayes.py` / `ipd_bayes_plus.py` | Bayes/BayesPlus |
| `triple_hybrid.py` | Triple-majority |
| `train_joint_final_resp.py` | Final+Resp3 联合策略 |
| `train_multiobj.py` | 多目标训练与参数配置 |
| `train_bayes_priors.py` | Bayes 先验参数化 |
| `bayesplus_responsive3.cpp` | BayesPlus / Responsive3 的 C++17 实现 |
| `BAYES_RESPONSIVE_FLOW.md` | 两套 C++ 策略的流程图 |

### 实验结果

| 文件 | 内容 |
|---|---|
| `FINAL_COMPARISON.md` | 各测试集主表 |
| `SUITE_RESULTS.md` | 9 个测试集结果 |
| `OPPONENT_PERFORMANCE.md` | 85 类对手逐类型表现 |
| `MATCHUP_RESULTS.md` | 早期策略对阵汇总 |
| `ADVANCED_RESULTS.md` | 训练/优化/盲测全过程 |
| `matchup_triple_results.csv` | 主策略逐类型原始数据 |
| `final_benchmark.json` | 统一四分布数据 |
| `suite_results.json` | 新测试集数据 |
| `blind_test.py` | 盲测代码 |
| `final_benchmark.py` | 四分布统一测试代码 |
| `run_suite_benchmark.py` | 9 个测试集代码 |

### 注意

`assoc_containers.cpp` 与本项目无关，是原有工作区文件，不要提交或修改。

---

## 5. 复现命令

### 快速自测

```bash
python3 ipd_strategy.py
python3 ipd_early.py
python3 blind_test.py --fields 10 --seed 20261025
```

### 统一四分布

```bash
python3 final_benchmark.py --fields 20 --workers 24
```

### 9 个测试集

```bash
python3 run_suite_benchmark.py --fields 12 --workers 24
```

### 逐类型对阵

```bash
python3 matchup_triple.py
```

### C++ 编译

```bash
g++ -std=c++17 -O2 -DRESP_BAYES_DEMO bayesplus_responsive3.cpp -o demo
./demo
```

---

## 6. 接口与移植注意

Python 策略统一使用：

```text
C = 0
D = 1
day = 1..100
mine/theirs = 到此日之前的动作
```

每场比赛开始必须调用一次 `reset()`。

### C++ 现状

- `ipd_strategy.cpp`：Final，但需要把参数改成 Final-tuned；
- `bayesplus_responsive3.cpp`：Responsive3 + BayesPlus，无外部依赖；
- **目前没有 C++ 版 Triple-majority**。  
  如果比赛要求 C++ 且想提交 Triple，需要：
  1. 把 Final-tuned 也移植到 C++；
  2. 把 Final、Responsive3、BayesPlus 三个动作做多数投票；
  3. 保持三者各自独立状态。

C++ 的 BayesPlus 模型库和 Python 版语义对应，但浮点/模型顺序可能造成几分差异。

---

## 7. 风险与注意事项

1. **没有真实 300 人数据**  
   所有结果来自合成对手库；盲测也只是分布内的未见种子，不等于真实比赛。

2. **分布依赖很强**  
   - Harvest 在末期收割分布最强；
   - Bayes 在 Grim/惩罚型最强；
   - Triple/Final 更均衡；
   - 不要用单一分布的结果直接外推。

3. **Triple 的复杂度**  
   每天运行三个策略，代码量和状态管理明显高于单 Final；需要确认比赛是否限制运行时间/代码大小。

4. **Bayes 有严重局部弱点**  
   对 RemorsefulProber 约 108，对 ProbeBot 约 115；这也是建议多数投票/不单独使用 Bayes 的原因。

5. **随机性**  
   Final/Responsive3 有随机原谅；报告中的种子固定为 `20261024`。  
   实际提交应固定种子，保证可复现。

6. **第 100 天**  
   所有候选策略都应在最后一天选择 D；如果移植时漏掉，会损失最后一天收益。

7. **Base Final 与 Final-tuned 不同**  
   `ipd_strategy.py` / `ipd_strategy.cpp` 默认参数仍是旧版；提交前必须改为 tuned 参数，或明确选择默认版。

---

## 8. 交接检查清单

- [ ] 确认比赛接口：函数签名、历史格式、是否允许自定义类和 reset。
- [ ] 确认是否允许每天运行三个子策略（Triple）。
- [ ] 确定提交版本：Triple / Final-tuned / Final+Resp3。
- [ ] 如果提交 C++，完成 Final-tuned 和 Triple 的 C++ 合并。
- [ ] 固定随机种子。
- [ ] 用最终版本重新跑一次 `blind_test.py`，确认没有配置遗漏。
- [ ] 检查第 1 天和第 100 天行为。
- [ ] 保存最终提交文件、配置 JSON 和运行日志。
- [ ] 不要提交无关的 `assoc_containers.cpp`。
