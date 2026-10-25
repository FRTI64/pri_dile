# 对抗性剥削策略（AdversarialPolicy）

## 1. 结论

在不改动神经网络权重的前提下，给 MLP/GRU 神经策略加了一个**事后剥削模块**，
并用约束对抗调参选择参数。结果：

- 四分布 20 场平均从 **548.15 → 552.24**（激进版，+4.09）；
- 盲测从 **555.10 / 8.30 → 559.51 / 8.10**；
- 四个分布全部上涨，没有分布均分下降；
- 不主动探测、不触发 Grim：所有规则只在**对手已经先背叛之后**才生效。

两个档位：

| 配置 | 四分布均值 | 盲测均分 | 盲测排名 | 特点 |
|---|---:|---:|---:|---|
| 保守版（默认） | 551.62 | 557.44 | 8.30 | 逐类型风险更小 |
| **激进版** | **552.24** | **559.51** | **8.10** | Adv-MLP 内分数最高，少量 wild 类型有小损失 |
| GRU 保守版 | 551.43 | 557.00 | 8.30 | 需要 GRU 权重 |
| GRU 激进版 | 551.69 | 559.05 | 8.10 | 需要 GRU 权重 |

### Adv-RL（本轮新增，最高四分布）

把 base 换成完整 PPO 训练的 RL 策略，规则不变：

| 配置 | 四分布均值 | 盲测均分 | 盲测排名 | 特点 |
|---|---:|---:|---:|---|
| **Adv-RL 激进** | **554.29** | **562.91** | **8.00** | 最高四分布 |
| Adv-RL 保守 | 553.73 | 561.05 | 8.20 | 逐类型风险略小 |

配置在 `configs/adversarial_rl_policy_config*.json`，base 为
`weights/rl_policy_weights_bc.json`（run4 BC 初始化 + hard）。最终从零 + hard
的单独 RL 盲测更高（563.23），但套上激进规则后 four-dist 553.08，因此正式
Adv-RL 仍使用 BC base。C++ 版本在 `cpp/adversarial_rl.cpp`，BC base 编译时加
`-DNEURAL_POLICY_USE_RL_BC_WEIGHTS`。详见 `docs/RL_POLICY.md`。

---

## 2. 为什么不做主动探测

先测试了“在某个时间点主动背叛一次”的探针：

| 方案 | 原始 | 末期收割 | 高 Grim | 宽+聪明 | 均值 | 最差分布 Δ |
|---|---:|---:|---:|---:|---:|---:|
| 不探测 | 551.71 | 549.85 | 551.86 | 549.14 | 550.64 | 0 |
| 第 20 天探测 | 528.34 | 533.17 | **441.55** | 529.95 | 508.25 | **-110.30** |
| 第 50 天探测 | 534.16 | 538.14 | 478.49 | 535.76 | 521.64 | -73.37 |
| 第 80 天探测 | 543.63 | 545.05 | 521.58 | 542.00 | 538.06 | -30.28 |
| 第 95 天探测 | 551.43 | 548.21 | 545.76 | 548.94 | 548.59 | -6.09 |

在对手第一次背叛之前，AllC、TFT、Grim、末期收割者的历史完全一样。主动探测
会触发 Grim，单场损失约 500 分，即使只探测一次也不划算。因此最终方案放弃主动
探测，只利用“对手已经先背叛”后的信息。

---

## 3. 两条剥削规则

### 3.1 末期收割规则

```text
对手第一次背叛发生在第 50 天或之后
且对手下一回合仍然背叛
=> 之后永久背叛
```

为什么安全：

- 只有在对手先 D 之后才触发，不会触发 Grim；
- 连续两次晚期背叛是“按日程收割”的强信号；
- 对手如果是 TFT/Joss 的偶发噪声，通常不会连续两次 D；
- 对 LateD10/15/20/25/30/40/50 的分数分别提升约 4–31 分。

### 3.2 非响应型规则

```text
n_c >= 8 且 n_d >= 3
且 pD - pC <= 0.1
=> 之后永久背叛
```

其中：

```text
pC = P(对手 D | 我上回合 C)
pD = P(对手 D | 我上回合 D)
```

若对手的背叛概率不随我们的动作变化，它是非响应型（Random、Bully、AllD 等），
永久背叛通常是最佳反应。

这条规则会剥削：

- `Bully`：约 +170 分；
- `Random.90`（90% 合作）：约 +50–60 分；
- 部分 Random / 非响应型对手。

### 3.3 为什么不会碰 Grim

两条规则都要求 `D in theirs`，即对手已经背叛。当前神经策略在对手首次背叛前
不会背叛，因此 Grim 永远不会先 D；规则不会改变对 Grim 的 604 分合作结果。

---

## 4. 约束对抗调参

`ipd/training/adversarial.py` 是黑盒对抗训练/调参，不是再训练网络权重：

1. 固定 base 神经策略，在搜索赛场上评估四分布；
2. 随机采样剥削参数 `(gap, min_c, min_d, late_day, repair)`；
3. 只保留 **四个分布都没有掉分** 的候选；
4. 在更大的验证赛场重跑前几名；
5. 输出最佳配置。

```bash
python3 -m ipd train adversarial --arch mlp --seed 20261026
python3 -m ipd train adversarial --arch gru --seed 20261026
# RL base (Adv-RL): deployable MLP export or actor-critic checkpoint
python3 -m ipd train adversarial --arch rl \
    --weights weights/rl_policy_weights_bc.json --seed 20261026
```

搜索空间包含：

```text
gap       : None, 0.0, 0.05, 0.10, 0.15, 0.20
min_c     : 6, 8, 10, 12
min_d     : 2, 3, 4
late_day  : None, 40, 45, 50, 55, 60, 65, 70
repair    : True / False
```

约束是“验证分布均分不下降”，不是“每个对手类型都不下降”。因此激进版在
`MemOne-rand`、`Extort3`、`NoisyTFT.10/20` 等 wild/forgiving 类型上可能有
几分损失，换取四分布和盲测的更高总分。保守版把 `gap` 限制在 0.1、`late_day=50`，
逐类型风险更小。

---

## 5. 测试结果

### 5.1 四分布（20 场 × 299 人，种子 20261026）

| 策略 | 原始混合 | 末期收割 | 高 Grim | 宽+聪明 | 均值 | 最差分布 |
|---|---:|---:|---:|---:|---:|---:|
| Neural 基线 | 549.92 | 542.85 | 550.20 | 549.63 | 548.15 | 542.85 |
| **Adv-MLP 保守** | 552.24 | 547.98 | 553.31 | 552.97 | 551.62 | 547.98 |
| **Adv-MLP 激进** | **552.64** | **548.26** | **553.33** | **554.72** | **552.24** | **548.26** |
| GRU 基线 | 549.37 | 542.74 | 550.21 | 548.93 | 547.81 | 542.74 |
| Adv-GRU 保守 | 551.82 | 547.84 | 553.26 | 552.79 | 551.43 | 547.84 |
| Adv-GRU 激进 | 551.45 | 547.63 | 553.29 | 554.41 | 551.69 | 547.63 |
| Triple-majority | 549.21 | 542.54 | 549.96 | 548.32 | 547.51 | 542.54 |

四个分布全部上涨；最低单场分也普遍上涨（例如激进 MLP 的 late 最低分从 511.64
升到 519.25）。

### 5.2 盲测（10 个未见过的 299 人赛场）

| 策略 | 均分 | 平均排名 | 第 1 名 | 前 10 | 最差 |
|---|---:|---:|---:|---:|---:|
| Neural 基线 | 555.10 | 8.30 | 7/10 | 7/10 | 27 |
| Adv-MLP 保守 | 557.44 | 8.30 | 7/10 | 7/10 | 27 |
| **Adv-MLP 激进** | **559.51** | **8.10** | 7/10 | 7/10 | **25** |
| GRU 基线 | 555.22 | 8.30 | 7/10 | 7/10 | 27 |
| Adv-GRU 保守 | 557.00 | 8.30 | 7/10 | 7/10 | 27 |
| Adv-GRU 激进 | 559.05 | 8.10 | 7/10 | 7/10 | 25 |
| Triple-majority | 552.52 | 9.20 | 6/10 | 7/10 | 33 |

---

## 6. 使用

### 6.1 Python

> 源码在 `ipd/`。在仓库根目录直接运行，或先执行 `pip install -e .`。

```python
from ipd.policies.adversarial import AdversarialPolicy

p = AdversarialPolicy.from_config(
    "configs/adversarial_policy_config_aggressive.json",
    seed=20261024,
)
p.reset()                               # 每场必须调用
action = p.move(day, mine, theirs)
# 或 action = p.act(t, mine, theirs)
```

自检：

```bash
python3 -m ipd self-test --candidates adv-mlp --config configs/adversarial_policy_config.json
```

### 6.2 测试

```bash
# 四分布
python3 -m ipd benchmark --arch adversarial \
    --config configs/adversarial_policy_config_aggressive.json --fields 20 --workers 20

# 盲测
python3 -m ipd blind --arch adversarial \
    --config configs/adversarial_policy_config_aggressive.json --fields 10 --workers 12

# 也可以接入原版脚本
python3 -m ipd benchmark --candidates base --neural --neural-arch adversarial \
    --neural-config configs/adversarial_policy_config_aggressive.json --fields 20
python3 -m ipd blind --candidates classic --neural --neural-arch adversarial \
    --neural-config configs/adversarial_policy_config_aggressive.json --fields 10
```

### 6.3 配置

| 文件 | 内容 |
|---|---|
| `configs/adversarial_policy_config.json` | MLP 保守版（默认） |
| `configs/adversarial_policy_config_aggressive.json` | MLP 激进版（最高分） |
| `configs/adversarial_gru_policy_config.json` | GRU 保守版 |
| `configs/adversarial_gru_policy_config_aggressive.json` | GRU 激进版 |

---

## 7. 文件

| 文件 | 内容 |
|---|---|
| `ipd/policies/adversarial.py` | 剥削 wrapper，支持 MLP/GRU base |
| `ipd/training/adversarial.py` | 约束对抗调参 |
| `configs/adversarial_policy_config*.json` | MLP 配置 |
| `configs/adversarial_gru_policy_config*.json` | GRU 配置 |
| `results/adversarial_search_results.json` | 约束对抗调参的搜索/验证记录 |
| `results/adversarial_benchmark.json` | MLP 保守版四分布结果 |
| `results/adversarial_aggressive_benchmark.json` | MLP 激进版四分布结果 |
| `results/adversarial_gru_benchmark.json` | GRU 保守版四分布结果 |
| `results/adversarial_gru_aggressive_benchmark.json` | GRU 激进版四分布结果 |
| `results/blind_adversarial_results.json` | MLP 保守版盲测 |
| `results/blind_adversarial_aggressive_results.json` | MLP 激进版盲测 |
| `results/blind_adversarial_gru_results.json` | GRU 保守版盲测 |
| `results/blind_adversarial_gru_aggressive_results.json` | GRU 激进版盲测 |
| `results/adversarial_rl_benchmark.json` | Adv-RL 激进版四分布 |
| `results/blind_adversarial_rl_results.json` | Adv-RL 激进版盲测 |
| `results/adversarial_rl_conservative_benchmark.json` | Adv-RL 保守版四分布 |
| `results/blind_adversarial_rl_conservative_results.json` | Adv-RL 保守版盲测 |

---

## 8. 风险

1. **约束是分布级别，不是逐对手级别。**  
   激进版在少量 wild/forgiving 类型上可能掉几分，但四分布和盲测总分上涨。

2. **规则依赖对手先背叛。**  
   对 AllC/Grim 这类从不先 D 的对手，仍然是 604 分合作，不会额外剥削；
   这是为了避免触发 Grim 的必要代价。

3. **随机性。**  
   基策略按概率采样；提交时固定 seed，并重新跑盲测。

4. **C++ 版本。**
   Adv-RL 已有 C++ 版本（`cpp/adversarial_rl.cpp`）；Adv-MLP/Adv-GRU 仍只有
   Python wrapper。如果比赛要求 C++ 且想提交 Adv-MLP，需要把两条规则移植到
   `NeuralPolicy` 外层。

5. **权重和配置依赖。**  
   需要 `weights/neural_policy_weights.json` / `weights/gru_policy_weights.json`
   或 `weights/rl_policy_weights.json`，以及对应 `adversarial_*_config.json`；
   单文件提交时需要内嵌。
