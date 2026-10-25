# 100 天囚徒困境项目交接文档

## 目录结构

```text
ipd/         规范 Python 包：core / policies / evaluation / training / tools
cpp/         C++17 推理实现与生成的头文件
configs/     对抗策略配置（与 weights/ 配对）
weights/     训练好的 MLP/GRU/RL 权重 JSON
results/     基准、盲测、训练曲线、搜索记录（JSON/CSV）
docs/        方法、结果、流程图与结构规范
tests/       分层测试（core / policies / evaluation / training）
README.md    项目总览与快速开始
```

分层与接口规范见 `docs/STRUCTURE.md`。新命令：

```bash
python3 -m ipd benchmark --candidates core --fields 20
python3 -m ipd blind --candidates core --fields 10
python3 -m ipd train rl --device auto
python3 -m ipd self-test --candidates all
```

所有命令默认在仓库根目录执行；路径由 `ipd.paths` 统一解析。如需从其他目录
运行，请先执行 `pip install -e .`。

## 0. 本次新增工作摘要

本轮在原有规则策略之外，新增了 MLP/GRU/完整 RL 三类学习策略及对应的约束剥削
版本，并补齐了 C++ 推理、测试和文档。

| 工作 | 主要文件 | 四分布均值 | 盲测均分 / 排名 |
|---|---|---:|---:|
| MLP 神经策略（BC + DAgger） | `ipd/training/neural.py` / `ipd/policies/neural.py` | 548.15 | 555.10 / 8.30 |
| GRU-Net 版本 | `ipd/training/gru.py` / `ipd/policies/gru.py` | 547.81 | 555.22 / 8.30 |
| 约束对抗剥削（MLP 激进） | `ipd/policies/adversarial.py` / `configs/adversarial_policy_config_aggressive.json` | **552.24** | **559.51 / 8.10** |
| 约束对抗剥削（GRU 激进） | `configs/adversarial_gru_policy_config_aggressive.json` | 551.69 | 559.05 / 8.10 |
| **完整 RL（PPO，从零 + hard，最终）** | `ipd/training/rl.py` / `ipd/policies/rl.py` | **553.33** | **563.23 / 8.00** |
| 完整 RL（PPO，从零 base） | `ipd/training/rl.py --from-scratch` | 551.87 | 559.27 / 8.30 |
| 完整 RL（PPO，BC 初始化） | `ipd/training/rl.py` / `ipd/policies/rl.py` | 550.50 | 558.60 / 8.30 |
| **Adv-RL 激进** | `ipd/policies/adversarial.py` + `configs/adversarial_rl_policy_config_aggressive.json` | **554.29** | **562.91 / 8.00** |

关键点：

- 训练使用 PyTorch，自动选择 NPU → CUDA → MPS → CPU；推理不需要 PyTorch。
- MLP/GRU 都提供纯 Python 和 C++17 推理；C++ 共用 `cpp/ipd_features.hpp`。
- 对抗模块只做“对手先背叛之后”的事后剥削，不主动探测，因此不会触发 Grim。
- 对抗版的约束是“四个验证分布均分不下降”，不是“每个对手类型都不下降”。
  激进版在少量 wild 类型上可能掉几分，保守版更稳；详见 `docs/ADVERSARIAL_POLICY.md`。
- 完整 RL 版本用 PPO 直接优化 100 天总分，不再依赖行为克隆；训练时 24 个
  worker 生成 rollout，GPU 只做更新，推理仍不需要 PyTorch。
- RL 默认保留“不主动探测”的安全规则，并用 wide 加权、BC KL 正则和 hard 对手
  挖掘压低尾部风险；`--allow-probe` 可关闭安全规则做对照。
- **最终的单独 RL**（从零 + hard）四分布 553.33、盲测 **563.23 / 8.00**，
  是目前最高盲测，也是最好的单网络策略。
- **Adv-RL** 把 BC 初始化的 RL 作为 base，再套用两条约束剥削规则；四分布
  **554.29**、盲测 562.91 / 8.00，是最高四分布。两者都超过此前最好的
  Adv-MLP（552.24 / 559.51）。
- 本轮大部分训练/测试在 24 核 CPU + RTX 5090 Laptop 上完成；最终从零 + hard
  的 RL 在 CPU 上完成。NPU 代码路径已实现但未实机验证。

复现入口：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt

python3 -m ipd train neural --device auto
python3 -m ipd train gru --device auto
python3 -m ipd train adversarial --arch mlp --seed 20261026
# 也可以直接执行训练文件：
# python3 ipd/training/neural.py --device auto
# python3 ipd/training/search/multiobj.py --workers 24
# RL 阶段 1：wide 加权 + BC KL
python3 -m ipd train rl --device auto --iterations 100 \
  --dist-weights 1,1,1,2 --ent-coef 0.0001 --lr 2e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --out weights/rl_policy_weights_stage1.json \
  --checkpoint-out weights/rl_policy_checkpoint_stage1.json \
  --history-out results/rl_training_history_stage1.json

# RL 阶段 2：BC base + hard（Adv-RL 的 base）
python3 -m ipd train rl --device auto --iterations 60 \
  --init-weights weights/rl_policy_checkpoint_stage1.json \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 --ent-coef 0.0001 --lr 1e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --select-metric blind --out weights/rl_policy_weights_bc.json \
  --checkpoint-out weights/rl_policy_checkpoint_bc.json \
  --history-out results/rl_training_history_bc.json

# RL 阶段 3：从零 base（30 iter）
python3 -m ipd train rl --device auto --from-scratch \
  --iterations 30 --fields-per-dist 5 --n 80 \
  --dist-weights 1,1,1,2 --ent-coef 0.001 --lr 3e-4 \
  --center-advantages --select-metric avg \
  --out weights/rl_policy_weights_scratch_base.json \
  --checkpoint-out weights/rl_policy_checkpoint_scratch_base.json \
  --history-out results/rl_training_history_scratch_base.json

# RL 阶段 4：从零 + hard（最终单独 RL）
python3 -m ipd train rl --device auto --iterations 40 \
  --init-weights weights/rl_policy_checkpoint_scratch_base.json \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 --ent-coef 0.0001 --lr 1e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --select-metric blind --out weights/rl_policy_weights.json \
  --checkpoint-out weights/rl_policy_checkpoint.json \
  --history-out results/rl_training_history.json

python3 -m ipd benchmark --arch adversarial \
  --config configs/adversarial_policy_config_aggressive.json --fields 20 --workers 20
python3 -m ipd benchmark --arch rl --fields 20 --workers 20
python3 -m ipd blind --arch adversarial \
  --config configs/adversarial_policy_config_aggressive.json --fields 10 --workers 12
python3 -m ipd blind --arch rl --fields 10 --workers 12
python3 -m ipd benchmark --arch adv-rl \
  --config configs/adversarial_rl_policy_config_aggressive.json --fields 20 --workers 20
python3 -m ipd blind --arch adv-rl \
  --config configs/adversarial_rl_policy_config_aggressive.json --fields 10 --workers 12
```

---

## 1. 结论与推荐

当前候选方案：

| 方案 | 复杂度 | 盲测均分 | 盲测平均排名 | 适用 |
|---|---:|---:|---:|---|
| **RL-PPO（从零 + hard，最终）** | 中 | **563.23** | **8.00** | **最高盲测，最佳单网络策略** |
| **Adv-RL 激进** | 中 | **562.91** | **8.00** | **最高四分布，RL 权重+配置** |
| Adv-RL 保守 | 中 | 561.05 | 8.20 | RL 版，逐类型风险略小 |
| **Adv-MLP 激进** | 中 | 559.51 | 8.10 | 此前最高分，需权重+配置 |
| RL-PPO（从零 base） | 中 | 559.27 | 8.30 | 从零训练的中间阶段 |
| **Adv-GRU 激进** | 中 | **559.05** | **8.10** | GRU 版最高分 |
| RL-PPO（BC 初始化） | 中 | 558.60 | 8.30 | Adv-RL 的 base |
| **Adv-MLP 保守** | 中 | 557.44 | 8.30 | 逐类型风险更小 |
| **Adv-GRU 保守** | 中 | 557.00 | 8.30 | GRU 保守版 |
| **GRU-Net-GPU/NPU** | 中 | 555.22 | 8.30 | 可携带权重、想要循环状态 |
| **Neural-GPU/NPU** | 中 | 555.10 | 8.30 | 可携带权重、想要更简单的 MLP |
| **Triple-majority** | 高 | 554.99 | 8.30 | 规则策略里盲测较强 |
| **Final-tuned** | 低 | 549.39 | 10.40 | 实现最简单、综合稳定 |
| **Final+Resp3** | 中 | 551.75 | 8.90 | 简单联合的折中 |

**建议：**

- 如果**盲测分数**是主要指标，且想用单一网络：优先 **RL-PPO 从零 + hard**
  （553.33 / 563.23 / 8.00）；它没有外层规则，是当前最高盲测；
- 如果**四分布均值**是主要指标：优先 **Adv-RL 激进**（554.29 / 562.91 / 8.00），
  base 使用 `weights/rl_policy_weights_bc.json`；
- 想要更低逐类型风险：**Adv-RL 保守**（553.73 / 561.05 / 8.20）；
- 如果不想用外层剥削规则：**RL-PPO 从零 + hard** 或纯从零 base
  （551.87 / 559.27 / 8.30）；
- 如果不想用 RL 权重，退回 **Adv-MLP 激进**（552.24 / 559.51 / 8.10）；
- 如果希望保留 GRU 循环状态：用 **Adv-GRU 激进** 或保守版；
- 如果比赛允许每天运行多个子策略：**Triple-majority**；
- 如果只能提交纯规则/无权重：**Final-tuned** 或 **Triple-majority**；
- 如果不希望三重策略复杂，但想要联合收益：用 **Final+Resp3**。

> 神经/RL/对抗版本的盲测来自统一的 `ipd/evaluation/blind.py`；该脚本复用了
> 对手循环赛，与更早的独立盲测结果不能直接混用。同一脚本内
> RL-PPO（从零 + hard）≈ Adv-RL > Adv-MLP > Adv-GRU > RL-PPO（BC 初始化）>
> Neural/GRU > Triple。
> 对抗版本的约束是“四个分布均不下降”，不是“每个对手类型都不下降”。详见
> `docs/NEURAL_POLICY.md`、`docs/GRU_POLICY.md`、`docs/RL_POLICY.md`、
> `docs/ADVERSARIAL_POLICY.md`。

不要直接使用 `ipd/core/machine.py` 中的默认参数当作 Final-tuned；默认是早期版本。Final-tuned 的参数在下文和 JSON 中。

---

## 2. 核心测试结果

### 2.1 统一四分布测试

| 策略 | 原始混合 | 末期收割 | 高 Grim | 宽+聪明 | 四分布均值 |
|---|---:|---:|---:|---:|---:|
| **Adv-RL 激进** | **555.76** | **549.19** | **553.39** | **558.82** | **554.29** |
| Adv-RL 保守 | 556.10 | 549.25 | 553.43 | 556.16 | 553.73 |
| **RL-PPO（从零 + hard，最终）** | **555.88** | **548.41** | **552.09** | **556.95** | **553.33** |
| **Adv-MLP 激进** | **552.64** | **548.26** | 553.33 | **554.72** | **552.24** |
| Adv-MLP 保守 | 552.24 | 547.98 | 553.31 | 552.97 | 551.62 |
| Adv-GRU 激进 | 551.45 | 547.63 | 553.29 | 554.41 | 551.69 |
| **RL-PPO（从零 base）** | **553.73** | **547.55** | **552.22** | **553.99** | **551.87** |
| **RL-PPO（BC 初始化 + hard）** | **554.08** | **544.29** | **550.34** | **553.30** | **550.50** |
| Neural-sample | 549.92 | 542.85 | 550.20 | 549.63 | 548.15 |
| GRU-sample | 549.37 | 542.74 | 550.21 | 548.93 | 547.81 |
| Triple-majority | 549.21 | 542.54 | 549.96 | 548.32 | 547.51 |
| Final-tuned | 549.30 | 542.83 | 549.69 | 543.56 | 546.35 |
| Final+Resp3 | 548.33 | 542.56 | 550.24 | 547.68 | 547.20 |
| Resp3-tuned | 547.69 | 542.49 | 549.84 | 543.77 | 545.95 |
| Bayes-tuned | **550.19** | **546.23** | **554.16** | 526.24 | 544.21 |

最终 RL / Adv-RL 比 Neural/GRU 高约 5–6 分；Adv-MLP 比 Neural/GRU 高约
3.5–4 分。Neural/GRU 与 Triple 同档。对抗版的约束是四个分布均不下降，
逐类型风险见 `docs/ADVERSARIAL_POLICY.md`。

### 2.2 盲测（10 个未见过的 300 人赛场）

| 策略 | 均分 | 平均排名 | 第 1 名 | 前 10 |
|---|---:|---:|---:|---:|
| **Triple-majority** | **554.99** | **8.30** | 7/10 | 7/10 |
| Final+Resp3 | 551.75 | 8.90 | 7/10 | 7/10 |
| Final-tuned | 549.39 | 10.40 | 5/10 | 6/10 |
| Resp3-tuned | 547.84 | 11.40 | 4/10 | 7/10 |
| Bayes-tuned | 527.67 | 64.40 | 0/10 | 0/10 |

Neural/GRU/RL/Adv 的盲测用 `ipd/evaluation/blind.py` 单独跑（复用对手循环赛、每个候选深拷贝同一赛场）：

| 策略 | 均分 | 平均排名 | 第 1 名 | 前 10 | 最差 |
|---|---:|---:|---:|---:|---:|
| **RL-PPO（从零 + hard，最终）** | **563.23** | **8.00** | **7/10** | **7/10** | **25** |
| **Adv-RL 激进** | **562.91** | **8.00** | **7/10** | **7/10** | **25** |
| Adv-RL 保守 | 561.05 | 8.20 | 7/10 | 7/10 | 26 |
| **Adv-MLP 激进** | **559.51** | **8.10** | 7/10 | 7/10 | 25 |
| RL-PPO（从零 base） | 559.27 | 8.30 | 7/10 | 7/10 | 27 |
| Adv-GRU 激进 | 559.05 | 8.10 | 7/10 | 7/10 | 25 |
| RL-PPO（BC 初始化 + hard） | 558.60 | 8.30 | 7/10 | 7/10 | 27 |
| Adv-MLP 保守 | 557.44 | 8.30 | 7/10 | 7/10 | 27 |
| Adv-GRU 保守 | 557.00 | 8.30 | 7/10 | 7/10 | 27 |
| GRU-sample | 555.22 | 8.30 | 7/10 | 7/10 | 27 |
| Neural-sample | 555.10 | 8.30 | 7/10 | 7/10 | 27 |
| Triple-majority | 552.52 | 9.20 | 6/10 | 7/10 | 33 |
| Final-tuned | 547.35 | 13.20 | 2/10 | 5/10 | 30 |
| Bayes-tuned | 527.67 | 65.80 | 0/10 | 0/10 | 100 |

两个脚本的绝对分数不要直接混用；上表只用于说明同一脚本内的相对顺序。

完整结果：

- `docs/FINAL_COMPARISON.md`
- `results/final_benchmark.json`
- `docs/SUITE_RESULTS.md`
- `docs/OPPONENT_PERFORMANCE.md`
- `docs/ADVANCED_RESULTS.md`
- `docs/NEURAL_POLICY.md` / `results/neural_benchmark.json` / `results/neural_blind_results.json`

---

## 3. 策略定义

### 3.1 Final-tuned（推荐单策略）

机制：`ipd/core/machine.py` 中的 `Machine`，但使用以下参数：

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

实现：`ipd/core/opponents/reciprocal.py` 的 `Responsive3`；候选入口在
`ipd/policies/candidates/resp3_tuned.py`。

### 3.3 BayesPlus-tuned

使用 `ipd/core/bayes_plus.py` 的模型库，参数先验倍率：

```text
coop_scale = 1.8369823500713045
punitive_scale = 2.4705958305576154
late_scale = 0.5216017158776368
noise_scale = 0.5817914534720606
probe_scale = 3.315461421931959
NONRESP_THRESHOLD = 0.27974873408266854
PUNITIVE_THRESHOLD = 0.7531347429866336
```

实现：`ipd/policies/candidates/bayes_tuned.py`。

**注意：** Bayes 对 Grim/punitive 很强，但在宽+聪明和盲测中明显掉队；不适合单独提交。

### 3.4 Triple-majority

实现：`ipd/policies/candidates/triple_majority.py`。

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

实现：`ipd/policies/candidates/hybrid.py`。

最佳配置：

- Final-tuned；
- **Responsive3 使用默认参数**，不是 tuned；
- 两者动作一致时采用该动作；
- 不一致时选择背叛（`p=0`）。

### 3.6 Neural-GPU/NPU

实现：

- `ipd/training/neural.py`：PyTorch 训练，自动选择 CUDA / NPU / MPS / CPU；
- `ipd/policies/neural.py`：纯 Python 推理，类 `NeuralPolicy`；
- `weights/neural_policy_weights.json`：已训练好的权重（约 243 KB）。

机制：

- 85 维历史特征 → 64 → 32 → 1 的 MLP，输出 `P(D)`；
- 训练信号来自 `Triple-majority` 的行为克隆 + 1 轮 DAgger；
- 默认按概率采样，保持教师的随机原谅行为；
- 第 1 天 C、第 100 天 D、对手从未 D 时 C（与 Final 的“不主动探测”一致）。

关键指标：

```text
验证集动作一致率 = 99.22%
验证集 BCE = 0.0222
采样温度 = 0.95
参数量 = 7,617
训练设备 = CUDA (RTX 5090 Laptop)
```

它是单策略，不需要同时运行 Final/Resp3/Bayes 三套状态；代价是需要携带权重文件。

### 3.7 GRU-Net-GPU/NPU

实现：

- `ipd/training/gru.py`：PyTorch 训练，自动选择 CUDA / NPU / MPS / CPU；
- `ipd/policies/gru.py`：纯 Python 推理，类 `GRUPolicy`；
- `cpp/gru_policy.cpp` / `cpp/gru_policy_weights.hpp`：C++17 推理；
- `weights/gru_policy_weights.json`：已训练权重（约 395 KB）。

机制：

- 85 维特征 → `GRU(32)` → 32 → 1，12,513 参数；
- 与 MLP 版同一套教师（Triple）和 DAgger 数据；
- 只对“对手已出现 D、且不是第 1/100 天”的时间步计算损失；
- 维护 32 维隐藏状态，因此每场必须 `reset()`，且同一时间步不要重复调用；
- 边界规则与 MLP 版相同（第 1 天 C、第 100 天 D、对手未 D 时 C）。

关键指标：

```text
验证集掩码动作一致率 = 97.21%
验证集掩码 BCE = 0.0803
采样温度 = 1.00
参数量 = 12,513
训练设备 = CUDA (RTX 5090 Laptop)
```

GRU 与 MLP 同档：四分布 547.81 vs 548.15，盲测 555.22 vs 555.10。

### 3.8 AdversarialPolicy（约束剥削）

实现：

- `ipd/policies/adversarial.py`：包装 MLP/GRU，加入两条事后剥削规则；
- `ipd/training/adversarial.py`：四分布不掉分约束下的黑盒对抗调参；
- `configs/adversarial_policy_config*.json`：MLP 保守/激进配置；
- `configs/adversarial_gru_policy_config*.json`：GRU 保守/激进配置。

两条规则：

1. 对手首次 D 发生在第 50 天及以后，且下一回合继续 D → 永久 D；
2. `n_c>=8, n_d>=3` 且 `pD-pC<=gap` → 永久 D。

两条规则都要求对手已经先 D，因此不会触发 Grim。主动探测已被实验否决：
第 20/50/80 天探测会让 grim30 掉 110/73/30 分。

结果（激进 MLP）：

```text
四分布：552.24（基线 548.15，+4.09）
盲测：559.51 / 8.10（基线 555.10 / 8.30）
```

约束是“四个分布均不下降”，不是“每个对手类型都不下降”。激进版在
`MemOne-rand`、`Extort3`、`NoisyTFT.10/20` 等 wild 类型上可能掉几分；
保守版（gap=0.1, late_day=50, repair=false）逐类型风险更小。

详见 `docs/ADVERSARIAL_POLICY.md`。

### 3.9 RL-PPO（完整强化学习）

实现：

- `ipd/training/rl.py`：PPO 训练、24 worker rollout、验证与导出；
- `ipd/policies/rl.py`：纯 Python actor-critic 推理与部署格式导出；
- `weights/rl_policy_checkpoint.json` / `weights/rl_policy_weights.json`：最终
  （从零 + hard）checkpoint / 部署 MLP；
- `weights/rl_policy_checkpoint_bc.json` / `weights/rl_policy_weights_bc.json`：
  BC 初始化 + hard（Adv-RL base）；
- `weights/rl_policy_checkpoint_scratch_base.json` /
  `weights/rl_policy_weights_scratch_base.json`：纯从零 base；
- `weights/rl_policy_checkpoint_stage1.json` / `weights/rl_policy_weights_stage1.json`：run3 阶段产物；
- `cpp/rl_policy_weights.hpp` / `cpp/rl_policy_weights_bc.hpp`：C++ 权重。

MDP：

- 状态：与 BC/GRU 相同的 85 维历史特征；
- 动作：C/D；
- 奖励：每轮 `payoff/100`，回合 100 轮；
- 对手：四个官方分布 + 可选 hard 对手混合。

训练：

- 策略主干 85→64→32→1，价值主干独立 85→64→32→1；
- PPO clip=0.2、GAE(gamma=1, lambda=0.95)、每场优势中心化；
- `bc_kl_coef=0.1` 把策略拉向初始 BC 策略，抑制过度剥削；
- `--dist-weights 1,1,1,2` 提高 wide（盲测分布）权重；
- `--train-dists ...,hard` 加入 ProbeBot、EarlyD1/10/25、Bully 等 hard 对手；
- 默认保留第 1 天 C、对手未 D 时 C、第 100 天 D 的安全规则。

结果（最终从零 + hard）：

```text
四分布：555.88 / 548.41 / 552.09 / 556.95，均值 553.33
盲测：563.23 / 8.00（BC 555.10 / 8.30）
```

这是目前最高盲测和最好的单网络策略。训练路径：

```text
run3（BC + wide + KL）        549.55 / 557.95
run4（BC + hard）             550.50 / 558.60   -> Adv-RL base
run5（从零 base，30 iter）    551.87 / 559.27
run6（从零 + hard，40 iter）  553.33 / 563.23   -> 最终
```

从零训练完全不加载 BC，只复用固定特征标准化，说明 PPO 确实在利用奖励信号学习。
主要尾部风险是 `Rand3->TFT`（BC 约 554，最终 RL 约 347）；`ProbeBot` 已从
run2 的 143.5 修复到 603，`EarlyD25` 基本持平。

### 3.10 Adv-RL（RL + 约束剥削）

实现：

- `ipd/policies/adversarial.py` 的 `base_arch: "rl"` 分支；
- `configs/adversarial_rl_policy_config_aggressive.json`（激进）；
- `configs/adversarial_rl_policy_config.json`（保守）；
- `cpp/adversarial_rl.cpp`（C++17 版本）。

规则与 Adv-MLP 完全相同，base 使用 `weights/rl_policy_weights_bc.json`
（run4 BC 初始化 + hard）：

1. 首次 D 在 `late_day` 之后且下一回合继续 D → 永久 D；
2. `n_c>=min_c, n_d>=min_d` 且 `pD-pC<=gap` → 永久 D。

激进配置：`gap=0.2, min_c=12, min_d=2, late_day=40, repair=true`。

结果：

```text
四分布：555.76 / 549.19 / 553.39 / 558.82，均值 554.29
盲测：562.91 / 8.00（7/10 第一，7/10 前 10）
```

这是最高四分布候选，已经超过 Adv-MLP 激进（552.24 / 559.51）。保守版为
553.73 / 561.05 / 8.20，逐类型风险略小。

把规则套到最终 scratch-hard 单独 RL 上：保守版 553.73 / 562.79，激进版四分布
553.08，均不如 BC base；所以正式 Adv-RL 配置仍使用
`weights/rl_policy_weights_bc.json`。

详见 `docs/RL_POLICY.md`。

---

## 4. 重要文件

### 核心策略

| 文件 | 内容 |
|---|---|
| `ipd/core/machine.py` | Final 机制（默认参数不是 tuned） |
| `cpp/ipd_strategy.cpp` | Final 的 C++ 版本（默认参数不是 tuned） |
| `ipd/core/opponents/basic.py` | 基础对手（AllC/TFT/Grim/GTFT/...） |
| `ipd/core/opponents/reciprocal.py` | Responsive3、ProbeBot、GTFT 等 |
| `ipd/core/opponents/bank.py` | 85 类扩展对手、采样器 |
| `ipd/core/bayes.py` / `ipd/core/bayes_plus.py` | Bayes/BayesPlus |
| `ipd/policies/candidates/triple_majority.py` | Triple-majority |
| `ipd/policies/candidates/` | 每个参赛/评估候选一个独立文件，互不 import |
| `ipd/evaluation/split.py` | 按候选/对手拆分的评估驱动 |
| `configs/eval/*.json` | 每个候选独立参数的评估配置 |
| `cpp/policies/*.hpp` | 按策略/模型拆分的 C++ 头文件 |
| `cpp/opponents/*.hpp` | 按对手拆分的 C++ 头文件 |
| `ipd/training/neural.py` | GPU/NPU 神经策略训练（PyTorch + DAgger） |
| `requirements-train.txt` | 神经策略训练依赖（仅 PyTorch） |
| `ipd/policies/neural.py` | 神经策略纯 Python 推理（无需 PyTorch） |
| `cpp/neural_policy.cpp` | MLP 策略 C++17 推理（无外部依赖） |
| `cpp/ipd_features.hpp` | MLP/GRU 共用的 C++ 特征提取 |
| `weights/neural_policy_weights.json` | 已训练的 MLP 策略权重 |
| `cpp/neural_policy_weights.hpp` | 自动生成的 C++ 权重 |
| `ipd/tools/export_cpp.py` | MLP 权重 JSON → C++ 头文件 |
| `ipd/training/gru.py` | GRU-Net 训练（PyTorch + DAgger） |
| `ipd/policies/gru.py` | GRU-Net 纯 Python 推理 |
| `cpp/gru_policy.cpp` | GRU-Net C++17 推理 |
| `weights/gru_policy_weights.json` / `.hpp` | GRU-Net 权重 / C++ 权重 |
| `ipd/tools/export_cpp.py` | GRU 权重 JSON → C++ 头文件 |
| `ipd/policies/adversarial.py` | 约束剥削 wrapper（MLP/GRU） |
| `ipd/training/adversarial.py` | 四分布不掉分约束下的对抗调参 |
| `ipd/training/rl.py` | 完整 PPO 强化学习训练（24 worker rollout + GPU 更新） |
| `ipd/policies/rl.py` | RL actor-critic 纯 Python 推理与部署导出 |
| `weights/rl_policy_checkpoint.json` / `weights/rl_policy_weights.json` | 最终从零 + hard 单独 RL |
| `weights/rl_policy_checkpoint_bc.json` / `weights/rl_policy_weights_bc.json` | BC 初始化 + hard（Adv-RL base） |
| `weights/rl_policy_checkpoint_scratch_base.json` / `weights/rl_policy_weights_scratch_base.json` | 纯从零 base |
| `weights/rl_policy_checkpoint_stage1.json` / `weights/rl_policy_weights_stage1.json` | run3 阶段产物 |
| `cpp/neural_policy.hpp` | BC/RL 共用的 C++ MLP 推理类 |
| `cpp/adversarial_rl.cpp` | Adv-RL 的 C++17 实现 |
| `cpp/rl_policy_weights.hpp` | 最终从零 RL 的 C++ 权重头文件 |
| `cpp/rl_policy_weights_bc.hpp` | Adv-RL BC base 的 C++ 权重头文件 |
| `configs/adversarial_rl_policy_config*.json` | Adv-RL 保守/激进配置 |
| `docs/RL_POLICY.md` | RL 的 MDP、训练、结果与风险 |
| `configs/adversarial_policy_config*.json` | MLP 保守/激进配置 |
| `configs/adversarial_gru_policy_config*.json` | GRU 保守/激进配置 |
| `docs/NEURAL_POLICY.md` / `docs/GRU_POLICY.md` / `docs/ADVERSARIAL_POLICY.md` | 三种策略族的方法、接口、结果、风险 |
| `ipd/training/search/joint.py` | Final+Resp3 联合策略 |
| `ipd/training/search/multiobj.py` | 多目标训练与参数配置 |
| `ipd/training/search/bayes_priors.py` | Bayes 先验参数化 |
| `cpp/bayesplus_responsive3.cpp` | BayesPlus / Responsive3 的 C++17 实现 |
| `docs/BAYES_RESPONSIVE_FLOW.md` | 两套 C++ 策略的流程图 |

### 实验结果

| 文件 | 内容 |
|---|---|
| `docs/FINAL_COMPARISON.md` | 各测试集主表 |
| `docs/SUITE_RESULTS.md` | 9 个测试集结果 |
| `docs/OPPONENT_PERFORMANCE.md` | 85 类对手逐类型表现 |
| `docs/MATCHUP_RESULTS.md` | 早期策略对阵汇总 |
| `docs/ADVANCED_RESULTS.md` | 训练/优化/盲测全过程 |
| `results/benchmark.json` | 当前统一四分布结果（全部候选） |
| `results/blind_results.json` | 当前统一盲测结果（全部候选） |
| `results/matchup_triple_results.csv` | 主策略逐类型原始数据 |
| `results/final_benchmark.json` | 规则基线四分布数据 |
| `results/neural_benchmark.json` | MLP 策略四分布对比数据 |
| `results/neural_blind_results.json` | MLP 策略盲测排名数据 |
| `results/neural_training_history.json` | MLP 策略训练/验证曲线 |
| `results/gru_benchmark.json` | GRU-Net 四分布对比数据 |
| `results/blind_gru_results.json` | GRU-Net 盲测排名数据 |
| `results/gru_training_history.json` | GRU-Net 训练/验证曲线 |
| `results/adversarial_search_results.json` | 对抗调参搜索/验证记录 |
| `results/adversarial_benchmark.json` | Adv-MLP 保守版四分布 |
| `results/adversarial_aggressive_benchmark.json` | Adv-MLP 激进版四分布 |
| `results/adversarial_gru_benchmark.json` | Adv-GRU 保守版四分布 |
| `results/adversarial_gru_aggressive_benchmark.json` | Adv-GRU 激进版四分布 |
| `results/blind_adversarial_results.json` | Adv-MLP 保守版盲测 |
| `results/blind_adversarial_aggressive_results.json` | Adv-MLP 激进版盲测 |
| `results/blind_adversarial_gru_results.json` | Adv-GRU 保守版盲测 |
| `results/blind_adversarial_gru_aggressive_results.json` | Adv-GRU 激进版盲测 |
| `results/rl_training_history.json` | 最终从零 + hard 训练/验证曲线 |
| `results/rl_training_history_bc.json` | BC 初始化 + hard（Adv-RL base）曲线 |
| `results/rl_training_history_scratch_base.json` | 纯从零 base 曲线 |
| `results/rl_training_history_stage1.json` | run3 训练/验证曲线 |
| `results/rl_benchmark.json` | 最终从零 + hard 四分布结果 |
| `results/blind_rl_results.json` | 最终从零 + hard 盲测结果 |
| `results/rl_bc_benchmark.json` | BC 初始化 + hard 四分布结果 |
| `results/blind_rl_bc_results.json` | BC 初始化 + hard 盲测结果 |
| `results/rl_scratch_base_benchmark.json` | 纯从零 base 四分布结果 |
| `results/blind_rl_scratch_base_results.json` | 纯从零 base 盲测结果 |
| `results/adversarial_rl_benchmark.json` | Adv-RL 激进四分布 |
| `results/blind_adversarial_rl_results.json` | Adv-RL 激进盲测 |
| `results/adversarial_rl_conservative_benchmark.json` | Adv-RL 保守四分布 |
| `results/blind_adversarial_rl_conservative_results.json` | Adv-RL 保守盲测 |
| `results/rl_search_results.json` | RL 各轮 run 的配置与分数汇总 |
| `results/suite_results.json` | 新测试集数据 |
| `ipd/evaluation/blind.py` | 盲测代码（规则 + 学习策略） |
| `ipd/evaluation/benchmark.py` | 四分布统一测试代码 |
| `ipd/evaluation/suite.py` | 9 个测试集代码 |

---

## 5. 复现命令

### 快速自测

```bash
python3 -m ipd test
python3 -m ipd self-test --candidates all
python3 -m ipd blind --candidates core --fields 10 --seed 20261025
```

### 统一四分布

```bash
python3 -m ipd benchmark --candidates base --fields 20 --workers 24
```

### 神经策略

```bash
# 训练（需要 PyTorch；CPU 也能跑，但 GPU/NPU 更快）
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-train.txt
python3 -m ipd train neural --device auto --seed 20261025
python3 -m ipd train gru --device auto --seed 20261025
python3 -m ipd train adversarial --arch mlp --seed 20261026
python3 -m ipd train rl --device auto --iterations 100 \
  --dist-weights 1,1,1,2 --ent-coef 0.0001 --lr 2e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --out weights/rl_policy_weights_stage1.json \
  --checkpoint-out weights/rl_policy_checkpoint_stage1.json \
  --history-out results/rl_training_history_stage1.json

# 从零训练（最佳单一 RL 的第一阶段）
python3 -m ipd train rl --device auto --from-scratch \
  --iterations 30 --fields-per-dist 5 --n 80 \
  --dist-weights 1,1,1,2 --ent-coef 0.001 --lr 3e-4 \
  --center-advantages --select-metric avg \
  --out weights/rl_policy_weights_scratch_base.json \
  --checkpoint-out weights/rl_policy_checkpoint_scratch_base.json \
  --history-out results/rl_training_history_scratch_base.json

# 推理自检（不需要 PyTorch）
python3 -m ipd self-test --candidates neural --weights weights/neural_policy_weights.json
python3 -m ipd self-test --candidates gru --weights weights/gru_policy_weights.json
python3 -m ipd self-test --candidates adv-mlp --config configs/adversarial_policy_config.json
python3 -m ipd self-test --candidates rl --weights weights/rl_policy_weights.json
python3 -m ipd self-test --candidates rl --weights weights/rl_policy_weights_bc.json
python3 -m ipd self-test --candidates rl --weights weights/rl_policy_weights_scratch_base.json
python3 -m ipd self-test --candidates adv-rl --config configs/adversarial_rl_policy_config_aggressive.json

# 四分布与盲测（不需要 PyTorch）
python3 -m ipd benchmark --fields 20 --workers 20
python3 -m ipd benchmark --arch gru --fields 20 --workers 20
python3 -m ipd benchmark --arch adversarial --config configs/adversarial_policy_config_aggressive.json --fields 20 --workers 20
python3 -m ipd benchmark --arch rl --fields 20 --workers 20
python3 -m ipd benchmark --arch adv-rl --config configs/adversarial_rl_policy_config_aggressive.json --fields 20 --workers 20
python3 -m ipd blind --fields 10 --workers 12
python3 -m ipd blind --arch gru --fields 10 --workers 12
python3 -m ipd blind --arch adversarial --config configs/adversarial_policy_config_aggressive.json --fields 10 --workers 12
python3 -m ipd blind --arch rl --fields 10 --workers 12
python3 -m ipd blind --arch adv-rl --config configs/adversarial_rl_policy_config_aggressive.json --fields 10 --workers 12
```

### 9 个测试集

```bash
python3 -m ipd suite --fields 12 --workers 24
```

### 逐类型对阵

```bash
python3 -m ipd matchup --candidates final-tuned,resp3-tuned,bayes-tuned,hybrid,triple-majority
```

### C++ 编译

```bash
g++ -std=c++17 -O2 -DRESP_BAYES_DEMO cpp/bayesplus_responsive3.cpp -o demo
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

- `cpp/policies/final_policy.hpp`：Final（`FinalParams` 可切换 base/tuned），
  `cpp/ipd_strategy.cpp` 保留为演示入口；
- `cpp/policies/responsive3_policy.hpp` / `bayes_plus_policy.hpp`：
  Responsive3 + BayesPlus，无外部依赖；
- `cpp/policies/neural_policy.hpp` / `gru_policy.hpp` / `adversarial_rl_policy.hpp`：
  学习策略和 Adv-RL；
- `cpp/opponents/*.hpp`：演示/测试用对手，每个对手独立头文件；
- **目前没有 C++ 版 Triple-majority**。  
  如果比赛要求 C++ 且想提交 Triple，需要：
  1. 把 Final-tuned 也移植到 C++；
  2. 把 Final、Responsive3、BayesPlus 三个动作做多数投票；
  3. 保持三者各自独立状态。
- **C++ 版 RL-PPO 已提供**：`weights/rl_policy_weights.json` 与
  `cpp/rl_policy_weights.hpp`；编译时加 `-DNEURAL_POLICY_USE_RL_WEIGHTS`：
  `g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o rl_demo`。
  头文件里的 `SAFE` 常量与 Python checkpoint 的 `safe` 字段一致。
- **C++ 版 Adv-RL 已提供**：`cpp/adversarial_rl.cpp`。
  最终从零 RL base：
  `g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_demo`；
  BC base（对应 554.29 / 562.91）：
  `g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_BC_WEIGHTS -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_bc_demo`。
- **C++ 版 NeuralPolicy 已提供**：`cpp/neural_policy.cpp` + `cpp/neural_policy_weights.hpp`。
  重新训练权重后先运行 `python3 -m ipd export --arch mlp` 刷新头文件，再编译：
  `g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo`。
  C++ 与 Python 版做过 200 组随机历史的特征/概率对齐检查，最大差异约 `5e-10`。
- **C++ 版 GRU-Net 已提供**：`cpp/gru_policy.cpp` + `cpp/gru_policy_weights.hpp`。
  编译前运行 `python3 -m ipd export --arch gru`；命令：
  `g++ -std=c++17 -O2 -DGRU_POLICY_DEMO cpp/gru_policy.cpp -o gru_demo`。
  与 Python 版做过 60 组随机历史的逐步概率对齐，最大差异约 `5e-10`。
- 两种 C++ 策略共用 `cpp/ipd_features.hpp` 的 85 维特征提取。
- **Adv-MLP / Adv-GRU 目前只有 Python 版本**；**Adv-RL 已有 C++ 版本**
  （`cpp/adversarial_rl.cpp`）。如果比赛要求 C++ 且想提交 Adv-MLP，需要在
  `NeuralPolicy` 外层再加同样的两条剥削规则。

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
   `ipd/core/machine.py` / `cpp/policies/final_policy.hpp` 的默认参数仍是旧版；
   提交前使用 `FinalParams::tuned()` / `DEFAULT_FINAL_TUNED`，或明确选择默认版。

8. **Neural 依赖权重文件与训练数据分布**
   `NeuralPolicy` 必须加载 `weights/neural_policy_weights.json`；如果比赛只允许单文件，需要把权重内嵌。
   它是 Triple 的蒸馏学生，真实对手分布偏离训练分布时不保证超过 Triple。
   推理本身不需要 PyTorch，但训练需要。

9. **Neural 的 NPU 路径未实机验证**
   代码支持 `--device npu`，本轮只在 CUDA 上完成训练和测试。

10. **GRU-Net 是有状态的**
    每场必须调用 `reset()`；同一时间步不要重复调用 `act()`，否则隐藏状态会前进两次。
    它的优势相对 MLP 不显著，但纯 Python 推理更慢，C++ 权重也更大（约 192 KB）。

11. **两种神经策略都需要权重文件**
    `weights/neural_policy_weights.json` 或 `weights/gru_policy_weights.json`。单文件提交时需要内嵌。

12. **对抗版本的约束是分布级别，不是逐类型级别**
    Adv 激进版在 `MemOne-rand`、`Extort3`、`NoisyTFT.10/20` 等 wild 类型上可能
    掉几分；但四个分布和盲测总分均上涨。若要求逐类型不掉分，使用保守版或纯
    Neural/GRU。Adv-MLP/Adv-GRU 目前没有 C++ 版本；Adv-RL 有。

13. **RL 会放大尾部风险**
    最终 RL 在 `EarlyD1/10`、`Bully`、`Random.90`、`LateD` 上大幅超过 BC；
    `ProbeBot` 已从 run2 的 143.5 修复到 603，`EarlyD25` 基本持平。最大回退是
    `Rand3->TFT`（BC 约 554，最终 RL 约 347），`Extort3`、`Random.50/70` 也有
    小幅回退。如果比赛字段中这些类型比例异常高，优先 Adv-RL（BC base）或
    BC/GRU。

14. **RL 需要 checkpoint 或部署权重**
    最终单独 RL：`weights/rl_policy_checkpoint.json` + `weights/rl_policy_weights.json`；
    Adv-RL base：`weights/rl_policy_weights_bc.json`；纯从零 base：
    `weights/rl_policy_weights_scratch_base.json`。Adv-RL 还需要
    `configs/adversarial_rl_policy_config_aggressive.json`。单文件提交时需要
    内嵌对应 MLP 权重，或退回 Adv-MLP。默认 `safe=true`；只有显式训练
    `--allow-probe` 才会关闭主动探测限制。

---

## 8. 交接检查清单

- [ ] 确认比赛接口：函数签名、历史格式、是否允许自定义类和 reset。
- [ ] 确认是否允许每天运行三个子策略（Triple）。
- [ ] 确认是否允许提交权重文件；如果只允许单文件，决定是否把 Neural/GRU 权重内嵌。
- [ ] 确定提交版本：**RL-PPO（从零 + hard）** / **Adv-RL** / Adv-MLP / Neural /
      GRU-Net / Triple / Final-tuned / Final+Resp3。
- [ ] 如果提交 Neural/GRU/Adv/RL，先做无 PyTorch 自检：
      `python3 -m ipd self-test --candidates neural`、
      `python3 -m ipd self-test --candidates gru`、
      `python3 -m ipd self-test --candidates adv-mlp`、
      `python3 -m ipd self-test --candidates rl` 或
      `python3 -m ipd self-test --candidates adv-rl`。
- [ ] 如果提交 GRU/Adv-GRU，确认每场调用 `reset()`，且同一时间步只调用一次 `act()`。
- [ ] 如果提交 Adv/Adv-RL，确认权重和对应 `adversarial_*_config.json` 一起提交。
- [ ] 如果提交 C++，完成 Final-tuned 和 Triple 的 C++ 合并，或使用现有
      Neural/GRU/RL/Adv-RL C++ 版本（最终 RL 加 `-DNEURAL_POLICY_USE_RL_WEIGHTS`，
      Adv-RL BC base 加 `-DNEURAL_POLICY_USE_RL_BC_WEIGHTS`，两者都加
      `-DADVERSARIAL_RL_DEMO` 时启用 Adv-RL wrapper）。
- [ ] 固定随机种子。
- [ ] 用最终版本重新跑一次 `python3 -m ipd blind`，确认没有配置遗漏。
- [ ] 检查第 1 天和第 100 天行为。
- [ ] 保存最终提交文件、配置 JSON 和运行日志。
