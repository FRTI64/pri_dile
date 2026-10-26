# 完整强化学习策略（PPO）

本文记录 `ipd/training/rl.py` 的完整 RL 训练流程。它不是行为克隆，也不是
DAgger：策略直接以 100 天总分（每轮收益之和）为目标，通过 PPO 更新。

当前结果：最终单独 RL（从零 + hard 微调）四分布 **553.33**、盲测
**563.23 / 8.00**，是最高盲测分数，也是最好的单网络策略；Adv-RL（BC 初始化
base）四分布 **554.29**、盲测 **562.91 / 8.00**，是最高四分布分数。两者都超过
此前最好的 Adv-MLP（552.24 / 559.51）。

## 1. 为什么需要完整 RL

已有的 `NeuralPolicy` / `GRUPolicy` 是 **模仿学习**：

- 先让 `Triple-majority` 教师走棋；
- 用行为克隆（BC）拟合教师动作；
- 用一轮 DAgger 修正学生自己走出的状态。

这能复制教师，但不能超过教师，也不会主动发现教师没有利用的收益。完整 RL 把
“对手行为”当作环境的一部分，直接优化：

```text
最大化 E[ sum_{t=1..100} payoff(a_t, opponent_t) ]
```

因此它可以学到教师没有写死的规则，例如：

- 对 `EarlyD10` 这种“先背叛 10 轮、之后永远合作”的对手，持续剥削；
- 对 `Bully`、`Random.90` 等类型调整惩罚/原谅节奏；
- 在 `late` 和 `wide` 分布中提高末期收益。

## 2. MDP 定义

| 元素 | 定义 |
|---|---|
| 状态 | `neural_policy.extract_features()` 的 85 维历史特征 |
| 动作 | `C=0`、`D=1` |
| 奖励 | 当前轮收益 `payoff(a, b) / 100`（缩放只影响数值稳定性） |
| 回合 | 100 轮，`t=0..99` |
| 对手 | 从训练分布中采样的 80 个对手组成一个 field；每场独立 |
| 目标 | 最大化 100 轮收益总和 |

动作边界（默认，`safe=True`）：

```text
第 1 天 C
对手从未 D 时 C（不主动探测）
第 100 天 D（最后一天背叛是严格占优）
```

其余时间由网络输出 `P(D)` 并采样。`--allow-probe` 会去掉前两条，让 RL 自己学习
是否值得早期探测；默认保留，因为 `grim30` 中主动探测的尾部风险很大。

## 3. PPO 实现

### 3.1 网络

- 共享输入：85 维特征；
- **策略主干**：85 → 64 → 32（tanh）→ 1 个 logit；
- **价值主干**：独立 85 → 64 → 32（tanh）→ 1 个 value；
- 策略主干和价值主干分开，避免价值回归的大梯度污染策略表示；
- 部署时只导出策略主干 + 策略头，格式与 `neural_policy_weights.json` 完全一致；
- 策略网络 7,617 参数，actor-critic 合计 15,234 参数（与 BC MLP 同规模）。

### 3.2 目标函数

```text
L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    + 0.5 * MSE(V, return)
    - 0.0001 * entropy
    + 0.1 * KL(pi || pi_ref)
```

- `ratio = exp(log pi(a|s) - log pi_old(a|s))`；
- GAE：`gamma=1.0`、`lambda=0.95`（有限 100 轮，不做折扣）；
- **优势中心化**：每场减去该场优势均值，消除“对手是 AllC 还是 Grim”带来的
  方差，不改变最优策略；
- 价值头预测绝对回报，因此 GAE 仍然使用绝对 value；
- `pi_ref` 是初始 BC 策略（或上一阶段 RL checkpoint），KL 项是稳定性正则，
  不是行为克隆损失；
- 硬规则决定的动作（第 1 天、未探测、第 100 天）不进入策略损失，只用价值损失。

### 3.3 训练数据

默认四个官方分布等权：

```text
original, late, grim30, wide
```

实践中 `wide` 最接近盲测（`sample_wide(seed, n, 0.15)`），因此推荐：

```text
--dist-weights 1,1,1,2
```

`--train-dists ...,hard` 会加入一个 **hard** 混合，专门采样容易让 RL 过度剥削
的对手：

```text
ProbeBot, RemorsefulProber, Prober, Tester,
EarlyD1/10/25, Bully, Rand3->TFT, Random.90,
Extort1.5/3, MemOne-rand, NoisyTFT.20, Joss.20, Grim
```

hard 只用于训练，验证始终是四个官方分布。

`--train-dists ...,robust` 是 E1 新增的 **domain-randomized** 训练分布
（`ipd/evaluation/robust.py`）：Extort/Random/LateD/NoisyTFT/Joss/
DynamicWindow/Kamikaze/Harvest 的参数每次采样，另有 TFT/Grim/AllD/Bully/AllC
anchor。E1a/E1b 未过 Gate（held-out +1.36/+1.94）；E1c 从 E1b 出发用
`repair` 分布修复 random 尾部，held-out v2 +3.10、official +0.45、
blind +0.80，held-out v3 final +3.91，判 **Go candidate**。权重
`weights/rl_policy_weights_repair.json` 注册为 `rl-repair`，但主权重
未替换。详见 `docs/e1-robust-finetune.md`。

P3 新增 frozen policy pool 训练入口：
`--opponent-weights path1,path2` 让 RL learner 直接对 RL/Neural/GRU/SelfPlay
权重训练；`--opponent-fraction` 控制 official 分布与 frozen pool 的混合；
`--opponent-penalty λ` 把 reward 改成 `own - λ*opponent`，用于
anti-exploitation blue response。`--no-validation` 适合 red best response
（固定预算、不做官方分布选模）。详见 `docs/p3-red-league.md`。

### 3.4 并行与设备

- 24 个标准库 worker 生成 rollout（不导入 PyTorch）；
- 主进程在 CUDA / NPU / MPS / CPU 上做 PPO 更新；
- 每个 worker 返回打包的 float32 特征、动作、旧 logprob、value、reward、
  advantage、return；
- 每 5 个 iteration 在固定验证种子上评估，按 `--select-metric` 保存最佳
  checkpoint。

`--select-metric`：

| 值 | 公式 | 用途 |
|---|---|---|
| `balanced` | `0.75*avg + 0.25*worst` | 四分布均衡 |
| `avg` | `avg` | 只看四分布均值 |
| `wide` | `0.5*avg + 0.5*wide` | 偏向盲测分布 |
| `blind` | `0.4*avg + 0.4*wide + 0.2*worst` | 盲测优先且保留尾部 |

其他可选参数：

- `--freeze-policy-trunk`：只训练策略头和价值网络，做保守消融；
- `--save-all-checkpoints DIR`：保存每次验证的 checkpoint，便于离线挑选；
- `--allow-probe`：关闭“不主动探测”的安全规则；
- `--from-scratch`：不加载 BC 权重，从随机策略开始训练。

## 4. 训练命令

### 4.1 从 BC 策略开始（推荐）

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-train.txt

python3 -m ipd train rl \
  --device auto --workers 24 \
  --iterations 100 --val-every 5 \
  --fields-per-dist 6 --n 80 \
  --val-fields-per-dist 4 --val-n 100 \
  --dist-weights 1,1,1,2 \
  --ent-coef 0.0001 --lr 2e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --select-metric balanced \
  --out weights/rl_policy_weights_stage1.json \
  --checkpoint-out weights/rl_policy_checkpoint_stage1.json \
  --history-out results/rl_training_history_stage1.json
```

### 4.2 hard 对手微调

```bash
python3 -m ipd train rl \
  --device auto --workers 24 \
  --iterations 60 --val-every 5 \
  --fields-per-dist 5 --n 80 \
  --val-fields-per-dist 4 --val-n 100 \
  --init-weights weights/rl_policy_checkpoint_stage1.json \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 \
  --ent-coef 0.0001 --lr 1e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --select-metric blind \
  --out weights/rl_policy_weights_bc.json \
  --checkpoint-out weights/rl_policy_checkpoint_bc.json \
  --history-out results/rl_training_history_bc.json
```

阶段 1 把 run3 checkpoint 写到 `weights/rl_policy_checkpoint_stage1.json`；
阶段 2 用它初始化，输出 `weights/rl_policy_checkpoint_bc.json` 和
`weights/rl_policy_weights_bc.json`（Adv-RL 的 base）。最终单独 RL 的从零训练
流程见 4.3。

### 4.3 从零开始（最终单独 RL）

阶段 1：从随机策略训练 base（实际用 30 iter、CPU）：

```bash
python3 -m ipd train rl \
  --device cpu --workers 16 \
  --from-scratch --iterations 30 \
  --fields-per-dist 5 --n 80 \
  --ent-coef 0.001 --lr 3e-4 \
  --center-advantages \
  --dist-weights 1,1,1,2 --select-metric avg \
  --out weights/rl_policy_weights_scratch_base.json \
  --checkpoint-out weights/rl_policy_checkpoint_scratch_base.json \
  --history-out results/rl_training_history_scratch_base.json
```

阶段 2：hard 对手微调（实际用 40 iter、CPU）：

```bash
python3 -m ipd train rl \
  --device cpu --workers 16 \
  --iterations 40 --val-every 5 \
  --fields-per-dist 5 --n 80 \
  --val-fields-per-dist 3 --val-n 100 \
  --init-weights weights/rl_policy_checkpoint_scratch_base.json \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 \
  --ent-coef 0.0001 --lr 1e-4 --target-kl 0.02 \
  --center-advantages --bc-kl-coef 0.1 \
  --select-metric blind \
  --out weights/rl_policy_weights.json \
  --checkpoint-out weights/rl_policy_checkpoint.json \
  --history-out results/rl_training_history.json
```

### 4.4 允许主动探测

```bash
python3 -m ipd train rl \
  --device auto --workers 24 \
  --allow-probe --iterations 100 \
  --dist-weights 1,1,1,2 \
  --ent-coef 0.0001 --lr 2e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --out weights/rl_policy_weights_probe.json \
  --checkpoint-out weights/rl_policy_checkpoint_probe.json \
  --history-out results/rl_training_history_probe.json
```

## 5. 结果

### 5.1 统一四分布（20 个赛场 × 299 人）

| 策略 | 原始混合 | 末期收割 | 高 Grim | 宽+聪明 | 四分布均值 |
|---|---:|---:|---:|---:|---:|
| **Adv-RL 激进（BC base）** | **555.76** | **549.19** | **553.39** | **558.82** | **554.29** |
| Adv-RL 保守（BC base） | 556.10 | 549.25 | 553.43 | 556.16 | 553.73 |
| **RL-PPO（从零 + hard，最终）** | **555.88** | **548.41** | **552.09** | **556.95** | **553.33** |
| Adv-MLP 激进 | 552.64 | 548.26 | 553.33 | 554.72 | 552.24 |
| RL-PPO（从零 base） | 553.73 | 547.55 | 552.22 | 553.99 | 551.87 |
| RL-PPO（run4 BC + hard） | 554.08 | 544.29 | 550.34 | 553.30 | 550.50 |
| RL-PPO（run3 wide + KL） | 551.66 | 543.81 | 550.28 | 552.44 | 549.55 |
| Neural MLP（BC） | 549.92 | 542.85 | 550.20 | 549.63 | 548.15 |
| Triple-majority | 549.21 | 542.54 | 549.96 | 548.32 | 547.51 |

### 5.2 盲测（10 个未见的 300 人赛场）

| 策略 | 均分 | 平均排名 | 第 1 名 | 前 10 | 最差 |
|---|---:|---:|---:|---:|---:|
| **RL-PPO（从零 + hard，最终）** | **563.23** | **8.00** | **7/10** | **7/10** | **25** |
| **Adv-RL 激进（BC base）** | **562.91** | **8.00** | **7/10** | **7/10** | **25** |
| Adv-RL 保守（BC base） | 561.05 | 8.20 | 7/10 | 7/10 | 26 |
| Adv-MLP 激进 | 559.51 | 8.10 | 7/10 | 7/10 | 25 |
| RL-PPO（从零 base） | 559.27 | 8.30 | 7/10 | 7/10 | 27 |
| RL-PPO（run4 BC + hard） | 558.60 | 8.30 | 7/10 | 7/10 | 27 |
| RL-PPO（run3 wide + KL） | 557.95 | 8.30 | 7/10 | 7/10 | 27 |
| Neural MLP（BC） | 555.10 | 8.30 | 7/10 | 7/10 | 27 |
| Triple-majority | 552.52 | 9.20 | 6/10 | 7/10 | 33 |

最终单独 RL 比 BC 高约 **8.1 分盲测、5.2 分四分布**；Adv-RL 比 Adv-MLP 高
3.4 分盲测、2.05 分四分布。也就是说，**完整 RL 已经超过了此前最好的 Adv-MLP
候选**，而且单网络版本还拿到了最高盲测。

### 5.3 Adv-RL：RL + 约束剥削

Adv-RL 直接复用 `ipd/policies/adversarial.py` 的两条规则，base 使用
`weights/rl_policy_weights_bc.json`（run4 BC 初始化 + hard 微调）：

- 对手首次 D 在 `late_day` 之后，且下一回合继续 D → 永久 D；
- `n_c>=min_c, n_d>=min_d` 且 `pD-pC<=gap` → 永久 D。

激进配置：

```text
gap=0.2, min_c=12, min_d=2, late_day=40, repair=true
```

两条规则都要求对手已经先 D，因此不会触发 Grim。配置：

- `configs/adversarial_rl_policy_config_aggressive.json`；
- `configs/adversarial_rl_policy_config.json`（保守版）。

把规则套到最终 scratch-hard 单独 RL 上：保守版 553.73 / 562.79，激进版四分布
553.08，均不如 BC base；因此正式 Adv-RL 配置仍使用
`weights/rl_policy_weights_bc.json`。

### 5.4 逐类型行为

最终 RL（scratch-hard）与 BC 的主要差异（8 次重复，固定种子）：

| 对手 | BC | RL（最终） | Adv-RL（BC base） | 说明 |
|---|---:|---:|---:|---|
| EarlyD10 | 557.0 | 905.0 | 893.0 | RL 学会持续剥削“先 D 后永久 C” |
| EarlyD1 | 602.0 | 954.0 | 882.0 | 同上 |
| Bully | 594.0 | 882.0 | 747.0 | 对方退缩后继续施压 |
| Random.90 | 789.0 | 849.0 | 832.0 | 利用高合作率随机对手 |
| ProbeBot | 593.0 | 603.0 | 598.0 | hard 挖掘后不再灾难性掉分 |
| EarlyD25 | 770.0 | 774.0 | 758.0 | 基本持平 |
| Extort1.5 / Extort3 | 470.0 / 202.0 | 462.0 / 181.0 | 475.0 / 229.0 | 轻度回退 |
| **Rand3->TFT** | **554.0** | **347.0** | **516.0** | **最大尾部风险** |

等类型均值（85 类对手等权，单次重复会抖动）：

```text
BC               541.8
Adv-MLP          546.6
RL（最终）       558.4
Adv-RL（BC base）558.9
```

结论：完整 RL 的收益主要来自“发现 BC 没有利用的非响应型对手”（EarlyD1/10、
Bully、Random.90、LateD），代价是 `Rand3->TFT` 这种“随机开局后转 TFT”的对手
可能被过度惩罚。hard 挖掘把 `ProbeBot` 从 run2 的 143.5 修复到 603，但
`Rand3->TFT` 仍是最大尾部风险；如果比赛字段中该类型比例异常高，应使用
Adv-RL（BC base）或 BC/GRU。

### 5.5 从零训练消融

为了验证这不是“BC 换皮”，用 `--from-scratch` 从随机策略开始训练，完全不加载
BC 权重（只复用固定特征标准化）。两个阶段：

| 阶段 | iteration | 四分布均值 | 盲测均分 | 盲测排名 |
|---|---:|---:|---:|---:|
| 从零 base | 30 | 551.87 | 559.27 | 8.30 |
| 从零 + hard 微调（最终） | 40 | **553.33** | **563.23** | **8.00** |
| BC 初始化 + hard（run4） | 60 | 550.50 | 558.60 | 8.30 |

从零训练只用了约 1/3 的 iteration 和更少的 field，就已经超过 BC 初始化版本；
加上 hard 微调后，单独 RL 的盲测达到 563.23，超过所有 Adv 版本。这说明 PPO
确实在利用奖励信号学习，而不是复刻教师。

权重：

- 从零 base：`weights/rl_policy_weights_scratch_base.json`；
- 最终从零 + hard：`weights/rl_policy_weights.json`；
- BC 初始化 + hard：`weights/rl_policy_weights_bc.json`（Adv-RL base）。

把 Adv-RL 规则套到最终从零模型上，保守版四分布 553.73、盲测 562.79，激进版
四分布 553.08，均不如 BC base 的 Adv-RL（554.29 / 562.91）；因此正式 Adv-RL
配置仍使用 `weights/rl_policy_weights_bc.json`。

## 6. 导出与 C++ 推理

训练输出两个文件：

| 文件 | 内容 |
|---|---|
| `weights/rl_policy_checkpoint.json` | actor-critic checkpoint（策略 + 价值） |
| `weights/rl_policy_weights.json` | 可直接部署的 MLP（策略头） |

部署 MLP 与 `NeuralPolicy` 完全兼容：

```bash
python3 -m ipd self-test --candidates rl --weights weights/rl_policy_weights.json
```

生成 C++ 头文件：

```bash
python3 -m ipd export --arch mlp \
  --weights weights/rl_policy_weights.json \
  --out cpp/rl_policy_weights.hpp

python3 -m ipd export --arch mlp \
  --weights weights/rl_policy_weights_bc.json \
  --out cpp/rl_policy_weights_bc.hpp

g++ -std=c++17 -O2 \
  -DNEURAL_POLICY_USE_RL_WEIGHTS -DNEURAL_POLICY_DEMO \
  cpp/neural_policy.cpp -o rl_demo
./rl_demo
```

`cpp/neural_policy.cpp` 会根据头文件里的 `SAFE` 常量决定是否执行“不主动探测”
规则。RL 权重与 BC 权重使用同一套 85 维特征和 3 层 MLP 格式。

Adv-RL（RL + 两条规则）的 C++ 版本：

```bash
# 最终从零 RL base（四分布 553.08 / 保守版 553.73）
g++ -std=c++17 -O2 \
  -DNEURAL_POLICY_USE_RL_WEIGHTS -DADVERSARIAL_RL_DEMO \
  cpp/adversarial_rl.cpp -o adv_rl_demo
./adv_rl_demo

# BC 初始化 base（对应 Python 的 554.29 / 562.91）
g++ -std=c++17 -O2 \
  -DNEURAL_POLICY_USE_RL_BC_WEIGHTS -DADVERSARIAL_RL_DEMO \
  cpp/adversarial_rl.cpp -o adv_rl_bc_demo
./adv_rl_bc_demo
```

`cpp/rl_policy_weights.hpp` 是最终从零模型，`cpp/rl_policy_weights_bc.hpp`
是 Adv-RL 的 BC 初始化 base。

## 7. 评估命令

```bash
# 四分布，20 个赛场
python3 -m ipd benchmark --arch rl \
  --weights weights/rl_policy_checkpoint.json \
  --fields 20 --workers 20

# 正式盲测
python3 -m ipd blind --arch rl \
  --weights weights/rl_policy_checkpoint.json \
  --fields 10 --workers 12

# argmax 对照
python3 -m ipd benchmark --arch rl \
  --weights weights/rl_policy_checkpoint.json \
  --neural-mode argmax --fields 20 --workers 20

# Adv-RL 激进（config 的 base_weights 会解析到 rl_policy_weights_bc.json）
python3 -m ipd benchmark --arch adv-rl \
  --config configs/adversarial_rl_policy_config_aggressive.json \
  --fields 20 --workers 20
python3 -m ipd blind --arch adv-rl \
  --config configs/adversarial_rl_policy_config_aggressive.json \
  --fields 10 --workers 12
```

## 8. 风险与限制

1. **没有真实 300 人比赛数据**  
   训练和评估都来自合成对手库；盲测只是分布内未见种子。

2. **RL 会放大尾部风险**  
   最终 scratch-hard 在 `Rand3->TFT` 上约 347，而 BC 约 554，单类型回退很大；
   `Extort3`、`Random.50/70`、部分 `Periodic/Alternator` 也有小幅回退。
   hard 挖掘已经把 `ProbeBot` 从 143.5 修复到 603，`EarlyD25` 基本持平。
   如果比赛字段中 `Rand3->TFT` 比例异常高，应使用 Adv-RL（BC base）或 BC/GRU。

3. **最高盲测与最高四分布不是同一个模型**  
   最终单独 RL 盲测 **563.23 / 8.00**，是最高盲测；Adv-RL（BC base）四分布
   **554.29**，是最高四分布、盲测 562.91。两者差距很小，按比赛评分方式选择。

4. **权重文件必需**  
   部署需要 `rl_policy_weights.json`；单文件提交时需要内嵌权重。

5. **训练随机性**  
   PPO 对种子敏感；所有 run 使用 `--seed 20261030`，评估种子固定为
   benchmark 20261026 / blind 20261025。

6. **`--allow-probe` 尚未证明有收益**  
   主动探测在 `grim30` 中风险很大；默认不启用。

7. **设备与 NPU**  
   run2/3/4 在 CUDA 上训练；最终从零 + hard 在 CPU 上完成。设备选择支持 NPU，
   但本轮未实机验证。

## 9. 文件

| 文件 | 内容 |
|---|---|
| `ipd/training/rl.py` | PPO 训练、rollout、验证、导出 |
| `ipd/policies/rl.py` | 纯 Python actor-critic 推理与部署格式导出 |
| `ipd/policies/neural.py` | 部署 MLP 推理（支持 `safe` 字段） |
| `ipd/evaluation/benchmark.py` | 四分布 benchmark，`--arch rl` |
| `ipd/evaluation/blind.py` | 盲测，`--arch rl` |
| `weights/rl_policy_checkpoint.json` | 最终（从零 + hard）actor-critic checkpoint |
| `weights/rl_policy_weights.json` | 最终（从零 + hard）部署 MLP |
| `weights/rl_policy_checkpoint_bc.json` / `weights/rl_policy_weights_bc.json` | run4 BC 初始化 + hard（Adv-RL base） |
| `weights/rl_policy_checkpoint_scratch_base.json` / `weights/rl_policy_weights_scratch_base.json` | 纯从零 base（30 iter） |
| `weights/rl_policy_checkpoint_stage1.json` / `weights/rl_policy_weights_stage1.json` | run3 阶段 checkpoint / 权重 |
| `cpp/neural_policy.hpp` | MLP 推理类（BC/RL 共用） |
| `cpp/adversarial_rl.cpp` | Adv-RL 的 C++17 实现 |
| `cpp/rl_policy_weights.hpp` | 最终从零 RL 的 C++ 权重 |
| `cpp/rl_policy_weights_bc.hpp` | Adv-RL BC base 的 C++ 权重 |
| `configs/adversarial_rl_policy_config*.json` | Adv-RL 保守/激进配置（BC base） |
| `results/rl_training_history.json` | 最终从零 + hard 训练/验证曲线 |
| `results/rl_training_history_bc.json` | run4 BC 初始化训练曲线 |
| `results/rl_training_history_scratch_base.json` | 纯从零 base 训练曲线 |
| `results/rl_training_history_stage1.json` | run3 训练/验证曲线 |
| `results/rl_benchmark.json` | 最终从零 + hard 四分布结果 |
| `results/blind_rl_results.json` | 最终从零 + hard 盲测结果 |
| `results/rl_bc_benchmark.json` | run4 BC 初始化四分布结果 |
| `results/blind_rl_bc_results.json` | run4 BC 初始化盲测结果 |
| `results/rl_scratch_base_benchmark.json` | 纯从零 base 四分布结果 |
| `results/blind_rl_scratch_base_results.json` | 纯从零 base 盲测结果 |
| `results/adversarial_rl_benchmark.json` | Adv-RL 激进四分布 |
| `results/blind_adversarial_rl_results.json` | Adv-RL 激进盲测 |
| `results/adversarial_rl_conservative_benchmark.json` | Adv-RL 保守四分布 |
| `results/blind_adversarial_rl_conservative_results.json` | Adv-RL 保守盲测 |
| `results/rl_search_results.json` | 各轮 RL 的配置与分数汇总 |
