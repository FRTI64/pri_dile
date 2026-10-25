# GPU/NPU 训练的神经策略（NeuralPolicy）

## 1. 一句话结论

新增了一个由 GPU/NPU 训练的神经网络策略 `NeuralPolicy`：

- 训练：PyTorch，自动选择 CUDA / Ascend NPU / MPS / CPU；
- 推理：**纯 Python 标准库**，不需要安装 PyTorch；
- 结构：85 维历史特征 → 64 → 32 → 1 的 MLP，约 7.6k 参数；
- 一个权重文件即可提交，不需要像 Triple 那样每天运行三个子策略；
- 在本次 20 场四分布测试中平均 548.15，略高于 Triple-majority 的 547.51；
- 在 10 个完全未参与训练/调参的 300 人盲测赛场中均分 555.10、平均排名 8.30。

这不是“手工规则换皮”：策略的动作概率由神经网络输出，训练数据通过行为克隆 +
DAgger 从现有最强规则策略生成，网络参数全部由 GPU 优化得到。

> 另见 `docs/GRU_POLICY.md`：同一套特征和训练流程也可以训练 **GRU-Net** 版本
> （85→GRU(32)→32→1）。GRU 版盲测 555.22 / 8.30，四分布 547.81；
> 与本文的 MLP 版同档，按需选择。
>
> 另见 `docs/ADVERSARIAL_POLICY.md`：在 MLP/GRU 之上加入事后剥削模块，并做约束对抗
> 调参。激进版四分布 **552.24**、盲测 **559.51 / 8.10**，四个分布均不下降。

---

## 2. 方法

### 2.1 教师与训练目标

默认教师是 `Triple-majority`（Final-tuned + Responsive3-tuned + BayesPlus-tuned
多数投票）。对每个访问到的状态：

1. 计算 85 维特征；
2. 查询教师策略在当前历史上的动作（C/D）；
3. 用二分类交叉熵训练网络预测 `P(D)`。

第一轮只使用教师自己走出的状态；随后执行 DAgger：

- 用当前网络与对手对局；
- 在**网络实际访问到的状态**上重新查询教师；
- 把新样本加入训练集，重新训练。

本轮配置生成了 432,000 个训练样本（教师 288k + DAgger 144k）和 96,000 个
独立验证样本。最终验证集：

```text
BCE = 0.0222
动作一致率 = 99.22%
D 类 precision = 0.9675
D 类 recall = 0.9825
采样温度 = 0.95
```

### 2.2 网络与特征

特征全部由历史 `mine/theirs` 计算，不读取对手源码或身份，包括：

- 当前天数、剩余天数、终局窗口；
- 双方最近 8 步动作；
- 总体/近 3/5/10/20 步合作率；
- `pC = P(对手 D | 我上步 C)`、`pD = P(对手 D | 我上步 D)`；
- 样本量、响应差距、是否已有足够证据；
- 连续 C/D 长度、交替率；
- 四种联合动作（CC/CD/DC/DD）的总体与近 10 步频率；
- 对手首次/最近 D 的时间、我方首次/最近 D 的时间、响应延迟；
- 当前比分、上一步收益、最近是否互相背叛；
- 近 10 步条件响应概率。

网络：

```text
85 -> Linear(64) -> tanh -> Linear(32) -> tanh -> Linear(1) -> sigmoid
```

推理时输出 `P(D)`，默认按概率采样，与教师的随机原谅行为一致。也可以切换到
`argmax` 模式，但在本项目的测试中 `sample` 更好。

### 2.3 安全规则

为了不因为蒸馏误差在开局就触发 Grim 类永久惩罚，推理层加了三条规则：

1. 第 1 天合作；
2. 第 100 天背叛；
3. **对手从未背叛过时合作**。

第 3 条与 Final/Responsive3/BayesPlus 的“不主动探测”一致：在对手第一次 D
之前，无法区分 AllC、TFT、Grim 和末期收割者，提前背叛的期望收益为负。

---

## 3. 设备支持

`ipd/training/neural.py --device auto` 按以下顺序选择：

1. Ascend NPU（安装 `torch_npu` 后）；
2. NVIDIA CUDA；
3. Apple MPS；
4. CPU。

本轮实际训练设备：

```text
cuda (NVIDIA GeForce RTX 5090 Laptop GPU)
torch 2.14.1+cu130
```

NPU 路径使用同一套 `torch.device("npu")` 代码，但本轮没有 Ascend 实机，
因此未做实机验证；首次在 NPU 上运行时建议先用小样本跑通。

安装与训练：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-train.txt
# Ascend NPU 还需要与 torch 版本匹配的 torch_npu
# pip install torch_npu

# CUDA / NPU / CPU 自动选择
python3 -m ipd train neural --device auto \
    --fields-per-dist 12 --dagger-fields-per-dist 6 \
    --n 60 --epochs 40 --workers 20

# 指定 NPU
python3 -m ipd train neural --device npu

# 指定 CPU
python3 -m ipd train neural --device cpu --epochs 10
```

训练脚本读取：

- `results/train_multiobj_results.json`（Final/Responsive3 调参结果）；
- `results/train_bayes_priors_results.json`（Bayes 先验调参结果）。

输出：

- `weights/neural_policy_weights.json`：策略权重（约 243 KB）；
- `results/neural_training_history.json`：训练曲线、验证指标、教师配置。

---

## 4. 测试结果

### 4.1 四分布统一测试

与 `ipd/evaluation/benchmark.py` 相同的 20 个赛场 × 299 人、相同种子
（`20261026`），每格是每对手平均分：

| 策略 | 原始混合 | 末期收割 | 高 Grim | 宽+聪明 | 四分布均值 | 最差分布 |
|---|---:|---:|---:|---:|---:|---:|
| Final-tuned | 549.30 | 542.83 | 549.69 | 543.56 | 546.35 | 542.83 |
| Resp3-tuned | 547.69 | 542.49 | 549.84 | 543.77 | 545.95 | 542.49 |
| Bayes-tuned | 550.19 | 546.23 | **554.16** | 526.24 | 544.21 | 526.24 |
| Triple-majority | 549.21 | 542.54 | 549.96 | 548.32 | 547.51 | 542.54 |
| **Neural-sample** | **549.92** | **542.85** | **550.20** | **549.63** | **548.15** | **542.85** |

Neural 在四个分布上都略高于 Triple，但差距只有约 0.6 分，属于同档水平；
主要优势是**单策略、单权重文件、推理零依赖**，而不是压倒性强度。

另外对 85 类对手各 16 局做了一次逐类型抽查：Neural 等权平均 548.46，
Triple 547.35，Final 551.08。Neural 最弱的类型与 Triple 基本一致
（AllD 约 95、Random.05 约 134、Extort3 约 180），没有出现新的灾难性对手类型。

### 4.2 盲测

`ipd/evaluation/blind.py` 使用 10 个全新种子、未参与训练/调参的 299 人赛场，
并复用对手之间的循环赛结果以降低计算量：

| 策略 | 均分 | 平均排名 | 第 1 名 | 前 10 | 最差 |
|---|---:|---:|---:|---:|---:|
| Final-tuned | 547.35 | 13.20 | 2/10 | 5/10 | 30 |
| Resp3-tuned | 548.42 | 11.60 | 3/10 | 6/10 | 34 |
| Bayes-tuned | 527.67 | 65.80 | 0/10 | 0/10 | 100 |
| Triple-majority | 552.52 | 9.20 | 6/10 | 7/10 | 33 |
| **Neural-sample** | **555.10** | **8.30** | **7/10** | **7/10** | **27** |

注意：这里的基线数字与 `HANDOVER.md` 中旧盲测（Triple 554.99 / 8.30）略有
差异，因为 `ipd/evaluation/blind.py` 复用了对手循环赛，并对每个候选使用深拷贝的
同一初始赛场；这是为了在同等计算量下比较多个候选。跨脚本的绝对分数不要直接
混用，趋势和相对差距可以参考。

统一盲测入口为 `python3 -m ipd blind`；该入口复用了对手循环赛，因此绝对分数不能与更早的独立盲测结果直接混用。

---

## 5. 推理接口

> 源码在 `ipd/`。在仓库根目录直接运行，或先执行 `pip install -e .`。

```python
from ipd.policies.neural import NeuralPolicy

p = NeuralPolicy("weights/neural_policy_weights.json", seed=20261024)
p.reset()                               # 每场开始调用

# 比赛接口：day 从 1 到 100，mine/theirs 为之前的历史
action = p.move(day, mine, theirs)

# 或使用与本项目其他策略一致的 act 接口（t 从 0 开始）
action = p.act(t, mine, theirs)
```

动作约定与其他文件一致：`C = 0`，`D = 1`。

快速自检：

```bash
python3 -m ipd self-test --candidates neural --weights weights/neural_policy_weights.json
```

如果比赛必须提交 C++，已经提供无依赖的 C++17 版本：

- `cpp/neural_policy.cpp`：MLP 推理 + 可选 demo；
- `cpp/ipd_features.hpp`：与 GRU-Net 共用的 85 维特征提取；
- `cpp/neural_policy_weights.hpp`：由 `ipd/tools/export_cpp.py` 从 JSON 生成；
- 动作接口：`act(t, mine, theirs)` 或 `move(day, mine, theirs)`；
- 构造函数第二个参数为 `false` 时使用 argmax，不再按概率采样。

```bash
python3 -m ipd export --arch mlp     # 重新训练后刷新权重头文件
g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo
./neural_demo
```

C++ 实现已与 Python 版做过 200 组随机历史的特征/概率对齐检查，最大差异约
`5e-10`。demo 对 AllC/TFT/Grim 均为 604 分，对 AllD 为 95 分，与 Python 版一致。

如果比赛只允许提交一个 `.cpp` 文件，可以把 `cpp/neural_policy_weights.hpp` 和
`cpp/ipd_features.hpp` 的内容直接粘贴到 `cpp/neural_policy.cpp` 顶部，替换对应的
`#include`。

---

## 6. 复现命令

```bash
# 训练（需要 PyTorch）
. .venv/bin/activate
python3 -m ipd train neural --device auto --seed 20261025

# 四分布统一测试（不需要 PyTorch）
python3 -m ipd benchmark --fields 20 --n 299 --workers 20

# 也可以直接把 Neural 加进现有统一测试
python3 -m ipd benchmark --candidates base --fields 20 --workers 24 --neural

# 盲测排名（不需要 PyTorch）
python3 -m ipd blind --fields 10 --n 299 --workers 12

# 或把 Neural 加进原版 ipd/evaluation/blind.py（会跑完整 300 人循环赛，较慢）
python3 -m ipd blind --candidates classic --fields 10 --neural

# argmax 对照
python3 -m ipd benchmark --fields 8 --neural-mode argmax

# 生成 C++ 权重头文件并编译 demo
python3 -m ipd export --arch mlp
g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo
./neural_demo
```

结果文件：

| 文件 | 内容 |
|---|---|
| `requirements-train.txt` | 训练依赖（仅 PyTorch） |
| `weights/neural_policy_weights.json` | 可直接加载的权重 |
| `cpp/neural_policy.cpp` | C++17 推理实现（含 demo） |
| `cpp/neural_policy_weights.hpp` | 由 JSON 生成的 C++ 权重 |
| `ipd/tools/export_cpp.py` | 权重 JSON → C++ 头文件 |
| `results/neural_training_history.json` | 训练/验证曲线与元数据 |
| `results/neural_benchmark.json` | 四分布对比结果 |
| `results/neural_blind_results.json` | 盲测排名结果 |

---

## 7. 风险与后续

1. **学生不会稳定超过教师**  
   Neural 的动作来自教师蒸馏；本次比 Triple 高约 0.6 分，不能保证在真实比赛
   中复现。若真实对手分布与训练分布差异很大，优先考虑 Triple/Final。

2. **随机性**  
   默认 `sample` 模式按 `P(D)` 采样；提交时应固定 `seed`，并重跑一次盲测确认。

3. **权重文件是必需产物**  
   比赛接口如果只允许提交单个源码文件，需要把权重内嵌到 Python 文件，或改回
   规则策略。

4. **纯 Python 推理速度**  
   本机约 1.5 秒/100 场；单场 100 天约 15 ms，正常比赛规模足够。若需要更快，
   可以直接使用已提供的 C++ 版本，或减小隐藏层/量化权重。

5. **NPU 未实机验证**
   设备选择和训练代码已支持 `torch_npu`，但本轮只在 CUDA 上完成训练。

6. **C++ 源码体积**
   `cpp/neural_policy_weights.hpp` 约 120 KB。如果比赛限制源码大小，可以减小隐藏层、
   量化权重，或只提交 Python 版本。

---

## 8. 后续：完整强化学习

BC + DAgger 只能复制教师。`ipd/training/rl.py` 用 PPO 直接优化比赛收益，
最终从零 + hard 的单独 RL 盲测 563.23 / 8.00；加 `AdversarialPolicy` 后四分布
554.29、盲测 562.91。详见 `docs/RL_POLICY.md`。
