# GRU-Net 策略（GRUPolicy）

## 1. 一句话结论

在原有 MLP 神经策略之外，新增了一个 **GRU-Net** 版本：

- 输入仍是 85 维历史特征，但网络维护 32 维循环隐藏状态；
- 结构：`85 -> GRU(32) -> MLP(32) -> 1`，12,513 个参数；
- 训练：PyTorch，自动选择 CUDA / Ascend NPU / MPS / CPU；
- 推理：纯 Python 标准库 + 无依赖 C++17；
- 四分布 20 场平均 **547.81**，盲测均分 **555.22 / 平均排名 8.30**；
- 与 MLP 版同档：20 场略低 0.34 分，盲测略高 0.12 分。

GRU-Net 的价值不是“必然更强”，而是提供了另一种策略族：它可以通过隐藏状态
学习“惩罚中 / 测试中 / 剥削中”这类模式，而不完全依赖手工汇总特征。

> 另见 `docs/ADVERSARIAL_POLICY.md`：GRU-Net 也可以套用事后剥削模块。激进版四分布
> **551.69**、盲测 **559.05 / 8.10**，四个分布均不下降。

---

## 2. 与 MLP 版的区别

| | NeuralPolicy (MLP) | GRUPolicy (GRU-Net) |
|---|---|---|
| 输入 | 85 维手工特征 | 同样 85 维特征 |
| 状态 | 无状态 | 32 维隐藏状态，跨 100 天保留 |
| 结构 | 85→64→32→1 | 85→GRU(32)→32→1 |
| 参数量 | 7,617 | 12,513 |
| 训练损失 | 全部时间步 | 只训练网络实际控制的时间步 |
| 盲测均分 | 555.10 | **555.22** |
| 四分布均值 | **548.15** | 547.81 |

### 2.1 为什么对 GRU 使用掩码损失

推理层有三条固定规则：

1. 第 1 天合作；
2. 第 100 天背叛；
3. 对手从未背叛时合作。

这些动作不需要网络决定。因此 GRU 训练时只对满足以下条件的时间步计算 BCE：

```text
对手已经出现过 D
且 t > 0
且 t < 99
```

这样模型容量集中在真正需要判断的“对手首次背叛之后”的状态。
验证集上的掩码指标：

```text
BCE = 0.0803
动作一致率 = 97.21%
D 类 precision = 0.9667
D 类 recall = 0.9831
采样温度 = 1.00
```

注意：这个 97.21% 与 MLP 的 99.22% 不是同一批状态，不能直接横向比较；
MLP 的指标包含大量简单的“对手未背叛 → C”状态。

### 2.2 训练数据

与 MLP 版相同：

- 教师：`Triple-majority`；
- 行为克隆 + 1 轮 DAgger；
- 432,000 个时间步（4,320 场比赛 × 100 天）；
- 96,000 个独立验证时间步；
- 训练设备：CUDA RTX 5090 Laptop，`torch 2.14.1+cu130`。

DAgger 时学生会以自己的隐藏状态走棋，教师在同一历史上给出标签。

---

## 3. 测试结果

### 3.1 四分布统一测试

与 `ipd/evaluation/benchmark.py` 相同的 20 个赛场 × 299 人、种子 `20261026`：

| 策略 | 原始混合 | 末期收割 | 高 Grim | 宽+聪明 | 均值 | 最差分布 |
|---|---:|---:|---:|---:|---:|---:|
| **GRU-sample** | 549.37 | 542.74 | **550.21** | **548.93** | 547.81 | 542.74 |
| Neural-sample | **549.92** | **542.85** | 550.20 | 549.63 | **548.15** | **542.85** |
| Triple-majority | 549.21 | 542.54 | 549.96 | 548.32 | 547.51 | 542.54 |
| Final-tuned | 549.30 | 542.83 | 549.69 | 543.56 | 546.35 | 542.83 |

GRU 与 MLP 都在 547.5–548.2 区间，差距小于 0.4 分，属于噪声范围。

### 3.2 盲测

`ipd/evaluation/blind.py --arch gru`，10 个未参与训练/调参的 299 人赛场：

| 策略 | 均分 | 平均排名 | 第 1 名 | 前 10 | 最差 |
|---|---:|---:|---:|---:|---:|
| **GRU-sample** | **555.22** | **8.30** | 7/10 | 7/10 | 27 |
| Neural-sample (MLP) | 555.10 | 8.30 | 7/10 | 7/10 | 27 |
| Triple-majority | 552.52 | 9.20 | 6/10 | 7/10 | 33 |
| Final-tuned | 547.35 | 13.20 | 2/10 | 5/10 | 30 |

两者盲测排名完全相同，GRU 均分只高 0.12 分；不应据此断言 GRU 显著更强。

---

## 4. 使用

### 4.1 Python 推理

> 源码在 `ipd/`。在仓库根目录直接运行，或先执行 `pip install -e .`。

```python
from ipd.policies.gru import GRUPolicy

p = GRUPolicy("weights/gru_policy_weights.json", seed=20261024)
p.reset()                               # 每场开始必须调用

action = p.move(day, mine, theirs)      # day = 1..100
# 或
action = p.act(t, mine, theirs)         # t = 0..99
```

自检：

```bash
python3 -m ipd self-test --candidates gru --weights weights/gru_policy_weights.json
```

**重要：GRUPolicy 是有状态的。** 每场比赛必须调用 `reset()`；同一时间步不要
重复调用 `act()`，否则隐藏状态会前进两次，行为会偏离训练。

### 4.2 训练

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-train.txt

python3 -m ipd train gru --device auto \
    --fields-per-dist 12 --dagger-fields-per-dist 6 \
    --n 60 --epochs 40 --hidden 32 --layers 1
```

输出：

- `weights/gru_policy_weights.json`
- `results/gru_training_history.json`

### 4.3 测试

```bash
# 四分布
python3 -m ipd benchmark --arch gru --fields 20 --workers 20

# 盲测
python3 -m ipd blind --arch gru --fields 10 --workers 12

# 也可以把 GRU 加进原版脚本
python3 -m ipd benchmark --candidates base --neural --neural-arch gru --fields 20
python3 -m ipd blind --candidates classic --neural --neural-arch gru --fields 10
```

### 4.4 C++ 推理

```bash
python3 -m ipd export --arch gru
g++ -std=c++17 -O2 -DGRU_POLICY_DEMO cpp/gru_policy.cpp -o gru_demo
./gru_demo
```

demo 输出：

```text
features=85 hidden=32 layers=1
AllC     C= 99 D=  1 score= 604
AllD     C=  5 D= 95 score=  95
TFT      C= 99 D=  1 score= 604
Grim     C= 99 D=  1 score= 604
```

C++ 与 Python 版做过 60 组随机历史的逐步概率对齐，最大差异约 `5e-10`。
如果比赛只允许一个 `.cpp`，把 `cpp/gru_policy_weights.hpp` 和 `cpp/ipd_features.hpp`
的内容粘贴到 `cpp/gru_policy.cpp` 顶部即可。

注意：C++ 版目前只支持 **1 层 GRU**（默认配置）。Python 版支持多层；如果训练了
`--layers 2`，请使用 Python 版，或重新用 `--layers 1` 训练后再导出 C++。

---

## 5. 文件

| 文件 | 内容 |
|---|---|
| `ipd/training/gru.py` | GPU/NPU 训练（GRU + DAgger） |
| `ipd/policies/gru.py` | 纯 Python 推理 |
| `cpp/gru_policy.cpp` | C++17 推理（含 demo） |
| `weights/gru_policy_weights.json` | 训练好的权重（约 395 KB） |
| `cpp/gru_policy_weights.hpp` | C++ 权重（约 192 KB） |
| `ipd/tools/export_cpp.py` | 权重 JSON → C++ 头文件 |
| `results/gru_benchmark.json` | 四分布结果 |
| `results/blind_gru_results.json` | 盲测结果 |
| `results/gru_training_history.json` | 训练/验证曲线 |

---

## 6. 风险与注意事项

1. **有状态**：必须每场 `reset()`；不要对同一状态重复调用 `act()`。
2. **优势不显著**：盲测只比 MLP 高 0.12 分，四分布低 0.34 分；优先按实现复杂度
   和比赛限制选择，不要把 GRU 当成必然升级。
3. **纯 Python 推理更慢**：GRU 每步都要更新隐藏状态；对从不背叛的对手也会做
   循环计算。本机约 2.5 秒/100 场，仍可用于常规比赛规模。
4. **C++ 体积**：权重头文件约 192 KB，比 MLP 版更大。
5. **NPU 未实机验证**：训练代码支持 `--device npu`，本轮只在 CUDA 上训练。
6. **权重依赖**：GRU 版需要 `weights/gru_policy_weights.json`；单文件提交时需内嵌权重。
