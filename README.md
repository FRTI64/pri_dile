# 100 天囚徒困境策略项目

本项目实现并比较了多类 100 天迭代囚徒困境（IPD）策略：规则策略、贝叶斯策略、
多数投票融合策略，以及可导出到 C++ 的 MLP / GRU-Net 神经策略和约束对抗剥削版本。

当前推荐版本和完整实验结论见 **[HANDOVER.md](HANDOVER.md)**。

## 目录结构

```text
.
├── src/                  Python 源码：策略、对手库、训练、评估、C++ 导出
├── cpp/                  C++17 推理实现、共用特征头文件和生成的权重头文件
├── configs/              对抗策略配置（需与 weights/ 中的权重配对使用）
├── weights/              训练好的 MLP / GRU 权重（JSON）
├── results/              基准测试、盲测、训练曲线和搜索记录（JSON / CSV）
├── docs/                 方法说明、结果表、流程图
├── tests/                结构与策略接口冒烟测试
├── HANDOVER.md           交接文档：推荐方案、参数、风险、检查清单
└── requirements-train.txt
```

需要读写项目文件的 Python 脚本通过 `src/paths.py` 解析路径，因此既可以在仓库
根目录运行，也可以从其他目录运行：

```bash
python3 src/blind_test.py --fields 10 --seed 20261025
```

如果在自己的代码中 `import` 这些模块，需要把 `src/` 加入模块搜索路径：

```bash
PYTHONPATH=src python3 -c "from neural_policy import NeuralPolicy"
```

## 快速开始

### 冒烟测试

```bash
python3 -m unittest discover -s tests -v
```

### 规则策略与盲测

```bash
python3 src/ipd_strategy.py
python3 src/blind_test.py --fields 10 --seed 20261025
python3 src/final_benchmark.py --fields 20 --workers 24
python3 src/run_suite_benchmark.py --fields 12 --workers 24
```

### 神经策略推理（不需要 PyTorch）

```bash
python3 src/neural_policy.py weights/neural_policy_weights.json
python3 src/gru_policy.py weights/gru_policy_weights.json
python3 src/adversarial_policy.py configs/adversarial_policy_config.json
```

### 训练（需要 PyTorch）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt

python3 src/train_neural_policy.py --device auto
python3 src/train_gru_policy.py --device auto
python3 src/train_adversarial_policy.py --arch mlp --seed 20261026
```

### 四分布与盲测

```bash
python3 src/benchmark_neural_policy.py --fields 20 --workers 20
python3 src/benchmark_neural_policy.py --arch gru --fields 20 --workers 20
python3 src/benchmark_neural_policy.py --arch adversarial \
  --config configs/adversarial_policy_config_aggressive.json --fields 20 --workers 20

python3 src/blind_test_neural.py --fields 10 --workers 12
python3 src/blind_test_neural.py --arch adversarial \
  --config configs/adversarial_policy_config_aggressive.json --fields 10 --workers 12
```

### C++ 推理

```bash
# MLP / GRU（权重头文件已生成在 cpp/）
g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo
g++ -std=c++17 -O2 -DGRU_POLICY_DEMO cpp/gru_policy.cpp -o gru_demo
./neural_demo
./gru_demo

# BayesPlus / Responsive3
g++ -std=c++17 -O2 -DRESP_BAYES_DEMO cpp/bayesplus_responsive3.cpp -o demo
./demo
```

重新训练权重后，刷新 C++ 头文件：

```bash
python3 src/export_neural_cpp.py   # weights/neural_policy_weights.json -> cpp/neural_policy_weights.hpp
python3 src/export_gru_cpp.py      # weights/gru_policy_weights.json    -> cpp/gru_policy_weights.hpp
```

## 结果摘要

统一四分布测试（20 个赛场 × 299 人）：

| 策略 | 四分布均值 | 盲测均分 / 平均排名 |
|---|---:|---:|
| **Adv-MLP 激进** | **552.24** | **559.51 / 8.10** |
| Adv-GRU 激进 | 551.69 | 559.05 / 8.10 |
| Adv-MLP 保守 | 551.62 | 557.44 / 8.30 |
| GRU-Net | 547.81 | 555.22 / 8.30 |
| Neural MLP | 548.15 | 555.10 / 8.30 |
| Triple-majority | 547.51 | 552.52 / 9.20 |
| Final-tuned | 546.35 | 549.39 / 10.40 |

不同盲测脚本的绝对分数不可直接混用；详细说明见 `HANDOVER.md`。

## 文档索引

| 文档 | 内容 |
|---|---|
| [HANDOVER.md](HANDOVER.md) | 交接总览、推荐方案、参数、风险、检查清单 |
| [docs/IPD_README.md](docs/IPD_README.md) | 全部模块和实验脚本清单 |
| [docs/NEURAL_POLICY.md](docs/NEURAL_POLICY.md) | MLP 策略：训练、接口、结果、风险 |
| [docs/GRU_POLICY.md](docs/GRU_POLICY.md) | GRU-Net 策略 |
| [docs/ADVERSARIAL_POLICY.md](docs/ADVERSARIAL_POLICY.md) | 约束对抗剥削规则与调参 |
| [docs/FINAL_COMPARISON.md](docs/FINAL_COMPARISON.md) | 各测试集主表 |
| [docs/SUITE_RESULTS.md](docs/SUITE_RESULTS.md) | 9 个测试集结果 |
| [docs/OPPONENT_PERFORMANCE.md](docs/OPPONENT_PERFORMANCE.md) | 85 类对手逐类型表现 |
| [docs/ADVANCED_RESULTS.md](docs/ADVANCED_RESULTS.md) | P0/P1/P2 完整过程 |
| [docs/MATCHUP_RESULTS.md](docs/MATCHUP_RESULTS.md) | 早期对阵汇总 |
| [docs/BAYES_RESPONSIVE_FLOW.md](docs/BAYES_RESPONSIVE_FLOW.md) | BayesPlus / Responsive3 C++ 流程图 |

## 注意事项

- `src/ipd_strategy.py` / `cpp/ipd_strategy.cpp` 的默认参数是早期版本，不是 Final-tuned。
- GRU 是有状态策略：每场必须 `reset()`，同一时间步只调用一次 `act()`。
- 所有候选策略都应在第 100 天选择背叛。
- 对抗策略目前只有 Python 版；C++ 版需要在外层补充两条剥削规则。
