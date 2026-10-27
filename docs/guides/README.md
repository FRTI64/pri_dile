# 指南

面向任务的 how-to 文档。命令均在仓库根目录执行；需要从其他目录运行时先
`pip install -e .`。

| 文档 | 适用场景 |
|---|---|
| [evaluation.md](evaluation.md) | 跑 Protocol v3 dev/test、legacy 四分布、suite、matchup、league |
| [training.md](training.md) | 训练 MLP/GRU/RL/self-play/opponent-model；复现 A60/E1c |
| [deployment.md](deployment.md) | Python 推理、C++ 编译、权重导出、单文件提交 |
| [split-evaluation.md](split-evaluation.md) | 为每个候选设置独立参数并拆分结果 |
| [axelrod.md](axelrod.md) | 可选 Axelrod 对手库（243/224/cheating） |

相关参考：

- 命令与参数：[../reference/cli.md](../reference/cli.md)
- 指标定义：[../reference/metrics.md](../reference/metrics.md)
- 配置与目录：[../reference/configuration.md](../reference/configuration.md)
