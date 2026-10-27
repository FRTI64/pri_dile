# 运维与交接

面向“接手、提交、复现、promotion”的操作文档。

| 文档 | 内容 |
|---|---|
| [handover.md](handover.md) | 当前结论、评估协议、权重、实验账本、风险、检查清单 |
| [next-session.md](next-session.md) | 下一 session 的起始 prompt、任务优先级、数据纪律 |

相关入口：

- 评估指南：[../guides/evaluation.md](../guides/evaluation.md)
- 训练指南：[../guides/training.md](../guides/training.md)
- 部署指南：[../guides/deployment.md](../guides/deployment.md)
- 决策记录：[../decisions/](../decisions/)

**Promotion 规则：** 未独立复现、未确认 C++ 导出/第 1/100 天行为/checksum 前，
不要替换 `weights/rl_policy_weights.json`。详见 handover。
