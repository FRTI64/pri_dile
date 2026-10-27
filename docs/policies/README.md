# 策略方法

每个策略族一篇方法文档，包含原理、接口、训练、结果和风险。

| 文档 | 策略 | 当前状态 |
|---|---|---|
| [rl.md](rl.md) | PPO / 85 维 MLP | 主权重、A60、RL-Repair 的基础 |
| [neural.md](neural.md) | MLP BC + DAgger | 可部署；排名低于 RL 系列 |
| [gru.md](gru.md) | GRU-Net | 有状态；排名低于 RL 系列 |
| [adversarial.md](adversarial.md) | 约束剥削 wrapper | Adv-MLP/GRU/RL；anti-exploitation 未解决 |
| [selfplay.md](selfplay.md) | 21 维非对称自博弈 | 最终 BC 蒸馏；改进 No-Go |

统一接口、候选注册和目录分层见
[../architecture/overview.md](../architecture/overview.md)。当前候选排名见
[../../README.md](../../README.md) 和 [../operations/handover.md](../operations/handover.md)。
