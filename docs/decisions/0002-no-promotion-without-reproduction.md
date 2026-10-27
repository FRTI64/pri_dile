# ADR 0002: 未独立复现不得替换主权重

- **Status:** Accepted（test 消费部分由 ADR 0010 补充）
- **Date:** 2026-10-27
- **Deciders:** IPD project

## Context

A60（`rl-hard60`）和 E1c（`rl-repair`）在多个指标上超过当前主权重
`weights/rl_policy_weights.json`。但：

- 主权重是当前提交 fallback，替换会影响所有下游评估和 C++ 导出；
- 新候选的提升主要来自 held-out 和有限 seed，仍可能过拟合；
- 训练随机性、policy RNG、held-out 复用都会造成假阳性；
- 比赛提交是一次性的，错误 promotion 的成本高于继续保留候选。

## Decision

新候选只有满足以下条件才允许替换主权重：

1. 用独立 seeds 复现 Protocol v4 的 `dev_param` / `dev_family` /
   `dev_complex` / `dev_ood`；test 消费已由 ADR 0010 永久锁定，
   provisional test 不能作为独立 final；
2. 确认 C++ 导出、第 1/100 天行为、安全规则和权重 checksum；
3. 不出现 family 崩塌或 anti-exploitation 明显退化；
4. 得到明确确认，并更新 `README.md`、`handover.md`、checksum 和报告。

在此之前：

- RL-Repair 只注册为 `rl-repair`，不覆盖主权重；
- A60 作为 `rl-hard60` fallback；
- 文档中“最佳候选”和“主权重”必须分开写。

## Consequences

- 当前提交默认仍是 RL-PPO；
- RL-Repair 的排名和 held-out 结果保留，但不等于已发布；
- promotion 是一个显式工程事件，而不是训练脚本的副作用。

## References

- [../operations/handover.md](../operations/handover.md)
- [../research/e1-robust-finetune.md](../research/e1-robust-finetune.md)
- `results/robust_20261026/README.md`
