# ADR 0001: 用冻结 league + held-out 作为主评估

- **Status:** Accepted
- **Date:** 2026-10-26
- **Deciders:** IPD project

## Context

早期选模只依赖官方四分布和 10-field blind。随着候选从规则策略扩展到
RL / Adv / SelfPlay，出现了三个问题：

- 不同实验使用不同 fields、seed 和 policy seed，数字不可比；
- 只看均值会掩盖 family 崩塌和尾部风险；
- 蓝军池 cross-play 退化为互相合作，无法测量被针对性剥削的风险。

## Decision

从 E0 开始，候选比较的主口径是 **冻结 league**：

- 固定候选 spec、权重路径、suite、seed 和 policy seed；
- official / blind 使用 20 fields × 299；
- 报告 `min_norm_family`、CVaR10、per-family、E4 red；
- 新候选必须通过 validation，再用一次性 held-out final test；
- 旧 `final` suite 因反复使用，降级为 validation。

标准 CLI 仍保留，用于快速烟测和独立复现，但不作为唯一选模依据。

## Consequences

- 数字更稳、可复现，但评测成本更高；
- 不能把 10-field、标准 CLI、league 的数字混表；
- 每次新实验必须记录协议和 seed；
- `heldout_v2` 后来也作为 validation 使用，`heldout_v3` 已被 E1c 消耗，
  后续 final test 需要新建 `heldout_v4`。

## References

- [../research/e0-league.md](../research/e0-league.md)
- [../reference/metrics.md](../reference/metrics.md)
- `results/e0_league_20261026/README.md`
