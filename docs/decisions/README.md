# 架构决策记录（ADR）

记录高成本、不可逆或容易反复争论的工程决策。格式参考 Michael Nygard 的 ADR：
**Context → Decision → Consequences**。

## 索引

| ADR | 决策 | 状态 |
|---|---|---|
| [0001](0001-frozen-league-evaluation.md) | 用冻结 league + held-out 作为主评估 | Accepted |
| [0002](0002-no-promotion-without-reproduction.md) | 未独立复现不得替换主权重 | Accepted |
| [0003](0003-anti-exploitation-unsolved.md) | 不把 red 训练收益当成 anti-exploitation | Accepted |
| [0004](0004-archive-no-go-research.md) | No-Go 实验归档而非删除 | Accepted |
| [0005](0005-protocol-v2-splits.md) | 用 Protocol v2 取代混用的 official / heldout 口径 | Accepted |
| [0006](0006-protocol-v3-complex-sets.md) | Protocol v3 扩大并复杂化 train/dev/test | Accepted |
| [0007](0007-protocol-v4-axelrod-scale.md) | Protocol v4 接入 Axelrod 并扩大赛场 | Accepted |
| [0008](0008-protocol-v4-style-expansion.md) | Protocol v4 新增风格集与旧 family 扩展 | Accepted |
| [0009](0009-protocol-v4-level-complexity.md) | Protocol v4 按思考等级/复杂度划分对手集 | Accepted |
| [0010](0010-lock-test-consumption.md) | 永久锁定 test_v7 / test_red 消费 | Superseded by 0012 |
| [0011](0011-stop-blue-red-gating.md) | 停止用 blue-red 对战作为评估/promotion 门槛 | Accepted |
| [0012](0012-unlock-test-v7.md) | 解禁 test_v7 一次性消费；test_red 继续锁定 | Accepted（v4 legacy） |
| [0013](0013-protocol-v5-resplit.md) | Protocol v5 重划 train/dev/test；新 final test_v8 | Accepted |

## 新 ADR 模板

```markdown
# ADR NNNN: 标题

- Status: Proposed | Accepted | Superseded by ADR-XXXX
- Date: YYYY-MM-DD
- Deciders: ...

## Context
为什么需要决策？有哪些约束和备选？

## Decision
选了什么？边界是什么？

## Consequences
正面/负面后果、后续工作、需要更新的文档。
```
