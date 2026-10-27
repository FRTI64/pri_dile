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
