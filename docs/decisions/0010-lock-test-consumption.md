# ADR 0010：永久锁定 test 消费

- 状态：Accepted（2026-10-27）
- 关联：`configs/splits/protocol_v4/test_policy.json`、
  `ipd/evaluation/protocol.py`、`ipd/evaluation/protocol_cli.py`

## 背景

Protocol v4 的 `test_v7` 和 `test_red` 是一次性 held-out 集。之前的
`--no-consume` 是每次运行手动选择的，容易因为漏传参数而误消费；一旦
usage ledger 写入，test 集就失去独立性，只能重建 `test_v8`。

## 决策

1. 用 `configs/splits/protocol_v4/test_policy.json` 保存消费策略，固定
   `consume_test=false`。
2. `protocol eval` / `protocol red` 在 test tier 上读取该策略；锁定后自动
   进入 provisional 模式，不调用 `guard_test_usage` / `record_usage`。
3. CLI 不提供解锁参数；`--acknowledge-once` / `--force` 不能覆盖锁定，
   只打印警告。policy 缺失、不可读或格式错误时 fail closed。
4. provisional 结果写入 `*_test_provisional.json` / `*_provisional.json`，
   并标记 `usage_consumed=false`、`selection_eligible=false`、
   `test_consumption_locked=true`。

## 后果

- 任何误传参数的 CI / 烟测都不会消耗 test 集。
- `test_v7` / `test_red` 只能用于诊断，不能作为 final、排名或选模依据；
  若未来确实需要 final，必须先显式修改 policy 文件并重新审计。
- `results/protocol_v4/usage.json` 不再因 test 运行新增记录。
- 文档、CLI 帮助和测试需同步说明“锁定”语义。
