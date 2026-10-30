# ADR 0012：解禁 test_v7 一次性消费

- 状态：Accepted（2026-10-30；v4 legacy，v5 见 ADR 0013）
- 关联：`configs/splits/protocol_v4/test_policy.json`、
  `ipd/evaluation/protocol.py`、`ipd/evaluation/protocol_cli.py`
- 取代：[ADR 0010](0010-lock-test-consumption.md) 的“永久锁定”部分

## 背景

ADR 0010 从 2026-10-27 起把 `test_v7` / `test_red` 永久锁定，所有 test
运行都只能是 provisional。锁定期间没有写入
`results/protocol_v4/usage.json`，因此 `test_v7` 本身尚未被消费。

当前需要为最终候选做一次真正的 held-out final。继续把 `test_v7` 当作
provisional 会让 final 与选模脱节；同时 blue-red 已停用（ADR 0011），
`test_red` 不再需要解禁。

## 决策

1. `test_policy.json` 增加按 tier 的 `consume_tiers`：
   `test_v7=true`、`test_red=false`；`consume_test` 保留为全局 fallback，
   仓库内固定为 `false`（fail closed）。
2. `test_consumption_allowed(tier=...)` 优先读取 `consume_tiers`；未列出的
   tier 才回退到 `consume_test`。缺失、不可读、格式错误的 policy 仍然
   视为锁定。
3. `protocol eval --tiers test_v7` 现在必须显式传 `--acknowledge-once`
   才会消费；`--no-consume` 继续提供 provisional 诊断，不写 ledger。
4. 消费写入 `results/protocol_v4/usage.json`。同一候选再次运行需要
   `--force`，新记录标记 `contaminated=true`；`--force` 不能绕过锁定。
5. `test_red` 继续锁定；`protocol red --tier test_red` 仍只能 provisional。

## 后果

- 可以对最终候选运行一次真正的 `test_v7` final；结果标记
  `usage_consumed=true`、`selection_eligible=true`。
- `test_v7` 一旦消费，后续新候选需要新的 held-out `test_v8`（或重新审计
  扩充分层）；不能再把已消费的 `test_v7` 当作独立 final。
- provisional 诊断仍可随时用 `--no-consume` 复现，且不污染 ledger。
- 文档、CLI 帮助和测试同步从“永久锁定”改为“按 tier 解禁 +
  一次性消费”语义。
