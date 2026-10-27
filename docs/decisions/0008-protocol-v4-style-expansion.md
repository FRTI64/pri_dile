# ADR 0008：Protocol v4 新增风格集与旧 family 扩展

- 状态：Accepted（2026-10-27）
- 关联：`docs/reference/protocol-v4.md`、`ipd/evaluation/protocol.py`

## 背景

Protocol v4 已有 6 个 complex family，但风格仍集中在 phase/adaptive/probe
一类；旧 family 的参数网格也还有空档。`test_v7` 尚未消费，可以在不破坏
baseline 的情况下扩展 catalog。

## 决策

1. 新增 6 个行为风格 family：
   - `noise_adaptive`：对孤立噪声宽恕、对持续背叛惩罚；
   - `reputation_grim`：按累计合作率进入/退出 Grim；
   - `contrition_cycle`：自己背叛后道歉/修复；
   - `endgame_hunter`：末局前 TFT，末局收割或抢先惩罚；
   - `pattern_breaker`：检测短周期并打反周期；
   - `conditional_extort`：只剥削高合作率对手。
2. 为旧 family 增加新对手：
   - nice / forgiving / punitive / defect / late / random / wild /
     adaptive / extort / temporal / noisy。
3. catalog 从 597 扩到 **747 configs**；`test_v7` 的 complex 份额现在覆盖
   12 个 family。
4. 继续使用 2048 人 / dev 30 fields / test 50 fields；`test_v7` 未消费。

## 后果

- v4 baseline 仍需重跑；旧 v3 数字只作 legacy。
- `test_v7` 已包含新风格，消费后下一候选需要 `test_v8`。
- catalog 更大，evaluation 成本上升；但 `test_v7` 仍是 top 3 一次性运行。
