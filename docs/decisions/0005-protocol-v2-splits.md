# ADR 0005：用 Protocol v2 取代混用的 official / heldout 口径

- 状态：Accepted（2026-10-27）
- 相关：`docs/reference/protocol-v4.md`、`ipd/evaluation/protocol.py`

## 背景

旧评估同时存在四种事实上的 split：

1. `official` 四分布既是训练分布又是报告分布；
2. `blind` 只换 seed；
3. `heldout_v2` 被多个实验反复用于选模；
4. `final` suite 被多次 A/B/C 使用。

这导致“official 提升”不能区分拟合与泛化，且 validation 已经泄漏。
继续在旧表上比较候选，会把选择偏差当成策略改进。

## 决策

1. 以 **对手 config**（family + type + 参数 + 行为 seed）为划分单位，
   不再只换 field seed。
2. 固定四层：`train` / `dev_param` / `dev_family` / `test_v5`。
3. 参数化 family 按 70/15/15 做 family-local 分层划分；train/dev/test
   都覆盖完整参数范围，但不共享具体 config。
4. `dev_family` 和 `test_v5` 额外保留训练中未出现的 family。
5. `test_v5` 由 60% seen-family test + 25% test-only + 15% anchors 组成，
   使用 manifest hash 和 usage ledger 保证一次性。
6. 旧 `official` / `blind` 结果保留为 legacy，不再作为“held-out”证据；
   新候选必须重新跑 dev，只有 top 3 可以消费 test_v5。

## 后果

- 旧 `official` / `blind` 数字不能与新表混用；所有候选需要重新建立
  baseline（本次选择：20 候选 dev + 前 3 test）。
- 训练默认入口从四分布改为 `train_v2`；validation 使用
  `dev_param` / `dev_family`。
- 任何 catalog 改动都会改变 manifest hash，必须重新审计并记录。
- 短期成本：需要重跑 baseline；长期收益：泛化结论可审计、validation
  不再被反复消费。
