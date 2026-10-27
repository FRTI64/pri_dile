# ADR 0009：Protocol v4 按思考等级/复杂度划分对手集

- 状态：Accepted（2026-10-27）
- 关联：`docs/reference/protocol-v4.md`、`ipd/evaluation/protocol.py`

## 背景

Protocol v4 已有 12 个 complex family（加入 lookahead/portfolio 后为 14）
和 66 个 curated Axelrod 策略（后续扩展到 224 个 fast 策略），但缺少按
“对手有多聪明/多复杂”组织的评估口径。只按 family 看，无法回答候选在不同
thinking level 和策略复杂度上的强弱。

## 决策

1. 给每个 catalog entry 增加 `thinking_level`（0–4）和 `complexity`
   （0–4）元数据。
2. 新增两个高阶 local family：
   - `lookahead`：估计对手合作率并做 2–3 步前瞻；
   - `portfolio`：对 TFT/Grim/Extort/Adaptive 等成员投票。
3. 增加 Axelrod 高阶快策略：Winner12/21、EvolvedHMM5/FSM/ANN noise、
   PSOGambler、KnowledgeableWorseAndWorse、ThueMorse、MEM2、ShortMem 等；
   不引入 long-run MetaWinner/MetaMajority。
4. 提供派生对手集 `dev_level0..4` / `dev_complexity0..4`，以及训练
   profile `train_level0..4` / `train_complexity0..4`。
5. catalog 扩到 **977 configs**；`test_v7` 仍未消费。

## 后果

- 可以单独评估某个思考等级或复杂度，而不只看 family 混合。
- catalog 更大，但 `test_v7` 仍只给 top 3 一次性使用。
- v4 baseline 尚未运行；下一候选消费 `test_v7` 后需要 `test_v8`。
