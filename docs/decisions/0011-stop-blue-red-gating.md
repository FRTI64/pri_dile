# ADR 0011: 停止用 blue-red 对战作为评估/ promotion 门槛

- Status: Accepted
- Date: 2026-10-29
- Deciders: 项目维护者

## Context

项目历史上把 blue-vs-red exploitability 作为 anti-exploitation 证据：

- E4 / P3 red pool：red 对蓝方接近 100% 胜率；
- PSRO / PSRO-scratch / PSRO-v4：red_win 可以到 0.0，但 dev_param
  大幅下降；
- 多轮 red 训练、PSRO oracle、gating/MoE 实验成本高，且没有转化为
  主评估分数的稳定提升；
- 用户明确决定：**以后不再考虑 blue-red 对战**。

## Decision

1. 未来的候选比较、选模和 promotion 不再使用 blue-red /
   exploitability / `dev_red` / `test_red` 作为门槛或证据。
2. 不再新增 red 训练、red pool、PSRO anti-exploitation 实验；
   已有代码、权重、结果保留用于历史复现，不删除。
3. 现有 `psro` / `psro-br` / `psro-scratch` / `psro-scratch-br` /
   `psro-v4` / `psro-v4-br` 继续注册为历史实验候选，但 red 指标
   不再影响推荐；它们的主指标只有 Protocol v4 dev/test。
4. 新的推荐/ promotion 依据改为：
   - Protocol v4 `dev_param` / `dev_family` / `dev_complex` /
     `dev_axelrod` / `dev_ood`；
   - `dev_level0..4` / `dev_complexity0..4` 分级诊断；
   - 独立 seed 复现；
   - Python / C++ parity 和部署检查；
   - 外部比赛结果（如有）。
5. `test_v7` / `test_red` 的锁定策略不变；`test_v7` 仍只做
   provisional 诊断，不参与选模。

## Consequences

- 文档中的 red / anti-exploitation 段落标记为 legacy；不再作为
  “未解决风险”或 promotion 阻塞项。
- PSRO 相关候选保留，但推荐表不再以 anti-exploitation 为卖点。
- 后续 session 不应再安排 red / blue-red 任务；除非用户明确重新开启。
- 历史结果仍在 `results/` 和 `docs/research/`，可用于复现，但不再
  参与当前决策。
