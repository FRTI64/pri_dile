# ADR 0004: No-Go 实验归档，不删除原始记录

- **Status:** Accepted
- **Date:** 2026-10-27
- **Deciders:** IPD project

## Context

项目经历了 E0–E2、P3、P2 C1–C3、SelfPlay 等多轮实验，其中大量结论是
No-Go。早期文档逐渐出现：

- 旧结果与新结果混在同一页；
- “当前最佳”被后续候选覆盖，但历史表格仍被当成当前排名；
- 候选数量从 17 增到 20，66 个权重变体的扫描结果分散在多个文件；
- README 同时承担总览、教程、结果表和历史记录，越来越难维护。

## Decision

- 采用分层文档：`architecture/`、`guides/`、`reference/`、`policies/`、
  `research/`、`operations/`、`history/`、`decisions/`；
- No-Go 实验保留在 `research/`，原始表格不删除；
- 过期结果移到 `history/`，顶部加“历史快照”和替代链接；
- 当前结论只写在根 README、handover、reference 和 architecture；
- 每个实验的报告仍放在 `results/<experiment>/README.md`，保留原始数据。

## Consequences

- 旧链接需要更新，历史 checksum 对应的路径会失效；
- 新读者可以按“指南/参考/研究/历史”选择阅读深度；
- 文档不再把历史数字误当当前排名；
- 需要持续维护 `docs/README.md` 和 `CHANGELOG.md`。

## References

- [../README.md](../README.md)
- [../history/README.md](../history/README.md)
- [../../CHANGELOG.md](../../CHANGELOG.md)
