# ADR 0013：Protocol v5 重划 train / dev / test

- 状态：Accepted（2026-10-30）
- 关联：`ipd/evaluation/protocol.py`、`configs/splits/protocol_v5/`、
  `docs/reference/protocol-v5.md`
- 取代：Protocol v4 作为当前主口径（v4 进入 legacy）

## 背景

Protocol v4 的 `test_v7` 已在 ADR 0012 中解禁，继续在 v4 上叠加新候选
会让 test 复用并失去独立性。需要一套新的 train/dev/test 分配，同时
保留 v4 的 catalog、赛场规模和选模流程。

catalog 没有新增策略类型；因此 v5 不是“新数据”，而是**同一 catalog 的
新划分**。

## 决策

1. 保留 977-config catalog、224 个 Axelrod 策略 + 15 个 AxelrodMix、
   14 个 complex family，以及 2048 人 / dev 30 fields / test 50 fields。
2. 用 `SPLIT_SALT = "protocol_v5"` 改变 family-local shuffle；405/977
   个 config 的 tier 与 v4 不同。
3. 新 final 为 `test_v8`；v4 的 `test_v7`、v4 manifest 和全部 v4 结果
   冻结为 legacy。
4. `test_v8` 与 `test_red` 初始都锁定；需要 final 时显式修改
   `configs/splits/protocol_v5/test_policy.json`，再用
   `--acknowledge-once` 消费并写 `results/protocol_v5/usage.json`。
5. 训练侧新增 `train_v5` profile；`train_v4` / `train_v3` 保留为
   历史别名，解析到当前 v5 train。
6. CLI 默认协议、manifest、usage ledger、结果路径全部切到
   `protocol_v5`；旧 v4 路径不再写入。

## 后果

- v5 必须重新跑完整 dev baseline 和 final，不能沿用 v4 排名。
- v5 与 v4 共用 catalog，因此 v5 `test_v8` 对旧 v4 候选**不是**
  held-out；新候选必须从 v5 train 重新训练/选模。
- `test_v8` 消费后，下一候选需要 `test_v9` 或重新审计扩充分层。
- v4 manifest / `test_v7` 结果仍保留在仓库中，只用于历史复现，不能与
  v5 数字混表。
