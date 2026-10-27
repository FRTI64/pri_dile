# ADR 0007：Protocol v4 接入 Axelrod 并扩大赛场

- 状态：Accepted（2026-10-27）
- 关联：`docs/reference/protocol-v4.md`、`ipd/evaluation/protocol.py`

## 背景

Protocol v3 的 532 个 config 仍以本地规则/策略为主，缺少 Axelrod 的经典、
进化、ZD 和元策略；默认 299 人 / 20 fields / 40 fields 的规模也不足以覆盖
更大的对手库。v3 的 `test_v6` 已被消费。

## 决策

1. 新建 **Protocol v4**；v3 的 manifest、`test_v6` 和 baseline 保留为 legacy。
2. 增加 50 个精选 **Axelrod 4.14.0** 策略和 15 个 **AxelrodMix** 组合：
   - `AxelrodMix` 每场从 2–4 个 Axelrod 策略中抽一个；
   - 组合覆盖 TitForTat/Grudger/Adaptive、ZD、Meta、Evolved 等。
3. 新增 `train_axelrod` / `dev_axelrod` / `test_axelrod` 分组；
   `train_v4` 和 `test_v7` 按比例混入 Axelrod。
4. 默认赛场扩大：**2048 人**，dev **30 fields**，test **50 fields**，
   red **30 seeds**。
5. final 改为 `test_v7`；Axelrod 为可选依赖，缺失时 audit 给 warning，
   实际评估报清晰错误。

## 后果

- v3 的 299/20/40 baseline 不能当作 v4 排名；v4 baseline 需要另行运行。
- 完整 v4 tier 需要 `requirements-axelrod.txt`；manifest 不依赖安装。
- Axelrod Meta 策略较慢；日常可用精选策略/AxelrodMix，完整 bank 只做定向分析。
- `test_v7` 消费后，下一候选需要 `test_v8`。
