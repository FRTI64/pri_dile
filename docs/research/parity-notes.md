# Parity notes：Plan 3 规则批处理

本文件记录 Plan 3 原生规则 bank 的已知差异、RNG 契约和 fallback。

## 1. RNG 契约

- 每个对手对象在构建后保存 `rng.getstate()`（CPython MT19937，625 个
  int：624 个 state word + index）。
- `rules.cpp` 复刻 CPython 3.14 的：
  - `genrand_uint32`（twist/temper）；
  - `random()`（两次 32-bit 输出组合为 53-bit double）；
  - `getrandbits(k)`（`genrand_uint32() >> (32-k)`）；
  - `_randbelow(n)`（`n.bit_length()` 位拒绝采样）。
- `rb_reset_all` 恢复构建时快照，而不是调用 Python `reset()`；因此
  field 复用、多个 candidate 复用同一 bank 时与 Plan 1 的
  `snapshot_field_rng`/`restore_field_rng` 语义一致。
- 所有支持的类型——包括随机规则——在固定 seed 下逐动作一致；测试
  `test_fast_opponents.ParityTests` 覆盖 719 个 config × 100 轮。

## 2. 已修复的 RNG 共享 bug

在 `ipd/evaluation/protocol.py` 中，以下复合规则把 `rng` 作为
`HarvestFactor` 的第 4 个位置参数传入：

```python
HarvestFactor(start, repairable, p_defect, rng)
```

`HarvestFactor.__init__` 的第 4 个位置参数是 `last_d`，不是 `rng`。结果：

- `last_d` 被设为一个 truthy 的 `Random` 对象（行为等同 True）；
- 子规则获得一个新的、OS-seeded 的 `random.Random()`；
- `VotingPortfolio` 的 harvest 成员与 adaptive 成员使用不同 RNG 流，
  旧结果不可复现。

受影响位置：`PhaseSwitcher`、`AdaptiveMix`、`ProbeExploit`、`MetaMix`、
`VotingPortfolio`。已统一改为 `rng=rng` 关键字。因为这些规则本身是
概率性的，且旧实现没有稳定基线，本次修复不改变任何确定性候选的结果；
它让随机结果可复现，并使原生 bank 能逐位匹配 Python。

## 3. Fallback（保留原 Python 对象）

| 类型 | 原因 |
|---|---|
| `RandomFSM` | 每场随机生成转移表；当前未编码进 C ABI |
| `ProbeBot` | 复杂前缀状态机；train/dev 少量，未移植 |
| `Axelrod` / `AxelrodMix` | 外部库类级状态；保持 CPU 隔离 |
| `PolicyFinal` / `PolicyTriple` | 随机规则策略，Plan 2 决定保留 CPU |
| `PolicyRL` / `PolicyNeural` / `PolicyGRU` / `PolicySelfPlay` | Plan 2 GPU/CPU policy 路径 |

Fallback 行由 `RuleBank` 逐行调用原对象，RNG 和 `reset()` 行为不变。
`by_opponent` 聚合、field 顺序、名字都不变。

## 4. 未覆盖的实例

`test_v7` 按权重：

- 规则实例覆盖 99.4%；
- 非 Axelrod 实例（含 policy，policy 由 Plan 2 加速）覆盖 99.5%；
- 唯一未覆盖的规则 family 是 `RandomFSM`（0.44%）。

## 5. 小 batch / 浮点

- 规则 bank 不涉及浮点推理，只有标量概率比较；没有 float32/float64
  边界问题。
- B<128 时 ctypes 每轮调用开销大于原生循环收益；`VectorEnv` 默认
  不启用，`BatchOpponentField auto` 在 fast 行 <90% 时也不启用。
