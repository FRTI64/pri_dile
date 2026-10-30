# ADR 0014: 推荐 RL-FullGuard-C，保留 RL-PPO 主权重

- Status: Accepted
- Date: 2026-10-30
- Deciders: 项目用户 + maintainer

## Context

Protocol v5 dev baseline 的共同弱点是 `probe_exploit`（min_norm ~0.934）；
随后 `late` 又成为旧 `min_norm ≥ 0.99` 门槛的最后一个瓶颈。

`rl-full-guard` / `rl-full-guard-c` 用反应式 guard 修复了这两个窗口：

- `rl-full-guard`（V5-FT 基座）：dev 五项 min_norm 全部 ≥ 1.0，
  dev_param 533.04 / 532.43，test provisional 496.36 / 496.40；
- `rl-full-guard-c`（V5-FT-Complex 基座）：dev_complex 487.09 /
  489.36、dev_ood 525.41 / 525.49，五项 min_norm ≥ 1.0，
  test provisional 496.40 / 496.44；
- `test_v8` final 已用 `--acknowledge-once` 消费一次：
  **RL-FullGuard-C 496.40**（RL-PPO 487.83），ledger 已记录；
- C++ `RLFullGuardPolicy` 已移植，4 组固定序列 Python/C++ parity
  通过；
- 主权重 `weights/rl_policy_weights.json`（RL-PPO）未改动，
  sha256 `bc582056…`。

FullGuard-C 是 **meta policy**（V5-FT-Complex 基座 + guard），
不能简单用单个 MLP 权重替代；覆盖主权重会丢失 guard 并改变
`rl` baseline 的语义。

## Decision

1. **推荐提交候选：`rl-full-guard-c`**（Python + C++）；
2. **保留 `weights/rl_policy_weights.json` 作为 RL-PPO baseline /
   fallback，不覆盖**；
3. 部署使用 `weights/rl_policy_weights_v5_ft_complex.json` +
   `cpp/policies/rl_full_guard_policy.hpp`；
4. `test_v8` 已消费；后续新候选需要 `test_v9`；
5. 所有文档的“当前建议”以 FullGuard-C 为首选，但保留 RL-PPO /
   RL-Switch / GRU-DAgger / RL-Wide160 作为 fallback。

## Consequences

- README / handover / candidates / protocol-v5 需要把 FullGuard-C
  标为推荐提交候选；主权重 checksum 保持不变；
- C++ 部署入口为 `RLFullGuardPolicy`，权重头由
  `ipd/tools/export_full_guard_cpp.py` 生成；
- 如果比赛接口只允许提交单个 MLP 权重，则只能提交 V5-FT-Complex
  （无 guard），此时 dev gate 不再成立，需要重新评估；
- 继续优化 test 需要新训练/新候选，并只能使用 `test_v9` 作为
  held-out；不要再在已消费的 `test_v8` 上调参；
- FullGuard-C 相对 V5-FT 的 test 增益仍小（约 +0.12），
  不应把它描述为显著超越；推荐理由是 dev gate、complex/OOD 和
  鲁棒性。
