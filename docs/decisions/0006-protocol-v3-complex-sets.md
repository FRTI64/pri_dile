# ADR 0006：Protocol v3 扩大并复杂化 train/dev/test

- 状态：Accepted（2026-10-27）
- 关联：`docs/reference/protocol-v3.md`、`ipd/evaluation/protocol.py`

## 背景

Protocol v2 的 catalog 只有 224 个 config，dev/test 的 seen-family 参数点约
25 个；对手以规则策略为主，缺少 phase/adaptive/probe/policy 对手。`test_v5`
已被消费，无法继续用于新候选。继续在 v2 上扩展会把“更多对手”与“test 复用”
混在一起。

## 决策

1. 新建 **Protocol v3**，v2 的 manifest 和 `test_v5` 结果保持不动，仅作
   legacy。
2. catalog 扩展到 532 个 config：
   - 标准 family 参数网格约 330 个；
   - 6 个 complex family（phase_switch / adaptive_mix / probe_exploit /
     delayed_extort / memory_two / meta_mix）约 96 个；
   - policy opponents（规则 / RL / MLP / GRU / SelfPlay）10 个；
   - test-only / dev-only / anchors 其余。
3. 新增 validation 层 `dev_complex`（复杂 family）和 `dev_ood`（混合偏移），
   使选模不只看标准 `dev_param`。
4. 新 final `test_v6`：35% seen test + 20% test-only + 20% complex +
   15% policy + 10% anchors；`test_v5` 不再使用。
5. 训练侧新增 `train_v3` / `train_easy` / `train_hard` / `train_complex` /
   `train_policy` 五个 profile。
6. `test_v6` / `test_red` 继续使用 manifest + usage ledger 一次性保护。

2026-10-27 baseline：RL-Repair test_v6 484.32 > A60 476.54 > Adv-RL 471.44；
Red-v5/v6 仍 100% 击败 top 3。

## 后果

- v2 结果不能与 v3 混表；新候选必须重跑 v3 dev。
- `test_v6` 消费后，下一个候选需要 `test_v7`。
- policy opponents 会增加评估成本，但能暴露规则对手看不到的弱点。
- complex / policy 的加入可能让绝对分数下降；比较应在同一 v3 协议内进行。
