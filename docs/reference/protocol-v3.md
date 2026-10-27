# Protocol v3：更大的 train / dev / test 划分

状态：**当前主口径**（2026-10-27 起）。Protocol v2 已进入 legacy，保留在
`results/protocol_v2/`；实现入口：`ipd/evaluation/protocol.py`。

## 1. 为什么从 v2 升级

v2 解决了 official/blind 与训练同源的问题，但仍有局限：

- catalog 只有 224 个 config，dev/test 的 seen-family 参数点约 25 个；
- 对手以规则策略为主，缺少复杂的 phase/adaptive/probe/policy 对手；
- 只有一套 validation（`dev_param` + `dev_family`），无法分别看复杂对手和
  OOD 混合；
- `test_v5` 已被消费，不能继续用于新候选。

v3 扩展到 **532 个 config**，增加 complex families、policy opponents、
`dev_complex`、`dev_ood` 和 `test_v6`；训练侧增加多个 curriculum profile。

## 2. 划分

| 层 | 用途 | 组成 | 规模 |
|---|---|---|---|
| `train` | 训练 | 标准 family + complex + policy + anchors 的混合 | 251 + 74 + 4 + 5 |
| `dev_param` | 选模：参数 held-out | 标准 family 的 dev config + anchors | 54 + 5 |
| `dev_family` | 选模：family held-out | `mirror_adaptive` / `delayed_grim` + anchors | 14 + 5 |
| `dev_complex` | 选模：复杂 family | complex families 的 dev config + anchors | 17 + 5 |
| `dev_ood` | 选模：OOD 混合 | dev / dev_complex / dev_family 的加权混合 | - |
| `test_v6` | 一次性 final | 35% seen test + 20% test-only + 20% complex + 15% policy + 10% anchors | 56 + 36 + 15 + 6 + 5 |

固定 anchor：`AllC`、`AllD`、`TFT`、`Grim`、`Random.50`。它们在每层都出现，
行为固定，不参与 config 泄漏检查，但单独报告。

held-out family：

- `dev_family`：`mirror_adaptive`、`delayed_grim`
- `test_v6` test-only：`early_harvest`、`frequent_kamikaze`、
  `probe_harvest`、`hyper_extort`、`early_probe`、`adaptive_probe`

complex families（train/dev/test 参数不相交）：

```text
phase_switch  adaptive_mix  probe_exploit
delayed_extort  memory_two  meta_mix
```

policy opponents：

- train：RL-BC、SelfPlay-BC、Neural、GRU
- test：Final-tuned、Triple-majority、RL-scratch、SelfPlay、Neural-retrain、
  GRU-retrain

`test_v6` 的 policy 只使用不与当前 top 3 直接相同的权重，避免把候选自身当作
test opponent。

## 3. 训练 profile

```bash
python3 -m ipd train rl --train-dists train_v3 \
  --dists dev_param,dev_family,dev_complex,dev_ood ...
```

| profile | 内容 |
|---|---|
| `train_v3` | 标准 + complex + policy + anchors 的完整混合 |
| `train_easy` | nice / forgiving / punitive / defect / late + anchors |
| `train_hard` | noisy / random / extort / temporal / wild / adaptive + anchors |
| `train_complex` | complex families + anchors |
| `train_policy` | policy opponents only |

## 4. 命令

```bash
# 审计 / manifest
python3 -m ipd protocol audit
python3 -m ipd protocol manifest
python3 -m ipd protocol audit --json --out results/protocol_v3/audit.json

# dev：所有候选，四个 validation tier
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_ood \
  --fields 20 --n 299 --workers 20 --baseline RL-PPO \
  --out results/protocol_v3/protocol_v3_dev.json

# test_v6：只对预注册 top 3 跑一次
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v6 --fields 40 --n 299 --workers 20 \
  --acknowledge-once \
  --out results/protocol_v3/protocol_v3_test.json

# red
python3 -m ipd protocol red --candidates all --tier dev_red --seeds 20
python3 -m ipd protocol red --candidates <top3> --tier test_red \
  --seeds 20 --acknowledge-once
```

`test_v6` / `test_red` 用 `results/protocol_v3/usage.json` 锁定；重复运行需
`--force`，并标记 `contaminated`。当前 manifest：
`configs/splits/protocol_v3/manifest.json`，catalog sha256 前缀
`00b435c7...`。

## 5. 选模规则

1. `dev_family` 过滤 family 崩塌；
2. `dev_complex` 过滤复杂对手崩坏；
3. `dev_ood` 看混合/OOD 稳定性；
4. 以 `dev_param` 为主排序，结合 `min_norm` / `worst_family`；
5. 预注册 top 3 后才运行 `test_v6`；
6. `test_v6` 结果不参与下一轮调参。

## 6. 报告规则

- `dev_param` / `dev_family` / `dev_complex` / `dev_ood` 的 mean、min、p05；
- per-family、`worst_family`、`min_norm`；
- test_v6 的 seen / test-only / complex / policy 分项；
- red blue / red win / diff；
- manifest hash、usage ledger、权重 checksum。

## 7. 当前限制

- 对手仍是合成库；test_v6 不是真实 300 人比赛数据。
- catalog 中 dev/test 的 seen-family config 仍有限（约 50 个），扩大网格会
  改变 manifest hash。
- Red-v5 / Red-v6 是 P3 legacy artifact；正式提交前应训练新的 red pair。
- `test_v6` 已消费，下一候选需要新建 `test_v7`。

## 8. 2026-10-27 baseline

| 候选 | dev_param | dev_family | dev_complex | dev_ood | test_v6 |
|---|---:|---:|---:|---:|---:|
| RL-Repair | **532.44** | 582.92 | **420.38** | **489.84** | **484.32** |
| RL-PPO-Hard60 | 527.97 | 582.94 | 410.09 | 485.12 | 476.54 |
| Adv-RL | 526.58 | 583.20 | 415.19 | 487.22 | 471.44 |
| RL-PPO | 524.37 | 582.72 | 397.90 | 478.61 | - |

- top 3 由 `dev_param` mean 预注册；RL-Repair 在 test_v6 上显著优于 A60
  （+7.78，CI [7.02, 8.58]）和 Adv-RL（+12.88，CI [11.65, 14.08]）。
- Red-v5 / Red-v6 仍 100% 击败 top 3；anti-exploitation 未解决。
- 完整 20 候选表：`results/protocol_v3/README.md`。
