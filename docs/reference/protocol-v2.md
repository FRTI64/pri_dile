# Protocol v2：train / dev / test 划分

状态：**新主口径**。旧的 `official` / `blind` 表进入 legacy，只用于历史复现，
不再作为“泛化”证据。实现入口：`ipd/evaluation/protocol.py`。

## 1. 为什么重划

旧口径的问题：

- `official` 的四个分布（`original/late/grim30/wide`）同时是 RL 训练分布，
  只是换 seed，属于 in-distribution；
- `blind` 只换 seed，对手 family / 参数范围仍与训练重叠；
- `heldout_v2` 被 E1/E2/P3/P2 反复用于选模，事实上已经是 validation；
- `heldout_v3` 已被 E1c 消费；`final` suite 也被多次 A/B/C 使用；
- 没有 manifest / usage ledger，split 定义散落在多个模块和 config 中。

Protocol v2 把“对手实例”作为划分单位：family + type + 参数 + 行为 seed。
同一个 family 可以在 train/dev/test 出现，但具体 config 不重叠；`dev_family`
和 `test_v5` 还额外包含训练中完全未出现的 family。

## 2. 四层

| 层 | 用途 | 组成 | 规模（当前 manifest） |
|---|---|---|---|
| `train` | 训练/拟合 | 已见 family 的 train catalog + 5 个 anchor | 113 configs |
| `dev_param` | 选模 | 已见 family 的 dev catalog + anchors | 25 configs |
| `dev_family` | 选模 | `mirror_adaptive` / `delayed_grim` + anchors | 10 configs |
| `test_v5` | 一次性 final | 60% seen-family test + 25% test-only families + 15% anchors | 26 + 30 + 5 configs |

固定 anchor：`AllC`、`AllD`、`TFT`、`Grim`、`Random.50`。它们在每层都出现，
行为固定，因此不参与 config 泄漏检查，但单独报告。

held-out family：

- `dev_family`：`mirror_adaptive`、`delayed_grim`
- `test_v5`：`early_harvest`、`frequent_kamikaze`、`probe_harvest`、
  `hyper_extort`、`early_probe`、`adaptive_probe`

`dev_family` 的 family 不出现在 `train` / `dev_param` / `test_v5`；
`test_v5` 的 test-only family 不出现在 `train` / `dev_param` / `dev_family`。
`test_v5` 中 60% 的 seen-family config 与 train/dev 参数不相交。

## 3. 命令

```bash
# 审计：family / config / manifest / coverage
python3 -m ipd protocol audit
python3 -m ipd protocol audit --json --out results/protocol_v2/audit.json

# 冻结/校验 manifest
python3 -m ipd protocol manifest

# dev：所有候选，20 fields × 299
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family --fields 20 --n 299 --workers 20 \
  --seed 20261027 --baseline RL-PPO \
  --out results/protocol_v2/protocol_v2_dev.json

# test_v5：只对预注册的 top 3 跑一次
python3 -m ipd protocol eval --candidates RL-Repair,RL-PPO-Hard60,Adv-RL \
  --tiers test_v5 --fields 30 --n 299 --workers 20 \
  --seed 20261027 --acknowledge-once \
  --out results/protocol_v2/protocol_v2_test.json
```

`test_v5` 必须带 `--acknowledge-once`；usage ledger 在
`results/protocol_v2/usage.json`。同一候选重复运行时，`--force` 会把
新结果标记为 `contaminated`，不得再作为 final test。

Red exploitability：

```bash
# dev red：可反复用于诊断
python3 -m ipd protocol red --candidates all --tier dev_red --seeds 20

# test red：只在 final 时跑一次
python3 -m ipd protocol red --candidates <top3> --tier test_red \
  --seeds 20 --acknowledge-once
```

当前 `dev_red` / `test_red` 使用 P3 的 Red-v5 / Red-v6，彼此不重叠，但都是
legacy artifact；正式提交前应训练新的 protocol-specific red pair。

## 4. Manifest 与版本

- manifest：`configs/splits/protocol_v2/manifest.json`
- 当前 catalog：224 configs，`catalog_sha256` 前缀 `f6705fbb...`。
- catalog 由代码确定性生成；manifest 保存 `catalog_sha256` 和全部 config。
- 任何 family / 参数 / tier 改动都会改变 hash；审计会报 `catalog hash mismatch`。
- 训练入口：

```bash
python3 -m ipd train rl --train-dists train_v2 \
  --dists dev_param,dev_family --select-metric avg ...
```

`train_v2` 只从 `train` 层采样；`dev_param` / `dev_family` 可作为 validation。
训练代码不应接受 `test_v5` 作为 validation。

## 5. 选模规则

1. 先用 `dev_family` 过滤病理策略（例如 `min` 明显低于 anchor 合作线，
   或某 family 崩塌）；
2. 按 `dev_param` mean 排序；
3. 接近时比较 `min_norm`、`worst_family`、per-family；
4. 在跑 test 前预注册 top 3，不能看完 test 再换候选；
5. `test_v5` 只跑一次，结果不参与下一轮调参。

## 6. 报告规则

任何候选至少报告：

- `dev_param` / `dev_family` 的 mean、min、p05；
- per-family mean、`worst_family`、`min_norm`（相对 baseline）；
- 旧 `official` / `blind` 仅作为 legacy 对照，必须注明“不可与新表混用”；
- 最终候选才运行 `test_v5`，且只运行一次；
- 权重 checksum、manifest hash、usage ledger 路径。

## 7. 当前限制

- 对手仍是合成库；test_v5 不是真实 300 人比赛数据。
- 当前 catalog 的 dev/test config 数量有限（约 25 个 seen-family config），
  后续可通过扩大参数网格增加统计功效；扩大会改变 manifest hash。
- 第一版 test-only family 过于合作（所有候选都 595），已作为 pilot 归档到
  `results/protocol_v2/pilot_test_v5_prehard/`；当前 manifest 是修订版。
- `dev_red` / `test_red` 已接入 `protocol red`，但当前 red pair 是 P3 的
  Red-v5 / Red-v6；正式提交前应训练新的 protocol-specific red pair。
- `test_v5` 已消费，不得再运行；新候选需新建 `test_v6`。`test_red` 也已
  消费一次。

## 8. 2026-10-27 baseline

| 候选 | dev_param | dev_family | test_v5 | dev_red blue | test_red blue |
|---|---:|---:|---:|---:|---:|
| RL-Repair | 520.09 | 582.71 | **477.05** | 378.05 | 314.65 |
| RL-PPO-Hard60 | 514.18 | 582.73 | 469.91 | 378.00 | 319.00 |
| Adv-RL | **520.19** | 583.03 | 467.95 | 375.90 | 342.95 |
| RL-PPO | 510.55 | 582.53 | - | - | - |

- top 3 由 `dev_param` mean 预注册；RL-Repair 在 test 上显著优于 A60
  （+7.14，CI [6.44, 7.88]）和 Adv-RL（+9.10，CI [7.84, 10.41]）。
- Red-v5 / Red-v6 仍 100% 击败 top 3；anti-exploitation 未解决。
- 完整 20 候选表：`results/protocol_v2/README.md`。
