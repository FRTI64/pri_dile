# Protocol v4：Axelrod + 更大规模

状态：**当前主口径**（2026-10-27 起）。v2 / v3 均进入 legacy；实现入口：
`ipd/evaluation/protocol.py`。

## 1. 为什么从 v3 升级

v3 已有 532 个 config、complex / policy 对手，但仍有局限：

- 对手主要来自本地规则库，缺少 Axelrod 的经典/进化/元策略；
- 没有 Axelrod 策略的组合对手；
- 默认赛场仍是 299 人、dev 20 fields、test 40 fields，容量不足以覆盖
  更大的 catalog；
- `test_v6` 已被消费，不能继续用于新候选。

v4 增加：

- **50 个 Axelrod 4.14.0 策略**（经典、noisy、adaptive、ZD、meta、evolved）；
- **15 个 AxelrodMix 组合**（每场从 2–4 个 Axelrod 策略中抽一个）；
- `dev_axelrod` / `train_axelrod`；
- 默认赛场扩大到 **2048 人（2000+）**，dev **30 fields**，test **50 fields**；
- final 改为 `test_v7`。

catalog 从 532 扩到 **约 597 个 config**。

## 2. 划分

| 层 | 用途 | 组成 | 规模 |
|---|---|---|---|
| `train` | 训练 | 标准 + complex + policy + Axelrod + anchors | 251 + 74 + 4 + 45 + 5 |
| `dev_param` | 选模：标准参数 held-out | 标准 family + Axelrod + anchors | 54 + 10 + 5 |
| `dev_family` | 选模：family held-out | `mirror_adaptive` / `delayed_grim` | 14 + 5 |
| `dev_complex` | 选模：复杂 family | complex families | 17 + 5 |
| `dev_axelrod` | 选模：Axelrod | Axelrod 策略 + 组合 | 10 + 5 |
| `dev_ood` | 选模：偏移混合 | dev / complex / family / Axelrod | - |
| `test_v7` | 一次性 final | 30% seen + 20% test-only + 15% complex + 10% policy + 15% Axelrod + 10% anchors | 56 + 36 + 15 + 6 + 10 + 5 |

固定 anchor：`AllC`、`AllD`、`TFT`、`Grim`、`Random.50`。

held-out family：

- `dev_family`：`mirror_adaptive`、`delayed_grim`
- `test_v7` test-only：`early_harvest`、`frequent_kamikaze`、
  `probe_harvest`、`hyper_extort`、`early_probe`、`adaptive_probe`

complex families：

```text
phase_switch  adaptive_mix  probe_exploit
delayed_extort  memory_two  meta_mix
```

Axelrod 策略（50）：

```text
TitForTat / TitFor2Tats / TwoTitsForTat / Grudger / Forgiver /
ForgivingTitForTat / GTFT / ContriteTitForTat / FirmButFair /
WinStayLoseShift / GoByMajority / HardGoByMajority / SoftGrudger /
Grumpy / Prober / RemorsefulProber / NaiveProber / HardProber /
Adaptive / AdaptiveTitForTat / BackStabber / DoubleCrosser / Bully /
Predator / Ripoff / SneakyTitForTat / TrickyCooperator /
TrickyDefector / Alternator / CyclerCCD / CyclerCCCD /
RandomTitForTat / Random / Defector / Cooperator / ZDExtort2 /
ZDExtort3 / ZDExtortion / ZDGTFT2 / MetaMajority / MetaWinner /
MetaHunter / EvolvedFSM16 / EvolvedANN / PSOGambler1_1_1 /
Calculator / OmegaTFT / Stalker / Thumper / LimitedRetaliate
```

`AxelrodMix` 每场从组合中抽一个策略，例如：

```text
(TitForTat, Grudger, Adaptive)
(TitForTat, BackStabber, Prober)
(ZDExtort2, ZDExtort3, ZDExtortion)
(MetaWinner, MetaHunter, AdaptiveTitForTat)
```

## 3. 依赖

Axelrod 是可选依赖：

```bash
python3 -m pip install -r requirements-axelrod.txt
```

`python3 -m ipd protocol audit` 在没有 Axelrod 时会给出 warning；实际跑
Protocol v4 tier 会报清晰的缺依赖错误。manifest 不依赖 Axelrod 安装。

## 4. 训练 profile

```bash
python3 -m ipd train rl --train-dists train_v4 \
  --dists dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --n 160 ...
```

| profile | 内容 |
|---|---|
| `train_v4` | 标准 + complex + policy + Axelrod + anchors |
| `train_easy` | nice / forgiving / punitive / defect / late + anchors |
| `train_hard` | noisy / random / extort / temporal / wild / adaptive + anchors |
| `train_complex` | complex families + anchors |
| `train_policy` | policy opponents only |
| `train_axelrod` | Axelrod 策略 + 组合 |

## 5. 命令

```bash
python3 -m ipd protocol audit
python3 -m ipd protocol manifest

# manifest：configs/splits/protocol_v4/manifest.json
# catalog sha256: d6a674aa...
# dev：默认 2048 人 × 30 fields
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v4/protocol_v4_dev.json

# test_v7：默认 2048 人 × 50 fields，top 3 一次性
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 \
  --acknowledge-once \
  --out results/protocol_v4/protocol_v4_test.json

# red
python3 -m ipd protocol red --candidates <top3> --tier dev_red --seeds 30
python3 -m ipd protocol red --candidates <top3> --tier test_red \
  --seeds 30 --acknowledge-once
```

## 6. 选模规则

1. `dev_family` 过滤 family 崩塌；
2. `dev_complex` 过滤复杂对手崩坏；
3. `dev_axelrod` 看 Axelrod 泛化；
4. `dev_ood` 看混合/偏移稳定性；
5. 以 `dev_param` 为主排序，结合 `min_norm` / `worst_family`；
6. 预注册 top 3 后才运行 `test_v7`；test 结果不参与下一轮调参。

## 7. 当前限制

- 对手仍是合成/Axelrod 库；test_v7 不是真实 300 人比赛数据。
- v4 的 2048/30/50 规模尚未跑新 baseline；`results/protocol_v3/` 的
  299/20/40 结果只作 legacy。当前不要把它当作 v4 排名。
- Axelrod 完整 Meta 策略较慢；日常可用 `AxelrodMix` / 精选策略，完整 bank
  只在定向分析时使用。
- `test_v7` 一旦消费，下一候选需要新建 `test_v8`。
- Red-v5 / Red-v6 仍是 P3 legacy artifact；anti-exploitation 未解决。
