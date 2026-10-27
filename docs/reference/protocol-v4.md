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

- **224 个 Axelrod 4.14.0 策略**（经典、noisy、adaptive、ZD、meta、evolved）；
- **15 个 AxelrodMix 组合**（每场从 2–4 个 Axelrod 策略中抽一个）；
- `dev_axelrod` / `train_axelrod`；
- 默认赛场扩大到 **2048 人（2000+）**，dev **30 fields**，test **50 fields**；
- final 改为 `test_v7`；v4 起 test 消费永久锁定，只做 provisional 诊断。

catalog 从 532 扩到 **977 个 config**，并新增 8 个复杂/高阶 family 和一批
旧 family 的新对手；另外按 **thinking level 0–4** 和 **complexity 0–4**
提供派生对手集。

## 2. 划分

| 层 | 用途 | 组成 | 规模 |
|---|---|---|---|
| `train` | 训练 | 标准 + complex + policy + Axelrod + anchors | 292 + 179 + 4 + 167 + 5 |
| `dev_param` | 选模：标准参数 held-out | 标准 family + Axelrod + anchors | 62 + 36 + 5 |
| `dev_family` | 选模：family held-out | `mirror_adaptive` / `delayed_grim` | 14 + 5 |
| `dev_complex` | 选模：复杂 family | 14 个 complex family | 39 + 5 |
| `dev_axelrod` | 选模：Axelrod | Axelrod 策略 + 组合 | 36 + 5 |
| `dev_ood` | 选模：偏移混合 | dev / complex / family / Axelrod | - |
| `test_v7` | 永久 held-out（provisional only） | 30% seen + 20% test-only + 15% complex + 10% policy + 15% Axelrod + 10% anchors | 61 + 36 + 40 + 6 + 36 + 5 |

固定 anchor：`AllC`、`AllD`、`TFT`、`Grim`、`Random.50`。

held-out family：

- `dev_family`：`mirror_adaptive`、`delayed_grim`
- `test_v7` test-only：`early_harvest`、`frequent_kamikaze`、
  `probe_harvest`、`hyper_extort`、`early_probe`、`adaptive_probe`

complex families：

```text
phase_switch       adaptive_mix       probe_exploit
delayed_extort     memory_two         meta_mix
noise_adaptive     reputation_grim    contrition_cycle
endgame_hunter     pattern_breaker    conditional_extort
lookahead          portfolio
```

新增复杂风格：

- `noise_adaptive`：对孤立噪声宽恕，对持续背叛惩罚；
- `reputation_grim`：按累计合作率决定是否永久 Grim；
- `contrition_cycle`：自己背叛后主动道歉/修复；
- `endgame_hunter`：末局前 TFT，末局收割或抢先惩罚；
- `pattern_breaker`：检测对手短周期并打反周期；
- `conditional_extort`：只剥削高合作率对手，否则 TFT；
- `lookahead`：估计对手合作率并做 2–3 步前瞻；
- `portfolio`：对 TFT / Grim / Extort / Adaptive 等成员投票。

旧 family 也加入了新对手：

- `nice`：Majority k=4/6；
- `forgiving`：ForgiveTFT missed=4、TFTNoisyForgive g=0.50、GenerousZD；
- `punitive`：Gradual、ProbabilisticGrim、ResetGrim、SpitefulGrim、
  DelayedPunisher、更多 GrimForgive；
- `defect`：RandomThenTFT、更多 EarlyDefector；
- `late`：LateD 98/99；
- `random`：更多 Periodic 模式；
- `wild`：更多 MemoryOne / Reactive / RandomFSM；
- `adaptive`：TFTProbeImmediate、更多 TFTNoisyForgive；
- `extort`：ExtortZD 1.0、GenerousZD 3.2/3.6；
- `temporal`：更多 Harvest / Dynamic / Kamikaze / ProbePhase / WindowTFT；
- `noisy`：NoisyTFT/Joss p=0.52/0.55。

## 2.1 按思考等级 / 策略复杂度

每个 catalog entry 带 `thinking_level` 和 `complexity`（0–4）。派生集合：

| 集合 | 含义 | dev configs |
|---|---|---:|
| `dev_level0` | 非策略型：AllC/AllD/Random/Periodic 等 | 6 |
| `dev_level1` | 固定启发式：TFT/Grim/GTFT/LateD 等 | 62 |
| `dev_level2` | 反应/自适应：Probe/Noisy/Responsive/Extort 等 | 44 |
| `dev_level3` | 对手感知/meta：AdaptiveMix/Phase/Reputation 等 | 29 |
| `dev_level4` | 搜索/组合/学习：Lookahead/Portfolio/MetaMix/Policy/evolved | 10 |
| `dev_complexity0` | 无状态/极简 | 6 |
| `dev_complexity1` | 简单反应式 | 28 |
| `dev_complexity2` | 有状态/参数化 | 75 |
| `dev_complexity3` | 复合/自适应 | 25 |
| `dev_complexity4` | 搜索/集成/学习/长程 | 17 |

```bash
# 只跑某一思考等级
python3 -m ipd protocol eval --candidates all --tiers dev_level3 \
  --fields 30 --n 2048 --workers 20 \
  --out results/protocol_v4/protocol_v4_dev_level3.json

# 只跑某一复杂度
python3 -m ipd protocol eval --candidates all --tiers dev_complexity4 \
  --fields 30 --n 2048 --workers 20 \
  --out results/protocol_v4/protocol_v4_dev_complexity4.json
```

训练侧对应 `train_level0..4` / `train_complexity0..4`，可直接传给
`--train-dists`。`test_v7` 仍按 family 混合；评估报告可再按 level/complexity
拆分分析。

Axelrod 策略（224：66 curated + 158 extra，extra 在 `axelrod_extra` family）：

```text
TitForTat / TitFor2Tats / TwoTitsForTat / Grudger / Forgiver /
ForgivingTitForTat / GTFT / ContriteTitForTat / FirmButFair /
WinStayLoseShift / GoByMajority / HardGoByMajority / SoftGrudger /
Grumpy / Prober / RemorsefulProber / NaiveProber / HardProber /
Adaptive / AdaptiveTitForTat / BackStabber / DoubleCrosser / Bully /
Predator / Ripoff / SneakyTitForTat / TrickyCooperator /
TrickyDefector / Alternator / CyclerCCD / CyclerCCCD /
RandomTitForTat / Random / Defector / Cooperator / ZDExtort2 /
ZDExtort3 / ZDExtortion / ZDGTFT2 / MetaHunter / EvolvedFSM16 /
EvolvedANN / PSOGambler1_1_1 /
Calculator / OmegaTFT / Stalker / Thumper / LimitedRetaliate /
Winner12 / Winner21 / EvolvedHMM5 / EvolvedFSM16Noise05 /
EvolvedANNNoise05 / PSOGambler2_2_2 / PSOGamblerMem1 /
KnowledgeableWorseAndWorse / ThueMorse / ThueMorseInverse /
LimitedRetaliate3 / Retaliate2 / Retaliate3 / HardTitFor2Tats /
SlowTitForTwoTats2 / DynamicTwoTitsForTat / MEM2 / ShortMem
```

`AxelrodMix` 每场从组合中抽一个策略，例如：

```text
(TitForTat, Grudger, Adaptive)
(TitForTat, BackStabber, Prober)
(ZDExtort2, ZDExtort3, ZDExtortion)
(Winner21, MetaHunter, AdaptiveTitForTat)
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
| `train_level0..4` | 按思考等级筛选 |
| `train_complexity0..4` | 按策略复杂度筛选 |

## 5. 命令

```bash
python3 -m ipd protocol audit
python3 -m ipd protocol manifest

# manifest：configs/splits/protocol_v4/manifest.json
# catalog sha256: beb594ab...
# dev：默认 2048 人 × 30 fields
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v4/protocol_v4_dev.json

# test_v7：默认 2048 人 × 50 fields；永久 provisional，不会消费
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 \
  --no-consume \
  --out results/protocol_v4/protocol_v4_test_provisional.json

# 只验证 pipeline，同样不会消费 test_v7：
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v7 --fields 1 --n 256 --workers 4 \
  --no-consume --out /tmp/test_v7_smoke.json

# red
python3 -m ipd protocol red --candidates <top3> --tier dev_red --seeds 30
python3 -m ipd protocol red --candidates <top3> --tier test_red \
  --seeds 30 --no-consume
```

## 6. 选模规则

1. `dev_family` 过滤 family 崩塌；
2. `dev_complex` 过滤复杂对手崩坏；
3. `dev_axelrod` 看 Axelrod 泛化；
4. `dev_ood` 看混合/偏移稳定性；
5. 以 `dev_param` 为主排序，结合 `min_norm` / `worst_family`；
6. `test_v7` / `test_red` 只做 provisional 诊断；test 结果不参与选模或调参。

`configs/splits/protocol_v4/test_policy.json` 将 `consume_test` 固定为
`false`，CLI 没有覆盖开关。test 运行必须显式传 `--no-consume`，结果会标记
`usage_consumed=false`、`selection_eligible=false`，并写入
`*_test_provisional.json` / `*_provisional.json`。`--acknowledge-once` /
`--force` 不能解锁；policy 缺失或损坏时也按锁定处理。

## 7. 当前限制

- 对手仍是合成/Axelrod 库；test_v7 不是真实 300 人比赛数据。
- 224 个 Axelrod 策略里包含 evolved/meta 等较慢策略；完整 test_v7 会比 v3
  慢，正式 final 前建议用 `--fields 1 --n 256 --no-consume` 做 pipeline
  烟测。
- v4 的 2048/30/50 规模尚未跑新 baseline；`results/protocol_v3/` 的
  299/20/40 结果只作 legacy。当前不要把它当作 v4 排名。
- Axelrod 完整 Meta 策略较慢；日常可用 `AxelrodMix` / 精选策略，完整 bank
  只在定向分析时使用。
- test 消费已永久锁定；`test_v7` / `test_red` 只作 provisional 诊断。
- Red-v5 / Red-v6 仍是 P3 legacy artifact；anti-exploitation 未解决。
