# Protocol v5：重新划分的 train / dev / test

状态：**当前主口径**（2026-10-30 起，ADR 0013）。v2 / v3 / v4 均进入
legacy；实现入口：`ipd/evaluation/protocol.py`。

## 1. 为什么从 v4 升级

v4 的 catalog（977 configs）、2048 人赛场、Axelrod / complex / policy
分层继续保留，但 v4 的 `test_v7` 已解禁，继续在 v4 上叠加新候选会让
test 复用并失去独立性。v5 的目标是提供一套新的 train/dev/test 分配：

- **catalog 保持 977 configs**：224 个 Axelrod 策略 + 15 个
  AxelrodMix、14 个 complex family、2048 人 / dev 30 fields /
  test 50 fields；
- **重新划分**：`SPLIT_SALT = "protocol_v5"` 改变 family-local shuffle，
  405/977 个 config 的 tier 与 v4 不同；
- **新 final**：`test_v8` 取代 v4 的 `test_v7`；`test_v7` 及全部 v4
  结果冻结为 legacy；
- `test_v8` / `test_red` 初始锁定；`test_v8` 已于 2026-10-30 消费一次，
  `test_red` 仍锁定。

> **跨版本泄漏警告：** v5 与 v4 使用同一 catalog。v4 的 train/dev
> config 可能进入 v5 test_v8，反之亦然。因此 `test_v8` 只对“在 v5
> train 上训练/选模”的候选构成 held-out；不要把旧 v4 候选的 v5 分数
> 当成独立 test 证据，也不要把 v4/v5 数字混表。

## 2. 划分

| 层 | 用途 | 组成 | 规模 |
|---|---|---|---|
| `train` | 训练 | 标准 + complex + policy + Axelrod + anchors | 292 + 179 + 4 + 167 + 5 |
| `dev_param` | 选模：标准参数 held-out | 标准 family + Axelrod + anchors | 62 + 36 + 5 |
| `dev_family` | 选模：family held-out | `mirror_adaptive` / `delayed_grim` | 14 + 5 |
| `dev_complex` | 选模：复杂 family | 14 个 complex family | 39 + 5 |
| `dev_axelrod` | 选模：Axelrod | Axelrod 策略 + 组合 | 36 + 5 |
| `dev_ood` | 选模：偏移混合 | dev / complex / family / Axelrod | - |
| `test_v8` | held-out final（一次性消费；`--no-consume` 可做 provisional） | 30% seen + 20% test-only + 15% complex + 10% policy + 15% Axelrod + 10% anchors | 61 + 36 + 40 + 6 + 36 + 5 |

固定 anchor：`AllC`、`AllD`、`TFT`、`Grim`、`Random.50`。

held-out family：

- `dev_family`：`mirror_adaptive`、`delayed_grim`
- `test_v8` test-only：`early_harvest`、`frequent_kamikaze`、
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
  --out results/protocol_v5/protocol_v5_dev_level3.json

# 只跑某一复杂度
python3 -m ipd protocol eval --candidates all --tiers dev_complexity4 \
  --fields 30 --n 2048 --workers 20 \
  --out results/protocol_v5/protocol_v5_dev_complexity4.json
```

训练侧对应 `train_level0..4` / `train_complexity0..4`，可直接传给
`--train-dists`。`test_v8` 仍按 family 混合；评估报告可再按 level/complexity
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
Protocol v5 tier 会报清晰的缺依赖错误。manifest 不依赖 Axelrod 安装。

## 4. 训练 profile

```bash
python3 -m ipd train rl --train-dists train_v5 \
  --dists dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --n 160 ...
```

| profile | 内容 |
|---|---|
| `train_v5` | 标准 + complex + policy + Axelrod + anchors |
| `train_easy` | nice / forgiving / punitive / defect / late + anchors |
| `train_hard` | noisy / random / extort / temporal / wild / adaptive + anchors |
| `train_complex` | complex families + anchors |
| `train_pattern_breaker` | 仅 `pattern_breaker` 的 `train_complex` 参数点（定向 hard mining） |
| `train_probe_pattern` | 仅 `probe_exploit` / `pattern_breaker` 的 `train_complex` 参数点（v5 定向训练） |
| `train_probe_only` | 仅 `probe_exploit` 的 `train_complex` 参数点（specialist 实验） |
| `train_weak_complex` | `pattern_breaker` / `adaptive_mix` / `probe_exploit` / `conditional_extort` / `lookahead` |
| `train_policy` | policy opponents only |
| `train_axelrod` | Axelrod 策略 + 组合 |
| `train_level0..4` | 按思考等级筛选 |
| `train_complexity0..4` | 按策略复杂度筛选 |

## 5. 命令

```bash
python3 -m ipd protocol audit
python3 -m ipd protocol manifest

# manifest：configs/splits/protocol_v5/manifest.json
# catalog sha256: c7aab4da...
# dev：默认 2048 人 × 30 fields
python3 -m ipd protocol eval --candidates all \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/protocol_v5/protocol_v5_dev.json

# test_v8 final：先在 test_policy.json 解禁，再一次性消费并写 usage ledger
python3 -m ipd protocol eval --candidates <final> \
  --tiers test_v8 --fields 50 --n 2048 --workers 20 \
  --acknowledge-once \
  --out results/protocol_v5/protocol_v5_test_final.json

# provisional 诊断：不消费 test_v8
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v8 --fields 50 --n 2048 --workers 20 \
  --no-consume \
  --out results/protocol_v5/protocol_v5_test_provisional.json

# 只验证 pipeline，同样不会消费 test_v8：
python3 -m ipd protocol eval --candidates <top3> \
  --tiers test_v8 --fields 1 --n 256 --workers 4 \
  --no-consume --out /tmp/test_v8_smoke.json

# 全 catalog 逐 config（diagnostic；不等于 field 抽样）
python3 -m ipd matchup --candidates all --opponent-set protocol --reps 16

# red
python3 -m ipd protocol red --candidates <top3> --tier dev_red --seeds 30
python3 -m ipd protocol red --candidates <top3> --tier test_red \
  --seeds 30 --no-consume
```

### 5.1 调度、RNG 与剖析（Plan 1）

`protocol eval` 现在按 `(field, candidate_chunk)` 调度：一个 worker 构建一次
field，串行评估 4–6 个候选，而不是每个候选都重建一次 field。

```bash
# 自动 worker 数：min(20, cpu_count - 2, RAM 预算)
python3 -m ipd protocol eval --candidates all --tiers dev_param \
  --fields 30 --n 2048 --workers auto \
  --candidate-chunk 6 --field-reuse auto --profile \
  --out results/protocol_v5/protocol_v5_dev_param.json
```

- **RNG 契约**：`reset()` 不会回退对手的 Mersenne Twister。复用 field 前，
  worker 保存每个对手/学习策略的 `rng.getstate()`，每个候选开始前恢复；
  候选 policy 仍按 `(candidate, field)` 重建，保持原有匹配顺序。这样复用与
  逐候选重建逐位一致。
- `--field-reuse auto`：能恢复 RNG 时复用；`on` 显式请求；`off` 退回每个
  候选重建。`Darwin` 等类级状态对手始终重建。
- `--candidate-chunk 6`：候选交错分组，避免长任务聚在同一 chunk。
- `--profile`：把 per-candidate / per-tier / per-family 耗时写到
  `profile_<result>.json`；结果 JSON 始终包含 `workers`、
  `worker_budget`、`peak_rss_mb`、`peak_pss_mb`（物理内存估计）、
  `peak_anon_mb` 和 `profile` 摘要。
- `--workers auto` 解析出的实际值、CPU/RAM 预算和候选分块都会写入结果文件，
  便于审计“workers / 耗时 / RSS”结论。

### 5.2 GPU sample（per-match sample RNG）

Plan 2 的 GPU 路径只支持 argmax；argmax 是另一个确定性策略，会让 sample
训练的学习型候选明显掉分。`--sample-rng per-match` 提供 opt-in 的
per-match 契约：每场用
`(base_seed, field_seed, candidate, match_index)` 派生独立 RNG，
CPU/GPU 使用同一组逐轮 uniform，因此可以逐动作 parity。它先于 v5 split
引入，因此文档中仍可能看到“v5 sample RNG”的旧称。

```bash
python3 -m ipd protocol eval --candidates all --tiers test_v8 \
  --fields 50 --n 2048 --workers 20 --gpu-workers 4 \
  --gpu-field-batch 4 \
  --mode sample --sample-rng per-match --gpu on --gpu-dtype float32 \
  --seed 20261027 --no-consume
```

- 默认 `--sample-rng v4` 是历史顺序 RNG；`per-match` 必须显式开启，
  不能与 v4 sample 表混用。
- 结果标记 `sample_rng=per-match`、`gpu_sample=true`、
  `sample_rng_version=5`；Protocol v5 split 另有
  `split_version=5` / `protocol_version=5`。
- Plan 6 把 sample policy 对手也接入 GPU，但只加速原本就在 GPU 的候选；
  24/24 候选分数与 Plan 5 baseline 逐位一致。**这些 Plan 5/6 数字是在
  v4 split 上得到的**，不能直接迁移到 v5；v5 需重新跑。
- `Adv-*`、PSRO meta-policy、`Final` / `Triple` 和规则/Axelrod 仍走
  CPU，避免改变它们的顺序 RNG 和分数。
- 详见 [plan5-gpu-sample.md](../research/plan5-gpu-sample.md) 与
  [plan6-python-bottleneck.md](../research/plan6-python-bottleneck.md)。

## 6. 选模规则

1. `dev_family` 过滤 family 崩塌；
2. `dev_complex` 过滤复杂对手崩坏；
3. `dev_axelrod` 看 Axelrod 泛化；
4. `dev_ood` 看混合/偏移稳定性；
5. 以 `dev_param` 为主排序，结合 `min_norm` / `worst_family`；
6. `test_v8` 只用于 final 报告，不参与调参或模型排序；`test_red` 仍只做
   provisional 诊断（blue-red 已停用，ADR 0011）。

`configs/splits/protocol_v5/test_policy.json` 用
`consume_tiers` 按 tier 控制消费：`test_v8=true`（2026-10-30 已消费
一次）、`test_red=false`；`consume_test` 是全局 fallback，仓库内固定为
`false`。v5 初次发布时 `test_v8` 保持锁定；需要 final 时才把对应 tier
改为 `true`。解禁后消费必须显式传 `--acknowledge-once`，结果写入
`results/protocol_v5/usage.json`，标记 `usage_consumed=true`、
`selection_eligible=true`。`--no-consume` 仍可生成
`*_test_provisional.json` / `*_provisional.json`（不写 ledger），
`--force` 只允许对已消费候选重跑并标记 `contaminated=true`，不能绕过锁定；
policy 缺失或损坏时按 fail closed 处理。`test_v8` 消费后，新候选需要
`test_v9` 或重新审计扩充分层。

## 7. 当前限制

- 对手仍是合成/Axelrod 库；test_v8 不是真实 300 人比赛数据。
- 224 个 Axelrod 策略里包含 evolved/meta 等较慢策略；完整 test_v8 会比 v3
  慢，正式 final 前建议用 `--fields 1 --n 256 --no-consume` 做 pipeline
  烟测。
- v4 的 top 7 dev 是 legacy 数字；v5 dev baseline 已跑（33 candidates，
  1662.15s，见 `results/protocol_v5/README.md`）。v5 的共同弱项是
  `probe_exploit`，旧 `min_norm ≥ 0.99` 门槛下没有 top candidate 通过；
  分级诊断（`protocol_v5_dev_levels.json`）显示 `pattern_breaker` 对
  RL-Wide160 / RL-Switch / GRU-DAgger 已经不弱，主要缺口是
  `probe_exploit`（约 116 分）。`train_probe_pattern` 定向 fine-tune
  可把 `pattern_breaker` 提到约 762，但 `probe_exploit` 仍约 118；
  probe-only specialist、v2 特征、精确规则都没有净收益。probe 专项暂停：
  **不采用 unsafe specialist / `--allow-probe` / 预判 gate**，也不为
  probe 修改 protocol 或 safety 规则；v5 保持 977 configs。2026-10-30
  又加入**反应式** `rl-probe-guard`：不预判、不关 safety，在观察到
  第 6/8 回合 probe 后先让 3 回合 C；dev probe 约 126、
  `dev_complex` min_norm 0.996（两个 seed），test_v8 provisional
  495.92 vs V5-FT 496.37。随后 `rl-full-guard` 用 probe `C,C,C` +
  late `D,C,C`，并要求连续 3 回合 D 确认；dev 五项 `min_norm`
  全部 ≥ 1.0（两个 seed），dev_param 533.04 / 532.43、dev_complex
  485.82 / 487.76，level / complexity 也全部高于 V5-FT。它是 v5
  首个通过旧 `min_norm ≥ 0.99` 的候选；test_v8 provisional
  496.36 / 496.40 vs V5-FT 496.37 / 496.22，尚未 consumption。
  `rl-full-guard-c` 用 V5-FT-Complex 基座，把 dev_complex 提到
  487.09 / 489.36、test 提到 496.40 / 496.44，五项 min_norm 仍
  全部 ≥ 1.0。FullGuard-C 已按 ADR 0014 作为推荐提交候选
  （test_v8 final 496.40 已消费），主权重保留 RL-PPO。
  不要把 v4 的 `test_v7` 或 dev 排名直接迁移到 v5。
- v5 与 v4 共用 catalog，只改变 tier 分配；`test_v8` 对旧 v4 候选不是
  held-out。新候选必须从 v5 `train` 重新训练/选模。
- Axelrod 完整 Meta 策略较慢；日常可用 `AxelrodMix` / 精选策略，完整 bank
  只在定向分析时使用。
- `test_v8` 已于 2026-10-30 消费一次（RL-FullGuard-C 496.40），
  `test_red` 仍锁定；新候选需要 `test_v9`。provisional 诊断仍用
  `--no-consume`，不写 ledger。
- Red-v5 / Red-v6 仍是 P3 legacy artifact；anti-exploitation 未解决。
