# Axelrod 内部矩阵迁移后的重训练与重评估

> 实验目录：`results/retrain_matrix_20261029/`、`weights/retrain_matrix_20261029/`
> 实验开始：2026-10-29 23:12（Asia/Shanghai），训练跨到 2026-10-30。
> 结论：初始 clean lineage **No-Go**；修复轮 PB 权重 3/13 后的
> **Dagger-Fix4** 达到 promotion 门槛（parity-level：两个训练 seed
> complex +0.02 / -0.22，独立 eval seed +0.14，min_norm 0.997，
> dev_axelrod -0.2 内）。新权重保留在
> `weights/retrain_matrix_20261029/`，未覆盖现有主权重。详见
> 第 8–9 节。

## 1. 背景与预检

仓库已把 Axelrod 对手的内部收益矩阵从 3/0/5/1 改为项目矩阵
6/0/10/1：

- `ipd/core/primitives.py`: R,S,T,P = 6,0,10,1；
- `ipd/core/opponents/axelrod_bank.py`: 通过
  `match_attributes["game"]` 把 6/0/10/1 传给所有 Axelrod 策略；
  ZD/LR 四向量按最大可行 phi 重算；
- `ipd/core/opponents/axelrod_legacy.py`: 按策略原理重写 6 组历史
  阈值；默认 3/0/5/1 下逐项还原原值。

本次没有修改矩阵、ZD phi 适配或 `axelrod_legacy.py`。工作树在实验
开始时已有上述迁移改动（dirty），实验只新增 results/weights 产物。

环境与预检：

| 项目 | 值 |
|---|---|
| git commit | `b0955cc1859c547fbb12837affd07330a34f839b` + matrix migration working tree |
| Python | 3.14.8 |
| Axelrod | 4.14.0 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU (24 GB) |
| `python3 -m ipd test` | **342 tests OK**（107.4 s） |
| `python3 -m ipd protocol audit` | **OK**（catalog 358 configs） |
| 内部矩阵断言 | **OK**：TitForTat `RPST() == (6,1,0,10)` |

产物目录：

```text
results/retrain_matrix_20261029/          # 日志、JSON、选模记录、报告
weights/retrain_matrix_20261029/          # 新部署权重（未覆盖 weights/ 主权重）
```

## 2. 矩阵变化的初步量化

### 2.1 同协议旧/新矩阵对照（受控）

为了把“矩阵效应”和“v5/v4 协议差异”分开，在 HEAD worktree（旧
Axelrod 内部 3/0/5/1；项目 payoff 仍是 6/0/10/1）与当前工作树
（新内部 6/0/10/1）上跑同一命令：15 fields × 2048、seed
20261027、v4 CPU sample、相同现有权重：

| 候选 | 矩阵 | dev_param | dev_axelrod | dev_ood |
|---|---|---:|---:|---:|
| RL-PPO | 旧 3/0/5/1 | 540.49 | 575.09 | 513.99 |
| RL-PPO | 新 6/0/10/1 | 541.35 | 580.22 | 514.58 |
| RL-PPO | diff | +0.85 | **+5.14** | +0.59 |
| RL-Pattern | 旧 3/0/5/1 | 548.46 | 583.21 | 520.43 |
| RL-Pattern | 新 6/0/10/1 | 548.85 | 584.89 | 520.44 |
| RL-Pattern | diff | +0.39 | **+1.69** | +0.01 |
| RL-Wide160 | 旧 3/0/5/1 | 548.28 | 583.38 | 522.30 |
| RL-Wide160 | 新 6/0/10/1 | 548.69 | 585.46 | 522.71 |
| RL-Wide160 | diff | +0.42 | **+2.09** | +0.41 |

受控对照确认：矩阵变化的收益集中在 `dev_axelrod`（+1.7～+5.1），
`dev_param`/`dev_ood` 只有 +0.0～+0.9；没有 >1 分退化。旧文档
v5 GPU 数字给出一致量级（RL-PPO +4.6、Wide160 +2.1），说明
v5/v4 协议差异不是主因。

### 2.2 旧文档 v5 参考

旧文档中的 v5 GPU 结果只列在这里，**不与新矩阵结果混表**：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| RL-PPO（旧 v5） | 540.35 | 582.53 | 440.31 | 574.64 | 513.48 |
| RL-Pattern（旧 v5） | 548.45 | 583.07 | 447.64 | 582.22 | 519.46 |
| RL-Wide160（旧 v5） | 548.26 | 583.08 | 454.31 | 582.38 | 521.37 |
| RL-Switch（旧 v5） | 548.58 | 582.91 | 454.24 | 582.86 | 521.42 |
| GRU-DAgger（旧 v5） | 548.58 | 582.90 | 455.13 | 582.49 | 521.54 |

新矩阵同候选的变化（新 v4 CPU − 旧 v5 GPU，仅作量级参考）：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| RL-PPO | +0.82 | +0.02 | -0.45 | **+4.56** | +0.39 |
| RL-Pattern | +0.30 | +0.02 | -0.08 | **+1.80** | +0.25 |
| RL-Wide160 | +0.29 | -0.00 | +0.10 | **+2.12** | +0.42 |
| RL-Switch | +0.39 | +0.00 | -0.00 | **+1.80** | +0.15 |
| GRU-DAgger | +0.42 | +0.03 | -0.20 | **+1.84** | +0.42 |

要点：

- `dev_axelrod` 普遍上移约 1.8–2.1 分（RL-PPO +4.6），超出 1 分噪声；
- `dev_param` / `dev_family` / `dev_complex` / `dev_ood` 基本在 ±0.5
  分内，没有出现 >1 分退化；
- 受控旧/新矩阵对照（2.1）与 v5 参考量级一致：矩阵效应本身是
  正向的，不是退化修复；
- 仍按任务要求重训包含 Axelrod 的 lineage（`train_v4` 含 10%
  `train_axelrod`，warm start 显式用 `train_axelrod`），以确认
  现有权重是否已足够、以及是否存在可复现的提升。

## 3. Phase 1：现有权重的 dev 重评（新矩阵）

命令：

```bash
python3 -m ipd protocol eval \
  --candidates rl,rl-repair,rl-pattern,rl-wide,rl-wide160,rl-switch,rl-dagger-gru,triple-v4 \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 \
  --baseline RL-PPO \
  --out results/retrain_matrix_20261029/baseline_dev.json
```

结果（seed 20261027，30 fields × 2048，v4 CPU sample）：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | min_norm |
|---|---:|---:|---:|---:|---:|---:|
| RL-PPO | 541.17 | 582.55 | 439.86 | 579.20 | 513.87 | baseline |
| RL-Repair | 547.99 | 582.78 | 434.06 | 583.39 | 515.33 | **0.471** |
| RL-Pattern | 548.75 | 583.09 | 447.56 | 584.02 | 519.71 | 0.848 |
| RL-Wide | 548.65 | 583.09 | 452.47 | 584.28 | 521.17 | 0.966 |
| RL-Wide160 | 548.55 | 583.08 | 454.41 | 584.50 | 521.79 | 0.997 |
| RL-Switch | 548.97 | 582.91 | 454.24 | 584.66 | 521.57 | 0.997 |
| GRU-DAgger | 549.00 | 582.93 | 454.93 | 584.33 | 521.96 | 0.997 |
| Triple-v4 | 538.58 | 583.07 | 447.08 | 568.55 | 514.14 | 0.876 |

`min_norm` 是各 tier 内相对 RL-PPO 的最差 family 归一化分数
（protocol `min_norm`）。RL-Repair 的 0.471 来自
`dev_complex / pattern_breaker`：234.6 vs RL-PPO 498.4。这是迁移前
就存在的 PB 漏洞，不是矩阵新引入的；RL-Pattern / Wide 系列仍需要
保留 PB 训练信号。

## 4. Phase 2：clean-lineage 重训

起点是 `weights/rl_policy_checkpoint_repair.json`（RL-Repair 训练
分布不含 Axelrod，是干净父模型）。所有中间产物写在
`results/retrain_matrix_20261029/`，最终部署权重写在
`weights/retrain_matrix_20261029/`。

### 4.1 RL-Pattern

20 iter entropy-regularized warm start（`train_complex` +
`train_axelrod`），再做 8 iter `train_v4` + `train_pattern_breaker`
定向训练：

```bash
python3 -m ipd train rl --device auto --iterations 20 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261029 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --train-dists original,late,grim30,wide,train_complex,train_axelrod \
  --dist-weights 1,1,1,2,2,1 \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --no-validation \
  --out results/retrain_matrix_20261029/pattern_warm_weights.json \
  --checkpoint-out results/retrain_matrix_20261029/pattern_warm_ac.json \
  --history-out results/retrain_matrix_20261029/pattern_warm_history.json

python3 -m ipd train rl --device auto --iterations 8 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261032 \
  --init-weights results/retrain_matrix_20261029/pattern_warm_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,4 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/retrain_matrix_20261029/pattern_ckpts \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/retrain_matrix_20261029/pattern_weights.json \
  --checkpoint-out results/retrain_matrix_20261029/pattern_ac.json \
  --history-out results/retrain_matrix_20261029/pattern_history.json
```

按原配方取 **`pattern_ckpts/iter_006.json`**，不使用 `--out` 的
iter8。训练内验证（2 fields × 256）selection：

| iter | 0 | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| balanced selection | 488.67 | 489.08 | 492.93 | 493.67 | **495.03** |

iter8 的验证 selection 更高，但为了复现原 RL-Pattern 配方并避免在
dev 上继续追高，仍固定选 iter6。

### 4.2 RL-Wide / RL-Wide160

```bash
python3 -m ipd train expand \
  --input results/retrain_matrix_20261029/pattern_ckpts/iter_006.json \
  --hidden 128 64 \
  --out results/retrain_matrix_20261029/wide_ac.json \
  --weights-out results/retrain_matrix_20261029/wide_weights.json \
  --seed 20261033

python3 -m ipd train rl --device auto --hidden 128 64 --iterations 8 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261034 \
  --init-weights results/retrain_matrix_20261029/wide_ac.json \
  --train-dists original,late,grim30,wide,train_v4 \
  --dist-weights 1,1,1,2,2 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/retrain_matrix_20261029/wide_gentle_ckpts \
  --bc-kl-coef 0.1 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/retrain_matrix_20261029/wide_gentle_weights.json \
  --checkpoint-out results/retrain_matrix_20261029/wide_gentle_ac.json \
  --history-out results/retrain_matrix_20261029/wide_gentle_history.json

python3 -m ipd train expand \
  --input results/retrain_matrix_20261029/wide_gentle_ckpts/iter_008.json \
  --hidden 128 128 \
  --out results/retrain_matrix_20261029/wide128_ac.json \
  --seed 20261035

python3 -m ipd train rl --device auto --hidden 128 128 --iterations 4 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261036 \
  --init-weights results/retrain_matrix_20261029/wide128_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/retrain_matrix_20261029/wide_balance_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/retrain_matrix_20261029/wide_balance_weights.json \
  --checkpoint-out results/retrain_matrix_20261029/wide_balance_ac.json \
  --history-out results/retrain_matrix_20261029/wide_balance_history.json

python3 -m ipd train expand \
  --input results/retrain_matrix_20261029/wide_balance_ckpts/iter_004.json \
  --hidden 160 128 \
  --out results/retrain_matrix_20261029/wide160_ac.json \
  --seed 20261039

python3 -m ipd train rl --device auto --hidden 160 128 --iterations 2 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261040 \
  --init-weights results/retrain_matrix_20261029/wide160_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/retrain_matrix_20261029/wide160_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/retrain_matrix_20261029/wide160_weights.json \
  --checkpoint-out results/retrain_matrix_20261029/wide160_ac_best.json \
  --history-out results/retrain_matrix_20261029/wide160_history.json
```

- gentle 8 iter 的验证没有改善（iter0 selection 490.97，iter8 489.62），
  仍按 docs 的 lineage 用 iter8 扩宽；这是本次最可疑的一步；
- 128/128 balance 取 `wide_balance_ckpts/iter_004.json`（history best
  iteration = 4）；
- 160/128 最终阶段取 **`wide160_ckpts/iter_001.json`**（history best
  iteration = 1；iter2 的 selection 更低）；
- 独立训练 seed：同一起点扩到 160/128（expand seed 20261042），
  2 iter balance 用 seed 20261041，取 `iter_001`。

### 4.3 RL-Switch

不训练新网络：前 6 回合用重训 RL-Wide160，若双方前 6 回合各
D>=2 且对手最近一步 C，余下回合用现有 RL-Repair。默认阈值保持
`window=6, mine>=2, opp>=2`。小规模 dev 扫描（5 fields × 1024，
`dev_param/dev_complex/dev_axelrod`）见
`switch_thresholds.json`；默认阈值在 dev_axelrod 最高（585.13），
complex 也最高（453.81），因此不改阈值。

### 4.4 GRU-DAgger

复用 `results/distill_20261043/dagger_switch.py`，只把教师替换为
重训 RL-Switch（重训 Wide160 + 现有 RL-Repair），其余超参数保持：

```bash
python3 results/retrain_matrix_20261029/dagger_switch_retrained.py \
  --student gru --rounds 2 --bc-fields 12 --dagger-fields 6 \
  --val-fields 3 --n 48 --hidden 64 --layers 1 \
  --epochs 30 --lr 1e-3 --seed 20261043 \
  --base-weights results/retrain_matrix_20261029/wide160_ckpts/iter_001.json \
  --alt-weights weights/rl_policy_weights_repair.json \
  --out weights/retrain_matrix_20261029/rl_dagger_gru_weights.json \
  --history-out results/retrain_matrix_20261029/dagger_history.json
```

第二个训练 seed 20261044 输出到
`weights/retrain_matrix_20261029/rl_dagger_gru_weights_seed20261044.json`。
两个 seed 的验证 BCE 分别为 0.1251 / 0.1211，switch 子集准确率
0.992 / 0.992。

### 4.5 选中的 checkpoint / seed 汇总

| 阶段 | 训练 seed | 选中 | 验证 selection |
|---|---:|---|---:|
| Pattern warm start | 20261029 | 20 iter（无验证） | — |
| Pattern targeted | 20261032 | `pattern_ckpts/iter_006.json` | 493.67（iter8 495.03 未用） |
| Wide 128/64 gentle | 20261034 | `wide_gentle_ckpts/iter_008.json` | 489.62（iter0 490.97） |
| Wide 128/128 balance | 20261036 | `wide_balance_ckpts/iter_004.json` | 489.58 |
| Wide160 main | 20261040 | `wide160_ckpts/iter_001.json` | 493.00 |
| Wide160 seed2 | 20261041 | `wide160_seed2_ckpts/iter_001.json` | 499.31 |
| GRU-DAgger main | 20261043 | 2 轮 DAgger 后最终 GRU | BCE 0.1251 |
| GRU-DAgger seed2 | 20261044 | 2 轮 DAgger 后最终 GRU | BCE 0.1211 |

## 5. Phase 3：Protocol v4 dev 复评

因为 stock `protocol eval` 只接受 registry 名称、不能同时给多个
新权重，使用 `eval_weights.py` 复用同一套 worker/aggregate/seed
schedule（seed 20261027，30 fields × 2048，v4 CPU sample）：

```bash
python3 results/retrain_matrix_20261029/eval_weights.py \
  --specs results/retrain_matrix_20261029/retrained_specs.json \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --baseline RL-PPO \
  --out results/retrain_matrix_20261029/retrained_dev.json
```

结果（seed 20261027，30 fields × 2048）：

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | min_norm |
|---|---:|---:|---:|---:|---:|---:|
| RL-PPO | 541.17 | 582.55 | 439.86 | 579.20 | 513.87 | 1.000 |
| RL-Pattern（现有） | 548.75 | 583.09 | 447.56 | 584.02 | 519.71 | 0.848 |
| RL-Wide160（现有） | 548.55 | 583.08 | 454.41 | 584.50 | 521.79 | 0.997 |
| RL-Switch（现有） | 548.97 | 582.91 | 454.24 | 584.66 | 521.57 | 0.997 |
| GRU-DAgger（现有） | 549.00 | 582.93 | 454.93 | 584.33 | 521.96 | 0.997 |
| RL-Pattern-RM | 548.97 | 583.03 | 448.24 | 584.07 | 519.90 | **0.836** |
| RL-Wide-RM | 548.76 | 583.02 | 451.26 | 583.93 | 520.84 | 0.916 |
| RL-Wide160-RM | 548.60 | 583.02 | 452.76 | 583.92 | 521.19 | 0.986 |
| RL-Wide160-RM-s2 | 548.51 | 583.02 | 453.41 | 583.91 | 521.42 | 0.996 |
| RL-Switch-RM | 548.85 | 582.87 | 452.99 | 583.92 | 521.11 | 0.985 |
| GRU-DAgger-RM | 548.88 | 582.89 | 453.54 | 583.87 | 521.46 | 0.996 |
| GRU-DAgger-RM-s2 | 548.84 | 582.90 | 453.48 | 583.74 | 521.36 | 0.996 |

相对各自现有同 lineage 候选的 diff：

| 候选 | 对照 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---|---:|---:|---:|---:|---:|
| RL-Pattern-RM | RL-Pattern | +0.22 | -0.05 | **+0.68** | +0.05 | +0.19 |
| RL-Wide-RM | RL-Wide160 | +0.21 | -0.05 | **-3.15** | -0.58 | -0.96 |
| RL-Wide160-RM | RL-Wide160 | +0.06 | -0.06 | **-1.65** | -0.59 | -0.60 |
| RL-Wide160-RM-s2 | RL-Wide160 | -0.04 | -0.06 | **-1.00** | -0.59 | -0.37 |
| RL-Switch-RM | RL-Switch | -0.12 | -0.04 | **-1.25** | -0.74 | -0.46 |
| GRU-DAgger-RM | GRU-DAgger | -0.12 | -0.04 | **-1.38** | -0.46 | -0.50 |
| GRU-DAgger-RM-s2 | GRU-DAgger | -0.15 | -0.03 | **-1.45** | -0.59 | -0.59 |

结论：只有 RL-Pattern-RM 没有 tier 级回退（complex +0.68），但它的
`dev_axelrod` 仅 +0.05、`dev_param` +0.22，且 `min_norm` 0.836
（pattern_breaker）远低于 0.99。Wide / Switch / Dagger 重训线在
`dev_complex` 全部回退 ≥1 分，不能 promotion。

### 5.1 dev_axelrod per-family

| 候选 | axelrod | axelrod_extra | AxelrodMix | anchor |
|---|---:|---:|---:|---:|
| RL-Pattern | 564.89 | 632.66 | 593.39 | 489.28 |
| RL-Pattern-RM | 565.38 | 632.89 | 592.90 | 488.87 |
| RL-Pattern-RM diff | +0.49 | +0.24 | -0.50 | -0.41 |
| RL-Wide160 | 564.56 | 631.53 | 596.62 | 489.40 |
| RL-Wide160-RM | 565.14 | 632.20 | 593.31 | 488.92 |
| RL-Wide160-RM diff | +0.57 | +0.67 | **-3.31** | -0.48 |
| RL-Wide160-RM-s2 | 565.13 | 632.09 | 593.40 | 488.97 |
| RL-Wide160-RM-s2 diff | +0.57 | +0.55 | **-3.22** | -0.43 |
| RL-Switch | 565.28 | 631.55 | 596.69 | 488.22 |
| RL-Switch-RM | 565.55 | 631.97 | 593.37 | 487.96 |
| RL-Switch-RM diff | +0.28 | +0.42 | **-3.32** | -0.26 |
| GRU-DAgger | 565.20 | 631.25 | 595.88 | 488.31 |
| GRU-DAgger-RM | 565.51 | 631.67 | 593.53 | 488.02 |
| GRU-DAgger-RM diff | +0.31 | +0.42 | **-2.35** | -0.30 |
| GRU-DAgger-RM-s2 | 565.45 | 631.35 | 593.44 | 488.07 |
| GRU-DAgger-RM-s2 diff | +0.25 | +0.10 | **-2.44** | -0.24 |

解读：

- 单策略 `axelrod` 与 `axelrod_extra` 普遍 +0.1～+0.8，说明新矩阵
  下重训没有破坏单策略对局；
- 但 `AxelrodMix` 一致回退 2.4～3.3 分，抵消了单策略收益，导致
  tier 总分略低于现有权重；Pattern-RM 的 mix 只 -0.50，是唯一
  接近持平的；
- ZD / Meta / 历史阈值策略的定向 panel 见 5.3（直接对局诊断，
  不参与选模）。

### 5.2 dev_complex per-family（关键 family）

| 候选 | pattern_breaker | lookahead | adaptive_mix | phase_switch | probe_exploit |
|---|---:|---:|---:|---:|---:|
| RL-Pattern | 422.42 | 353.42 | 291.92 | 491.81 | 276.27 |
| RL-Pattern-RM | 418.09 | 365.30 | 293.55 | 491.48 | 276.65 |
| RL-Pattern-RM diff | -4.33 | **+11.88** | +1.63 | -0.33 | +0.38 |
| RL-Wide160 | 516.23 | 378.16 | 294.98 | 491.07 | 272.07 |
| RL-Wide160-RM | 491.53 | 373.01 | 297.54 | 488.91 | 272.66 |
| RL-Wide160-RM diff | **-24.70** | -5.15 | +2.56 | -2.16 | +0.59 |
| RL-Wide160-RM-s2 | 505.01 | 373.23 | 297.19 | 487.87 | 271.67 |
| RL-Wide160-RM-s2 diff | **-11.22** | -4.93 | +2.21 | -3.20 | -0.40 |
| RL-Switch | 512.94 | 377.35 | 296.47 | 491.06 | 271.54 |
| RL-Switch-RM | 493.28 | 375.44 | 297.46 | 488.55 | 272.73 |
| RL-Switch-RM diff | **-19.67** | -1.91 | +0.99 | -2.50 | +1.20 |
| GRU-DAgger | 519.77 | 382.04 | 295.67 | 491.32 | 272.14 |
| GRU-DAgger-RM | 499.03 | 379.91 | 297.12 | 488.94 | 272.24 |
| GRU-DAgger-RM diff | **-20.75** | -2.13 | +1.45 | -2.38 | +0.10 |
| GRU-DAgger-RM-s2 | 499.31 | 377.95 | 296.92 | 489.11 | 272.53 |
| GRU-DAgger-RM-s2 diff | **-20.46** | -4.09 | +1.25 | -2.21 | +0.39 |

Pattern-RM 把 `lookahead` 拉起 +11.88，但 `pattern_breaker` 仍只有
418.09（现有 422.42），所以 min_norm 仍 0.836。Wide 线的 PB 回退
是本次所有 ≥1 分 complex 退化的主因。

### 5.3 ZD / Meta / 历史阈值定向 panel

直接对局诊断（每个策略 20 个 seed × 100 回合，分数尺度 0–1000；
不是 Protocol 采样，不参与选模）：

| 候选 | ZD (10) | Meta (11) | 历史阈值 (7) |
|---|---:|---:|---:|
| RL-PPO | 325.68 | 351.19 | 582.61 |
| RL-Pattern | 340.43 | 357.08 | 591.07 |
| **RL-Pattern-RM** | **342.50** | **360.19** | **592.29** |
| RL-Wide160 | 337.96 | 362.34 | 592.94 |
| **RL-Wide160-RM** | **338.88** | **363.02** | **593.89** |
| RL-Switch | 342.34 | 363.66 | 592.94 |
| GRU-DAgger | 342.59 | 360.95 | 592.21 |
| **GRU-DAgger-RM** | **343.14** | **362.04** | **593.95** |

解读：

- 重训确实让 ZD / Meta / 历史阈值三组都略有提升（Pattern 的 Meta
  +3.11，Dagger 的 historical +1.74，Wide160 的 historical +0.95）；
- 但 Protocol `dev_axelrod` 的 `AxelrodMix` 子族回退 2.4–3.3，
  抵消了这些单策略收益；这说明重训不是“没适应新矩阵”，而是
  优化目标与 protocol 的 mix 采样不一致；
- panel 只用于解释方向，不改变初始 lineage 的 No-Go 判定。

## 6. 独立 seed

独立 eval seed（20261029，30 fields × 2048，同样的 v4 CPU sample）：

| 候选 | eval seed | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | min_norm |
|---|---|---:|---:|---:|---:|---:|---:|
| RL-Pattern-RM | 20261027 | 548.97 | 583.03 | 448.24 | 584.07 | 519.90 | 0.836 |
| RL-Pattern-RM | 20261029 | 547.32 | 582.72 | 447.44 | 583.30 | 521.41 | 0.830 |
| RL-Wide-RM | 20261027 | 548.76 | 583.02 | 451.26 | 583.93 | 520.84 | 0.916 |
| RL-Wide-RM | 20261029 | 546.99 | 582.73 | 451.02 | 583.15 | 522.35 | 0.924 |
| RL-Wide160-RM | 20261027 | 548.60 | 583.02 | 452.76 | 583.92 | 521.19 | 0.986 |
| RL-Wide160-RM | 20261029 | 546.84 | 582.71 | 452.69 | 583.17 | 522.63 | 0.996 |
| RL-Wide160-RM-s2 | 20261027 | 548.51 | 583.02 | 453.41 | 583.91 | 521.42 | 0.996 |
| RL-Wide160-RM-s2 | 20261029 | 546.74 | 582.72 | 453.19 | 583.18 | 522.56 | 0.997 |
| RL-Switch-RM | 20261027 | 548.85 | 582.87 | 452.99 | 583.92 | 521.11 | 0.985 |
| RL-Switch-RM | 20261029 | 547.10 | 582.60 | 452.81 | 583.20 | 522.69 | 0.994 |
| GRU-DAgger-RM | 20261027 | 548.88 | 582.89 | 453.54 | 583.87 | 521.46 | 0.996 |
| GRU-DAgger-RM | 20261029 | 547.14 | 582.60 | 453.48 | 583.11 | 522.79 | 0.996 |
| GRU-DAgger-RM-s2 | 20261027 | 548.84 | 582.90 | 453.48 | 583.74 | 521.36 | 0.996 |
| GRU-DAgger-RM-s2 | 20261029 | 547.13 | 582.62 | 453.54 | 583.00 | 522.77 | 0.996 |

独立 seed 的字段整体偏移（RL-PPO 自身 param -1.67、ood +1.44），
但同一权重在 complex 上几乎不变（Wide160-RM -0.07；
GRU-DAgger-RM -0.07；两个 160 训练 seed 仍都是 ~452–453），
说明“重训 complex 落后现有 Wide160 约 1.4 分”不是单次字段
噪声。独立 seed 下与现有同 lineage 候选的直接对照（20261029）：

| 候选 | 对照 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---|---:|---:|---:|---:|---:|
| RL-Pattern-RM | RL-Pattern | +0.38 | -0.05 | +0.14 | +0.07 | +0.02 |
| RL-Wide-RM | RL-Wide160 | +0.26 | -0.05 | **-3.22** | -0.50 | -0.77 |
| RL-Wide160-RM | RL-Wide160 | +0.12 | -0.07 | **-1.55** | -0.48 | -0.49 |
| RL-Wide160-RM-s2 | RL-Wide160 | +0.01 | -0.06 | **-1.05** | -0.47 | -0.56 |
| RL-Switch-RM | RL-Switch | -0.16 | -0.02 | **-1.46** | -0.47 | -0.56 |
| GRU-DAgger-RM | GRU-DAgger | -0.02 | -0.05 | **-1.29** | -0.40 | -0.38 |
| GRU-DAgger-RM-s2 | GRU-DAgger | -0.03 | -0.04 | **-1.23** | -0.51 | -0.40 |

方向完全复现：Wide/Switch/Dagger 在独立 seed 下 complex 仍回退
1.2–3.2 分；Pattern-RM 仍然只是持平（+0.14 complex、+0.07
axelrod）。因此“重训更好”的方向不可复现，No-Go 不是单 seed 噪声。

## 7. test_v7 provisional

dev 选出的 top2 按 4 个非 family tier 的均值排序（family 几乎恒定）：

| 排名 | 候选 | mean(param, complex, axelrod, ood) |
|---|---|---:|
| 1 | GRU-DAgger-RM | 526.94 |
| 2 | GRU-DAgger-RM-s2 | 526.86 |
| 3 | RL-Wide160-RM-s2 | 526.81 |
| 4 | RL-Switch-RM | 526.72 |

只对 top2 跑 `test_v7` provisional（50 fields × 2048，`--no-consume`；
wrapper 不写 usage ledger）：

| 候选 | test_v7 | min field | p05 | worst family |
|---|---:|---:|---:|---:|
| GRU-DAgger-RM | 487.90 | 479.67 | 482.25 | 130.33 |
| GRU-DAgger-RM-s2 | 487.70 | 479.98 | 482.23 | 130.56 |

参照（旧矩阵、仅历史对照，不混表）：现有 GRU-DAgger 的 test_v7
provisional 为 487.89 / 487.02（两个 seed），RL-Switch 488.26。
重训 top2 的 test 没有崩坏，基本持平；但 test 永久锁定，不能作为
promotion 依据，也没有参与初始 lineage 的 No-Go 判定。

## 8. 修复轮（2026-10-30）

初始 clean lineage 判定 No-Go 后，继续做了定向修复。根因不是矩阵
迁移，而是重训线的 `pattern_breaker`（PB）信号不足：

- Pattern iter6 的 PB 只有 418（quick eval），而 iter8 的 PB 是
  507；原配方固定 iter6，导致后续 Wide 线的 PB 起点偏低；
- 128/128 balance 的 PB 权重只有 1/11，最终 160/128 阶段也没有
  把 PB 拉回现有 Wide160 的水平；
- 初始 Wide160-RM PB 491.5 vs 现有 516.2（-24.7），是 complex
  回退的主因。

### 8.1 修复尝试

| 尝试 | 改动 | 关键结果 | 判定 |
|---|---|---|---|
| Fix1 | 补上原文档缺失的 128/128 gentle 阶段 | quick complex 452.80，PB 484.2 | 仍低于现有，失败 |
| Fix2 | 从 Fix1 做 PB 4/13，4 iter，取 iter3 | full complex **456.71**，但 min_norm **0.9495**（lookahead 347.3） | min_norm 失败 |
| Fix3 | PB 2/13，4 iter，取 iter2 | full complex 454.39，min_norm **0.9869** | min_norm 失败 |
| Fix4-iter1 | PB 3/13，3 iter，取 iter1 | full complex 453.46，min_norm **0.9972**；但独立 seed complex -1.18 | 主 seed 过、独立 seed 边缘失败 |
| Fix4-iter2 | 同上，取 iter2 | full complex 454.82，min_norm **0.9783** | min_norm 失败 |
| **Dagger-Fix4** | 从 Fix4-iter1 蒸馏 GRU-DAgger（2 轮 DAgger） | full complex **454.95**，min_norm **0.9971**；seed2 complex **454.90**，min_norm 0.9970 | **通过** |

关键取舍：PB 权重越高，PB 越高，但 lookahead 越低。Fix2 的 PB
616 换来 lookahead 347（min_norm 0.9495）；Fix4-iter1 的 PB 511、
lookahead 368（min_norm 0.9972）是能过 0.99 的平衡点；
Dagger-Fix4 在这个平衡点上进一步把 PB 提到 533.7，同时 lookahead
保持 369.3。

### 8.2 Dagger-Fix4 全量 dev

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | min_norm |
|---|---:|---:|---:|---:|---:|---:|
| GRU-DAgger（现有） | 549.00 | 582.93 | 454.93 | 584.33 | 521.96 | 0.9966 |
| Dagger-Fix4（train seed 20261048） | 548.83 | 582.88 | **454.95** | 584.15 | 521.82 | **0.9971** |
| diff | -0.17 | -0.05 | **+0.02** | -0.18 | -0.14 | +0.0005 |
| Dagger-Fix4-s2（train seed 20261049） | 548.91 | 582.88 | 454.71 | 584.17 | 521.77 | **0.9972** |
| diff | -0.09 | -0.05 | -0.22 | -0.16 | -0.19 | +0.0006 |

两个训练 seed 的 complex 为 +0.02 / -0.22，均在 1 分内；PB
533.74 / 523.98（现有 519.77），lookahead 369.27 / 375.65
（现有 382.04），但都高于 RL-PPO 的 365.74。训练 seed 之间的差异
说明这是 parity-level 修复，而不是稳定的大幅提升。

### 8.3 独立 seed（20261029）

| 候选 | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood | min_norm |
|---|---:|---:|---:|---:|---:|---:|
| GRU-DAgger（现有） | 547.16 | 582.65 | 454.76 | 583.51 | 523.17 | 0.9965 |
| Dagger-Fix4 | 547.15 | 582.61 | **454.90** | 583.35 | 523.12 | **0.9970** |
| diff | -0.01 | -0.04 | **+0.14** | -0.16 | -0.05 | +0.0005 |

两个 eval seed 的 complex 都是正方向（+0.02 / +0.14），其它 tier
全部在 1 分内；这是“parity-level 但方向可复现”的修复。

### 8.4 test_v7 provisional（top2，未消费）

dev top2：Dagger-Fix4、Fix4-Wide160。50 fields × 2048，
`--no-consume`：

| 候选 | test_v7 | min field | p05 | worst family |
|---|---:|---:|---:|---:|
| **Dagger-Fix4** | 487.77 | 479.45 | 482.34 | 130.82 |
| Fix4-Wide160 | 486.32 | 478.21 | 480.68 | 130.92 |

参照现有 GRU-DAgger 487.89 / 487.02、RL-Switch 488.26；Dagger-Fix4
没有崩坏，处于同一水平。test 仅诊断，不参与选模。

## 9. 判定（更新）

**初始 clean lineage：No-Go（保留原表，见第 5–6 节）。**
Wide/Switch/Dagger 的初始重训线在 `dev_complex` 回退 1.25–3.15，
min_norm 0.836–0.986，不能 promotion。

**修复轮：Dagger-Fix4 达到 promotion 门槛，结论为 Go（候选级）。**

| 条件 | Dagger-Fix4 结果 |
|---|---|
| dev_axelrod 不低于 baseline（1 分内） | 通过：-0.18（主 seed）、-0.16（独立 seed） |
| param/family/complex/ood 无 >1 分退化 | 通过：最大退化 -0.22；complex +0.02（主）/ +0.14（独立 eval）/ -0.22（第二训练 seed） |
| min_norm ≥ 0.99 | 通过：0.9971 / 0.9970 / 0.9972 |
| 独立 seed 方向可复现 | 通过（parity）：两个训练 seed 与两个 eval seed 都在 ±0.25 内，没有复现初始 -1.3 级回退 |
| Python/C++ parity | 通过：新 GRU `max_abs_diff=0`；MLP/switch 也为 0 |
| 完整测试 | 通过：342 tests OK |
| test_v7 provisional 不崩坏 | 通过：487.77（诊断 only） |

结论：**建议把 Dagger-Fix4 作为 promotion 候选注册为
`GRU-DAgger-Fix4` / 新单模型候选**。它是从重训 Wide160（PB 3/13
修复）蒸馏出的 GRU，不是从旧权重 warm-start 的 adaptation；相对
现有 GRU-DAgger 的主要收益是 PB +14/+4、min_norm 略高，complex 仅
parity-level（+0.02 / -0.22 两个训练 seed；独立 eval +0.14），不是
大幅提升。

由于任务硬约束“不覆盖 weights/ 下现有主权重”，本次**没有**替换
`weights/rl_dagger_gru_weights.json` 或 `weights/rl_policy_weights.json`；
新权重保留在 `weights/retrain_matrix_20261029/rl_dagger_gru_weights_fix4.json`。
是否把它设为默认主权重，需要按 ADR 0002 显式确认后再做。

矩阵迁移本身可以保留：受控旧/新矩阵对照显示 `dev_axelrod`
+1.7～+5.1，其它 tier 稳定。初始重训失败是 PB 信号不足，修复轮
已经给出可通过门槛的候选。

### 9.1 部署一致性（诊断）

虽然初始 lineage 是 No-Go，修复轮仍对新权重做了 native/Python bit-exact parity
（`tests/test_policy_native.py`、`test_switch_cpp.py`、
`test_dagger_gru_cpp.py` 也在 342 tests 中通过）：

| 权重 | 类型 | max_abs_diff |
|---|---|---:|
| `rl_policy_weights_pattern_rm.json` | MLP | 0 |
| `rl_policy_weights_wide_rm.json` | MLP | 0 |
| `rl_policy_weights_wide160_rm.json` | MLP | 0 |
| `rl_policy_weights_wide160_rm_s2.json` | MLP | 0 |
| `rl_dagger_gru_weights.json` | GRU | 0 |
| `rl_dagger_gru_weights_seed20261044.json` | GRU | 0 |
| `rl_policy_weights_wide160_fix4.json` | MLP | 0 |
| `rl_dagger_gru_weights_fix4.json` | GRU | 0 |
| retrained RL-Switch | switch | 0 |

本次没有导出或覆盖 `cpp/` header；Dagger-Fix4 的 Python/native
parity 已为 0，但正式接 C++/比赛接口前仍需按 ADR 0002 重新导出并
复核第 1/100 天行为。

## 10. 复现与产物

### 10.1 关键产物

| 路径 | 内容 |
|---|---|
| `baseline_dev.json` | 现有 8 候选的新矩阵 dev（第 3 节） |
| `retrained_dev.json` | 重训 + 现有对照的 dev 合并结果；per-tier 原始文件 `retrained_dev_*.json` |
| `seed2_dev.json` / `seed2_current.json` | 独立 eval seed（20261029）下重训候选与现有候选 |
| `new_matrix_control_small.json` / `old_matrix_control_small.json` | 同协议旧/新矩阵受控对照（第 2.1 节） |
| `test_v7_provisional.json` / `test_v7_fix4_provisional.json` | 初始 top2 / 修复轮 top2 的 test_v7 provisional（均不消费） |
| `fix2_dev.json` / `fix3_dev.json` / `fix4_dev.json` / `dagger_fix4_dev.json` | 修复轮各自的 5-tier dev 结果 |
| `fix4_seed2.json` / `dagger_fix4_seed2.json` | 修复轮独立 eval seed |
| `dagger_fix4_s2_dev.json` | Dagger-Fix4 第二训练 seed（20261049）的全量 dev |
| `parity_retrained.json` | 新权重 native/Python parity（全部 0） |
| `summary.json` | 机器可读的关键数字 + 修复轮结论 |
| `switch_thresholds.json` | RL-Switch 阈值小扫描（默认阈值保留） |
| `axelrod_panel.json` | ZD / Meta / 历史阈值定向诊断 |
| `pattern_history.json` / `wide_gentle_history.json` / `wide_balance_history.json` / `wide160_history.json` | 训练曲线和 selection |
| `pattern_ckpts/` / `wide_gentle_ckpts/` / `wide_balance_ckpts/` / `wide160_ckpts/` / `wide160_seed2_ckpts/` | 初始 clean lineage checkpoints |
| `fix2_pb_ckpts/` / `fix3_pb_ckpts/` / `fix4_pb_ckpts/` | PB 修复轮 checkpoints |
| `fix4_pb_history.json` / `dagger_fix4_history.json` | 修复轮训练曲线 |
| `export_manifest.json` | 新部署权重路径与 sha256 |
| `run_phase2.sh` / `run_dagger.sh` / `run_dev_eval.sh` / `run_seed2_eval.sh` / `run_seed2_current.sh` / `run_matrix_control.sh` | 初始 lineage 与对照的完整命令 |
| `run_fix_lineage.sh` / `run_fix2_dev.sh` / `run_fix3_dev.sh` / `run_fix4_dev.sh` / `run_fix4_seed2.sh` / `run_dagger_fix4_dev.sh` / `run_dagger_fix4_seed2.sh` | 修复轮完整命令 |
| `parity_retrained.py` / `eval_axelrod_panel.py` / `eval_weights.py` | parity、panel、任意权重评测脚本 |

### 10.2 新权重（实验产物，未覆盖主权重）

| 文件 | sha256 |
|---|---|
| `weights/retrain_matrix_20261029/rl_policy_weights_pattern_rm.json` | `c454bd739bb8f5e08f1903c6caf3ca43b26dafc0369026f90049e94d3e6478e4` |
| `weights/retrain_matrix_20261029/rl_policy_weights_wide_rm.json` | `c2891dc529f5e495ad633e68ceebbf58bf87552436030fb6bfc8fd941ab499d3` |
| `weights/retrain_matrix_20261029/rl_policy_weights_wide160_rm.json` | `eff57924c5bb20797862821afdd6f717993c3341247101fc23740dbeeb1c3864` |
| `weights/retrain_matrix_20261029/rl_policy_weights_wide160_rm_s2.json` | `1c35dfd6f9afd2582b9f6c62e0dd98c3d0b54bd9a2ef6813e0895a27ad910996` |
| `weights/retrain_matrix_20261029/rl_policy_weights_wide160_fix4.json` | `e7186370a17ca50712cf48a6df63ce9923a86717d14f47dfaab542bcaa79d34d` |
| `weights/retrain_matrix_20261029/rl_dagger_gru_weights_fix4.json` | `3c6b25b393c44adcdc8e8e56bb47934e409f918ecfc7709bf18e34dc806f55c9` |
| `weights/retrain_matrix_20261029/rl_dagger_gru_weights_fix4_seed20261049.json` | `cf831f1114b0a4a5e00c342ece2ceb7cbfaa0069f9e64241c9cb392493c4f1c4` |
| `weights/retrain_matrix_20261029/rl_dagger_gru_weights.json` | `ea7317f3d9e0e86a5adc2f1875ca389e54011f917366bfcdd4657af047ce6859` |
| `weights/retrain_matrix_20261029/rl_dagger_gru_weights_seed20261044.json` | `d71c473d4ae04b95e46ec41d53a7371511c853d7b9cd01aaf84e7c57077b82a7` |

`weights/` 下原有主权重和所有现有候选权重均未修改。

RL-Switch-RM 是规则 meta（`rl-switch`），没有单独的权重文件：它由
`rl_policy_weights_wide160_rm.json`（或对应 checkpoint）+
`weights/rl_policy_weights_repair.json` 组成，阈值保持默认
`window=6, mine>=2, opp>=2`。

### 10.3 完整命令

见 `run_phase2.sh`、`run_dagger.sh`、`run_dev_eval.sh`、
`run_seed2_eval.sh`、`eval_weights.py`、`eval_axelrod_panel.py`。
所有 seed、迭代数、选中的 checkpoint 均已写在上面对应小节。
