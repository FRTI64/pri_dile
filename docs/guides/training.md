# 训练指南

训练需要 PyTorch；推理和 C++ 部署不需要。设备选择顺序为
**NPU → CUDA → MPS → CPU**（`--device auto`）；NPU 路径尚未实机验证。

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt
python3 -m ipd train --list
```

## 1. 任务总览

| 任务 | 入口 | 用途 |
|---|---|---|
| `neural` | `train neural` | MLP 行为克隆 + DAgger |
| `gru` | `train gru` | GRU-Net BC + DAgger |
| `rl` | `train rl` | PPO（主权重、A60、RL-Repair、red/blue response） |
| `psro` | `train psro` | PSRO：blue-vs-red 元博弈 + PPO best response |
| `adversarial` | `train adversarial` | 约束剥削规则搜索 |
| `opponent-model` | `train opponent-model` | C2/C3 对手模型 / gate |
| `selfplay` | `train selfplay` | 21 维非对称自博弈 |
| `selfplay-bc` | `train selfplay-bc` | 21 维 GRU 的 BC warm start |
| `selfplay-compare` | `train selfplay-compare` | fair held-out 对比 |
| `selfplay-red-eval` | `train selfplay-red-eval` | frozen red best response |
| `multiobj` / `wide` / `triple` / `bayes` / `joint` | `train <task>` | 规则参数搜索 |

所有任务接受 `--seed`、`--out`；学习任务额外接受 `--device`、
`--history-out`，并暴露 `main(argv=None)`。

## 2. MLP / GRU

```bash
python3 -m ipd train neural --device auto \
  --fields-per-dist 12 --dagger-fields-per-dist 6 \
  --n 60 --epochs 40 --workers 20

python3 -m ipd train gru --device auto \
  --fields-per-dist 12 --dagger-fields-per-dist 6 \
  --n 60 --epochs 40 --hidden 32 --layers 1
```

产物：

- `weights/neural_policy_weights.json` / `weights/gru_policy_weights.json`；
- `results/neural_training_history.json` / `results/gru_training_history.json`。

训练细节和结果见 [../policies/neural.md](../policies/neural.md) 与
[../policies/gru.md](../policies/gru.md)。

## 3. PPO

### 3.1 从零 + hard（主权重）

```bash
# 先训练从零 base（30 iter）
python3 -m ipd train rl --device auto --from-scratch \
  --iterations 30 --fields-per-dist 5 --n 80 \
  --dist-weights 1,1,1,2 --ent-coef 0.001 --lr 3e-4 \
  --center-advantages --select-metric avg \
  --out weights/rl_policy_weights_scratch_base.json \
  --checkpoint-out weights/rl_policy_checkpoint_scratch_base.json

# 再加 hard（40 iter）
python3 -m ipd train rl --device auto --iterations 40 \
  --init-weights weights/rl_policy_checkpoint_scratch_base.json \
  --train-dists original,late,grim30,wide,hard \
  --dist-weights 1,1,1,2,1 --ent-coef 0.0001 --lr 1e-4 \
  --center-advantages --bc-kl-coef 0.1 --target-kl 0.02 \
  --select-metric blind \
  --out weights/rl_policy_weights.json \
  --checkpoint-out weights/rl_policy_checkpoint.json \
  --history-out results/rl_training_history.json
```

### 3.2 A60（hard×2）

```bash
python3 -m ipd train rl --device auto --seed 20261030 --iterations 60 \
  --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,hard --dist-weights 1,1,1,2,2 \
  --init-weights weights/rl_policy_checkpoint.json \
  --bc-kl-coef 0.1 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 --select-metric avg \
  --out weights/rl_policy_weights_hard60.json \
  --checkpoint-out weights/rl_policy_checkpoint_hard60.json
```

### 3.3 E1c repair（RL-Repair）

```bash
python3 -m ipd train rl --device auto --seed 20261032 \
  --iterations 20 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,repair \
  --dist-weights 1,1,1,2,2 \
  --init-weights weights/rl_policy_checkpoint_robust_b.json \
  --bc-kl-coef 0.05 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --save-all-checkpoints results/robust_20261026/E1c_checkpoints \
  --out weights/rl_policy_weights_repair.json \
  --checkpoint-out weights/rl_policy_checkpoint_repair.json \
  --history-out results/robust_20261026/E1c_history.json
```

### 3.4 Protocol v5 训练入口

新候选不再用 `original,late,grim30,wide` 作为训练分布，而是：

```bash
python3 -m ipd train rl --device auto --iterations 40 \
  --train-dists train_v5 --n 160 \
  --dists dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --val-fields-per-dist 5 --val-n 160 --val-every 5 \
  --select-metric avg \
  --out weights/my_protocol_v5_policy.json \
  --checkpoint-out weights/my_protocol_v5_checkpoint.json
```

训练 profile：

- `train_v5`：标准 + complex + policy + Axelrod 的完整混合；
  `train_v4` / `train_v3` 是历史别名，解析到当前 v5 train；
- `train_easy` / `train_hard`：课程分段；
- `train_complex`：complex families；
- `train_pattern_breaker`：只含 `pattern_breaker` 的 `train_complex` 参数点；
- `train_probe_pattern`：只含 `probe_exploit` / `pattern_breaker` 的
  `train_complex` 参数点（v5 定向训练用）；
- `train_probe_only`：只含 `probe_exploit` 的 `train_complex` 参数点
  （probe specialist 实验用，含 anchors）；
- `train_weak_complex`：`pattern_breaker` / `adaptive_mix` /
  `probe_exploit` / `conditional_extort` / `lookahead`；
- `train_policy`：policy opponents only；
- `train_axelrod`：Axelrod 策略 + 组合。

`dev_param` / `dev_family` / `dev_complex` / `dev_axelrod` / `dev_ood`
只用于 validation。
不要传 `test_v8`，训练代码会拒绝它。

`train_v5` 和 `train_axelrod` 含 Axelrod 对手，需要先安装
`requirements-axelrod.txt`；不用 Axelrod 时改用 `train_easy` /
`train_hard` / `train_complex`。

另有训练专用分布 `hard` / `hard2` / `robust` / `repair` /
`extort_hard` / `periodic_hard`。`extort_hard` 连续采样 ExtortZD
chi 1.5–7.0，`periodic_hard` 采样 Kamikaze / Periodic /
DynamicWindow，两者都保留 anchors 且不进入 Protocol catalog。
实测在 RL-Wide160 上均为 No-Go（extort/periodic 小幅提升但
PB/complex 回退），见 [../research/extort-hard.md](../research/extort-hard.md)
和 [../research/periodic-hard.md](../research/periodic-hard.md)。

PPO 的 MDP、网络、损失、验证指标见 [../policies/rl.md](../policies/rl.md)；
E1 的 Gate 和结果见 [../research/e1-robust-finetune.md](../research/e1-robust-finetune.md)。

### 3.5 宽度扩展（net2wider）

`train expand` 把已有 MLP 的隐藏层加宽，避免从零重训：

```bash
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_pattern.json \
  --hidden 128 64 \
  --out /tmp/wide_ac.json \
  --weights-out /tmp/wide_weights.json
```

扩展后再用 `--hidden 128 64` / `--hidden 128 128` / `--hidden 160 128`
继续 PPO。当前最均衡的是 160/128；详细配方、消融和结果见
[../research/rl-wide160.md](../research/rl-wide160.md) 与
[../research/rl-wide.md](../research/rl-wide.md)。

输入特征 v2（85→97）也已接入：

```bash
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_wide160.json \
  --features v2 \
  --out /tmp/wide160_v2_ac.json

python3 -m ipd train rl --features v2 --hidden 160 128 \
  --init-weights /tmp/wide160_v2_ac.json ...
```

v2 在 RL-Wide160 上是 No-Go（dev_complex +2～4，但 dev_param/test
回退），见 [../research/feature-v2.md](../research/feature-v2.md)。

## 4. PSRO

完整 PSRO 使用独立的 blue population 和 red exploiter population：

1. cross-play 估计 `blue_score - red_score` 元博弈；
2. blue 求 maximin 混合，red 求对偶 maximin 混合；
3. 把 red 训练混合写成 `--opponent-pool` manifest；
4. 用现有 PPO trainer 训练一个 blue best response；
5. 加入 blue population，重新求解；可重复多轮。

```bash
python3 -m ipd train psro --config configs/psro/psro_20261027.json \
  --workers 20

# 更强 oracle（当前头部 blue；3 轮 × 30 iter、40% red、低 KL）
python3 -m ipd train psro --config configs/psro/psro_v4_20261029.json \
  --workers 20

# 只评估初始元博弈，不训练 oracle
python3 -m ipd train psro --config configs/psro/psro_20261027.json \
  --dry-run --out /tmp/psro_dry

# 诊断评估（小规模，不是完整 v4 baseline）
bash results/psro_20261027/run_eval.sh
```

产物：

- `results/psro_20261027/report.json`、`population.json`、每轮
  `br_XX.json` / `br_XX_history.json`；
- `weights/psro/meta_strategy.json`：blue/red 混合与 population；
- `weights/psro/psro_policy.json`：最后一轮 best response（`psro-br`）；
- 注册候选 `psro`（meta-policy，每场按 blue 混合采样）和
  `psro-br`（单策略）。

首轮结果：blue maximin 选择 Final-tuned + Triple-majority；PPO oracle
未进入 support，dev_param / held-out red 是明显 trade-off。**实验候选，
未 promotion**；方法与结果见 [../research/psro.md](../research/psro.md)。
若继续，优先改 oracle 或加入 red oracle（double oracle），不要重复同一配置。

### 4.1 从零 PSRO（existing models 只做 red）

```bash
# 1. 从零官方 base
python3 -m ipd train rl --device auto --from-scratch --iterations 30 \
  --fields-per-dist 5 --n 80 --dist-weights 1,1,1,2 \
  --ent-coef 0.001 --lr 3e-4 --center-advantages --select-metric avg \
  --seed 20262027 \
  --out weights/psro_scratch/base_policy.json \
  --checkpoint-out weights/psro_scratch/base_checkpoint.json \
  --history-out results/psro_scratch_20261027/base_history.json

# 2. PSRO：blue 只有 random/scratch base/BR；已有模型只进 red 池
python3 -m ipd train psro \
  --config configs/psro/psro_scratch_20261027.json --workers 20
```

`psro-scratch` / `psro-scratch-br` 是产物。诊断：`dev_param` 480.43 /
min_norm 0.55，但 `dev_red` 和 held-out Red-v5/v6 的 red win 降到 **0.0**
（首次翻转 red 结果）。official 代价太大，**未 promotion**；详见
[../research/psro-scratch.md](../research/psro-scratch.md)。

`configs/psro/psro_scratch_pilot_20261027.json` 是更激进的 pilot
（dev 473，red_win 0）；`psro_scratch_v2/v3` 是 ablation。

## 5. Self-play

```bash
# 200 episode smoke
python3 -m ipd train selfplay --smoke --device auto

# 完整三阶段课程
python3 -m ipd train selfplay --config configs/selfplay/full.json \
  --device auto --final-test

# 21 维 GRU 的 BC warm start
python3 -m ipd train selfplay-bc --device auto --episodes 4000 --epochs 10
```

注意：self-play 的改进实验已判 No-Go；最终 SelfPlay-GRU 是 BC 蒸馏策略，
不是从零 self-play 的产物。详见 [../policies/selfplay.md](../policies/selfplay.md)。

## 6. C2/C3 对手模型与 MoE

```bash
python3 -m ipd train opponent-model --data-mode mixed \
  --episodes 2 --explore-episodes 2 --counterfactual-episodes 1 \
  --counterfactual-branches 8 --counterfactual-horizon 6 \
  --counterfactual-prefix explore,base

python3 -m ipd moe-eval --suite validation,heldout_v2 --seeds 3 \
  --gate-model weights/opponent_gate_threeway.json \
  --safe-expert alld --safe-threshold 0.6 --min-day 10
```

C2/C3 全部离线 No-Go；这些命令只用于复现对照，不要继续扫 threshold。

## 7. 产物与纪律

- 权重写 `weights/`，实验报告/曲线写 `results/<experiment>/`；
- `.pt` checkpoint 被 `*.pt` 忽略，提交代码 + JSON 权重 + 报告；
- 不覆盖主权重，除非满足
  [../decisions/0002-no-promotion-without-reproduction.md](../decisions/0002-no-promotion-without-reproduction.md)；
- 训练结果必须报告 official、blind、held-out、CVaR、min_norm、red；
- 固定 seed 并记录完整命令；详细表格写入对应 `results/*/README.md`。
