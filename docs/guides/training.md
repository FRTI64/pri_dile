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

### 3.4 Protocol v3 训练入口

新候选不再用 `original,late,grim30,wide` 作为训练分布，而是：

```bash
python3 -m ipd train rl --device auto --iterations 40 \
  --train-dists train_v3 \
  --dists dev_param,dev_family,dev_complex,dev_ood \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --out weights/my_protocol_v3_policy.json \
  --checkpoint-out weights/my_protocol_v3_checkpoint.json
```

训练 profile：

- `train_v3`：标准 + complex + policy 的完整混合；
- `train_easy` / `train_hard`：课程分段；
- `train_complex`：complex families；
- `train_policy`：policy opponents only。

`dev_param` / `dev_family` / `dev_complex` / `dev_ood` 只用于 validation。
不要传 `test_v6`，训练代码会拒绝它。

PPO 的 MDP、网络、损失、验证指标见 [../policies/rl.md](../policies/rl.md)；
E1 的 Gate 和结果见 [../research/e1-robust-finetune.md](../research/e1-robust-finetune.md)。

## 4. Self-play

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

## 5. C2/C3 对手模型与 MoE

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

## 6. 产物与纪律

- 权重写 `weights/`，实验报告/曲线写 `results/<experiment>/`；
- `.pt` checkpoint 被 `*.pt` 忽略，提交代码 + JSON 权重 + 报告；
- 不覆盖主权重，除非满足
  [../decisions/0002-no-promotion-without-reproduction.md](../decisions/0002-no-promotion-without-reproduction.md)；
- 训练结果必须报告 official、blind、held-out、CVaR、min_norm、red；
- 固定 seed 并记录完整命令；详细表格写入对应 `results/*/README.md`。
