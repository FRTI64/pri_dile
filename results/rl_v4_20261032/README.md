# RL-Wide160（2026-10-31）

> 160/128 宽度细调：从 RL-Wide 128/128 扩展第一层到 160，再做 2 iter
> balance PPO（PB 权重 1/11）。**实验候选，未 promotion。**

## 结论

| 集 | RL-Pattern | RL-Wide 128/128 | **RL-Wide160** |
|---|---:|---:|---:|
| dev_param | **548.45** | 548.24 | 548.26 |
| dev_family | 583.07 | **583.08** | **583.08** |
| dev_complex | 447.64 | 452.79 | **454.31** |
| dev_axelrod | 582.22 | 582.30 | **582.38** |
| dev_ood | 519.46 | 520.71 | **521.37** |
| test_v7（provisional） | 485.43 | 485.50 | **485.62** |

`min_norm`（vs RL-PPO）：

| 候选 | dev_param | dev_complex | dev_ood |
|---|---:|---:|---:|
| RL-Wide | 0.9968 | 0.9726 | 0.9596 |
| **RL-Wide160** | 0.9970 | **0.9974** | **0.9966** |

- 相对 RL-Wide：dev_param +0.02、dev_complex **+1.52**、dev_ood **+0.66**、
  dev_axelrod +0.08；test_v7 +0.12；
- 相对 RL-Pattern：dev_param -0.19、dev_complex **+6.67**、dev_ood +1.91；
- complex / OOD 的 `min_norm` 基本回到 1.0，**没有新的最差 family**；
- `dev_red`（30 seeds）：blue 378.40 / red 711.73，red_win 仍 100%，
  anti-exploitation 未改善；
- `test_red` provisional（30 seeds）：blue 319.43 / red 742.43，
  red_win 100%；比 RL-Repair blue 高 6.63，但仍被 100% 击败；
- 分级诊断：`dev_level3` 451.52、`dev_level4` 447.54、
  `dev_complexity4` 401.27；低难度 tier 有 0.1–0.8 的小幅回退；
- legacy 20-field：四分布 avg **554.77**、blind **564.75**；
- C++ 导出验证通过（`N_HIDDEN0=160` / `N_HIDDEN1=128`）。

## 独立 seed（20261029）

| 候选 | dev_param | dev_complex | dev_ood | complex min_norm | ood min_norm |
|---|---:|---:|---:|---:|---:|
| RL-Wide | 546.60 | 452.05 | 522.14 | 0.9744 | 0.9453 |
| **RL-Wide160** | 546.39 | **454.02** | **522.99** | **0.9976** | **0.9971** |

dev_param 与 RL-Wide 基本持平（-0.21），complex/OOD 收益和 min_norm
改善均可复现。

## 文件

| 文件 | 内容 |
|---|---|
| `protocol_v4_dev_rl_wide160.json` | **官方 dev 结果**（5 候选） |
| `protocol_v4_test_v7_rl_wide160_provisional.json` | **官方 provisional test** |
| `protocol_v4_dev_repro_seed20261029_rl_wide160.json` | 独立 seed 复现 |
| `protocol_v4_dev_red_rl_wide160.json` | dev_red（30 seeds） |
| `protocol_v4_test_red_rl_wide160_provisional.json` | test_red provisional（未消费） |
| `protocol_v4_dev_level_rl_wide160.json` | `dev_level0..4` 分级诊断 |
| `protocol_v4_dev_complexity_rl_wide160.json` | `dev_complexity0..4` 分级诊断 |
| `wide160_ckpts/iter_001.json` | 最终 actor-critic checkpoint |
| `wide160_history.json` | 2 iter 训练曲线 |
| `wide160_iter1_*.json` | 固定 seed 逐 tier / test 诊断 |

最终部署权重：

- `weights/rl_policy_weights_wide160.json`
  sha256 `dca9006c9fce5da0a7a152ed1935ad1ebb8176c56a485772d5e77fccc0eb682d`
- `weights/rl_policy_checkpoint_wide160.json`
  sha256 `6e93b60f1ad6d13ad84e455c58a18641b07e9ea54c10ea6a5e58f908eea133ab`

## 复现

```bash
# 128/128 -> 160/128
python3 -m ipd train expand \
  --input weights/rl_policy_checkpoint_wide.json \
  --hidden 160 128 \
  --out results/rl_v4_20261032/wide160_ac.json \
  --seed 20261033

# 2 iter balance；最终取 iter_001
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 2 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261040 \
  --init-weights results/rl_v4_20261032/wide160_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,1 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 1 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261032/wide160_ckpts \
  --bc-kl-coef 0.1 --lr 3e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261032/wide160_weights_best.json \
  --checkpoint-out results/rl_v4_20261032/wide160_ac_best.json \
  --history-out results/rl_v4_20261032/wide160_history.json

# dev / test / red
python3 -m ipd protocol eval \
  --candidates RL-PPO,RL-Repair,RL-Pattern,RL-Wide,RL-Wide160 \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-PPO \
  --out results/rl_v4_20261032/protocol_v4_dev_rl_wide160.json

python3 -m ipd protocol eval \
  --candidates RL-PPO,RL-Repair,RL-Pattern,RL-Wide,RL-Wide160 \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 --candidate-chunk 1 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 2 --seed 20261027 --no-consume \
  --out results/rl_v4_20261032/protocol_v4_test_v7_rl_wide160_provisional.json

python3 -m ipd protocol red \
  --candidates RL-Wide160,RL-Wide,RL-Repair \
  --tier dev_red --seeds 30 --workers 20 \
  --out results/rl_v4_20261032/protocol_v4_dev_red_rl_wide160.json
```
