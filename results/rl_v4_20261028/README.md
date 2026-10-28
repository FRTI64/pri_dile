# RL-Pattern（2026-10-29）

> 实验候选 `rl-pattern`：从 RL-Repair / entropy fine-tune 出发，混入
> `train_v4` 与 `train_pattern_breaker`，修复 Protocol v4 的
> `pattern_breaker` 漏洞。**未 promotion。**

## 结论

| 集 | RL-PPO | RL-Repair | **RL-Pattern** |
|---|---:|---:|---:|
| dev_param | 540.35 | 547.96 | **548.45** |
| dev_family | 582.53 | 582.76 | **583.07** |
| dev_complex | 440.31 | 434.28 | **447.64** |
| dev_axelrod | 574.64 | **582.46** | 582.22 |
| dev_ood | 513.48 | 515.19 | **519.46** |
| test_v7（provisional） | 477.26 | **486.31** | 485.43 |

- dev：相对 RL-Repair 4/5 tier 提升，`dev_complex` +13.36、
  `dev_ood` +4.27、`dev_param` +0.49；`dev_axelrod` -0.24。
- test_v7：-0.88 vs RL-Repair，+8.17 vs RL-PPO；test 消费永久锁定，
  本次为 `--no-consume` provisional，不进入排名/选模。
- legacy 20-field 兼容性：四分布 avg **554.93**（RL-Repair 554.72），
  blind **564.91**（RL-Repair 564.68），第 1/100 天行为正常。
- C++ 导出兼容：`ipd export --arch mlp` 已对 pattern 权重验证通过
  （`ipd-neural-policy-v1`）。
- 主要修复：`pattern_breaker` 从 236.8 → 420.3（dev_complex）。
- 主要回退：`extort`、`axelrod`、`meta_mix`；test 上
  `hyper_extort` / `axelrod_extra` 仍有回退。

## 文件

| 文件 | 内容 |
|---|---|
| `ent01_proper_weights.json` / `ent01_proper_ac.json` | 20 iter entropy warm start 的部署权重 / actor-critic |
| `ent01_proper_history.json` | warm start 训练曲线 |
| `pb_v4_history.json` | 最终 8 iter 训练曲线 |
| `pb_v4_ckpts/iter_{002,004,006,008}.json` | 各 checkpoint（actor-critic） |
| `pb_v4_ckpts/*.eval.json` | 固定 seed argmax 诊断 |
| `protocol_v4_dev_rl_pattern.json` | **官方 dev 结果**（3 候选） |
| `protocol_v4_test_v7_rl_pattern_provisional.json` | **官方 provisional test 结果** |
| `test_v7_pb_*.json` | 最终候选 / PB-heavy 的直接 test 诊断 |
| `pb_heavy_proper_*` | PB-heavy 4 iter 消融 |
| `pb_mix_*` | train_complex + PB 消融 |

最终部署权重已复制到：

- `weights/rl_policy_weights_pattern.json`
  sha256 `5c50db0ed1c560e2d05cbc52a4006dc0b3e7fa55cf102f0ba932c1b7433619ae`
- `weights/rl_policy_checkpoint_pattern.json`
  sha256 `13f59b869e0cdc5380ba71b4ae97ca13fcd1bdca8b1553883ef4bb46fc413f42`

## 复现

```bash
# 1) entropy warm start
python3 -m ipd train rl --device cuda --iterations 20 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261029 \
  --init-weights weights/rl_policy_checkpoint_repair.json \
  --train-dists original,late,grim30,wide,train_complex,train_axelrod \
  --dist-weights 1,1,1,2,2,1 \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --no-validation \
  --out results/rl_v4_20261028/ent01_proper_weights.json \
  --checkpoint-out results/rl_v4_20261028/ent01_proper_ac.json \
  --history-out results/rl_v4_20261028/ent01_proper_history.json

# 2) train_v4 + pattern_breaker；最终取 iter_006
python3 -m ipd train rl --device cuda --iterations 8 \
  --fields-per-dist 2 --n 128 --workers 20 --seed 20261032 \
  --init-weights results/rl_v4_20261028/ent01_proper_ac.json \
  --train-dists original,late,grim30,wide,train_v4,train_pattern_breaker \
  --dist-weights 1,1,1,2,2,4 \
  --dists dev_param,dev_complex,dev_ood \
  --val-fields-per-dist 2 --val-n 256 --val-every 2 \
  --select-metric balanced \
  --save-all-checkpoints results/rl_v4_20261028/pb_v4_ckpts \
  --bc-kl-coef 0.05 --lr 5e-5 --ent-coef 0.01 --target-kl 0.02 \
  --out results/rl_v4_20261028/pb_v4_weights.json \
  --checkpoint-out results/rl_v4_20261028/pb_v4_ac.json \
  --history-out results/rl_v4_20261028/pb_v4_history.json

# 3) dev / provisional test
python3 -m ipd protocol eval --candidates RL-PPO,RL-Repair,RL-Pattern \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-PPO \
  --out results/rl_v4_20261028/protocol_v4_dev_rl_pattern.json

python3 -m ipd protocol eval --candidates RL-PPO,RL-Repair,RL-Pattern \
  --tiers test_v7 --fields 50 --n 2048 --workers 20 --candidate-chunk 2 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --no-consume \
  --out results/rl_v4_20261028/protocol_v4_test_v7_rl_pattern_provisional.json
```
