> **历史实验：基于 980-config 临时 catalog，已回滚；当前 v5 为 977 configs。**

# alld-probe unsafe specialist（2026-10-30）

> **实验，未注册、未 promotion。** 只训 `probe_exploit:alld` 训练点，
> 并关闭 safety 的“对手未 D 前主动 D”限制（`--allow-probe`）。

## 训练

```bash
python3 -m ipd train rl --device cuda --hidden 160 128 \
  --iterations 8 --fields-per-dist 2 --n 128 --workers 20 \
  --seed 20261058 \
  --init-weights weights/rl_policy_checkpoint_wide160.json \
  --train-dists train_probe_alld \
  --dists dev_complex --val-fields-per-dist 4 --val-n 512 \
  --val-every 1 --select-metric avg \
  --save-all-checkpoints results/protocol_v5/probe_alld_unsafe_20261030/ckpts \
  --bc-kl-coef 0.02 --lr 1e-4 --ent-coef 0.01 --target-kl 0.02 \
  --allow-probe \
  --out results/protocol_v5/probe_alld_unsafe_20261030/probe_alld_unsafe_weights.json \
  --checkpoint-out results/protocol_v5/probe_alld_unsafe_20261030/probe_alld_unsafe_ac.json \
  --history-out results/protocol_v5/probe_alld_unsafe_20261030/probe_alld_unsafe_history.json
```

## 结果

| 候选 | dev_complex | probe_exploit | pattern_breaker |
|---|---:|---:|---:|
| RL-Wide160（base） | 477.77 | 115.7 | 632.1 |
| ProbeAlldU-iter3 | 433.09 | **280.7** | 657.9 |
| ProbeAlldU-iter8 | 432.35 | 274.8 | 669.7 |

- probe 从 115.7 提升到 **280.7**，但整体 dev_complex 只有 433，不能单独提交。
- 280 依赖从第 0 回合开始主动 D；规则 gate 只能在首次 D 之后（t=5/7）
  识别，后验切换最多把 probe 拉到约 123–129，且会误伤其他 family。
- **结论：该方案不可采用。** 它关闭 safety、依赖开局主动 D，破坏其他 family；仅作失败实验记录，不进入候选或 gate。

## 文件

| 文件 | 内容 |
|---|---|
| `screen_dev_complex.json` | iter3/4/6/8 的 dev_complex 筛选 |
| `probe_alld_unsafe_history.json` | 训练历史 |
| `probe_alld_unsafe_weights.json` | best 导出权重 |
| `ckpts/` | iter1–8 |
