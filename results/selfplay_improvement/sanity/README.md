# Final sanity check — 2026-10-26

目的：冻结提交前做一次可复现检查，确认权重路径、benchmark/blind 复现、
checksum，以及 README/HANDOVER 的 No-Go 表述没有遗漏。

## 1. 复现命令

```bash
python3 -m ipd benchmark --candidates rl,adv-rl,selfplay \
  --fields 20 --workers 20 \
  --out results/selfplay_improvement/sanity/benchmark_20fields.json

python3 -m ipd blind --candidates rl,adv-rl,selfplay \
  --fields 10 --workers 10 \
  --out results/selfplay_improvement/sanity/blind_10fields.json
```

- benchmark：20 fields × 299，seed=20261026
- blind：10 fields × 299，seed=20261025

## 2. Benchmark 复现结果

| 策略 | original | late | grim30 | wide | avg | 记录值 |
|---|---:|---:|---:|---:|---:|---:|
| RL-PPO | 555.88 | 548.41 | 552.09 | 556.95 | **553.33** | 553.33 |
| Adv-RL | 555.76 | 549.19 | 553.39 | 558.82 | **554.29** | 554.29 |
| SelfPlay-GRU | 551.43 | 546.37 | 549.82 | 555.41 | **550.76** | 550.76 |

结论：与 `results/rl_benchmark.json`、`results/adversarial_rl_benchmark.json`、
`results/selfplay_final/benchmark_20fields.json` 完全一致，无漂移。

## 3. Blind 复现结果

| 策略 | score | mean rank | rank1 | top10 | worst | 记录值 |
|---|---:|---:|---:|---:|---:|---:|
| RL-PPO | **563.23** | 8.00 | 7/10 | 7/10 | 25 | 563.23 / 8.00 |
| Adv-RL | **562.91** | 8.00 | 7/10 | 7/10 | 25 | 562.91 / 8.00 |
| SelfPlay-GRU | **560.43** | 8.10 | 7/10 | 7/10 | 25 | 560.43 / 8.10 |

结论：与 `results/blind_rl_results.json`、
`results/blind_adversarial_rl_results.json`、`results/selfplay_final/blind.json`
完全一致，无漂移。

## 4. 冻结提交建议

### 主提交（score-only）：RL-PPO

```text
weights/rl_policy_weights.json
sha256 = bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4
```

期望结果：`553.33 / 563.23 / 8.00`。  
理由：held-out final min_norm = 1.000，是当前最稳的 generalist；官方分也只比
Adv-RL 低约 0.96。

### 21 维 / GRU / 无 PyTorch 备选：SelfPlay-GRU

```text
weights/selfplay_gru_policy.json
sha256 = e02e1854ebb376ba6fab83eef1d3d101093f0d6c9bb7bc3589a5aa02f58a68ea
```

期望结果：`550.76 / 560.43 / 8.10`。  
注意：它与 `weights/selfplay_bc_policy.json` 的 sha256 完全相同，确认是
**BC-only 蒸馏策略**，不是 self-play fine-tune 产物。报告必须保留这个表述。

### 条件提交：Adv-RL

```text
base:   weights/rl_policy_weights_bc.json
config: configs/adversarial_rl_policy_config_aggressive.json
```

期望结果：`554.29 / 562.91 / 8.00`。  
但 held-out final 为 `504.08`、min_norm `0.968`，低于 RL-PPO 的 `506.49 / 1.000`。
只有在“官方四分布构成就是比赛分布，且不看 held-out”时才选它；默认不选。

## 5. Checksum

完整列表见 `results/selfplay_improvement/sanity/checksums.txt`。关键项：

```text
rl_policy_weights.json          bc582056e09307210aff882c96dda4ad1be41df1dee8216cc167dc7af8e5c0a4
selfplay_gru_policy.json        e02e1854ebb376ba6fab83eef1d3d101093f0d6c9bb7bc3589a5aa02f58a68ea
rl_policy_weights_bc.json       4c5f6459a3e56c1067ef1eda3496490c2cc8223a786ee5ef52b633b538f1c11d
adversarial_rl_config_aggressive 65133b56be59feb55f544bdfd6a3277d2a4fb0f01454e25b427fb810218e468b
```

验证结果：

```text
sha256sum -c results/selfplay_improvement/sanity/checksums.txt
... all OK
```

测试：

```text
python3 -m unittest discover -s tests -v
Ran 71 tests in 3.707s
OK
```

## 6. 文档 No-Go 表述检查

已确认存在且一致：

- `README.md`：No-Go；E1a +0.43 mean / +0.016 min_norm；red best response
  100% 击败 BC/RL/Adv；指向 `results/selfplay_improvement/final_decision.md`。
- `HANDOVER.md`：No-Go；停止 self-play 方向，不替换现有权重。
- `docs/SELFPLAY.md`：No-Go 表格、E1a/E1b、E4 red best response、参数量更正。
- `results/selfplay_final/README.md`：No-Go、参数量 43,777/45,890 vs 7,617、
  held-out red 595 只是互惠型对照。
- `results/selfplay_improvement/final_decision.md`：完整 No-Go 决策树与 E4 表。

## 7. E4 negative result（必须保留）

| Blue | Blue mean | Red mean | Red win rate |
|---|---:|---:|---:|
| BC-only | 304.60 | 636.10 | 100% |
| Scratch self-play | 274.40 | 518.40 | 100% |
| E1a | 304.25 | 623.75 | 100% |
| RL-PPO | 306.95 | 648.45 | 100% |
| Adv-RL | 328.00 | 621.00 | 100% |

结论：目前没有任何策略解决针对性 best response 的 anti-exploitation；
旧 `final_test.json` 的 RL-BC `595 | 595` 只是互惠型对照，不能作为验收。

## 8. 冻结状态

- **不替换** `weights/selfplay_gru_policy.json`；
- **不覆盖** `results/selfplay_final/`；
- 主提交 RL-PPO，21 维备选 SelfPlay-GRU；
- Adv-RL 仅作为条件选项；
- 后续不再在当前比赛周期继续堆 self-play。
