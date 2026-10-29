# RL-Switch（2026-10-29）

> 窄 meta：RL-Wide160 默认；第 6 回合前双方各 D≥2 且对手最新 C 时，
> 切换到 RL-Repair。**当前最佳 v4 dev/test 候选，未 promotion。**

## 结果

| 集 | RL-Wide160 | RL-Switch |
|---|---:|---:|
| dev_param（seed 27） | 548.26 | **548.58** |
| dev_family | **583.08** | 582.91 |
| dev_complex | **454.31** | 454.24 |
| dev_axelrod | 582.38 | **582.86** |
| dev_ood | 521.37 | **521.42** |
| test_v7（seed 27，provisional） | 485.62 | **488.26** |
| test_v7（seed 29，provisional） | 484.61 | **487.51** |
| dev_param（seed 29） | 546.39 | **546.80** |
| Protocol nine-suite avg | 520.43 | **521.18** |
| 158-bank nine-suite avg | **554.45** | 554.37 |

- `min_norm`（vs RL-Wide160）：0.994–0.998；
- red：dev_red blue 378.40 相同；test_red blue 319.37 vs 319.43；
  red_win 仍 100%；
- 主要改善：Extort3 +21、Extort.F3 +15、NoisyTFT.20 +14、
  RemorsefulProber +12、EarlyProbe +10；
- 主要回退：FSM.F2/F3、Random.50/20/30、Prober/Tester（-5～-14）。

## 文件

| 文件 | 内容 |
|---|---|
| `protocol_v4_dev_rl_switch.json` | 官方 dev（seed 27） |
| `protocol_v4_test_v7_rl_switch_provisional.json` | 官方 test_v7（seed 27） |
| `protocol_v4_dev_repro_seed20261029_rl_switch.json` | 独立 dev seed |
| `protocol_v4_test_v7_repro_seed20261029_rl_switch.json` | 独立 test seed |
| `protocol_v4_dev_red_rl_switch.json` | dev_red |
| `protocol_v4_test_red_rl_switch_provisional.json` | test_red provisional |
| `matchup_protocol_switch.*` | protocol catalog 对比 |
| `matchup_all_switch.*` | 158 类 bank 对比 |
| `meta_probe.py` / `meta_safe_probe.py` / `meta_prefix_probe.py` / `meta_switch_probe.py` | meta 原型与消融 |

## 复现

见 `docs/research/rl-switch.md`。核心命令：

```bash
python3 -m ipd protocol eval --candidates RL-Wide160,RL-Switch \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --candidate-chunk 1 \
  --mode sample --sample-rng per-match --gpu on --gpu-workers 4 \
  --gpu-field-batch 4 --seed 20261027 --baseline RL-Wide160 \
  --out results/meta_20261039/protocol_v4_dev_rl_switch.json
```

注意：`rl-switch` 目前只有 Python 版；C++ 双权重移植尚未完成。
