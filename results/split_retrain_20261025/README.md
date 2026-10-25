# Split evaluation: retrain_20261025

- config: `/home/frti64/pri_dile/configs/eval/retrain_20261025.json`
- created: 2026-10-26T00:31:12

Each candidate has its own parameter set and result files.

## Benchmark

| candidate | params | avg | worst | files |
|---|---|---:|---:|---|
| Neural-retrain | 20 fields × 299, seed 20261026 | 547.93 | 542.74 | [result](by_candidate/Neural-retrain/benchmark.json), [opponents](by_candidate/Neural-retrain/opponents.json) |
| GRU-retrain | 20 fields × 299, seed 20261026 | 547.85 | 542.72 | [result](by_candidate/GRU-retrain/benchmark.json), [opponents](by_candidate/GRU-retrain/opponents.json) |
| Adv-MLP-retrain | 20 fields × 299, seed 20261026 | 552.02 | 548.03 | [result](by_candidate/Adv-MLP-retrain/benchmark.json), [opponents](by_candidate/Adv-MLP-retrain/opponents.json) |
| Adv-GRU-retrain | 20 fields × 299, seed 20261026 | 551.72 | 547.85 | [result](by_candidate/Adv-GRU-retrain/benchmark.json), [opponents](by_candidate/Adv-GRU-retrain/opponents.json) |
| RL-final-retrain | 20 fields × 299, seed 20261026 | 552.89 | 547.71 | [result](by_candidate/RL-final-retrain/benchmark.json), [opponents](by_candidate/RL-final-retrain/opponents.json) |
| RL-BC-retrain | 20 fields × 299, seed 20261026 | 550.66 | 544.54 | [result](by_candidate/RL-BC-retrain/benchmark.json), [opponents](by_candidate/RL-BC-retrain/opponents.json) |
| RL-scratch-base-retrain | 20 fields × 299, seed 20261026 | 552.01 | 547.51 | [result](by_candidate/RL-scratch-base-retrain/benchmark.json), [opponents](by_candidate/RL-scratch-base-retrain/opponents.json) |
| RL-stage1-retrain | 20 fields × 299, seed 20261026 | 549.55 | 543.81 | [result](by_candidate/RL-stage1-retrain/benchmark.json), [opponents](by_candidate/RL-stage1-retrain/opponents.json) |
| Adv-RL-retrain | 20 fields × 299, seed 20261026 | 554.15 | 549.25 | [result](by_candidate/Adv-RL-retrain/benchmark.json), [opponents](by_candidate/Adv-RL-retrain/opponents.json) |
| Adv-RL-conservative-retrain | 20 fields × 299, seed 20261026 | 553.39 | 549.10 | [result](by_candidate/Adv-RL-conservative-retrain/benchmark.json), [opponents](by_candidate/Adv-RL-conservative-retrain/opponents.json) |
| Adv-RL-final-retrain | 20 fields × 299, seed 20261026 | 553.14 | 548.41 | [result](by_candidate/Adv-RL-final-retrain/benchmark.json), [opponents](by_candidate/Adv-RL-final-retrain/opponents.json) |

## Blind

| candidate | params | score | mean rank | files |
|---|---|---:|---:|---|
| Neural-retrain | 10 fields × 299, seed 20261025 | 554.00 | 8.30 | [result](by_candidate/Neural-retrain/blind.json), [opponents](by_candidate/Neural-retrain/opponents.json) |
| GRU-retrain | 10 fields × 299, seed 20261025 | 555.19 | 8.30 | [result](by_candidate/GRU-retrain/blind.json), [opponents](by_candidate/GRU-retrain/opponents.json) |
| Adv-MLP-retrain | 10 fields × 299, seed 20261025 | 558.57 | 8.20 | [result](by_candidate/Adv-MLP-retrain/blind.json), [opponents](by_candidate/Adv-MLP-retrain/opponents.json) |
| Adv-GRU-retrain | 10 fields × 299, seed 20261025 | 559.18 | 8.10 | [result](by_candidate/Adv-GRU-retrain/blind.json), [opponents](by_candidate/Adv-GRU-retrain/opponents.json) |
| RL-final-retrain | 10 fields × 299, seed 20261025 | 562.22 | 8.10 | [result](by_candidate/RL-final-retrain/blind.json), [opponents](by_candidate/RL-final-retrain/opponents.json) |
| RL-BC-retrain | 10 fields × 299, seed 20261025 | 558.93 | 8.30 | [result](by_candidate/RL-BC-retrain/blind.json), [opponents](by_candidate/RL-BC-retrain/opponents.json) |
| RL-scratch-base-retrain | 10 fields × 299, seed 20261025 | 559.17 | 8.30 | [result](by_candidate/RL-scratch-base-retrain/blind.json), [opponents](by_candidate/RL-scratch-base-retrain/opponents.json) |
| RL-stage1-retrain | 10 fields × 299, seed 20261025 | 557.95 | 8.30 | [result](by_candidate/RL-stage1-retrain/blind.json), [opponents](by_candidate/RL-stage1-retrain/opponents.json) |
| Adv-RL-retrain | 10 fields × 299, seed 20261025 | 561.46 | 8.10 | [result](by_candidate/Adv-RL-retrain/blind.json), [opponents](by_candidate/Adv-RL-retrain/opponents.json) |
| Adv-RL-conservative-retrain | 10 fields × 299, seed 20261025 | 560.56 | 8.20 | [result](by_candidate/Adv-RL-conservative-retrain/blind.json), [opponents](by_candidate/Adv-RL-conservative-retrain/opponents.json) |
| Adv-RL-final-retrain | 10 fields × 299, seed 20261025 | 560.67 | 8.10 | [result](by_candidate/Adv-RL-final-retrain/blind.json), [opponents](by_candidate/Adv-RL-final-retrain/opponents.json) |

## Per-opponent files

`by_opponent/<opponent>.json` contains every candidate's score against that opponent type.  `by_candidate/<candidate>/opponents.json` contains all opponent types for one candidate.
