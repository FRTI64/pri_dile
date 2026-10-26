# E1 — Domain-randomized A60 fine-tune

- **Status:** E1a/E1b No-Go; **E1c Go candidate**（`rl-repair`）
- **Parent:** `RL-PPO-Hard60` / `weights/rl_policy_checkpoint_hard60.json`
- **New training distributions:** `robust` + `repair` (`ipd/evaluation/robust.py`)
- **Candidate:** `weights/rl_policy_weights_repair.json`（registered as `rl-repair`）
- **Final test:** `heldout_v3` used once by E1c（+3.91 vs A60）
- **Promotion:** not promoted; `weights/rl_policy_weights.json` unchanged

## 1. Hypothesis

A60 improved the fixed `hard` pool but still overfits a small set of parameter
points. Sampling the hard-opponent parameters continuously should improve
held-out families without changing the 85-dimensional architecture.

Randomized ranges (training only):

```text
Extort chi            [1.5, 3.8]
Random p              [0.01, 0.95]
LateD k               [20, 95]
NoisyTFT p            [0.01, 0.50]
Joss p                [0.01, 0.50]
DynamicWindow W       [3, 10]
Dynamic p_defect      [0.02, 0.60]
Dynamic forgiveness   [0.00, 0.80]
Kamikaze period       [8, 30]
Harvest start         [20, 90]
```

A small anchor block (TFT, Grim, AllD, Bully, AllC) prevents collapse into
permanent defection. The mixture is training-only and is not used for
reporting.

## 2. Training command

```bash
python3 -m ipd train rl --device auto --seed 20261030 \
  --iterations 40 --fields-per-dist 5 --n 80 --workers 20 \
  --train-dists original,late,grim30,wide,hard,robust \
  --dist-weights 1,1,1,2,1,2 \
  --init-weights weights/rl_policy_checkpoint_hard60.json \
  --bc-kl-coef 0.1 --lr 1e-4 --ent-coef 0.0001 --target-kl 0.02 \
  --val-fields-per-dist 5 --val-n 100 --val-every 5 \
  --select-metric avg \
  --out weights/rl_policy_weights_robust.json \
  --checkpoint-out weights/rl_policy_checkpoint_robust.json \
  --history-out results/robust_20261026/E1_history.json
```

Selection uses official validation only. `heldout_v2`/red are report-only for
the first E1 screen. Because `heldout_v2` is now used by E0/E1 comparison, a
separate one-shot final test uses the new **`heldout_v3`** suite
(`configs/eval/e1c_final_league.json`, seed base 21061026). Do not iterate on
`heldout_v3`.

### First training run (seed 20261030)

```text
initial (A60)  val avg=558.87  worst=553.45
best iter 40   val avg=559.07  worst=554.09
               original=554.09 late=560.05 grim30=554.99 wide=567.14
```

官方 validation 仅 +0.20 avg / +0.64 worst，属于噪声量级的正向。
robust 训练 reward 约 433–448，明显低于官方分布，说明随机化对手确实更难。

E0 风格评估（同 suite/seeds）：

```text
                 A60      RL-Robust    Δ
official        554.27     554.71     +0.44
blind           563.87     564.63     +0.76
validation      507.85     508.47     +0.62
heldout_v2      500.07     501.43     +1.36
min_norm        0.9928     0.9964     +0.0036
CVaR10          391.13     388.98     -2.15
E4 red blue     299.20     311.70     +12.50
```

held-out v2 的增益来自 extort +16.3、noisy +3.8，但 random -6.1、
adaptive -1.7；CVaR 反而下降。**未达到 +3 / +0.02 的 Gate，判 No-Go。**
E4 red 仍 100% 击败，blue mean 的提升不等于 anti-exploitation 解决。
`heldout_v3` 未运行，留给下一个通过 validation 的候选。

E1b（KL 0.05, lr 2e-4, seed 20261031，保存 8 个 checkpoint）：

```text
official validation: 559.01 -> 559.80 (best iter 40)
checkpoint selection (heldout_v2, fixed policy seed):
  A60 499.50 | E1b-040 502.54  (+3.05)
full E1b evaluation (index-based policy seed):
  official 554.66, blind 564.67, validation 509.99, heldout_v2 502.00
  min_norm 0.9892, CVaR10 388.81, red blue 310.90
  vs A60: +0.38 / +0.79 / +2.14 / +1.94 / -0.0036 / -2.32 / +11.70
heldout_v2 per-family: extort +20.5, noisy +4.4, random -6.3, adaptive -2.0
```

E1b 的 selection run 刚好 +3.05，但 full run 只有 +1.94；两次 validation
不一致，min_norm/CVaR 下降，random 仍回退。按保守规则判 **No-Go**，
不运行 heldout_v3。red blue +11.7 是正向信号，但 E4 red 仍 100% 击败。

### E1c repair（Go candidate）

从 E1b checkpoint 出发，用新的 `repair` 训练分布修复 random/adaptive 尾部：

```text
Random 5.0 (p 收窄到 [0.08, 0.92])
NoisyTFT 3.0, Joss 2.0, DynamicWindow 2.0, LateD 2.0
Extort 1.5, Kamikaze 1.0, Harvest 1.0
TFT 3.0, GTFT 2.0, Pavlov 2.0, AllC 2.0
Grim 1.0, AllD 1.0, Bully 1.0
```

训练 20 iter，lr 1e-4，KL 0.05。官方 validation 554.39→554.88；
脚本按 official 选 iter 15，但 heldout_v2 validation 选 **iter 20**。

```text
checkpoint selection (heldout_v2, fixed policy seed):
  A60 499.50 | E1b 502.54 | E1c-020 504.40  (+4.90 vs A60)
full E1c evaluation:
  official 554.72, blind 564.68, validation 509.93, heldout_v2 503.17
  min_norm 0.9930, CVaR10 390.39, red blue 303.45
  vs A60: +0.45 / +0.80 / +2.08 / +3.10 / +0.0002 / -0.74 / +4.25
standard CLI (independent policy seed):
  benchmark 20 fields 554.76; blind 564.61 / 5.75
heldout_v2 per-family: extort +25.6, random +0.6, noisy +4.0,
  temporal -0.1, adaptive -1.9
heldout_v3 final (one-shot):
  RL-Repair 509.88 vs A60 505.97 (+3.91)
  min_norm 0.9996 vs 0.9999; CVaR10 405.58 vs 405.65
  per-family: extort +23.4, random +4.7, noisy +2.5, temporal +2.2,
  adaptive -0.2, anchor -0.1
  red blue 306.35 vs 310.00; E4 red win 100%
```

**判定：Go candidate。** 两次 validation（+4.90 / +3.10）和一次 final
（+3.91）方向一致，官方/blind 不降，无 family 崩塌。注册 `rl-repair`，
但不覆盖主权重；anti-exploitation 仍未解决。`heldout_v3` 已被使用，
后续候选需要 heldout_v4。

## 4. Gate E1

Go only if all of the following hold:

1. `heldout_v2` (now validation) mean improves by at least +3 **or**
   `min_norm_family` improves by at least +0.02 vs A60; the one-shot
   `heldout_v3` final test must not contradict the direction;
2. official four-distribution mean is not lower than A60 by more than 2;
3. no held-out family collapses (CVaR10 and per-family checked);
4. E4 red blue mean is not worse than A60 by more than 10;
5. cross-play pool exploitability does not increase materially.

If the gate fails, keep A60 as the candidate and record a No-Go. Never
overwrite `weights/rl_policy_weights.json` from this experiment.

Observed results:

| Condition | E1a | E1b | E1c |
|---|---|---|---|
| +3 mean or +0.02 min_norm | FAIL (+1.36 / +0.0036) | FAIL (+1.94 / -0.0036) | **PASS (+3.10 / +0.0002)** |
| official not down >2 | PASS (+0.44) | PASS (+0.38) | PASS (+0.45) |
| no family collapse | PASS (random -6.1) | PASS (random -6.3) | PASS (adaptive -1.9) |
| red blue not worse >10 | PASS (+12.5) | PASS (+11.7) | PASS (+4.25) |
| heldout_v3 final direction | not run | not run | **PASS (+3.91)** |

E1c is a Go candidate; E1a/E1b are No-Go. `rl-repair` is registered but the
main weight is unchanged.

After the E1 candidate is frozen (no further tuning), run the one-shot final
test once:

```bash
python3 -m ipd league --config configs/eval/e1c_final_league.json \
  --workers 20 \
  --out results/robust_20261026/e1c_final_v3.json \
  --md-out results/robust_20261026/e1c_final_v3.md
```

## 5. Required report

- official four distributions (20 fields × 299),
- blind 20 fields,
- `validation`, new `heldout_v2` (20 seeds),
- per-family means, `min_norm_family`, CVaR10,
- E4 red blue mean / red win rate,
- cross-play vs RL-PPO, A60, RL-BC, Adv-RL, SelfPlay-GRU,
- checksums and a Go/No-Go conclusion.
