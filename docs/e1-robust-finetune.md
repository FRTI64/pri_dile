# E1 — Domain-randomized A60 fine-tune

- **Status:** E1a No-Go; E1b borderline No-Go（两次 validation 不一致）
- **Parent:** `RL-PPO-Hard60` / `weights/rl_policy_checkpoint_hard60.json`
- **New training distribution:** `robust` (`ipd/evaluation/robust.py`)
- **Artifacts:** `weights/rl_policy_weights_robust*.json`（研究用，未注册/未 promotion）
- **Final test:** `heldout_v3` 未运行，保留给下一个稳定通过 validation 的候选

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
(`configs/eval/e1_final_league.json`, seed base 21061026). Do not iterate on
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

| Condition | E1a | E1b |
|---|---|---|
| +3 mean or +0.02 min_norm | FAIL (+1.36 / +0.0036) | FAIL (+1.94 / -0.0036) |
| official not down >2 | PASS (+0.44) | PASS (+0.38) |
| no family collapse | PASS (random -6.1) | PASS (random -6.3) |
| red blue not worse >10 | PASS (+12.5) | PASS (+11.7) |

Both are No-Go; `heldout_v3` remains unused.

After the E1 candidate is frozen (no further tuning), run the one-shot final
test once:

```bash
python3 -m ipd league --config configs/eval/e1_final_league.json \
  --workers 20 \
  --out results/robust_20261026/final_heldout_v3.json \
  --md-out results/robust_20261026/final_heldout_v3.md
```

## 5. Required report

- official four distributions (20 fields × 299),
- blind 20 fields,
- `validation`, new `heldout_v2` (20 seeds),
- per-family means, `min_norm_family`, CVaR10,
- E4 red blue mean / red win rate,
- cross-play vs RL-PPO, A60, RL-BC, Adv-RL, SelfPlay-GRU,
- checksums and a Go/No-Go conclusion.
