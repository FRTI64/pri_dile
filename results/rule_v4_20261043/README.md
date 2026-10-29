# Triple-v4 — Protocol v4 rule-strategy optimization (2026-10-29)

> **Status:** registered rule candidate, **not promoted**.  Best pure-rule
> candidate on Protocol v4 dev, but still below RL-Repair / RL-PPO on the
> main `dev_param` ranking.  No weight files; no proactive probe.

## 1. What changed

`triple-v4` keeps the existing Triple-majority structure (Final + Responsive3
+ BayesPlus majority vote) and adds exactly one rule to the Final component:

```text
when n_c >= MIN_C and n_d >= MIN_D
     and p_d >= 0.70
     and 0.20 <= p_c <= 0.62
     and p_d - p_c >= 0.30
     and d_rate >= 0.15:
    if opponent last move == C: C
    else: C with probability 0.75, otherwise D
```

`p_c` / `p_d` are the existing conditional defection rates
(`P(opponent D | our previous C/D)`).  The branch calls an opponent
"noisy reciprocator" when it retaliates strongly after our defections but
still defects occasionally after our cooperation.  In that regime another
punishment round is usually worse than generous cooperation, so Final
forgives instead of entering a long punish/test loop.

The branch is deliberately narrow:

- clean TFT (`p_c≈0`, `p_d≈1`) is excluded by `p_c >= 0.20`;
- always-defect (`p_c≈p_d≈1`) is excluded by `p_d - p_c >= 0.30`;
- scheduled late defectors (`p_c≈p_d≈1` after their start) are excluded too;
- no proactive probe is added, so `dev_family` (`DelayedGrim` /
  `MirrorAdaptive`) is unchanged.

Implementation: `ipd/policies/candidates/triple_v4.py`, registered as
`triple-v4` / `Triple-v4`.

## 2. Protocol v4 dev (30 fields × 2048, seed 20261027)

Official command:

```bash
python3 -m ipd protocol eval \
  --candidates triple-majority,triple-v4 \
  --tiers dev_param,dev_family,dev_complex,dev_axelrod,dev_ood \
  --fields 30 --n 2048 --workers 20 --seed 20261027 \
  --baseline Triple-majority \
  --out results/rule_v4_20261043/protocol_v4_dev.json
```

| Candidate | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| Triple-majority | 536.87 | 583.06 | 441.17 | 566.93 | 511.62 |
| **Triple-v4** | **538.66** | **583.07** | **447.08** | **568.38** | **514.41** |
| Δ | **+1.79** | +0.01 | **+5.91** | **+1.45** | **+2.79** |

- `min_norm` vs Triple-majority: 0.9998 / 1.0000 / 0.9982 / 1.0000 / 0.9918.
  No worst-family collapse; the tiny `dev_ood` min_norm dip is
  `adaptive_mix` (-2.3 family mean).
- Every tier except `dev_family` improves; the candidate is strictly better
  on 4/5 dev tiers.

Largest family gains (dev_param / dev_complex / dev_ood):

| Family | Δ dev_param | Δ dev_complex | Δ dev_ood |
|---|---:|---:|---:|
| pattern_breaker | — | **+65.7** | **+77.9** |
| lookahead | — | **+18.5** | **+15.0** |
| extort | +5.2 | — | +6.1 |
| noisy | +5.5 | — | +5.1 |
| temporal | +5.0 | — | +4.8 |
| adaptive | +3.4 | — | +4.4 |
| probe_exploit | — | +3.1 | +3.4 |
| adaptive_mix | — | +3.3 | -2.3 |
| portfolio | — | -0.8 | -1.0 |

The gain is broad, not a single-family artifact; the only material
regression is `adaptive_mix` on `dev_ood`.

## 3. Independent seed 20261029

Same command with `--seed 20261029`:

| Candidate | dev_param | dev_family | dev_complex | dev_axelrod | dev_ood |
|---|---:|---:|---:|---:|---:|
| Triple-majority | 535.16 | 582.74 | 440.72 | 566.65 | 513.17 |
| **Triple-v4** | **536.88** | **582.77** | **446.51** | **568.41** | **516.09** |
| Δ | **+1.72** | +0.03 | **+5.78** | **+1.76** | **+2.92** |

The dev_complex / dev_ood gains replicate almost exactly; dev_param gain also
replicates (+1.72 vs +1.79).

## 4. Level / complexity diagnostics

30 fields × 2048, seed 20261027:

| Tier | Triple-majority | Triple-v4 | Δ |
|---|---:|---:|---:|
| dev_level0 | 608.87 | 608.94 | +0.06 |
| dev_level1 | 541.35 | 542.59 | +1.25 |
| dev_level2 | 538.87 | 540.67 | +1.80 |
| dev_level3 | 433.90 | 440.52 | +6.62 |
| dev_level4 | 438.63 | 450.55 | **+11.92** |
| dev_complexity0 | 608.87 | 608.94 | +0.06 |
| dev_complexity1 | 536.93 | 537.77 | +0.84 |
| dev_complexity2 | 520.92 | 523.28 | +2.36 |
| dev_complexity3 | 513.71 | 515.30 | +1.58 |
| dev_complexity4 | 382.67 | 391.71 | **+9.04** |

The improvement is concentrated in the high-thinking / high-complexity
tiers, which is where the noisy-reciprocator branch fires.

## 5. test_v7 provisional (locked)

50 fields × 2048, seed 20261027, `--no-consume`:

| Candidate | test_v7 | Δ |
|---|---:|---:|
| Triple-majority | 471.54 | — |
| **Triple-v4** | **473.68** | **+2.14** |

This is a provisional diagnostic only.  The test policy is permanently
locked (`consume_test=false`); the result is not selection-eligible and is
not used for ranking or promotion.

## 6. Position among candidates

- Best **pure-rule** candidate on v4 dev: `dev_param` 538.66 / `dev_complex`
  447.08 / `dev_ood` 514.41.
- Still below the learned head on sample-mode `dev_param`: RL-Repair 547.96,
  RL-Wide160 548.26, RL-PPO 540.35 (Protocol v4 sample table).
- Compared with the sample-mode RL-PPO baseline, Triple-v4 is slightly lower
  on `dev_param` (-1.7) but higher on `dev_complex` (+6.8) and `dev_ood`
  (+0.9); it is not a replacement for the learned candidates.
- No weight file, no GPU, no PyTorch inference; the strategy is a pure
  Python rule state machine and is easy to port to C++ if needed.

## 7. Limitations / next steps

- `dev_family` is unchanged by design; any proactive probe would damage the
  `DelayedGrim` / `MirrorAdaptive` guardrail (see `Harvest45` history).
- The remaining large rule gaps are `probe_exploit` (228.8 vs RL-Repair
  282.9), `extort` (469.0 vs 497.6), `late` (309.3 vs 322.3) and
  `axelrod_extra` (586.8 vs 627.1).  The noise branch only partially closes
  extort/noisy; do not add a global probe or a larger extort weight.
- The largest single gain is on the synthetic `pattern_breaker` /
  `lookahead` families.  Protocol v4 fields are synthetic; transfer to a
  real 300-player field is unproven.  Do not promote this candidate on the
  strength of the synthetic complex-family gain alone.
- If this line continues, target **explicit extortion/ZD detection** and
  **probe-style opening response**, not another parameter sweep of the same
  Triple knob.
- No C++ port yet.  If a rule submission is chosen, add a native
  implementation and action-parity tests before deployment.

## 8. Files

- Candidate: `ipd/policies/candidates/triple_v4.py`
- Registry: `ipd/policies/registry.py`
- Tests: `tests/test_candidates.py` (`TripleV4Tests`)
- Dev results: `results/rule_v4_20261043/protocol_v4_dev.json`
- Independent seed: `results/rule_v4_20261043/protocol_v4_dev_seed29.json`
- Level/complexity: `results/rule_v4_20261043/protocol_v4_dev_levels.json`
- test provisional: `results/rule_v4_20261043/protocol_v4_test_v7_provisional.json`
