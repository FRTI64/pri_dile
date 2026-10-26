# P2 C1 — Bayesian belief planner（offline prototype）

- **Status:** first offline screen complete — **No-Go**（2026-10-26）
- **Code:** `ipd/policies/belief_planner.py`,
  `ipd/evaluation/planner_eval.py`
- **Entry point:** `python3 -m ipd planner-eval`
- **Base / fallback:** `weights/rl_policy_weights_repair.json` (RL-Repair)
- **Promotion:** none; no candidate registered, no weights replaced

## 1. Design

The prototype keeps one particle per curated opponent type from
`ipd.core.opponents.bank.TYPE_BANK` (38 types covering nice/punitive/defect/
late/random/adaptive/wild).  Every round it:

1. scores the previous predictions against the opponent's latest observed
   action (`match_prob=0.85` / miss `0.15`);
2. normalises the particle weights;
3. advances every model to predict the current opponent action;
4. computes `p(D)` and a confidence `max(pD, 1-pD)`;
5. falls back to RL-Repair when confidence is below
   `--fallback-confidence`;
6. otherwise evaluates `C` vs `D` with a finite-horizon rollout
   (`--horizon`, `--top-k` hypotheses), using RL-Repair's deterministic
   argmax continuation as the future policy;
7. falls back if the expected gain over the base action is below `--margin`.

This is an offline research module.  It is deliberately **not** wired into the
submission candidate or the main weights.

## 2. Tests

```bash
python3 -m unittest tests.test_belief_planner -v
```

7 tests cover boundaries, reset, confidence fallback, belief concentration on
AllD, exploitation of AllD, determinism, curated library coverage and the
offline evaluator summary.

## 3. Screens

### 3.1 Aggressive planner（horizon 6, top-k 8, fallback 0.75）

1 seed per opponent, frozen validation / heldout_v2:

| Suite | base mean | planner mean | Δ | min_norm | override |
|---|---:|---:|---:|---:|---:|
| validation | 511.88 | 499.22 | **-12.65** | 0.847 | 9.4% |
| heldout_v2 | 507.79 | 503.43 | **-4.36** | 0.932 | 7.1% |

Per-family damage: validation extort **-69.9**, noisy -12.5; heldout_v2
extort -27.3, noisy -29.4.  The planner confuses extort/noisy opponents with
non-responsive exploiters and defects too often.

### 3.2 Conservative planner（horizon 4, top-k 6, fallback 0.90, margin 10）

| Suite | base mean | planner mean | Δ | min_norm | override |
|---|---:|---:|---:|---:|---:|
| validation | 511.88 | 510.03 | -1.85 | 0.971 | 0.3% |
| heldout_v2 | 507.79 | 510.90 | **+3.11** | 0.973 | 0.3% |

Heldout_v2 family deltas: random **+30.0**, adaptive +7.6, extort +2.6,
reciprocal/punitive 0, temporal **-6.8**, noisy **-11.7**.

Held-out red screen (20 seeds, same planner):

| Red | base blue | planner blue | Δ blue | base diff | planner diff |
|---|---:|---:|---:|---:|---:|
| E4-red-BC | 294.35 | 300.95 | **+6.60** | 415.5 | 403.0 |
| Red-RL-v5-heldout | 378.55 | 379.50 | +0.95 | 333.0 | 335.0 |
| Red-RL-v6-heldout | 314.85 | 317.80 | +2.95 | 428.5 | 424.0 |

Override rate is only 0.1%, so the red improvement is small and fragile.

## 4. Gate C assessment

Gate C asks for +10 on held-out red / dynamic / phase and official not down
more than 2:

- held-out red: E4 +6.6, v5 +0.95, v6 +2.95 — **below +10**;
- dynamic/phase proxy (`temporal_v2`): **-6.8**;
- heldout_v2 noisy: **-11.7**, min_norm 0.973;
- validation: -1.85 (inside the official-style tolerance, but no gain).

**No-Go for the first C1 prototype.**  The conservative variant shows that
belief + planning can find small useful overrides (random +30, E4 +6.6), but
the opponent-model likelihood does not separate noisy/extort from
non-responsive exploiters well enough.  More planning depth or a larger model
library is not the next lever; the failure is model discrimination.

## 5. Next steps

1. **C2 auxiliary opponent prediction**: add a learned head that predicts the
   opponent's next action / responsiveness; use it to gate planner overrides.
2. **C3 classifier + MoE**: train a type/responsiveness classifier and route
   only high-confidence subclasses to defensive/exploitative experts.
3. Keep RL-Repair as the base/fallback; offline planner evaluation remains the
   screen.  Do not change `weights/rl_policy_weights.json` without a separate
   validation + one-shot heldout_v4 test.

## 6. Reproduction

```bash
python3 -m unittest tests.test_belief_planner -v

# aggressive screen
python3 -m ipd planner-eval --suite validation,heldout_v2 --seeds 1 \
  --horizon 6 --top-k 8 --fallback-confidence 0.75 \
  --out results/p2_belief_20261026/planner_screen_h6.json

# conservative screen
python3 -m ipd planner-eval --suite validation,heldout_v2 --seeds 1 \
  --horizon 4 --top-k 6 --fallback-confidence 0.9 --margin 10 \
  --out results/p2_belief_20261026/planner_screen_conservative.json
```

## 7. Artifacts

```text
ipd/policies/belief_planner.py
ipd/evaluation/planner_eval.py
tests/test_belief_planner.py
results/p2_belief_20261026/
  planner_screen_h6.json
  planner_screen_conservative.json
  planner_red_screen.json
  README.md
  checksums.txt
```

`weights/rl_policy_weights.json` and
`weights/rl_policy_weights_repair.json` are unchanged.
