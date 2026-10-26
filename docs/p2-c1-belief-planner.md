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

1 seed per opponent, frozen validation / heldout_v2, **fair RNG** (separate
base instances with the same seed for base and planner):

| Suite | base mean | planner mean | Δ | min_norm | override |
|---|---:|---:|---:|---:|---:|
| validation | 511.88 | 499.77 | **-12.11** | 0.839 | 10.1% |
| heldout_v2 | 509.09 | 492.94 | **-16.15** | 0.846 | 6.9% |

Per-family damage: validation extort **-73.4**; heldout_v2 extort **-71.3**,
random -40.0.  The planner confuses extort/noisy opponents with non-responsive
exploiters and defects too often.

### 3.2 Conservative planner（horizon 4, top-k 6, fallback 0.90, margin 10）

| Suite | base mean | planner mean | Δ | min_norm | override |
|---|---:|---:|---:|---:|---:|
| validation | 511.88 | 509.69 | -2.19 | 0.971 | 0.3% |
| heldout_v2 | 509.09 | 511.15 | **+2.06** | 0.993 | 0.2% |

Heldout_v2 family deltas: adaptive +8.75, temporal +8.70, extort 0,
random/punitive/reciprocal 0, noisy **-3.00**, special 0.

Held-out red screen (20 seeds, fair RNG, same planner):

| Red | base blue | planner blue | Δ blue | base diff | planner diff |
|---|---:|---:|---:|---:|---:|
| E4-red-BC | 296.60 | 296.95 | +0.35 | 413.5 | 412.5 |
| Red-RL-v5-heldout | 375.10 | 375.10 | 0.00 | 342.0 | 342.0 |
| Red-RL-v6-heldout | 313.35 | 313.25 | -0.10 | 434.0 | 433.5 |

Override rate is only 0.05% on E4 and ~0 elsewhere; the red effect is
negligible.

> **RNG correction (2026-10-26):** an earlier version of this screen reused
> one base policy RNG stream for both arms, which produced spurious E4 +6.6
> and heldout_v2 +3.11.  The numbers above use independent same-seed base
> instances and supersede the first run.  The C2/C3 report keeps the same
> fair-RNG protocol.

## 4. Gate C assessment

Gate C asks for +10 on held-out red / dynamic / phase and official not down
more than 2:

- held-out red: E4 +0.35, v5 0.00, v6 -0.10 — **no meaningful gain**;
- dynamic/phase proxy (`temporal_v2`): +8.70 (below +10, one seed);
- heldout_v2 noisy: **-3.00**, min_norm 0.993;
- validation: -2.19 (just outside the official-style -2 tolerance).

**No-Go for the first C1 prototype.**  The conservative variant shows that
belief + planning can find small useful overrides (adaptive/temporal +8.7,
heldout mean +2.06), but the opponent-model likelihood does not separate
noisy/extort from non-responsive exploiters well enough.  More planning depth
or a larger model library is not the next lever; the failure is model
discrimination.

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
