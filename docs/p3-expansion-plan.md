# P3 expansion — multi-red pool + PSRO-lite blue response

- **Status:** complete — **No-Go** on Gate D (2026-10-26)
- **Parent:** `docs/p3-red-league.md`
- **Code:** `ipd/evaluation/policy_pool.py`, `ipd/training/league.py`,
  `ipd/training/rl.py`
- **Goal:** test whether a diverse red pool plus meta-Nash weighting can reach
  Gate D where the single-red league-lite response stopped at -12.8%.
- **Result:** `results/p3_league_20261026/README.md`

## 1. Why this experiment

The first P3 pass trained one 85-dim red against RL-PPO/A60/RL-Repair and then
trained a blue response against that single red.  It reduced held-out E4
exploitability by 12.8%, short of the -20% gate.  A single exploiter encourages
a narrow response; the next iteration uses several exploiters with different
blue training pools and a maximin mixture.

## 2. Red pool

Training reds (eligible for the blue opponent mixture):

| Red | Blue pool | Seed |
|---|---|---|
| Red-RL-v1 | RL-PPO / A60 / RL-Repair | 20261036 (existing) |
| Red-RL-v2 | Adv-RL / SelfPlay / Final / Triple | 20261039 |
| Red-RL-v3 | RL-PPO / A60 / RL-Repair / Blue-vs-Red-Comp-v1 | 20261040 |
| Red-RL-v4 | Adv-RL / SelfPlay / RL-Repair / Blue-vs-Red-Comp-v1 | 20261043 |

Held-out reds (never used for blue training or selection):

| Red | Blue pool | Seed |
|---|---|---|
| Red-RL-v5-heldout | RL-Repair / Adv-RL / SelfPlay / Final / Triple | 20261041 |
| Red-RL-v6-heldout | RL-PPO / A60 / RL-Repair / Adv-RL / SelfPlay / Blue-vs-Red-Comp-v1 | 20261042 |

All reds use the 85-dim PPO trunk, `--from-scratch`, `safe=False`
(`--allow-probe`), and a fixed 20/25-iteration budget without official
validation.

## 3. Meta weights

`python -m ipd meta` extracts the red×blue score-difference matrix from the red
evaluation report and computes a maximin mixture over the four training reds.
The columns are the learned blue population (RL-PPO / A60 / RL-Repair /
Adv-RL / SelfPlay / the previous league-lite response); rule baselines are
excluded because Gate D targets learned policies and their low exploitability
ceiling would distort the maximin.  The actual sampling weights are
`0.6 * maximin + 0.4 * uniform`, i.e. every training red keeps at least a 10%
share.  The uniform term is a diversity regulariser, not a performance claim.
Held-out reds are rejected if they appear in `--train-reds`.

## 4. Blue response

Two candidates are trained from RL-Repair:

```text
B1: 50% official/repair fields + 50% meta-red fields, reward = own - opponent
B2: 40% official/repair fields + 60% meta-red fields, reward = own - opponent
```

Both use `safe=True`, KL 0.05, lr 1e-4, 25 iterations.  **Only official
validation selects between B1/B2.**  The held-out red report is run exactly
once after selection.

## 5. Pre-registered Gate D

The candidate is Go only if all hold on `Blue-Meta-v2` vs `RL-Repair`:

1. mean exploitability `red_score - blue_score` across E4 / Red-v5 / Red-v6
   drops by **at least 20%**;
2. official four-distribution mean drops by **no more than 2**;
3. no official/blind/heldout_v2 family drops by more than **2%** relative to
   RL-Repair;
4. no single held-out red becomes more than **5%** easier than RL-Repair.

If the gate fails, keep RL-Repair and record No-Go.  Do not tune E2 risk
coefficients again; the next direction is P2 C1 (Bayesian belief planner).

## 6. Reproduction

```bash
bash results/p3_league_20261026/train_reds.sh
bash results/p3_league_20261026/train_heldout_reds.sh
bash results/p3_league_20261026/evaluate_red_pool.sh
bash results/p3_league_20261026/train_blue.sh
bash results/p3_league_20261026/train_blue_b3.sh
# select on official/blind/heldout_v2 + training reds v3/v4
python3 -m ipd league \
  --config configs/eval/p3_blue_checkpoint_selection.json \
  --skip-cross-play --workers 20 \
  --out results/p3_league_20261026/blue_checkpoint_selection.json
# then copy the selected B1 weights to blue_meta_v2 and run the one-shot test
bash results/p3_league_20261026/finalize_blue.sh
```

## 7. Result (No-Go)

- Maximin weights: v4 0.70, v1/v2/v3 0.10 each.
- Selected B1 (50% red, iter 5): official 554.999, blind 564.345,
  heldout_v2 502.938, min_norm 0.9679.
- Held-out exploitability 390.33 → 367.33 (**-5.9%**, gate -20%).
- E4 -10.5%, Red-v5 -1.7%, Red-v6 -4.7%; all held-out reds still win 100%.
- Official +0.24 (PASS), but `validation:extort` -3.21% (family FAIL).
- **Gate D FAIL.** Keep RL-Repair; do not register `rl-meta`. Next: P2 C1.
