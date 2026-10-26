# P3 — red/exploiter pool + league-lite

- **Status:** red training (first exploiter)
- **Code:** `ipd/evaluation/policy_pool.py`, `ipd/training/rl.py`
- **Goal:** produce a stronger red best response than E4 and measure
  exploitability against held-out blue policies. This is not a claim that
  anti-exploitation is solved.

## 1. Blue pool

Training pool (frozen):

```text
weights/rl_policy_weights.json          RL-PPO
weights/rl_policy_weights_hard60.json   RL-PPO-Hard60
weights/rl_policy_weights_repair.json   RL-Repair
```

Held-out blues for evaluation:

```text
Adv-RL, SelfPlay-GRU, Final-tuned, Triple-majority
```

## 2. Red training

The red learner is the same 85-dim RL-PPO trunk, initialised from scratch
(feature scaler only), with `--allow-probe` so it can proactively defect. It is
trained against the frozen blue pool with the task-based PPO loop.

```bash
python3 -m ipd train rl --device auto --seed 20261036 --from-scratch \
  --iterations 20 --fields-per-dist 20 --n 80 --workers 20 \
  --opponent-weights weights/rl_policy_weights.json,weights/rl_policy_weights_hard60.json,weights/rl_policy_weights_repair.json \
  --allow-probe --no-validation \
  --lr 2e-4 --ent-coef 0.01 --target-kl 0.03 \
  --out weights/red_pool/red_rl_v1.json \
  --checkpoint-out weights/red_pool/red_rl_checkpoint_v1.json \
  --history-out results/red_pool_20261026/red_train.json
```

No official validation/checkpoint selection is used for a red best response;
the budget is fixed. Checkpoints are evaluated afterwards against held-out
blues.

## 3. Evaluation

Create a league red-eval config with the existing blue candidates and two red
policies: E4 red (21-dim GRU) and the new 85-dim red RL. Run:

```bash
python3 -m ipd league --config configs/eval/red_pool_eval.json \
  --skip-suites --skip-cross-play --workers 20 \
  --out results/red_pool_20261026/red_eval.json
```

Report per red: blue mean, red mean, diff, red win rate. Compare against E4.
A new red that beats E4 pressure is a stronger exploiter, which means the
blue candidates are *more* exploitable — it does not solve anti-exploitation.

## 3.1 First red result

`Red-RL-v1` (85-dim, from scratch, safe=False) trained 20 iter against
RL-PPO/A60/RL-Repair. It is a **stronger exploiter than E4** on every
held-out blue:

| Blue | E4 red mean | Red-RL-v1 mean | E4 blue mean | Red-RL blue mean |
|---|---:|---:|---:|---:|
| RL-Repair | 703.8 | **727.6** | 296.8 | 312.1 |
| RL-PPO | 642.3 | **705.7** | 309.3 | 325.2 |
| RL-PPO-Hard60 | 662.3 | **716.7** | 309.8 | 332.2 |
| Adv-RL | 611.8 | **666.6** | 337.3 | 328.6 |
| SelfPlay-GRU | 656.6 | **718.5** | 306.1 | 322.0 |
| Final-tuned | 607.6 | **634.3** | 334.1 | 347.8 |
| Triple-majority | 577.9 | **620.0** | 335.9 | 331.0 |

Red win rate is 100% for Red-RL-v1 on all blues (E4 is 95–100%).
This is a first-milestone **Go for the red pool**, but it means the blue
candidates are more exploitable than the E4 result suggested.

## 3.2 Naive blue response (No-Go)

A standard own-score blue response against Red-RL-v1 improved its training
score from ~312 to ~343, but official validation dropped from 555.26 to
552.39 and the best checkpoint remained the initial RL-Repair. This own-score
objective does not target anti-exploitation.

The mixed league-lite response used:

```text
50% official/repair fields + 50% Red-RL-v1 fields
reward = own - 1.0 * opponent   (competitive/score-difference)
initial RL-Repair, safe=True, KL 0.05, 20 iter
best official val iter 10: 554.57
```

Held-out exploitability test (E4 red, not in blue training):

| Blue | E4 red blue mean | E4 red mean | diff |
|---|---:|---:|---:|
| RL-Repair | 296.8 | 703.8 | 407.0 |
| Blue-vs-Red-Comp | **311.5** | **666.5** | **355.0** |

- diff -12.8% (Gate D wants -20%): **No-Go**.
- official 554.74 (RL-Repair 554.76, -0.02), blind 563.80 (-0.81).
- E4 red still wins 100%; anti-exploitation is not solved.

The direction is positive, but a single training red is not enough. Next iteration
should expand the red pool (Adv-RL/SelfPlay exploiters, more checkpoints) and use
PSRO/meta-Nash weighting rather than a single-red response.

## 4. Gate P3 (first milestone)

- New red must be valid and beat at least the E4 red mean on held-out blues;
- no claim of solved anti-exploitation without a blue response that survives
  the new red.
- If the new red is weaker than E4, record No-Go and keep the E4 pool.

## 5. Tests

```bash
python3 -m unittest tests.test_policy_pool tests.test_training -v
```

Plus a one-iteration CPU smoke with `--opponent-weights`, checking history
contains `blue_pool` rewards and the exported red weights.
