# P2 C2/C3 — exploratory / counterfactual excitation data

- **Status:** data-layer Go; planner screen **No-Go**（2026-10-26）
- **Code:** `ipd/training/opponent_model.py`,
  `ipd/evaluation/planner_eval.py`
- **Entry point:** `python3 -m ipd train opponent-model`
- **Base / fallback:** `weights/rl_policy_weights_repair.json` (RL-Repair)
- **Promotion:** none; no candidate registered, no weights replaced
- **Result:** `results/p2_excitation_20261026/`

## 1. Problem

The first C2/C3 gate (86.6% on held-out types, 70.6% on heldout_v2)
failed because RL-Repair rarely defects early: cooperative/forgiving
histories had **0%** `excited` samples (`n_d >= 3` and `n_c >= 5`),
adaptive 23.7%, noisy 33.9%.  The gate therefore had almost no examples
of "what does this opponent do after I defect?"  All heldout_v2 false
positives came from adaptive/noisy.

## 2. Data

`exploration_actions()` produces open-loop blue schedules with
`single_probes`, `block_probes`, `random`, `alternating`, `late_block`
and `cooperate` templates.  Schedules are independent of the opponent,
so histories are not biased by the base policy.

`collect_counterfactual_dataset()` deep-copies the same opponent state
at stratified branch days, plays C vs D, and continues with C for
`horizon` rounds.  The branches share the opponent RNG stream
(common random numbers), so the reply difference is caused by blue's
action.  The pre-branch feature row is used for a C2 responsiveness
head (`responsive` if the two branches produce different numbers of D).

Training commands:

```bash
# base + exploratory trajectories
python3 -m ipd train opponent-model --data-mode base+explore \
  --episodes 2 --explore-episodes 2

# base + exploratory + counterfactual + responsiveness head
python3 -m ipd train opponent-model --data-mode mixed \
  --episodes 2 --explore-episodes 2 \
  --counterfactual-episodes 1 --counterfactual-branches 8 \
  --counterfactual-horizon 6 --counterfactual-prefix explore,base
```

## 3. Excitation

Overall training `excited` rate: **25.8% → 59.3%**.  Per class
(base → explore): cooperative **0.0% → 56.9%**, forgiving
**0.0% → 48.6%**, adaptive **23.7% → 68.8%**, noisy
**33.9% → 64.4%**, late 20.6% → 59.4%, punitive 11.1% → 63.7%.

Counterfactual branches: 536 pairs, 234 responsive (43.7%).
Responsive rates by class: cooperative 80.8%, noisy 89.6%,
punitive 48.2%, adaptive 38.9%, random 10.0%, late **0.0%**,
defect 16.7%.

## 4. Gate / response metrics

| Training data | Samples | Gate val | heldout_v2 acc/prec/rec | :explore | :counterfactual | Response |
|---|---:|---:|---|---|---|---:|
| old base | 13,400 | 86.6% | 70.6 / 70.2 / 38.7 | - | - | - |
| base+explore | 26,800 | 85.6% | **72.0 / 71.1 / 43.7** | 74.7 / 69.5 / 59.2 | - | - |
| mixed | 39,880 | 85.2% | **73.1 / 69.0 / 52.7** | 73.5 / 66.2 / 61.3 | **77.1 / 73.7 / 61.5** | **71.8%** |

The gate is better on excited states, but deployment trajectories still
lack excitation.  Mixed-data false-positive rates remain 25.8%
(adaptive) and 27.8% (noisy); random precision stays 100%.

## 5. Planner screen

Fair-RNG conservative planner (horizon 4, top-k 6, fallback 0.9,
margin 10):

| Planner | validation Δ | heldout_v2 Δ | min_norm |
|---|---:|---:|---:|
| old gated t0.7 (1 seed) | 0.00 | +1.20 | 0.999 |
| base+explore gate t0.7 (1 seed) | -0.34 | +1.15 | 0.998 |
| mixed CF gate t0.6 (1 seed) | -1.04 | +3.04 | 1.000 |
| mixed CF gate t0.6 (3 seeds) | **-0.44** | **+0.88** | 0.994 |
| ungated conservative (3 seeds) | -0.51 | +0.95 | 0.993 |

The 1-seed temporal +11.0 falls to **+0.45** with 3 seeds, so it is
noise.  Gated aggressive remains destructive (heldout_v2 -19.33,
extort -107.62).  Held-out red E4/v5/v6 all 0.00.

**Gate C fail:** no +10 on held-out red/dynamic/phase, heldout_v2 mean
far below +10.  No promotion; `heldout_v4` remains unused.

## 6. Diagnosis and next steps

1. Training excitation is fixed; **deployment excitation is not**.
   Static data cannot create response events when RL-Repair never
   defects.  Next: an early-game active probe (one deliberate D, then
   repair) before enabling exploitation.
2. Replace `allow/deny` with `safe_to_exploit / risky / unknown`;
   route only high-confidence safe states, unknown falls back to
   RL-Repair (three-way gate + MoE).
3. Train next-action / responsiveness jointly with the gate and
   evaluate on `our D → their next move` response events.  The current
   head (71.8% counterfactual accuracy) is not reliable enough alone.
4. Do not add planner horizon or tune risk coefficients; C1 planning
   remains the bottleneck.  Do not change main weights.

## 7. Reproduction

```bash
bash results/p2_excitation_20261026/run.sh
python3 -m unittest tests.test_opponent_model -v
```

## 8. Artifacts

```text
ipd/training/opponent_model.py
ipd/evaluation/planner_eval.py
tests/test_opponent_model.py
weights/opponent_*_explore.json
weights/opponent_*_cf.json
weights/opponent_responsiveness.json
results/p2_excitation_20261026/
docs/p2-c2c3-excitation.md
```

## 9. C3 follow-up

The next step, a three-way `safe/risky/unknown` gate + MoE, was implemented
and screened in `docs/p2-c3-threeway-moe.md` /
`results/p2_threeway_moe_20261026/`.  It improves precision on excited
states (safe precision 82.1%/82.6%) but abstains too often on deployment
trajectories (safe recall 20.7%).  The AllD MoE reaches 5-seed heldout_v2
+4.04 but validation only +0.53, with large seed variance; held-out red
is 0.00.  It remains **No-Go** for Gate C.
