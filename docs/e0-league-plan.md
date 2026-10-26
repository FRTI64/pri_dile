# E0 — Cross-play / PSRO / Exploitability Harness Plan

- **Status:** implemented; frozen run in `results/e0_league_20261026/`
- **Scope:** P0 evaluation/meta-game only. No weight replacement, no new
  training algorithm.
- **Goal:** make candidate comparisons trustworthy before E1/E2 training.
- **Baseline fallback:** `RL-PPO` (`weights/rl_policy_weights.json`) and
  `RL-PPO-Hard60` (`weights/rl_policy_weights_hard60.json`) remain untouched.

## 1. Frozen candidate pool

| Group | Candidate | Spec |
|---|---|---|
| rule | Final-tuned | `final-tuned` |
| rule | Resp3-tuned | `resp3-tuned` |
| rule | Triple-majority | `triple-majority` |
| learned | RL-PPO | `rl` + `weights/rl_policy_weights.json` |
| learned | RL-PPO-Hard60 | `rl-hard60` + `weights/rl_policy_weights_hard60.json` |
| learned | RL-BC | `rl` + `weights/rl_policy_weights_bc.json` |
| learned | Adv-RL | `adv-rl` + aggressive config |
| learned | SelfPlay-GRU | `selfplay` + `weights/selfplay_gru_policy.json` |
| learned | SelfPlay-BC | `selfplay` + `weights/selfplay_bc_policy.json` |

All candidates are built from the same serialisable `CandidateSpec` and are
reset between matches. Stochastic candidates use `mode=sample` unless a run
explicitly asks for `argmax`.

## 2. Fixed suites and seeds

| Suite | Contents | Seeds |
|---|---|---|
| `official` | four official distributions × 20 fields × 299 | `20261026` |
| `blind` | 20 unseen wide/smart fields × 299 | `20261025` |
| `validation` | 51 fixed opponents | `20261026 + i*7919` |
| `heldout_v2` | **new** 60 fixed opponents, new families/parameters | `20961026 + i*7919` |
| `red` | E4 red best response | `20561026 + i*7919` |

`heldout_v2` is created in this task and is **not** used for candidate
selection. It exists to report a genuinely fresh final hold-out. The older
`final` suite is now treated as validation because it was used for A/B/C
comparisons.

## 3. New modules

```text
ipd/evaluation/opponent_suite.py   # frozen suite definitions, incl. heldout_v2
ipd/evaluation/metagame.py         # payoff matrix, fictitious-play Nash,
                                   # Nash averaging, Elo, exploitability
ipd/evaluation/league.py           # config, cross-play, suite + red eval, CLI
configs/eval/e0_league.json        # frozen candidate/suite/seed config
tests/test_league.py               # tests written before implementation
tests/test_metagame.py
```

CLI:

```bash
python3 -m ipd league --config configs/eval/e0_league.json \
  --out results/e0_league_20261026/report.json
```

## 4. Metrics

- `mean`, per-family mean, `min_family_raw`, `min_norm_family` vs RL-PPO.
- `cvar10_family`: mean of the worst 10% family means.
- `cvar10_item`: mean of the worst 10% individual match scores.
- `sigma_family`, `max_minus_min`, `top_family_share`.
- Cross-play matrix in both orientations; symmetric pair payoff matrix.
- Elo from pairwise win rates; Nash averaging weights from the antisymmetric
  score-difference game.
- `meta_exploitability`: best pure-strategy gain against the Nash mixture.
- `pool_exploitability`: worst pool opponent gain against each candidate.
- Held-out red: blue mean, red mean, diff, red win rate, normalized blue score.

## 5. Gate E0

1. Unit tests pass.
2. The same command with the same config produces the same matrix and summary
   (within floating-point/order tolerance).
3. Every candidate has official/blind/heldout_v2/red metrics; missing data is
   an error, not zero.
4. Report includes per-family, `min_norm`, CVaR, cross-play Elo/Nash and
   exploitability.
5. No candidate is promoted in this phase.

## 6. Non-goals

- No PPO fine-tune, no domain randomization (E1), no Bayesian planner (C1).
- No statistical-TFT hardcode, no forced AllD switch, no dynamic first-strike.
- No use of `final_test`/`heldout_v2` for iterative model selection.

## 7. Follow-up after E0

If E0 passes, run E1 (domain-randomized A60 fine-tune) and compare on the same
frozen suites. E2/C1 only start after E0 shows a stable signal.
