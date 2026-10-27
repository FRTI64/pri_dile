# E0 — League / Cross-play / Exploitability Harness

- **Status:** implemented; full frozen run in `results/e0_league_20261026/`
- **Plan:** `docs/research/e0-league-plan.md`
- **Entry point:** `python3 -m ipd league`

This harness is the P0 gate before any new training method. It freezes every
candidate spec, opponent suite, seed and policy mode, then reports metrics that
the old four-distribution benchmark cannot show.

## 1. What it runs

```bash
python3 -m ipd league --config configs/eval/e0_league.json \
  --out results/e0_league_20261026/report.json
```

The config controls:

- candidate specs and exact weight/config paths;
- official four-distribution fields;
- blind fields;
- fixed opponent suites (`validation`, new `heldout_v2`);
- cross-play seeds;
- red best-response policies and seeds;
- per-candidate `policy_seed` (optional; useful for fair checkpoint
  comparisons where every checkpoint must use the same policy RNG seed);
- Nash fictitious-play budget.

Useful smoke/debug flags:

```bash
python3 -m ipd league --config configs/eval/e0_league.json \
  --only RL-PPO,RL-PPO-Hard60 --suites heldout_v2 \
  --cross-play-seeds 2 --red-seeds 2 --workers 4 \
  --out /tmp/e0_smoke/report.json --md-out /tmp/e0_smoke/README.md

python3 -m ipd league --config configs/eval/e0_league.json \
  --skip-suites --cross-play-seeds 2 --workers 4 \
  --out /tmp/e0_cross/report.json
```

## 2. Modules

| File | Purpose |
|---|---|
| `ipd/evaluation/opponent_suite.py` | frozen suites; new `heldout_v2` families and parameters |
| `ipd/evaluation/metagame.py` | antisymmetric payoff matrix, fictitious-play Nash averaging, Elo, exploitability |
| `ipd/evaluation/league.py` | config loader, cross-play/suite/red workers, summaries, CLI, Markdown report |
| `configs/eval/e0_league.json` | frozen candidate/suite/seed config |

P3 extends the harness with `ipd/training/league.py` and
`python3 -m ipd meta`: it extracts a red×blue score-difference matrix from a
red report, computes maximin weights, and writes a weighted opponent manifest
for `python3 -m ipd train rl --opponent-pool`.  See
`docs/research/p3-expansion.md`.

`heldout_v2` has seven families: `reciprocal_v2`, `punitive_v2`, `noisy_v2`,
`extort_v2`, `random_v2`, `temporal_v2`, `adaptive_v2`. Its parameter draws
are disjoint from the historical `final`/validation draws wherever practical.
It is a **report-only** suite; do not use it to iterate on model choice.

A second suite, `heldout_v3`, adds eight disjoint families
(`reciprocal_v3`, `punitive_v3`, `noisy_v3`, `extort_v3`, `random_v3`,
`temporal_v3`, `adaptive_v3`, `anchor_v3`). It is reserved for a one-shot
final test after a candidate passes validation. E1c used it once
(RL-Repair 509.88 vs A60 505.97), so **`heldout_v3` is no longer available**;
the next candidate that passes validation must use a new `heldout_v4`.

## 3. Metrics

- `mean`: mean score per match.
- `families`: per suite/family means, keyed as `suite:family`.
- `min_family_raw`: worst raw family mean (not comparable across families).
- `min_norm_family`: worst family mean divided by the same family's baseline
  (default `RL-PPO`) mean. This is the comparable tail metric.
- `cvar10_family`: mean of the worst 10% family means.
- `cvar10_item`: mean of the worst 10% individual match scores.
- `p05_item`, `sigma_family`, `max_minus_min`, `top_family_share`.
- Cross-play: full payoff matrix, win-rate matrix, Elo, Nash mixture, Nash
  exploitability, per-candidate pool exploitability.
- Red: blue mean, red mean, diff, red win rate, worst red, normalized blue
  score vs baseline.

## 4. Interpretation rules

1. `heldout_v2` and the historical `final` suite are not interchangeable.
   `final` is now validation because it was used in A/B/C decisions.
2. Cross-play Elo/Nash is a **pool-level** approximation. It is not proof
   against an arbitrary best response.
3. E4 red was trained against a frozen BC blue. It is a held-out stress test
   for non-BC candidates but does not solve anti-exploitation.
4. A candidate is not promoted by E0. E1/E2 must pass the same harness and
   the hard constraints before any weight replacement.
5. Compare `min_norm_family`, CVaR and per-family scores before the overall
   mean. Raw `min_family` is not comparable across families.

## 5. Frozen run result (2026-10-26)

See `results/e0_league_20261026/README.md` for the full table. Key points:

- E0 gate PASS; 9 candidates, official/blind/validation/heldout_v2/red complete.
- A60 overall 552.11 vs RL-PPO 550.80; heldout_v2 500.07 vs 494.97;
  CVaR10 391.13 vs 387.26. A60 min_norm 0.9928 because blind `wild`
  is slightly lower; E4 red blue 299.20 vs 310.55.
- Adv-RL has the best heldout_v2 mean (502.39) but a worse red/tail profile.
- SelfPlay-GRU/BC remain below RL-PPO on heldout_v2 and min_norm.
- Cross-play is degenerate: all candidates mutually cooperate (594–595),
  Elo ≈ 1000, Nash ≈ uniform, pool exploitability ≈ 0. A blue-only league
  cannot measure exploitability; P3 must include red/exploiter checkpoints.
- No weights were replaced.

E1 used the same harness for the domain-randomized A60 fine-tune. E1a was a
No-Go (held-out v2 +1.36), and E1b was also a No-Go (selection +3.05 but full
+1.94). E1c repaired the random family: held-out v2 +3.10, official +0.45,
blind +0.80, red blue +4.25, and the one-shot held-out v3 final was +3.91 vs
A60. E1c is a **Go candidate** (`rl-repair`); the main weight is unchanged.
See `results/robust_20261026/README.md`.

E2 added CVaR episode weighting, Dirichlet mixtures and worst-family DRO, but
E2a/E2b/E2c failed to improve held-out CVaR (from -0.5% to +1.2%) and were
No-Go. See `results/risk_20261026/README.md`.

## 6. Tests

```bash
python3 -m unittest tests.test_metagame tests.test_league -v
```

Coverage includes: antisymmetry, RPS Nash/uniformity, pure-strategy
exploitability, Elo ordering, deterministic Nash averaging, `heldout_v2`
uniqueness/determinism, cross-play matrix construction, `min_norm`/CVaR
summaries, red worker, config loading and a CLI smoke run.
