# Axelrod opponent-bank evaluation (2026-10-27)

Environment: `axelrod==4.14.0`, Python 3.14.8.

- `axelrod`: 243 ordinary strategies (`axelrod.strategies`, cheating strategies
  such as `Darwin` excluded).
- `axelrod-fast`: 224 strategies, excluding the 19 strategies Axelrod marks as
  `long_run_time` (mostly Meta players that simulate the whole bank each turn).

## Commands

```bash
# Four core candidates against all 224 fast Axelrod strategies.
python3 -m ipd benchmark --candidates core --dists axelrod-fast \
  --fields 1 --n 224 --workers 5 \
  --out results/axelrod_20261027/benchmark_core_fast.json

# Per-opponent matrix for Final-tuned, 224 fast strategies.
python3 -m ipd matchup --candidates final-tuned --opponent-set axelrod-fast \
  --reps 1 --workers 12 \
  --out results/axelrod_20261027/matchup_final_tuned_fast.csv

# Full 243-strategy smoke test with 10 rounds (not comparable to 100-round
# scores; used to exercise the long-running Meta strategies).
python3 -m ipd matchup --candidates final-tuned --opponent-set axelrod \
  --reps 1 --rounds 10 --workers 12 \
  --out results/axelrod_20261027/matchup_final_tuned_all_rounds10.csv

# Extra cheating set (4.14.0: Darwin only).
python3 -m ipd benchmark --candidates core \
  --dists axelrod-cheating --fields 1 --n 1 --workers 4 \
  --out results/axelrod_20261027/benchmark_core_cheating.json
python3 -m ipd matchup --candidates final-tuned \
  --opponent-set axelrod-cheating --reps 4 \
  --out results/axelrod_20261027/matchup_final_tuned_cheating.csv
```

## Results (100 rounds, 224 fast strategies, one field)

| Candidate | mean score/opponent |
|---|---:|
| Triple-majority | 560.95 |
| Resp3-tuned | 555.37 |
| Final-tuned | 555.27 |
| Bayes-tuned | 546.25 |

## Cheating set (Axelrod 4.14.0: Darwin only)

All four core candidates score 604.00 against Darwin in the one-field
benchmark (Final-tuned also scores 604.00 in the four-rep matchup).  Darwin's
class-level genome stays all-C against these candidates; the 604 is the usual
last-round defection, so this is a mechanism/robustness check rather than a
competitive ranking.  The set is built from `axelrod.cheating_strategies`
(4.14.0 contains Darwin only).  The legacy `Geller`/`MindReader` family is not
in 4.14.0 and is not supported by the current adapter because those classes
need direct access to the candidate's strategy object.

The per-opponent CSV/JSON for Final-tuned has one row per Axelrod strategy and
can be pivoted with:

```bash
python3 -m ipd.tools.pivot_matchup \
  --input results/axelrod_20261027/matchup_final_tuned_fast.csv \
  --json results/axelrod_20261027/matchup_final_tuned_fast.json \
  --wide-out /tmp/axelrod_wide.csv \
  --scores-out /tmp/axelrod_scores.csv
```

## Performance note

A one-candidate, one-field benchmark over the 224-strategy fast bank completes
in about 12 seconds; the four core candidates took about 12 seconds with 5
workers. The full 243 bank includes 19 long-running strategies (mostly Meta
players); with 100 rounds they dominate runtime (the 10-round smoke test above
already took several minutes). Use `axelrod-fast` for routine evaluations and
reserve the full `axelrod` bank for targeted long-running strategy analysis.
