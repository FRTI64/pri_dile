# Cross-repo 977 evaluation

- Generated: 2026-10-31T00:42:58+0800
- A: `/home/frti64/frti64_ipd` — 30 standalone strategies
- B: Protocol v4/v5 full catalog — 977 unique opponents
- Matches: 30 × 977 × 20 = 586,200
- Rounds per match: 100
- Payoff: T=10, R=6, P=1, S=0
- Base seed: 20261024
- Wall time: 989.5s
- Raw rows: 586,200; matchup rows: 29,310
- Catalog sha256: `c7aab4da622413460f8906a2846398d3e9466ed429ceab25dad197129eacf158`
- Git: `e1d98eab76f05448c4e063135a549bf194ab3940`

Every match uses fresh A/B instances.  Seeds are derived as
`crc32(f"{base_seed}|{A|B}|{candidate}|{opponent}|{rep}")`.

## Files

| file | content |
|---|---|
| `raw_reps.csv` | local-only (122 MB, ignored): one row per candidate × opponent × rep |
| `matchups.csv` | committed: one row per candidate × opponent, 20-rep mean/std/min/max |
| `matchups.json` | local-only (23 MB, ignored): same as `matchups.csv`, JSON records |
| `weighted_summary.csv` | per-candidate equal / protocol-v4 / protocol-v5 weighted scores |
| `candidates.csv` | A-side metadata |
| `opponents.csv` | B-side metadata, v4/v5 tier and entry weights |
| `run_config.json` | run parameters and checksums |
| `shards/` | resumable per-opponent shards (local, ignored by git) |

The full `raw_reps.csv` (122 MB) and `matchups.json` (23 MB) remain on disk
for local use but are ignored by git.  The repository's pre-commit hook
rejects binary blobs, so their `.gz` copies are ignored as well.  The
committed text artifacts are `matchups.csv` and the summary/metadata files;
regenerate the ignored files with `run_cross_repo.py --resume` or use the
local copies.

## Column notes

`raw_reps.csv` / `matchups.csv`:

- `candidate`, `candidate_class`, `registry_name`: A-side identity.
- `opponent_key`, `family`, `type`: B-side identity.
- `tier_v4`, `tier_v5`, `weight_v4`, `weight_v5`: protocol metadata.
- `score_a` / `score_b`: 100-turn total score (A/B side).
- `diff`: `score_a - score_b`.
- `a_coop_rate` / `b_coop_rate`: fraction of C moves.
- `score_*_mean/std/min/max`, `a_wins/b_wins/ties`: 20-rep summary.
- `reps_ok`: successful reps; `errors`: failed reps for that pair.

`matchups.csv` is the unaggregated per-pair table.  Max possible score is
`100 × T = 1000`, not `100 × R`.

## Weighting

`weighted_summary.csv` uses each manifest entry's `weight` field:
the family weight divided by the number of entries of that family in
its v4/v5 tier.  Weights are normalised over all 977 entries.  The
v4 and v5 columns therefore differ because the v4→v5 split moves 405
entries between train/dev/test tiers, not because the opponent
implementations differ.

## Weighted ranking

| # | candidate | v5 weighted | v4 weighted | equal weight |
|---:|---|---:|---:|---:|
| 1 | `rl-full-guard` | 507.199 | 503.624 | 509.034 |
| 2 | `rl-v5-ft-complex` | 507.164 | 506.027 | 510.180 |
| 3 | `rl-defection-guard` | 506.974 | 503.780 | 508.701 |
| 4 | `rl-wide160` | 506.498 | 504.235 | 508.501 |
| 5 | `rl-full-guard-c` | 506.367 | 505.722 | 509.875 |
| 6 | `rl-dagger-gru` | 506.311 | 505.572 | 509.347 |
| 7 | `rl-v5-ft` | 506.161 | 504.692 | 509.465 |
| 8 | `rl-gate` | 505.732 | 503.655 | 508.653 |
| 9 | `rl-probe-guard` | 505.693 | 503.789 | 508.698 |
| 10 | `rl-switch` | 505.211 | 504.645 | 509.477 |
| 11 | `rl-wide` | 504.808 | 503.242 | 507.493 |
| 12 | `rl-pattern` | 503.878 | 502.003 | 507.178 |
| 13 | `rl-repair` | 499.497 | 497.301 | 506.064 |
| 14 | `triple-v4` | 498.058 | 495.136 | 500.610 |
| 15 | `adv-rl` | 497.560 | 494.951 | 501.410 |
| 16 | `rl-hard60` | 497.051 | 495.978 | 501.100 |
| 17 | `triple-majority` | 496.063 | 490.858 | 498.406 |
| 18 | `rl` | 494.610 | 494.301 | 498.013 |
| 19 | `hybrid` | 493.821 | 489.503 | 497.312 |
| 20 | `neural` | 492.970 | 491.158 | 496.817 |
| 21 | `adv-mlp` | 492.969 | 490.191 | 495.119 |
| 22 | `gru` | 490.901 | 486.646 | 493.829 |
| 23 | `resp3-tuned` | 490.733 | 489.317 | 492.283 |
| 24 | `adv-gru` | 490.721 | 487.532 | 493.054 |
| 25 | `final-base` | 489.825 | 488.573 | 490.991 |
| 26 | `resp3-base` | 489.650 | 489.466 | 491.065 |
| 27 | `final-tuned` | 489.338 | 487.546 | 491.637 |
| 28 | `bayes-base` | 487.111 | 485.478 | 486.738 |
| 29 | `bayes-tuned` | 486.489 | 481.833 | 485.164 |
| 30 | `selfplay` | 485.615 | 485.140 | 489.047 |

## Notes

- B is the unique 977-entry union; v4/v5 copies are not duplicated.
- Excluded from B: 19 long-run Axelrod strategies, `Darwin`, and
  `Red-v5`/`Red-v6` red pools (not part of the 977 catalog).
- `matchups.csv` is intentionally unaggregated; the only
  cross-opponent aggregation is in `weighted_summary.csv`.
