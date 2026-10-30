# Raw per-repetition data

One CSV per A-side candidate.  Each file has the same columns as the combined
`raw_reps.csv` and contains:

- 977 opponents × 20 reps = **19,540 rows** per candidate
- 30 candidates × 19,540 = **586,200 rows** total

Files are grouped by candidate to keep each committed file below GitHub's
single-file limit.  The combined `raw_reps.csv` is local-only and ignored by
git because it is 122 MB.

## Rebuild the combined CSV

```bash
cd results/cross_repo_977_20261031
python raw_reps/merge_raw_reps.py
```

This writes `raw_reps.csv` next to the `raw_reps/` directory and checks that
the row count is 586,200.

## Columns

- `candidate`, `candidate_class`, `registry_name`: A-side identity.
- `opponent_index`, `opponent_key`, `family`, `type`: B-side identity.
- `tier_v4`, `tier_v5`, `weight_v4`, `weight_v5`, `family_weight`:
  protocol metadata.
- `rep`: repetition index 0–19.
- `seed_a`, `seed_b`: derived per-match seeds.
- `score_a`, `score_b`: 100-turn total scores.
- `diff`: `score_a - score_b`.
- `a_coop_rate`, `b_coop_rate`: fraction of C moves.
- `error`: empty on success.
