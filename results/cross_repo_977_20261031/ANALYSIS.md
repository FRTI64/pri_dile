# 30 candidates × 977 Protocol v4/v5 opponents — analysis

## Setup

- A: 30 standalone strategies from `/home/frti64/frti64_ipd/`.
- B: 977 unique Protocol v4/v5 opponent configs.
- 20 reps per A×B pair; 100 turns per match.
- Payoff: `T=10, R=6, P=1, S=0`.
- Total: 586,200 matches; 0 errors.
- Seeds: independent per `(candidate, opponent, rep)`.

`matchups.csv` has one unaggregated row per A×B pair. The only aggregate
table is `weighted_summary.csv`, which uses the v4/v5 manifest entry weights.

## Ranking

Top 10 by v5 weighted score:

| # | candidate | v5 weighted | v4 weighted | equal weight |
|---:|---|---:|---:|---:|
| 1 | `rl-full-guard` | **507.199** | 503.624 | 509.034 |
| 2 | `rl-v5-ft-complex` | 507.164 | **506.027** | **510.180** |
| 3 | `rl-defection-guard` | 506.974 | 503.780 | 508.701 |
| 4 | `rl-wide160` | 506.498 | 504.235 | 508.501 |
| 5 | `rl-full-guard-c` | 506.367 | 505.722 | 509.875 |
| 6 | `rl-dagger-gru` | 506.311 | 505.572 | 509.347 |
| 7 | `rl-v5-ft` | 506.161 | 504.692 | 509.465 |
| 8 | `rl-gate` | 505.732 | 503.655 | 508.653 |
| 9 | `rl-probe-guard` | 505.693 | 503.789 | 508.698 |
| 10 | `rl-switch` | 505.211 | 504.645 | 509.477 |

Headline:

- **v5 weighted**: `rl-full-guard` first (507.199), but only **0.035**
  above `rl-v5-ft-complex`; treat the top two as tied.
- **v4 weighted**: `rl-v5-ft-complex` first (506.027).
- **Equal weight**: `rl-v5-ft-complex` first (510.180).

The v4/v5 weighted rankings differ because 405/977 entries move between
train/dev/test tiers from v4 to v5. The opponent implementations are the
same; only the protocol weights differ.

## Opponent-family difficulty

Mean A score across all 30 candidates, by B family:

| family | mean A score | note |
|---|---:|---|
| `early_harvest` | 132.28 | lowest / hardest |
| `hyper_extort` | 157.68 | |
| `probe_exploit` | 191.56 | |
| `frequent_kamikaze` | 219.70 | |
| `adaptive_mix` | 286.72 | |
| `conditional_extort` | 309.40 | |
| `late` | 342.72 | |
| `phase_switch` | 376.32 | |
| `extort` | 379.52 | |
| `endgame_hunter` | 388.81 | |
| ... | ... | |
| `defect` | 703.49 | highest / most exploitable |
| `adaptive_probe` | 637.56 | |
| `delayed_grim` | 604.00 | |
| `nice` | 601.88 | |
| `punitive` | 601.00 | |

Patterns:

- Nearly every candidate's worst family is `early_harvest`
  (about 130 points) or `hyper_extort`.
- Nearly every learning candidate's best family is `defect`
  (about 650–750), driven by exploitable early defectors.
- `triple-majority` / `triple-v4` are the exception: their best family is
  `adaptive_probe` (617.17), not `defect`.
- The cooperative complex families (`noise_adaptive`, `memory_two`,
  `contrition_cycle`, `mirror_adaptive`, `reputation_grim`, `policy`)
  cluster near 595 points for most candidates.

## Extreme pairings

Highest A scores:

- `bayes-base` / `bayes-tuned` vs `defect:EarlyDefector:k=1`: **990.0**
- `bayes-base` / `bayes-tuned` vs `axelrod_extra:Axelrod:strategy=OppositeGrudger`: **990.0**
- `bayes-base` / `bayes-tuned` vs `axelrod_extra:Axelrod:strategy=WinShiftLoseStay`: **990.0**

Lowest A scores:

- `rl-repair` vs `axelrod_extra:Axelrod:strategy=Handshake`: **87.95**
- `rl-repair` vs `late:LateDefector:k=99`: **88.45**
- `rl-repair` vs `probe_exploit:ProbeExploit:chi=3.0|mode=alld|probe=2`: **88.50**

## Caveats

- 20 reps is enough to rank, but differences below roughly 1 point should be
  treated as noise, especially inside the top cluster.
- The three weighted columns answer different questions: v5 weighting is the
  current protocol's per-entry weighting; v4 weighting is the legacy split;
  equal weight treats all 977 opponents identically.
- Scores are 100-turn totals, with a maximum of `100 × T = 1000`.
- The raw 586,200-row table is split by candidate under `raw_reps/` and is
  also available locally as `raw_reps.csv`.

## Files

- `matchups.csv` / `matchups.json`: per-pair 20-rep summary.
- `weighted_summary.csv` / `.json`: per-candidate weighted summary.
- `raw_reps/`: raw per-rep data, one CSV per candidate.
- `README.md`: run configuration and column definitions.
