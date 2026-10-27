"""Benchmark, blind-test, suite and matchup harnesses."""
from __future__ import annotations

from .common import (
    FieldStats, MatchStats, aggregate, eval_task, eval_task_details,
    evaluate_field, evaluate_field_details, evaluate_spec_on_dist,
    multi_objective_fitness, quantile, run_match, seed_map,
)
from .distributions import (
    DISTS, FAMILY_MULT, N_PLAYERS, build_dist_field, build_field,
    build_train_dist_field, sample_blind_field, sample_suite_field, sample_wide,
)
from .metagame import (
    antisymmetrize, elo_ratings, exploitability, fictitious_play,
    nash_average,
)
from .opponent_suite import build_suite, suite_names
from .protocol import (
    PROTOCOL_NAME, TEST_TIER, audit_protocol, build_catalog,
    build_protocol_field, build_training_field, catalog_sha256,
    protocol_manifest,
)

__all__ = [
    "DISTS", "N_PLAYERS", "FAMILY_MULT", "build_field", "build_dist_field",
    "build_train_dist_field", "sample_wide", "sample_blind_field",
    "sample_suite_field", "FieldStats", "MatchStats", "run_match",
    "evaluate_field", "evaluate_field_details", "evaluate_spec_on_dist",
    "eval_task", "eval_task_details", "aggregate",
    "quantile", "multi_objective_fitness", "seed_map",
    "antisymmetrize", "elo_ratings", "exploitability", "fictitious_play",
    "nash_average", "build_suite", "suite_names", "PROTOCOL_NAME",
    "TEST_TIER", "audit_protocol", "build_catalog", "build_protocol_field",
    "build_training_field", "catalog_sha256", "protocol_manifest",
]
