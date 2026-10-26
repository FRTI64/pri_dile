"""Configuration for the self-play curriculum."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ipd.paths import ROOT, result_path, weights_path

from .opponents import PHASE_WEIGHTS


@dataclass
class PhaseConfig:
    name: str
    until_episodes: int
    red_fraction: tuple[float, float] = (0.0, 0.0)
    alpha: float = 0.85
    lam: float = 5.0
    weights: dict[str, float] | None = None


@dataclass
class PPOConfig:
    hidden: int = 64
    gru_layers: int = 2
    actor_hidden: int = 32
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    batch_episodes: int = 8
    blue_lr: float = 3e-4
    red_lr: float = 2e-4
    final_lr: float = 1e-5
    reward_scale: float = 0.01
    target_kl: float = 0.05
    center_advantages: bool = True
    entropy_by_phase: dict[str, float] | None = None
    bc_kl_coef: float = 0.0


@dataclass
class EvalConfig:
    fast_every_episodes: int = 200
    fast_opponents: int = 12
    fast_seeds: int = 3
    full_every_episodes: int = 1000
    full_seeds: int = 5
    final_seeds: int = 20
    warmup_episodes: int = 1000
    red_calibration_every_full: bool = True
    heldout_red_weights: str = field(
        default_factory=lambda: weights_path("rl_policy_checkpoint_bc.json"))
    # Red-line thresholds (100-round scores).
    tft_min: float = 590.0
    bully_min: float = 650.0
    noisy02_min: float = 560.0
    alld_min: float = 95.0
    min_family_min: float = 430.0
    heldout_red_min: float = 430.0
    failures_before_action: int = 3


@dataclass
class TrainConfig:
    seed: int = 20261026
    device: str = "auto"
    envs: int = 32
    rounds: int = 100
    total_episodes: int = 15000
    red_league_size: int = 8
    red_update_every: int = 3
    red_snapshot_every: int = 3
    phases: list[PhaseConfig] = field(default_factory=list)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    results_dir: str = field(
        default_factory=lambda: str(result_path("selfplay_20261026")))
    out: str = field(default_factory=lambda: weights_path(
        "selfplay_gru_policy.json"))
    checkpoint_out: str = field(default_factory=lambda: weights_path(
        "selfplay_checkpoint.pt"))
    log_every: int = 1

    def __post_init__(self) -> None:
        if not self.phases:
            self.phases = default_phases()

    @property
    def iterations(self) -> int:
        return max(1, int(round(self.total_episodes / self.envs)))

    def phase_for_episode(self, episodes_seen: int) -> PhaseConfig:
        for phase in self.phases:
            if episodes_seen < phase.until_episodes:
                return phase
        return self.phases[-1]

    def phase_index(self, episodes_seen: int) -> int:
        for i, phase in enumerate(self.phases):
            if episodes_seen < phase.until_episodes:
                return i
        return len(self.phases) - 1


def default_phases() -> list[PhaseConfig]:
    return [
        PhaseConfig("1a", 3000, (0.0, 0.0), 0.0, 0.0, PHASE_WEIGHTS["1a"]),
        PhaseConfig("1b", 8000, (0.0, 0.0), 0.0, 0.0, PHASE_WEIGHTS["1b"]),
        PhaseConfig("2", 13000, (0.15, 0.25), 0.85, 5.0, PHASE_WEIGHTS["2"]),
        PhaseConfig("3", 15000, (0.20, 0.30), 0.90, 10.0, PHASE_WEIGHTS["3"]),
    ]


def _merge_dataclass(obj: Any, data: dict[str, Any]) -> Any:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"unknown config field: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | None = None,
                base: TrainConfig | None = None) -> TrainConfig:
    cfg = base if base is not None else TrainConfig()
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = _merge_dataclass(cfg, raw)
    if overrides:
        cfg = _merge_dataclass(cfg, overrides)
    if cfg.phases:
        cfg.phases = [
            p if isinstance(p, PhaseConfig) else PhaseConfig(**p)
            for p in cfg.phases
        ]
    # Always keep the final phase covering the requested budget.
    if cfg.phases and cfg.total_episodes > cfg.phases[-1].until_episodes:
        cfg.phases[-1].until_episodes = cfg.total_episodes
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def smoke_config() -> TrainConfig:
    """Small end-to-end configuration used by tests and M2 smoke runs."""
    cfg = TrainConfig()
    cfg.envs = 8
    cfg.total_episodes = 200
    cfg.results_dir = str(ROOT / "results" / "selfplay_smoke")
    cfg.out = str(ROOT / "weights" / "selfplay_smoke_policy.json")
    cfg.checkpoint_out = str(ROOT / "weights" / "selfplay_smoke_checkpoint.pt")
    cfg.red_update_every = 1
    cfg.evaluation.fast_every_episodes = 50
    cfg.evaluation.fast_opponents = 6
    cfg.evaluation.fast_seeds = 1
    cfg.evaluation.full_every_episodes = 100
    cfg.evaluation.full_seeds = 1
    cfg.evaluation.warmup_episodes = 0
    cfg.evaluation.final_seeds = 1
    cfg.ppo.epochs = 2
    cfg.ppo.batch_episodes = 4
    return cfg


__all__ = [
    "PhaseConfig", "PPOConfig", "EvalConfig", "TrainConfig", "load_config",
    "default_phases", "smoke_config",
]
