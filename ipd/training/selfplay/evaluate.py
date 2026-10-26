"""Per-family validation, checkpoint selection, and one-shot final test."""
from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from ipd.core.primitives import C, D
from ipd.policies.rl import RLPolicy

from . import N_FEATURES
from .env import VectorEnv
from .models import GRUActorCritic
from .opponents import (
    BotSpec, build_calibration_suite, build_final_suite,
    build_validation_suite,
)


@dataclass
class EvalTask:
    name: str
    family: str
    seed: int
    factory: Callable[[random.Random], object]

    def build_factory(self):
        rng = random.Random(self.seed)

        def factory():
            return self.factory(rng)
        return factory


def suite_tasks(suite: Sequence[BotSpec], seeds: Sequence[int]) -> list[EvalTask]:
    return [EvalTask(item.name, item.family, seed, item.factory)
            for item in suite for seed in seeds]


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@torch.no_grad()
def evaluate_model(model: GRUActorCritic, tasks: Sequence[EvalTask], *,
                   device: str | torch.device = "cpu", mode: str = "sample",
                   batch_size: int = 32, seed: int = 20261026) -> list[dict]:
    """Evaluate a recurrent model on a list of independent episodes.

    Episodes are run in lock-step batches so the GPU sees a real batch at every
    round.  All opponents are fixed before the episode starts.
    """
    model = model.to(device).eval()
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for start in range(0, len(tasks), max(1, batch_size)):
        batch = list(tasks[start:start + max(1, batch_size)])
        env = VectorEnv([task.build_factory() for task in batch],
                        [task.family for task in batch])
        h = model.init_hidden(len(batch), device)
        for t in range(100):
            obs = torch.as_tensor(env.observations(), dtype=torch.float32,
                                  device=device)
            logits, values, h = model.step(obs, h)
            if mode == "argmax":
                actions = (logits >= 0.0).long().cpu().numpy()
            else:
                probs = _sigmoid_np(logits.detach().cpu().numpy())
                actions = (rng.random(len(batch)) < probs).astype(np.int64)
            env.step(actions.tolist())
        for task, env_item in zip(batch, env.envs):
            rows.append({
                "name": task.name,
                "family": task.family,
                "seed": task.seed,
                "score": env_item.own_score,
                "opponent_score": env_item.opponent_score,
                "opponent": env_item.opponent_name,
            })
    return rows


def summarize_rows(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"mean": 0.0, "families": {}, "min_family": 0.0,
                "sigma": 0.0, "vs_final": 0.0, "vs_triple": 0.0,
                "worst": 0.0, "n": 0}
    by_family: dict[str, list[float]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(float(row["score"]))
    family_means = {k: statistics.fmean(v) for k, v in by_family.items()}
    all_scores = [float(r["score"]) for r in rows]
    sigma = (statistics.pstdev(list(family_means.values()))
             if len(family_means) > 1 else 0.0)
    by_name = {r["name"]: float(r["score"]) for r in rows}
    final_scores = [v for k, v in by_name.items()
                    if k.startswith("Final-tuned")]
    triple_scores = [v for k, v in by_name.items()
                     if k.startswith("Triple-majority")]
    return {
        "mean": statistics.fmean(all_scores),
        "families": family_means,
        "min_family": min(family_means.values()),
        "sigma": sigma,
        "vs_final": statistics.fmean(final_scores) if final_scores else 0.0,
        "vs_triple": statistics.fmean(triple_scores) if triple_scores else 0.0,
        "worst": min(all_scores),
        "n": len(rows),
        "rows": list(rows),
    }


def passes_thresholds(summary: dict, config) -> tuple[bool, dict[str, bool]]:
    by_name = {r["name"]: float(r["score"]) for r in summary.get("rows", [])}
    eval_cfg = config.evaluation
    checks = {
        "tft": by_name.get("TFT", -1e9) >= eval_cfg.tft_min,
        "bully": by_name.get("Bully", -1e9) >= eval_cfg.bully_min,
        "noisy02": by_name.get("NoisyTFT.hold.02",
                               by_name.get("NoisyTFT.02", -1e9))
        >= eval_cfg.noisy02_min,
        "alld": by_name.get("AllD.hold", by_name.get("AllD", -1e9))
        >= eval_cfg.alld_min,
        "min_family": summary["min_family"] >= eval_cfg.min_family_min,
    }
    return all(checks.values()), checks


def selection_score(summary: dict) -> float:
    return (summary["mean"]
            + 0.5 * summary.get("vs_final", 0.0)
            + 0.5 * summary.get("vs_triple", 0.0)
            + 0.5 * summary["min_family"]
            - 2.0 * summary.get("sigma", 0.0))


def validation_report(model: GRUActorCritic, *, device, seeds: Sequence[int],
                      mode: str = "sample", batch_size: int = 32) -> dict:
    suite = build_validation_suite()
    tasks = suite_tasks(suite, seeds)
    rows = evaluate_model(model, tasks, device=device, mode=mode,
                          batch_size=batch_size)
    summary = summarize_rows(rows)
    return summary


FAST_NAMES = (
    "TFT", "Pavlov", "Grim", "Final-tuned", "Triple-majority",
    "NoisyTFT.02", "Bully", "Extort.hold.3.6", "Random.hold.0.03",
    "Random.hold.0.92", "AllD.hold", "LateD.hold.88",
)


def fast_validation(model: GRUActorCritic, *, device, opponents: int = 12,
                    seeds: Sequence[int] = (20261026,),
                    mode: str = "sample") -> dict:
    full = build_validation_suite()
    by_name = {item.name: item for item in full}
    suite = [by_name[name] for name in FAST_NAMES if name in by_name]
    for item in full:
        if len(suite) >= opponents:
            break
        if item not in suite:
            suite.append(item)
    suite = suite[:opponents]
    rows = evaluate_model(model, suite_tasks(suite, seeds),
                          device=device, mode=mode, batch_size=32)
    return summarize_rows(rows)


def calibration_report(model: GRUActorCritic, anchor: dict[str, float], *,
                       device, seeds: Sequence[int] = (20261026,),
                       mode: str = "sample") -> dict:
    suite = build_calibration_suite()
    rows = evaluate_model(model, suite_tasks(suite, seeds),
                          device=device, mode=mode, batch_size=32)
    summary = summarize_rows(rows)
    ratios = {family: summary["families"].get(family, 0.0) / max(1e-9, base)
              for family, base in anchor.items()}
    return {
        **summary,
        "anchor": dict(anchor),
        "ratios": ratios,
        "worst_ratio": min(ratios.values()) if ratios else 0.0,
    }


def heldout_red_tasks(seeds: Sequence[int],
                      checkpoint: str) -> list[EvalTask]:
    def factory_for(seed: int):
        def factory(_rng: random.Random):
            return RLPolicy(checkpoint, seed=seed, mode="sample")
        return factory

    return [EvalTask(f"RL-final-heldout.red.{seed}", "heldout_red", seed,
                     factory_for(seed))
            for seed in seeds]


def final_test(model: GRUActorCritic, *, device,
               seeds: Sequence[int] | None = None,
               heldout_red_weights: str | None = None,
               mode: str = "sample", batch_size: int = 32) -> dict:
    seeds = list(seeds or range(20262000, 20262020))
    suite = build_final_suite()
    rows = evaluate_model(model, suite_tasks(suite, seeds), device=device,
                          mode=mode, batch_size=batch_size)
    summary = summarize_rows(rows)
    red = {"mean": None, "rows": []}
    if heldout_red_weights:
        red_rows = evaluate_model(
            model, heldout_red_tasks(seeds[:min(20, len(seeds))],
                                     heldout_red_weights),
            device=device, mode=mode, batch_size=batch_size)
        red = summarize_rows(red_rows)
    return {
        "schema": "ipd-selfplay-final-test-v1",
        "suite_size": len(suite),
        "seeds": seeds,
        "summary": summary,
        "heldout_red": red,
    }


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


__all__ = [
    "EvalTask", "suite_tasks", "evaluate_model", "summarize_rows",
    "passes_thresholds", "selection_score", "validation_report",
    "fast_validation", "calibration_report", "heldout_red_tasks",
    "final_test", "save_json",
]
