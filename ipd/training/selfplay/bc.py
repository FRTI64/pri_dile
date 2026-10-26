#!/usr/bin/env python3
"""Optional behaviour-cloning warm start for the self-play GRU.

The self-play trainer can learn reciprocity from scratch, but 15k episodes are
often not enough to discover *conditional exploitation* (defect against
Random/Bully while preserving TFT/Grim).  This script distils the existing
RL-PPO policy into the 21-dimensional GRU as an optional initialisation::

    python3 -m ipd train selfplay-bc --device auto --episodes 4000
    python3 -m ipd train selfplay --config configs/selfplay/full.json \
      --init-checkpoint weights/selfplay_bc_init.pt --final-test

The teacher is only an initialisation; PPO self-play continues to update the
student against the same opponent population and red league.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ipd.paths import result_path, weights_path
from ipd.policies.rl import RLPolicy
from ipd.training.neural import describe_device, resolve_device

from .config import TrainConfig, load_config
from .models import export_weights, make_model
from .obs import features
from .opponents import make_training_bot, sample_family_labels

BC_WEIGHTS = {
    "good": 0.25, "top": 0.15, "noisy": 0.15,
    "extort": 0.10, "random": 0.15, "special": 0.20,
}


def collect_episodes(cfg: TrainConfig, teacher_weights: str, episodes: int,
                     seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    teacher = RLPolicy(teacher_weights, seed=seed, mode="argmax")
    rows: list[dict[str, Any]] = []
    for i in range(episodes):
        family = rng.choices(list(BC_WEIGHTS),
                             weights=list(BC_WEIGHTS.values()), k=1)[0]
        opponent = make_training_bot(rng, family)
        opponent.reset()
        teacher.reset()
        mine: list[int] = []
        theirs: list[int] = []
        obs: list[list[float]] = []
        actions: list[int] = []
        for t in range(cfg.rounds):
            obs.append(features(t, mine, theirs))
            a = int(teacher.act(t, mine, theirs))
            b = int(opponent.act(t, theirs, mine))
            actions.append(a)
            mine.append(a)
            theirs.append(b)
        rows.append({
            "obs": np.asarray(obs, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "family": family,
            "score": int(sum(6 if a == 0 and b == 0 else
                             0 if a == 0 else
                             10 if b == 0 else 1
                             for a, b in zip(mine, theirs))),
        })
        if (i + 1) % 500 == 0:
            print(f"  collected {i + 1}/{episodes} episodes", flush=True)
    return rows


def train_bc(cfg: TrainConfig, rows: Sequence[dict[str, Any]], *,
             epochs: int = 8, batch_size: int = 32, lr: float = 1e-3,
             device: str | torch.device = "cpu") -> tuple[Any, dict]:
    model = make_model(cfg, value_bias=0.0, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    obs = torch.as_tensor(np.stack([r["obs"] for r in rows]),
                          dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.stack([r["actions"] for r in rows]),
                              dtype=torch.float32, device=device)
    n = obs.shape[0]
    history: list[dict] = []
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(epochs):
        index = np.arange(n)
        rng.shuffle(index)
        losses: list[float] = []
        correct = total = 0
        model.train()
        for start in range(0, n, batch_size):
            idx = torch.as_tensor(index[start:start + batch_size],
                                  device=device)
            logits, _, _ = model.forward_sequence(obs[idx])
            loss = F.binary_cross_entropy_with_logits(logits, actions[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
            pred = (logits >= 0.0).float()
            correct += int((pred == actions[idx]).sum())
            total += int(actions[idx].numel())
        history.append({"epoch": epoch + 1, "loss": statistics.fmean(losses),
                        "accuracy": correct / max(1, total)})
        print(f"epoch {epoch + 1}/{epochs} "
              f"loss={history[-1]['loss']:.4f} "
              f"acc={history[-1]['accuracy']:.4f}", flush=True)
    return model, {"history": history}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Distil the existing RL policy into the 21-d self-play GRU.")
    ap.add_argument("--config", default="configs/selfplay/full.json")
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cuda", "npu", "mps", "cpu"))
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--teacher-weights", default=None)
    ap.add_argument("--out", default=None,
                    help="initialisation state dict for --init-checkpoint")
    ap.add_argument("--policy-out", default=None,
                    help="optional deployable JSON for direct evaluation")
    ap.add_argument("--history-out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.seed = args.seed
    device = resolve_device(args.device)
    teacher = args.teacher_weights or weights_path("rl_policy_checkpoint.json")
    out = args.out or weights_path("selfplay_bc_init.pt")
    policy_out = args.policy_out or weights_path("selfplay_bc_policy.json")
    history_out = args.history_out or result_path("selfplay_bc_history.json")
    print(f"device: {describe_device(device)} teacher={teacher}", flush=True)

    started = time.time()
    rows = collect_episodes(cfg, teacher, args.episodes, args.seed)
    mean_score = statistics.fmean(r["score"] for r in rows)
    print(f"teacher mean score on BC fields: {mean_score:.2f}", flush=True)
    model, metrics = train_bc(cfg, rows, epochs=args.epochs,
                              batch_size=args.batch_size, lr=args.lr,
                              device=device)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out)
    export = export_weights(model, meta={
        "algorithm": "bc-distillation",
        "teacher": teacher,
        "episodes": args.episodes,
        "teacher_score": mean_score,
        "metrics": metrics,
        "seed": args.seed,
    })
    Path(policy_out).parent.mkdir(parents=True, exist_ok=True)
    with open(policy_out, "w", encoding="utf-8") as fh:
        json.dump(export, fh, ensure_ascii=False, indent=2)
    with open(history_out, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"episodes": args.episodes,
                            "teacher_score": mean_score,
                            "seconds": time.time() - started},
                   "metrics": metrics}, fh, ensure_ascii=False, indent=2)
    print(f"written {out}", flush=True)
    print(f"written {policy_out}", flush=True)
    print(f"written {history_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
