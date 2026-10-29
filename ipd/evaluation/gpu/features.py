"""Exact batched feature extractors for the learned IPD policies.

Two contracts are implemented:

``features85``
    The 85-dimensional vector in :mod:`ipd.policies.neural` used by
    ``NeuralPolicy``, ``RLPolicy`` and ``GRUPolicy``.

``features21``
    The 21-dimensional ``selfplay-21-v1`` (and optional ``selfplay-21-v2``)
    observation used by ``SelfPlayGRUPolicy``.

Both a functional batch form and an incremental state object are provided.
The incremental form keeps lifetime counters, first/last defection positions,
run lengths and switch counts, so the per-round cost is independent of the
match length instead of recomputing the whole history.  The functional form is
used by tests and by short ad-hoc batches.

All tensors use the project action convention ``C = 0``, ``D = 1``.  Missing
history entries in the 21-dimensional observation are represented by ``-1``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

C, D = 0, 1

FEATURES85 = 85
FEATURES97 = 97
FEATURES21 = 21
_V2_PERIODS = (2, 3, 4, 5)
_V2_WINDOW = 8
_V2_TAIL = 10
ROUNDS = 100
_LOG101 = math.log(101.0)


def _as_float(seq: torch.Tensor, device: torch.device,
              dtype: torch.dtype) -> torch.Tensor:
    return seq.to(device=device, dtype=dtype)


def _mean_prefix(seq: torch.Tensor, t: int, window: int,
                 dtype: torch.dtype) -> torch.Tensor:
    """Mean of ``seq[:, max(0, t-window):t]``, 0.0 for an empty slice."""
    if t <= 0:
        return torch.zeros(seq.shape[0], device=seq.device, dtype=dtype)
    start = max(0, t - window)
    chunk = seq[:, start:t].to(dtype)
    return chunk.mean(dim=1)


def _last_moves(seq: torch.Tensor, t: int, k: int,
                dtype: torch.dtype) -> torch.Tensor:
    """Newest first: D=+1, C=-1, missing=0.  Returns ``[B, k]``."""
    b = seq.shape[0]
    out = torch.zeros(b, k, device=seq.device, dtype=dtype)
    for i in range(k):
        j = t - 1 - i
        if j >= 0:
            out[:, i] = torch.where(seq[:, j] == D,
                                    torch.ones((), device=seq.device,
                                               dtype=dtype),
                                    -torch.ones((), device=seq.device,
                                                dtype=dtype))
    return out


def _run_length(seq: torch.Tensor, t: int,
                dtype: torch.dtype) -> torch.Tensor:
    if t <= 0:
        return torch.zeros(seq.shape[0], device=seq.device, dtype=dtype)
    last = seq[:, t - 1:t]
    rev = seq[:, :t].flip(1)
    same = (rev == last).to(dtype)
    return same.cumprod(1).sum(1)


def _alternation(seq: torch.Tensor, t: int, window: int,
                 dtype: torch.dtype) -> torch.Tensor:
    b = seq.shape[0]
    if t < 2:
        return torch.zeros(b, device=seq.device, dtype=dtype)
    start = max(0, t - window)
    tail = seq[:, start:t]
    if tail.shape[1] < 2:
        return torch.zeros(b, device=seq.device, dtype=dtype)
    return (tail[:, 1:] != tail[:, :-1]).to(dtype).mean(1)


def _exact_alternation(seq: torch.Tensor, t: int, window: int,
                       dtype: torch.dtype) -> torch.Tensor:
    b = seq.shape[0]
    if t < window:
        return torch.zeros(b, device=seq.device, dtype=dtype)
    tail = seq[:, t - window:t]
    return (tail[:, 1:] != tail[:, :-1]).all(1).to(dtype)


def _switch_rate(seq: torch.Tensor, t: int,
                         dtype: torch.dtype) -> torch.Tensor:
    if t < 2:
        return torch.zeros(seq.shape[0], device=seq.device, dtype=dtype)
    return (seq[:, 1:t] != seq[:, :t - 1]).to(dtype).mean(1)


def _joint_counts(mine: torch.Tensor, theirs: torch.Tensor, t: int,
                  dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    m = mine[:, :t]
    th = theirs[:, :t]
    cc = ((m == C) & (th == C)).to(dtype).sum(1)
    cd = ((m == C) & (th == D)).to(dtype).sum(1)
    dc = ((m == D) & (th == C)).to(dtype).sum(1)
    dd = ((m == D) & (th == D)).to(dtype).sum(1)
    return cc, cd, dc, dd


def _payoff_tensor(mine: torch.Tensor, theirs: torch.Tensor,
                   dtype: torch.dtype) -> torch.Tensor:
    cc = (mine == C) & (theirs == C)
    dc = (mine == D) & (theirs == C)
    dd = (mine == D) & (theirs == D)
    return cc.to(dtype) * 6 + dc.to(dtype) * 10 + dd.to(dtype)


def _conditional_counts(mine: torch.Tensor, theirs: torch.Tensor, t: int,
                        dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    """Counts for transitions ``i=1..t-1``: mine[i-1] -> theirs[i]."""
    b = mine.shape[0]
    if t < 2:
        z = torch.zeros(b, device=mine.device, dtype=dtype)
        return z, z.clone(), z.clone(), z.clone()
    pm = mine[:, :t - 1]
    ct = theirs[:, 1:t]
    c_mask = pm == C
    d_mask = pm == D
    n_c = c_mask.to(dtype).sum(1)
    n_d = d_mask.to(dtype).sum(1)
    d_c = (c_mask & (ct == D)).to(dtype).sum(1)
    d_d = (d_mask & (ct == D)).to(dtype).sum(1)
    return n_c, n_d, d_c, d_d


def _recency(mine: torch.Tensor, theirs: torch.Tensor, t: int,
             dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    b = mine.shape[0]
    if t == 0:
        zeros = torch.zeros(b, device=mine.device, dtype=dtype)
        ones = torch.ones(b, device=mine.device, dtype=dtype)
        return ones.clone(), ones.clone(), ones.clone(), ones.clone(), \
            zeros.clone(), zeros.clone()
    m = mine[:, :t]
    th = theirs[:, :t]
    has_t_d = (th == D).any(1)
    has_m_d = (m == D).any(1)
    first_t = torch.where(has_t_d, (th == D).to(torch.int8).argmax(1),
                          torch.full((b,), -1, device=mine.device,
                                     dtype=torch.long))
    first_m = torch.where(has_m_d, (m == D).to(torch.int8).argmax(1),
                          torch.full((b,), -1, device=mine.device,
                                     dtype=torch.long))
    last_t = torch.where(has_t_d,
                         t - 1 - (th.flip(1) == D).to(torch.int8).argmax(1),
                         torch.full((b,), -1, device=mine.device,
                                    dtype=torch.long))
    last_m = torch.where(has_m_d,
                         t - 1 - (m.flip(1) == D).to(torch.int8).argmax(1),
                         torch.full((b,), -1, device=mine.device,
                                    dtype=torch.long))
    ft = first_t.to(dtype)
    fm = first_m.to(dtype)
    lt = last_t.to(dtype)
    lm = last_m.to(dtype)
    last_opp = torch.where(last_t >= 0, (t - lt) / 100.0,
                           torch.ones(b, device=mine.device, dtype=dtype))
    last_my = torch.where(last_m >= 0, (t - lm) / 100.0,
                          torch.ones(b, device=mine.device, dtype=dtype))
    first_opp = torch.where(first_t >= 0, (ft + 1.0) / 100.0,
                            torch.ones(b, device=mine.device, dtype=dtype))
    first_my = torch.where(first_m >= 0, (fm + 1.0) / 100.0,
                           torch.ones(b, device=mine.device, dtype=dtype))
    since_opp = torch.where(first_t >= 0, (t - ft) / 100.0,
                            torch.zeros(b, device=mine.device, dtype=dtype))
    gap = torch.where((first_m >= 0) & (first_t >= 0) & (first_m >= first_t),
                      (fm - ft) / 100.0,
                      torch.zeros(b, device=mine.device, dtype=dtype))
    return last_opp, last_my, first_opp, first_my, since_opp, gap


def features85(mine: torch.Tensor, theirs: torch.Tensor,
               t: int | None = None,
               dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Batched exact port of :func:`ipd.policies.neural.extract_features`.

    ``mine``/``theirs`` are ``[B, R]`` action tensors.  Only the first ``t``
    entries are used; ``t`` defaults to the full width.
    """
    if mine.shape != theirs.shape:
        raise ValueError("mine and theirs must have the same shape")
    if mine.dim() != 2:
        raise ValueError("histories must be [B, R]")
    if t is None:
        t = mine.shape[1]
    t = int(t)
    if t < 0 or t > mine.shape[1]:
        raise ValueError("t is outside the history buffer")
    device = mine.device
    b = mine.shape[0]
    one = torch.ones(b, device=device, dtype=dtype)
    zero = torch.zeros(b, device=device, dtype=dtype)

    def scalar(value: float) -> torch.Tensor:
        return torch.full((b,), float(value), device=device, dtype=dtype)

    f = [
        scalar(t / 99.0), scalar((99 - t) / 99.0),
        scalar(1.0 if t == 0 else 0.0),
        scalar(1.0 if t >= 99 else 0.0),
        scalar(1.0 if t >= 95 else 0.0),
        scalar(1.0 if t >= 90 else 0.0),
    ]
    f.append(_last_moves(mine, t, 8, dtype))
    f.append(_last_moves(theirs, t, 8, dtype))

    sum_mine = mine[:, :t].to(dtype).sum(1)
    sum_theirs = theirs[:, :t].to(dtype).sum(1)
    f += [
        sum_mine / max(1, t), sum_theirs / max(1, t),
        _mean_prefix(mine, t, 10, dtype),
        _mean_prefix(theirs, t, 3, dtype),
        _mean_prefix(theirs, t, 5, dtype),
        _mean_prefix(theirs, t, 10, dtype),
        _mean_prefix(theirs, t, 20, dtype),
    ]

    n_c, n_d, d_c, d_d = _conditional_counts(mine, theirs, t, dtype)
    p_c_raw = torch.where(n_c > 0, d_c / n_c.clamp_min(1), zero)
    p_d_raw = torch.where(n_d > 0, d_d / n_d.clamp_min(1), zero)
    p_c = (d_c + 1.0) / (n_c + 2.0)
    p_d = (d_d + 1.0) / (n_d + 2.0)
    f += [
        p_c_raw, p_d_raw, p_c, p_d, p_d - p_c, p_d_raw - p_c_raw,
        ((n_c >= 5) & (n_d >= 3)).to(dtype),
        n_c.clamp_max(20) / 20.0, n_d.clamp_max(20) / 20.0,
        torch.log1p(n_c) / _LOG101, torch.log1p(n_d) / _LOG101,
    ]

    has_d_theirs = (sum_theirs > 0).to(dtype)
    has_d_mine = (sum_mine > 0).to(dtype)
    has_c_theirs = (sum_theirs < t).to(dtype)
    f += [
        has_d_theirs, has_d_mine, has_c_theirs,
        sum_theirs / 100.0, sum_mine / 100.0, scalar(t / 99.0),
    ]

    f += [
        _run_length(theirs, t, dtype).clamp_max(10) / 10.0,
        _run_length(mine, t, dtype).clamp_max(10) / 10.0,
        _alternation(theirs, t, 10, dtype),
        _alternation(theirs, t, 20, dtype),
        _alternation(mine, t, 10, dtype),
        _exact_alternation(theirs, t, 10, dtype),
        _switch_rate(theirs, t, dtype),
        _switch_rate(mine, t, dtype),
        ((theirs[:, t - 5:t] == C).all(1).to(dtype)
         if t >= 5 else zero),
        ((theirs[:, t - 5:t] == D).all(1).to(dtype)
         if t >= 5 else zero),
    ]

    if t:
        a, bb = mine[:, t - 1], theirs[:, t - 1]
        f += [
            ((a == C) & (bb == C)).to(dtype),
            ((a == C) & (bb == D)).to(dtype),
            ((a == D) & (bb == C)).to(dtype),
            ((a == D) & (bb == D)).to(dtype),
        ]
    else:
        f += [zero, zero, zero, zero]

    cc, cd, dc, dd = _joint_counts(mine, theirs, t, dtype)
    denom = float(max(1, t))
    f += [cc / denom, cd / denom, dc / denom, dd / denom]

    start10 = max(0, t - 10)
    cc10, cd10, dc10, dd10 = _joint_counts(
        mine[:, start10:t], theirs[:, start10:t], t - start10, dtype)
    d10 = float(max(1, min(10, t)))
    f += [cc10 / d10, cd10 / d10, dc10 / d10, dd10 / d10]

    f += list(_recency(mine, theirs, t, dtype))

    if t:
        last_m = mine[:, t - 1]
        last_t = theirs[:, t - 1]
        last_pay = _payoff_tensor(last_m, last_t, dtype)
        score = _payoff_tensor(mine[:, :t].reshape(-1),
                               theirs[:, :t].reshape(-1), dtype)
        score = score.reshape(b, t).sum(1)
        last3_dd = (
            (mine[:, t - 3:t] == D) & (theirs[:, t - 3:t] == D)
        ).all(1).to(dtype) if t >= 3 else zero
        f += [
            score / 1000.0,
            last_pay / 10.0,
            torch.where(last_m == C, one, -one),
            torch.where(last_t == D, one, -one),
            last3_dd,
        ]
        f += [
            ((last_m == D) & (last_t == C)).to(dtype),
            ((last_m == C) & (last_t == D)).to(dtype),
            ((last_m == D) & (last_t == D)).to(dtype),
        ]
    else:
        f += [zero, zero, zero, zero, zero, zero, zero, zero]

    start = max(1, t - 10)
    if start < t:
        pm = mine[:, start - 1:t - 1]
        ct = theirs[:, start:t]
        c_mask = pm == C
        d_mask = pm == D
        nc10 = c_mask.to(dtype).sum(1)
        nd10 = d_mask.to(dtype).sum(1)
        dc10r = (c_mask & (ct == D)).to(dtype).sum(1)
        dd10r = (d_mask & (ct == D)).to(dtype).sum(1)
    else:
        nc10 = nd10 = dc10r = dd10r = zero
    f += [
        (dc10r + 1.0) / (nc10 + 2.0),
        (dd10r + 1.0) / (nd10 + 2.0),
        (dd10r + 1.0) / (nd10 + 2.0)
        - (dc10r + 1.0) / (nc10 + 2.0),
    ]

    out = torch.cat([
        item.reshape(b, -1) if item.dim() > 1 else item.reshape(b, 1)
        for item in f
    ], dim=1)
    if out.shape[1] != FEATURES85:  # pragma: no cover - internal invariant
        raise AssertionError(f"features85 produced {out.shape[1]} columns")
    return out


def _period_rate_tensor(seq: torch.Tensor, t: int, period: int,
                        window: int, dtype: torch.dtype) -> torch.Tensor:
    b = seq.shape[0]
    if t <= period:
        return torch.zeros(b, device=seq.device, dtype=dtype)
    start = max(0, t - window)
    tail = seq[:, start:t]
    length = tail.shape[1]
    if length <= period:
        return torch.zeros(b, device=seq.device, dtype=dtype)
    return (tail[:, period:] == tail[:, :-period]).to(dtype).mean(1)


def _v2_tail_tensor(mine: torch.Tensor, theirs: torch.Tensor, t: int,
                    dtype: torch.dtype) -> torch.Tensor:
    """Mirror / exploit / forgive rates over the last 10 rounds."""
    b = mine.shape[0]
    zero = torch.zeros(b, device=mine.device, dtype=dtype)
    start = max(1, t - _V2_TAIL)
    n = t - start
    if n <= 0:
        return torch.stack([zero, zero, zero, zero], dim=1)
    prev_m = mine[:, start - 1:t - 1]
    cur_t = theirs[:, start:t]
    prev_t = theirs[:, start - 1:t - 1]
    cur_m = mine[:, start:t]
    return torch.stack([
        (cur_t == prev_m).to(dtype).mean(1),
        (cur_m == prev_t).to(dtype).mean(1),
        ((cur_m == C) & (cur_t == D)).to(dtype).mean(1),
        ((cur_m == D) & (cur_t == C)).to(dtype).mean(1),
    ], dim=1)


def features97(mine: torch.Tensor, theirs: torch.Tensor,
               t: int | None = None,
               dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """85 v1 features plus 12 periodicity / response features."""
    if t is None:
        t = mine.shape[1]
    t = int(t)
    base = features85(mine, theirs, t, dtype)
    tail = [
        _period_rate_tensor(theirs, t, p, _V2_WINDOW, dtype)
        for p in _V2_PERIODS
    ] + [
        _period_rate_tensor(mine, t, p, _V2_WINDOW, dtype)
        for p in _V2_PERIODS
    ]
    out = torch.cat([base] + [item.reshape(-1, 1) for item in tail]
                    + [_v2_tail_tensor(mine, theirs, t, dtype)], dim=1)
    if out.shape[1] != FEATURES97:  # pragma: no cover
        raise AssertionError(f"features97 produced {out.shape[1]} columns")
    return out


class FeatureState85:
    """Incremental 85-feature state for lock-step batch simulation.

    ``append`` consumes one round of actions for every row.  ``features``
    returns the same vector as :func:`ipd.policies.neural.extract_features`
    for the history *before* the current action.
    """

    n_features = FEATURES85

    def __init__(self, batch: int, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float64, rounds: int = ROUNDS):
        self.device = torch.device(device)
        self.dtype = dtype
        self.rounds = int(rounds)
        self.reset(batch)

    def reset(self, batch: int | None = None) -> None:
        if batch is not None:
            self.batch = int(batch)
        b = self.batch
        dev = self.device
        self.t = 0
        self.mine = torch.zeros(b, self.rounds, dtype=torch.int8, device=dev)
        self.theirs = torch.zeros(b, self.rounds, dtype=torch.int8,
                                  device=dev)
        self.sum_mine = torch.zeros(b, dtype=torch.long, device=dev)
        self.sum_theirs = torch.zeros(b, dtype=torch.long, device=dev)
        self.cc = torch.zeros(b, dtype=torch.long, device=dev)
        self.cd = torch.zeros(b, dtype=torch.long, device=dev)
        self.dc = torch.zeros(b, dtype=torch.long, device=dev)
        self.dd = torch.zeros(b, dtype=torch.long, device=dev)
        self.score = torch.zeros(b, dtype=torch.long, device=dev)
        self.n_c = torch.zeros(b, dtype=torch.long, device=dev)
        self.n_d = torch.zeros(b, dtype=torch.long, device=dev)
        self.d_c = torch.zeros(b, dtype=torch.long, device=dev)
        self.d_d = torch.zeros(b, dtype=torch.long, device=dev)
        self.first_mine_d = torch.full((b,), -1, dtype=torch.long, device=dev)
        self.first_theirs_d = torch.full((b,), -1, dtype=torch.long,
                                         device=dev)
        self.last_mine_d = torch.full((b,), -1, dtype=torch.long, device=dev)
        self.last_theirs_d = torch.full((b,), -1, dtype=torch.long,
                                        device=dev)
        self.run_mine = torch.zeros(b, dtype=torch.long, device=dev)
        self.run_theirs = torch.zeros(b, dtype=torch.long, device=dev)
        self.switch_mine = torch.zeros(b, dtype=torch.long, device=dev)
        self.switch_theirs = torch.zeros(b, dtype=torch.long, device=dev)

    def append(self, mine: torch.Tensor, theirs: torch.Tensor) -> None:
        mine = mine.to(device=self.device, dtype=torch.int8)
        theirs = theirs.to(device=self.device, dtype=torch.int8)
        t = self.t
        if t >= self.rounds:
            raise RuntimeError("history buffer is full")
        if t > 0:
            prev_m = self.mine[:, t - 1]
            prev_t = self.theirs[:, t - 1]
            self.switch_mine += (mine != prev_m).long()
            self.switch_theirs += (theirs != prev_t).long()
            self.run_mine = torch.where(mine == prev_m, self.run_mine + 1,
                                        torch.ones_like(self.run_mine))
            self.run_theirs = torch.where(theirs == prev_t,
                                          self.run_theirs + 1,
                                          torch.ones_like(self.run_theirs))
            # Transition mine[t-1] -> theirs[t] (and vice versa for the
            # conditional block; only the blue-perspective statistics are
            # needed by the 85-feature contract).
            m_prev = self.mine[:, t - 1]
            is_c = m_prev == C
            self.n_c += is_c.long()
            self.n_d += (~is_c).long()
            self.d_c += (is_c & (theirs == D)).long()
            self.d_d += ((~is_c) & (theirs == D)).long()
        else:
            self.run_mine.fill_(1)
            self.run_theirs.fill_(1)

        self.mine[:, t] = mine
        self.theirs[:, t] = theirs
        self.sum_mine += mine.long()
        self.sum_theirs += theirs.long()
        self.cc += ((mine == C) & (theirs == C)).long()
        self.cd += ((mine == C) & (theirs == D)).long()
        self.dc += ((mine == D) & (theirs == C)).long()
        self.dd += ((mine == D) & (theirs == D)).long()
        self.score += _payoff_tensor(mine, theirs, torch.long)

        first_missing = self.first_mine_d < 0
        self.first_mine_d = torch.where(first_missing & (mine == D),
                                        torch.full_like(self.first_mine_d, t),
                                        self.first_mine_d)
        first_missing = self.first_theirs_d < 0
        self.first_theirs_d = torch.where(
            first_missing & (theirs == D),
            torch.full_like(self.first_theirs_d, t), self.first_theirs_d)
        self.last_mine_d = torch.where(mine == D,
                                       torch.full_like(self.last_mine_d, t),
                                       self.last_mine_d)
        self.last_theirs_d = torch.where(
            theirs == D, torch.full_like(self.last_theirs_d, t),
            self.last_theirs_d)
        self.t += 1

    def any_defect(self, side: str = "theirs") -> torch.Tensor:
        """Boolean mask: has the requested side ever defected?"""
        if side == "theirs":
            return self.sum_theirs > 0
        return self.sum_mine > 0

    def features(self) -> torch.Tensor:
        t = self.t
        dev = self.device
        dtype = self.dtype
        b = self.batch
        one = torch.ones(b, device=dev, dtype=dtype)
        zero = torch.zeros(b, device=dev, dtype=dtype)

        def scalar(value: float) -> torch.Tensor:
            return torch.full((b,), float(value), device=dev, dtype=dtype)

        f = [
            scalar(t / 99.0), scalar((99 - t) / 99.0),
            scalar(1.0 if t == 0 else 0.0),
            scalar(1.0 if t >= 99 else 0.0),
            scalar(1.0 if t >= 95 else 0.0),
            scalar(1.0 if t >= 90 else 0.0),
        ]
        f.append(_last_moves(self.mine, t, 8, dtype))
        f.append(_last_moves(self.theirs, t, 8, dtype))

        sum_m = self.sum_mine.to(dtype)
        sum_t = self.sum_theirs.to(dtype)
        f += [
            sum_m / max(1, t), sum_t / max(1, t),
            _mean_prefix(self.mine, t, 10, dtype),
            _mean_prefix(self.theirs, t, 3, dtype),
            _mean_prefix(self.theirs, t, 5, dtype),
            _mean_prefix(self.theirs, t, 10, dtype),
            _mean_prefix(self.theirs, t, 20, dtype),
        ]

        n_c = self.n_c.to(dtype)
        n_d = self.n_d.to(dtype)
        d_c = self.d_c.to(dtype)
        d_d = self.d_d.to(dtype)
        p_c_raw = torch.where(n_c > 0, d_c / n_c.clamp_min(1), zero)
        p_d_raw = torch.where(n_d > 0, d_d / n_d.clamp_min(1), zero)
        p_c = (d_c + 1.0) / (n_c + 2.0)
        p_d = (d_d + 1.0) / (n_d + 2.0)
        f += [
            p_c_raw, p_d_raw, p_c, p_d, p_d - p_c, p_d_raw - p_c_raw,
            ((n_c >= 5) & (n_d >= 3)).to(dtype),
            n_c.clamp_max(20) / 20.0, n_d.clamp_max(20) / 20.0,
            torch.log1p(n_c) / _LOG101, torch.log1p(n_d) / _LOG101,
        ]

        f += [
            (sum_t > 0).to(dtype), (sum_m > 0).to(dtype),
            (sum_t < t).to(dtype),
            sum_t / 100.0, sum_m / 100.0, scalar(t / 99.0),
        ]

        f += [
            self.run_theirs.to(dtype).clamp_max(10) / 10.0,
            self.run_mine.to(dtype).clamp_max(10) / 10.0,
            _alternation(self.theirs, t, 10, dtype),
            _alternation(self.theirs, t, 20, dtype),
            _alternation(self.mine, t, 10, dtype),
            _exact_alternation(self.theirs, t, 10, dtype),
            (self.switch_theirs.to(dtype) / max(1, t - 1)
             if t >= 2 else zero),
            (self.switch_mine.to(dtype) / max(1, t - 1)
             if t >= 2 else zero),
            ((self.theirs[:, t - 5:t] == C).all(1).to(dtype) if t >= 5
             else zero),
            ((self.theirs[:, t - 5:t] == D).all(1).to(dtype) if t >= 5
             else zero),
        ]

        if t:
            a = self.mine[:, t - 1]
            bb = self.theirs[:, t - 1]
            f += [
                ((a == C) & (bb == C)).to(dtype),
                ((a == C) & (bb == D)).to(dtype),
                ((a == D) & (bb == C)).to(dtype),
                ((a == D) & (bb == D)).to(dtype),
            ]
        else:
            f += [zero, zero, zero, zero]

        denom = float(max(1, t))
        f += [
            self.cc.to(dtype) / denom, self.cd.to(dtype) / denom,
            self.dc.to(dtype) / denom, self.dd.to(dtype) / denom,
        ]

        start10 = max(0, t - 10)
        cc10, cd10, dc10, dd10 = _joint_counts(
            self.mine[:, start10:t], self.theirs[:, start10:t],
            t - start10, dtype)
        d10 = float(max(1, min(10, t)))
        f += [cc10 / d10, cd10 / d10, dc10 / d10, dd10 / d10]

        # Recency / first-defection timing from incremental positions.
        ft = self.first_theirs_d.to(dtype)
        fm = self.first_mine_d.to(dtype)
        lt = self.last_theirs_d.to(dtype)
        lm = self.last_mine_d.to(dtype)
        has_t = self.first_theirs_d >= 0
        has_m = self.first_mine_d >= 0
        last_opp = torch.where(has_t, (t - lt) / 100.0,
                               torch.ones((), device=dev, dtype=dtype))
        last_my = torch.where(has_m, (t - lm) / 100.0,
                              torch.ones((), device=dev, dtype=dtype))
        first_opp = torch.where(has_t, (ft + 1.0) / 100.0,
                                torch.ones((), device=dev, dtype=dtype))
        first_my = torch.where(has_m, (fm + 1.0) / 100.0,
                               torch.ones((), device=dev, dtype=dtype))
        since_opp = torch.where(has_t, (t - ft) / 100.0,
                                torch.zeros((), device=dev, dtype=dtype))
        gap = torch.where(has_m & has_t & (self.first_mine_d
                                           >= self.first_theirs_d),
                          (fm - ft) / 100.0,
                          torch.zeros((), device=dev, dtype=dtype))
        f += [last_opp, last_my, first_opp, first_my, since_opp, gap]

        if t:
            last_m = self.mine[:, t - 1]
            last_t = self.theirs[:, t - 1]
            last_pay = _payoff_tensor(last_m, last_t, dtype)
            last3_dd = (
                (self.mine[:, t - 3:t] == D)
                & (self.theirs[:, t - 3:t] == D)
            ).all(1).to(dtype) if t >= 3 else zero
            f += [
                self.score.to(dtype) / 1000.0,
                last_pay / 10.0,
                torch.where(last_m == C, one, -one),
                torch.where(last_t == D, one, -one),
                last3_dd,
                ((last_m == D) & (last_t == C)).to(dtype),
                ((last_m == C) & (last_t == D)).to(dtype),
                ((last_m == D) & (last_t == D)).to(dtype),
            ]
        else:
            f += [zero, zero, zero, zero, zero, zero, zero, zero]

        start = max(1, t - 10)
        if start < t:
            pm = self.mine[:, start - 1:t - 1]
            ct = self.theirs[:, start:t]
            c_mask = pm == C
            d_mask = pm == D
            nc10 = c_mask.to(dtype).sum(1)
            nd10 = d_mask.to(dtype).sum(1)
            dc10r = (c_mask & (ct == D)).to(dtype).sum(1)
            dd10r = (d_mask & (ct == D)).to(dtype).sum(1)
        else:
            nc10 = nd10 = dc10r = dd10r = zero
        f += [
            (dc10r + 1.0) / (nc10 + 2.0),
            (dd10r + 1.0) / (nd10 + 2.0),
            (dd10r + 1.0) / (nd10 + 2.0)
            - (dc10r + 1.0) / (nc10 + 2.0),
        ]

        out = torch.cat([
            item.reshape(b, -1) if item.dim() > 1 else item.reshape(b, 1)
            for item in f
        ], dim=1)
        if out.shape[1] != FEATURES85:  # pragma: no cover
            raise AssertionError(
                f"FeatureState85 produced {out.shape[1]} columns")
        return out


class FeatureState97:
    """Incremental 97-feature state (v1 state + 12 v2 features)."""

    n_features = FEATURES97

    def __init__(self, batch: int, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float64, rounds: int = ROUNDS):
        self.state = FeatureState85(batch, device=device, dtype=dtype,
                                    rounds=rounds)
        self.device = self.state.device
        self.dtype = self.state.dtype
        self.rounds = self.state.rounds
        self.batch = self.state.batch

    def reset(self, batch: int | None = None) -> None:
        self.state.reset(batch)
        self.batch = self.state.batch

    def append(self, mine: torch.Tensor, theirs: torch.Tensor) -> None:
        self.state.append(mine, theirs)

    def any_defect(self, side: str = "theirs") -> torch.Tensor:
        return self.state.any_defect(side)

    def features(self) -> torch.Tensor:
        t = self.state.t
        base = self.state.features()
        tail = [
            _period_rate_tensor(self.state.theirs, t, p, _V2_WINDOW,
                                self.dtype)
            for p in _V2_PERIODS
        ] + [
            _period_rate_tensor(self.state.mine, t, p, _V2_WINDOW,
                                self.dtype)
            for p in _V2_PERIODS
        ]
        out = torch.cat([base] + [item.reshape(-1, 1) for item in tail]
                        + [_v2_tail_tensor(self.state.mine,
                                           self.state.theirs, t,
                                           self.dtype)], dim=1)
        if out.shape[1] != FEATURES97:  # pragma: no cover
            raise AssertionError(
                f"FeatureState97 produced {out.shape[1]} columns")
        return out


def features21(mine: torch.Tensor, theirs: torch.Tensor,
               t: int | None = None, version: str = "selfplay-21-v1",
               dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Batched exact port of ``ipd.policies.selfplay_gru_obs.features``."""
    if mine.shape != theirs.shape or mine.dim() != 2:
        raise ValueError("histories must be equal [B, R] tensors")
    if t is None:
        t = mine.shape[1]
    t = int(t)
    if t < 0 or t > mine.shape[1]:
        raise ValueError("t is outside the history buffer")
    if version not in ("selfplay-21-v1", "selfplay-21-v2"):
        raise ValueError(f"unsupported observation version: {version}")
    b = mine.shape[0]
    dev = mine.device
    out = torch.zeros(b, FEATURES21, device=dev, dtype=dtype)
    out[:, 0] = t / 100.0
    out[:, 1] = (100 - t) / 100.0
    for lag in range(1, 5):
        j = 6 + 2 * (lag - 1)
        out[:, j] = -1.0
        out[:, j + 1] = -1.0
    if t:
        m = mine[:, :t]
        th = theirs[:, :t]
        lm, lt = m[:, -1], th[:, -1]
        out[:, 2] = ((lm == C) & (lt == C)).to(dtype)
        out[:, 3] = ((lm == C) & (lt == D)).to(dtype)
        out[:, 4] = ((lm == D) & (lt == C)).to(dtype)
        out[:, 5] = ((lm == D) & (lt == D)).to(dtype)
        for lag in range(1, 5):
            j = 6 + 2 * (lag - 1)
            if t >= lag:
                out[:, j] = m[:, -lag].to(dtype)
                out[:, j + 1] = th[:, -lag].to(dtype)
        denom = float(max(1, t))
        opp_def = th.to(dtype).sum(1) / denom
        own_def = m.to(dtype).sum(1) / denom
        start10 = max(0, t - 10)
        out[:, 16] = th[:, start10:t].to(dtype).mean(1)
        if version == "selfplay-21-v2":
            n_c, n_d, d_c, d_d = _conditional_counts(m, th, t, dtype)
            p_c = torch.where(n_c > 0, d_c / n_c.clamp_min(1),
                              torch.zeros((), device=dev, dtype=dtype))
            p_d = torch.where(n_d > 0, d_d / n_d.clamp_min(1), p_c)
            out[:, 17] = p_d - p_c
        else:
            out[:, 17] = own_def - opp_def
        out[:, 14] = opp_def
        out[:, 15] = own_def
        out[:, 18] = ((m == C) & (th == C)).to(dtype).sum(1) / denom
        out[:, 19] = ((m == D) & (th == D)).to(dtype).sum(1) / denom
    out[:, 20] = 1.0 if t >= 90 else 0.0
    return out


class FeatureState21:
    """Incremental ``selfplay-21-v1/v2`` observation state."""

    n_features = FEATURES21

    def __init__(self, batch: int, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float64, rounds: int = ROUNDS,
                 version: str = "selfplay-21-v1"):
        if version not in ("selfplay-21-v1", "selfplay-21-v2"):
            raise ValueError(f"unsupported observation version: {version}")
        self.device = torch.device(device)
        self.dtype = dtype
        self.rounds = int(rounds)
        self.version = version
        self.reset(batch)

    def reset(self, batch: int | None = None) -> None:
        if batch is not None:
            self.batch = int(batch)
        b = self.batch
        dev = self.device
        self.t = 0
        self.mine = torch.zeros(b, self.rounds, dtype=torch.int8, device=dev)
        self.theirs = torch.zeros(b, self.rounds, dtype=torch.int8,
                                  device=dev)
        self.sum_mine = torch.zeros(b, dtype=torch.long, device=dev)
        self.sum_theirs = torch.zeros(b, dtype=torch.long, device=dev)
        self.cc = torch.zeros(b, dtype=torch.long, device=dev)
        self.dd = torch.zeros(b, dtype=torch.long, device=dev)
        self.n_c = torch.zeros(b, dtype=torch.long, device=dev)
        self.n_d = torch.zeros(b, dtype=torch.long, device=dev)
        self.d_c = torch.zeros(b, dtype=torch.long, device=dev)
        self.d_d = torch.zeros(b, dtype=torch.long, device=dev)

    def append(self, mine: torch.Tensor, theirs: torch.Tensor) -> None:
        mine = mine.to(device=self.device, dtype=torch.int8)
        theirs = theirs.to(device=self.device, dtype=torch.int8)
        t = self.t
        if t >= self.rounds:
            raise RuntimeError("history buffer is full")
        if t > 0:
            m_prev = self.mine[:, t - 1]
            is_c = m_prev == C
            self.n_c += is_c.long()
            self.n_d += (~is_c).long()
            self.d_c += (is_c & (theirs == D)).long()
            self.d_d += ((~is_c) & (theirs == D)).long()
        self.mine[:, t] = mine
        self.theirs[:, t] = theirs
        self.sum_mine += mine.long()
        self.sum_theirs += theirs.long()
        self.cc += ((mine == C) & (theirs == C)).long()
        self.dd += ((mine == D) & (theirs == D)).long()
        self.t += 1

    def any_defect(self, side: str = "theirs") -> torch.Tensor:
        if side == "theirs":
            return self.sum_theirs > 0
        return self.sum_mine > 0

    def features(self) -> torch.Tensor:
        t = self.t
        dev = self.device
        dtype = self.dtype
        b = self.batch
        out = torch.zeros(b, FEATURES21, device=dev, dtype=dtype)
        out[:, 0] = t / 100.0
        out[:, 1] = (100 - t) / 100.0
        for lag in range(1, 5):
            j = 6 + 2 * (lag - 1)
            out[:, j] = -1.0
            out[:, j + 1] = -1.0
        if t:
            m = self.mine[:, :t]
            th = self.theirs[:, :t]
            lm, lt = m[:, -1], th[:, -1]
            out[:, 2] = ((lm == C) & (lt == C)).to(dtype)
            out[:, 3] = ((lm == C) & (lt == D)).to(dtype)
            out[:, 4] = ((lm == D) & (lt == C)).to(dtype)
            out[:, 5] = ((lm == D) & (lt == D)).to(dtype)
            for lag in range(1, 5):
                j = 6 + 2 * (lag - 1)
                if t >= lag:
                    out[:, j] = m[:, -lag].to(dtype)
                    out[:, j + 1] = th[:, -lag].to(dtype)
            denom = float(max(1, t))
            opp_def = self.sum_theirs.to(dtype) / denom
            own_def = self.sum_mine.to(dtype) / denom
            start10 = max(0, t - 10)
            out[:, 16] = th[:, start10:t].to(dtype).mean(1)
            if self.version == "selfplay-21-v2":
                n_c = self.n_c.to(dtype)
                n_d = self.n_d.to(dtype)
                d_c = self.d_c.to(dtype)
                d_d = self.d_d.to(dtype)
                zero = torch.zeros((), device=dev, dtype=dtype)
                p_c = torch.where(n_c > 0, d_c / n_c.clamp_min(1), zero)
                p_d = torch.where(n_d > 0, d_d / n_d.clamp_min(1), p_c)
                out[:, 17] = p_d - p_c
            else:
                out[:, 17] = own_def - opp_def
            out[:, 14] = opp_def
            out[:, 15] = own_def
            out[:, 18] = self.cc.to(dtype) / denom
            out[:, 19] = self.dd.to(dtype) / denom
        out[:, 20] = 1.0 if t >= 90 else 0.0
        return out


__all__ = [
    "C", "D", "FEATURES21", "FEATURES85", "FEATURES97",
    "FeatureState21", "FeatureState85", "FeatureState97",
    "features21", "features85", "features97",
]
