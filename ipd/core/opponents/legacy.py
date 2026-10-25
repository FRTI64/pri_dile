"""Legacy fixed-weight field helpers kept for the historical CLI."""
from __future__ import annotations

import argparse
import random
from typing import Callable

from ..primitives import C, D, Strategy, payoff
from .basic import (
    AllC, AllD, Bully, FirmButFair, Grim, Joss, Mistrust, Pavlov, Prober,
    RandomMoves, SuspiciousTFT, TFT, TFT2, TFTLastD, Tester,
    WinStayLoseShift2,
)
from .defectors import (
    Alternator, EarlyDefector, Gradual, LateDefector, RandomP,
)
from .reciprocal import (
    AdaptiveProbe, ContriteTFT, DelayedTFT, ForgiveTFT, GenerousTFT,
    GrimForgive, LastKDefect, Majority, NoisyTFT, ProbeBot, RandomThenTFT,
    RemorsefulProber, Responsive, Responsive2, Responsive3, TFT2LastD,
    TFTForgiveOnce, TFTNoisyForgive, TFTProbe, TwoStrike,
)

# Factories: (weight, name, factory)
FIELD_SPECS: list[tuple[float, str, Callable]] = [
    (7.0, "AllC", lambda r: AllC(r)),
    (7.0, "AllD", lambda r: AllD(r)),
    (18.0, "TFT", lambda r: TFT(r)),
    (7.0, "TFT-lastD", lambda r: TFTLastD(r)),
    (4.0, "sTFT", lambda r: SuspiciousTFT(r)),
    (7.0, "Grim", lambda r: Grim(r)),
    (6.0, "GTFT.05", lambda r: GenerousTFT(0.05, r)),
    (4.0, "GTFT.15", lambda r: GenerousTFT(0.15, r)),
    (5.0, "TFT2", lambda r: TFT2(r)),
    (7.0, "Pavlov", lambda r: Pavlov(r)),
    (4.0, "Random.5", lambda r: RandomMoves(0.5, r)),
    (3.0, "Random.2", lambda r: RandomMoves(0.2, r)),
    (3.0, "Joss.05", lambda r: Joss(0.05, r)),
    (2.0, "Joss.15", lambda r: Joss(0.15, r)),
    (2.0, "Prober", lambda r: Prober(r)),
    (2.0, "Tester", lambda r: Tester(r)),
    (2.0, "FirmButFair", lambda r: FirmButFair(r)),
    (3.0, "Mistrust", lambda r: Mistrust(r)),
    (2.0, "Bully", lambda r: Bully(r)),
    (2.0, "WSLS2", lambda r: WinStayLoseShift2(r)),
    (2.0, "ContriteTFT", lambda r: ContriteTFT(r)),
    (1.0, "Alternator", lambda r: Alternator(r)),
    (2.0, "TFT-last2D", lambda r: LastKDefect(2, r)),
    (2.0, "NoisyTFT", lambda r: NoisyTFT(0.05, r)),
    (2.0, "Rand3->TFT", lambda r: RandomThenTFT(3, 0.5, r)),
    (2.0, "Majority3", lambda r: Majority(3, r)),
    (2.0, "TFTForgiveOnce", lambda r: TFTForgiveOnce(r)),
    (1.0, "GrimForgive", lambda r: GrimForgive(0.05, r)),
    (2.0, "LateD10", lambda r: LateDefector(10, r)),
    (2.0, "LateD25", lambda r: LateDefector(25, r)),
    (1.0, "EarlyD10", lambda r: EarlyDefector(10, r)),
    (1.0, "Gradual", lambda r: Gradual(r)),
    (2.0, "TFT-delay2", lambda r: DelayedTFT(2, r)),
    (1.0, "RemorsefulProber", lambda r: RemorsefulProber(r)),
    (1.0, "Grudger2", lambda r: Grudger2(r)),
    (1.0, "Random.1", lambda r: RandomP(0.1, r)),
    (1.0, "Random.8", lambda r: RandomP(0.8, r)),
    (1.0, "GTFT.33", lambda r: TFTNoisyForgive(1/3, r)),
]

WEIGHT_TOTAL = sum(w for w, _, _ in FIELD_SPECS)


def make_field(n: int, rng: random.Random) -> list[Strategy]:
    specs = FIELD_SPECS
    weights = [w for w, _, _ in specs]
    out = []
    for _ in range(n):
        _, _, f = rng.choices(specs, weights=weights, k=1)[0]
        out.append(f(rng))
    return out


def score_against(factory: Callable, field: list[Strategy],
                  rounds=100) -> tuple[int, dict]:
    me = factory(random.Random(1234567))
    by_name: dict[str, list[int]] = {}
    total = 0
    for opp in field:
        me.reset()
        opp.reset()
        mh: list[int] = []
        oh: list[int] = []
        s = 0
        for t in range(rounds):
            x = me.act(t, mh, oh)
            y = opp.act(t, oh, mh)
            mh.append(x)
            oh.append(y)
            s += payoff(x, y)
        total += s
        by_name.setdefault(opp.name, []).append(s)
    return total, {k: sum(v) / len(v) for k, v in by_name.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=299)
    ap.add_argument("--fields", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=100)
    args = ap.parse_args()

    candidates = [
        ("TFT", lambda r: TFT(r)),
        ("TFT-lastD", lambda r: TFTLastD(r)),
        ("TFT2", lambda r: TFT2(r)),
        ("TFT2-lastD", lambda r: TFT2LastD(r)),
        ("TwoStrike2", lambda r: TwoStrike(2, True, r)),
        ("TwoStrike3", lambda r: TwoStrike(3, True, r)),
        ("ForgiveTFT1", lambda r: ForgiveTFT(True, 1, r)),
        ("ForgiveTFT2", lambda r: ForgiveTFT(True, 2, r)),
        ("GTFT.05-lastD", lambda r: GenerousTFT(0.05, True, r)),
        ("GTFT.1-lastD", lambda r: GenerousTFT(0.1, True, r)),
        ("GTFT.2-lastD", lambda r: GenerousTFT(0.2, True, r)),
        ("Responsive", lambda r: Responsive(g=0.4, rng=r)),
        ("Resp3", lambda r: Responsive3(rng=r)),
        ("ProbeBot", lambda r: ProbeBot(rng=r)),
        ("Resp2-u1", lambda r: Responsive2(1.0, 0.10, rng=r)),
        ("Resp2-u.85", lambda r: Responsive2(0.85, 0.10, rng=r)),
        ("Resp2-u.7", lambda r: Responsive2(0.7, 0.10, rng=r)),
        ("ContriteTFT", lambda r: ContriteTFT(r)),
        ("Probe-first", lambda r: TFTProbe(0, 2, True, r)),
        ("Probe-10", lambda r: TFTProbe(10, 2, True, r)),
        ("Probe-20", lambda r: TFTProbe(20, 2, True, r)),
        ("Probe-30", lambda r: TFTProbe(30, 2, True, r)),
        ("Probe-40", lambda r: TFTProbe(40, 2, True, r)),
        ("Probe-50", lambda r: TFTProbe(50, 2, True, r)),
        ("Probe-60", lambda r: TFTProbe(60, 2, True, r)),
        ("Probe-70", lambda r: TFTProbe(70, 2, True, r)),
        ("Probe-80", lambda r: TFTProbe(80, 2, True, r)),
        ("Probe-90", lambda r: TFTProbe(90, 2, True, r)),
        ("Probe-95", lambda r: TFTProbe(95, 2, True, r)),
        ("Adaptive-60", lambda r: AdaptiveProbe(60, 2, True, r)),
    ]

    master = random.Random(args.seed)
    totals = {name: [] for name, _ in candidates}
    per_type: dict[str, dict[str, list[int]]] = {name: {} for name, _ in candidates}
    for fi in range(args.fields):
        field = make_field(args.n, random.Random(master.random()))
        for name, fac in candidates:
            total, by = score_against(fac, field, args.rounds)
            totals[name].append(total)
            for k, v in by.items():
                per_type[name].setdefault(k, []).append(v)
        print(f"field {fi+1}/{args.fields} done", flush=True)

    print("\n=== mean totals over fields ===")
    for name, vals in sorted(totals.items(), key=lambda kv: sum(kv[1])/len(kv[1]), reverse=True):
        mean = sum(vals) / len(vals)
        lo, hi = min(vals), max(vals)
        print(f"{name:18} mean={mean:8.1f}  min={lo:5d} max={hi:5d}  avg/opp={mean/args.n:6.2f}")

    print("\n=== per-opponent-type averages (best few) ===")
    best_names = sorted(totals, key=lambda n: sum(totals[n]) / len(totals[n]), reverse=True)[:5]
    types = sorted(per_type[best_names[0]])
    print(f"{'type':18}" + "".join(f"{n:>14}" for n in best_names))
    for typ in types:
        row = f"{typ:18}"
        for n in best_names:
            vals = per_type[n].get(typ, [])
            row += f"{(sum(vals)/len(vals) if vals else float('nan')):14.1f}"
        print(row)


if __name__ == "__main__":
    main()


__all__ = ["FIELD_SPECS", "WEIGHT_TOTAL", "make_field", "score_against", "main"]
