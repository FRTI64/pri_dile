"""History features shared by the P2 search skeleton and rule policies."""
from __future__ import annotations

from .simulator import C, D

# Feature names/order.  All numeric features are normalised to [0, 1].
FEATURES = [
    "day", "remaining", "my_last", "opp_last", "ever_opp_d",
    "ever_my_d", "opp_d_rate", "recent5_d", "recent10_d",
    "pC", "pD", "gap", "recent_pC", "recent_pD", "recent_gap",
    "opp_c_run", "opp_d_run", "my_c_run", "mutual_dd",
    "alternating8", "first_d_day", "time_since_d",
]
F = {name: i for i, name in enumerate(FEATURES)}


def feature_vector(day: int, mine: list[int], theirs: list[int]) -> list[float]:
    """Compact normalised history features (1-based ``day``)."""
    n = len(theirs)
    day0 = day - 1
    my_last = mine[-1] if mine else C
    opp_last = theirs[-1] if theirs else C
    ever_opp_d = int(D in theirs)
    ever_my_d = int(D in mine)
    d_rate = sum(theirs) / n if n else 0.0
    recent5 = theirs[-5:]
    recent10 = theirs[-10:]
    r5 = sum(recent5) / len(recent5) if recent5 else 0.0
    r10 = sum(recent10) / len(recent10) if recent10 else 0.0

    n_c = n_d = d_c = d_d = 0
    for i in range(1, n):
        if mine[i - 1] == C:
            n_c += 1
            d_c += theirs[i] == D
        else:
            n_d += 1
            d_d += theirs[i] == D
    p_c = d_c / n_c if n_c else 0.5
    p_d = d_d / n_d if n_d else 0.5
    gap = (p_d - p_c + 1.0) / 2.0

    rc = rd = rdc = rdd = 0
    start = max(1, n - 12)
    for i in range(start, n):
        if mine[i - 1] == C:
            rc += 1
            rdc += theirs[i] == D
        else:
            rd += 1
            rdd += theirs[i] == D
    rp_c = rdc / rc if rc else 0.5
    rp_d = rdd / rd if rd else 0.5
    rgap = (rp_d - rp_c + 1.0) / 2.0

    oc = od = 0
    for x in reversed(theirs):
        if x == C:
            if od:
                break
            oc += 1
        else:
            if oc:
                break
            od += 1
    mc = 0
    for x in reversed(mine):
        if x == C:
            mc += 1
        else:
            break
    mutual_dd = float(n >= 2 and mine[-1] == D and theirs[-1] == D)
    alt = float(n >= 8 and all(theirs[i] != theirs[i - 1]
                               for i in range(n - 7, n)))
    first_d = 100
    for i, x in enumerate(theirs):
        if x == D:
            first_d = i + 1
            break
    since = (day0 - first_d) if first_d < 100 and day0 >= first_d else 0

    return [
        day0 / 100.0, (100 - day) / 100.0, float(my_last), float(opp_last),
        float(ever_opp_d), float(ever_my_d), d_rate, r5, r10,
        p_c, p_d, gap, rp_c, rp_d, rgap,
        min(oc, 100) / 100.0, min(od, 100) / 100.0, min(mc, 100) / 100.0,
        mutual_dd, alt, first_d / 100.0, since / 100.0,
    ]


def feature_value(day: int, mine: list[int], theirs: list[int], name: str) -> float:
    return feature_vector(day, mine, theirs)[F[name]]
