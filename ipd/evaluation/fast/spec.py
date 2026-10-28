"""Metadata and parameter encoding for the fast rule bank.

The fast path is a native C++ interpreter for the *rule* opponents used by
Protocol v4 and by the self-play trainer.  Learned policies, Axelrod
strategies, and the two deliberately irregular rule families
(``RandomFSM`` / ``VotingPortfolio``) stay on their original Python objects.

A :class:`FastRuleSpec` is attached to a freshly built opponent as
``bot._fast_spec``.  It contains only plain Python values, so it can cross a
process-pool boundary.  :func:`encode_spec` converts it into the flat arrays
that the C ABI in ``rules.cpp`` accepts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ipd.core.primitives import C, D

# Keep this order in sync with ``enum RuleType`` in rules.cpp.  The native
# test asserts that ``native.type_count()`` equals ``len(TYPE_IDS)``.
TYPE_NAMES: tuple[str, ...] = (
    "AllC",
    "AllD",
    "TFT",
    "Grim",
    "RandomMoves",
    "TFT2",
    "TFTLastD",
    "Pavlov",
    "ContriteTFT",
    "WSLS2",
    "TFTForgiveOnce",
    "TFT2LastD",
    "TwoTitsForTat",
    "SuspiciousTFT",
    "Majority",
    "GenerousTFT",
    "TwoStrike",
    "ForgiveTFT",
    "TFTNoisyForgive",
    "GenerousZD",
    "Grudger2",
    "FirmButFair",
    "Mistrust",
    "Gradual",
    "GrudgerK",
    "GrimForgive",
    "ProbabilisticGrim",
    "ResetGrim",
    "SpitefulGrim",
    "DelayedPunisher",
    "Bully",
    "Prober",
    "Tester",
    "RemorsefulProber",
    "EarlyDefector",
    "RandomThenTFT",
    "LateDefector",
    "Periodic",
    "MemoryOne",
    "Reactive",
    "ExtortZD",
    "HarvestFactor",
    "DynamicWindow",
    "Kamikaze",
    "ProbePhase",
    "WindowTFT",
    "NoisyTFT",
    "Joss",
    "Responsive3",
    "Responsive",
    "Responsive2",
    "OmegaTFTLite",
    "TFTProbe",
    "AdaptiveProbe",
    "PhaseSwitcher",
    "AdaptiveMix",
    "ProbeExploit",
    "DelayedExtort",
    "MemoryTwo",
    "MetaMix",
    "NoiseAdaptive",
    "ReputationGrim",
    "ContritionCycle",
    "EndgameHunter",
    "PatternBreaker",
    "ConditionalExtort",
    "LookaheadOpponent",
    "MirrorAdaptive",
    "DelayedGrim",
    "AllCWithLastD",
    "VotingPortfolio",
)
TYPE_IDS: dict[str, int] = {name: index for index, name in enumerate(TYPE_NAMES)}

# Family-level aliases.  They are accepted by the encoder but are not part of
# the raw type list above unless they map to one of its entries.
ALIASES: dict[str, str] = {
    "RandomP": "RandomMoves",
    "GTFT": "GenerousTFT",
    "TTFT": "TwoTitsForTat",
    "LastKDefect": "TFTLastD",  # only when k == 1; see _normalise
    "Grudge2": "Grudger2",
    "TFTProbeImmediate": "TFTProbe",
    "GTFTNoisyForgive": "TFTNoisyForgive",
    "WeakTFT": "NoisyTFT",
    "WinStayLoseShift2": "WSLS2",
}

# String modes used by the complex families.  Keep in sync with the codes in
# rules.cpp.
FIRST_CODES = {"tft": 0, "coop": 1, "probe": 2}
SECOND_CODES = {"extort": 0, "harvest": 1, "grim": 2, "alld": 3, "tft": 4}
EXPLOIT_CODES = {"extort": 0, "harvest": 1, "alld": 2, "tft": 3}
META_CODES = {"tft": 0, "grim": 1, "extort": 2, "harvest": 3,
              "coop": 4, "random": 5, "alld": 6}
PORTFOLIO_CODES = {"tft": 0, "grim": 1, "extort": 2, "harvest": 3,
                   "adaptive": 4, "probe": 5, "random": 6, "coop": 7,
                   "alld": 8}


@dataclass(frozen=True)
class FastRuleSpec:
    """A serialisable description of one rule opponent."""

    type_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    name: str = ""
    family: str = ""

    def with_details(self, **extra: Any) -> "FastRuleSpec":
        if not extra:
            return self
        merged = dict(self.params)
        merged.update(extra)
        return FastRuleSpec(self.type_name, merged, self.name, self.family)


UNSUPPORTED_TYPES = frozenset({
    "PolicyFinal", "PolicyTriple", "PolicyRL", "PolicyNeural", "PolicyGRU",
    "PolicySelfPlay", "Axelrod", "AxelrodMix", "RandomFSM", "ProbeBot",
})


def attach_fast_spec(bot: Any, type_name: str, family: str = "",
                     name: str | None = None,
                     **params: Any) -> Any:
    """Attach a :class:`FastRuleSpec` to ``bot`` and return it."""
    original = type_name
    if type_name == "TFTProbeImmediate":
        type_name = "TFTProbe"
        if "probe" in params and "probe_round" not in params:
            params["probe_round"] = params.pop("probe")
    type_name = ALIASES.get(type_name, type_name)
    if type_name in UNSUPPORTED_TYPES or type_name not in TYPE_IDS:
        return bot
    # Fill in defaults that the catalog builder chooses positionally (for
    # example GenerousTFT(..., last_d=True)) or randomly at construction
    # (MetaMix.mode).  Explicit catalog parameters win on conflict.
    try:
        inferred = infer_fast_spec(bot)
    except Exception:
        inferred = None
    merged = dict(inferred.params) if inferred is not None else {}
    merged.update(params)
    try:
        bot._fast_spec = FastRuleSpec(
            type_name=type_name, params=merged,
            family=family or str(getattr(bot, "family", "")),
            name=name or str(getattr(bot, "name", original)))
    except Exception:
        # Some policy wrappers use slots; metadata is optional.
        pass
    return bot


def get_fast_spec(bot: Any) -> FastRuleSpec | None:
    """Return attached metadata, or ``None`` when the object is not fast."""
    spec = getattr(bot, "_fast_spec", None)
    if isinstance(spec, FastRuleSpec):
        return spec if spec.type_name in TYPE_IDS else None
    return None


def infer_fast_spec(bot: Any) -> FastRuleSpec | None:
    """Best-effort fallback for objects created without explicit metadata.

    This is used by tests and by callers that build opponents directly from
    the core bank.  Protocol/training builders always attach an explicit spec,
    so this function is not on the production hot path.
    """
    explicit = get_fast_spec(bot)
    if explicit is not None:
        return explicit
    cls = type(bot).__name__
    params: dict[str, Any] = {}
    type_name = ALIASES.get(cls, cls)

    if cls == "LastKDefect":
        k = int(getattr(bot, "k", 1))
        if k != 1:
            return None
        type_name = "TFTLastD"
        params = {"k": k}
    elif cls in ("RandomMoves", "RandomP"):
        params = {"p": float(getattr(bot, "p", 0.5))}
    elif cls == "GRUPolicy" or cls.startswith("Policy"):
        return None
    elif cls in ("NeuralPolicy", "RLPolicy", "SelfPlayGRUPolicy"):
        return None
    elif cls in ("FinalTunedPolicy", "TripleMajorityPolicy",
                 "PSROMetaPolicy", "AdvMLPPolicy", "AdvGRUPolicy",
                 "AdvRLPolicy"):
        return None
    elif cls == "GenerousTFT":
        params = {"g": float(getattr(bot, "g", 0.1)),
                  "last_d": bool(getattr(bot, "last_d", False))}
    elif cls == "TwoStrike":
        params = {"strikes": int(getattr(bot, "strikes", 2)),
                  "last_d": bool(getattr(bot, "last_d", True))}
    elif cls == "ForgiveTFT":
        params = {"missed_limit": int(getattr(bot, "missed_limit", 1)),
                  "last_d": bool(getattr(bot, "last_d", True))}
    elif cls == "TFTNoisyForgive":
        params = {"g": float(getattr(bot, "g", 1 / 3))}
    elif cls == "ExtortZD":
        params = {"chi": float(getattr(bot, "chi", 2.0))}
    elif cls == "GenerousZD":
        params = {"chi": float(getattr(bot, "kappa", 2.0))}
    elif cls == "GrudgerK":
        params = {"k": int(getattr(bot, "k", 3))}
    elif cls == "GrimForgive":
        params = {"g": float(getattr(bot, "g", 0.05))}
    elif cls in ("ProbabilisticGrim", "SpitefulGrim"):
        params = {"patience": int(getattr(bot, "patience", 2))}
        if cls == "ProbabilisticGrim":
            params["p_punish"] = float(getattr(bot, "p_punish", 0.85))
    elif cls == "ResetGrim":
        params = {"patience": int(getattr(bot, "patience", 2)),
                  "cooldown": int(getattr(bot, "cooldown", 12))}
    elif cls == "DelayedPunisher":
        params = {"patience": int(getattr(bot, "patience", 3)),
                  "punish_rounds": int(getattr(bot, "punish_rounds", 10))}
    elif cls == "EarlyDefector":
        params = {"k": int(getattr(bot, "k", 20))}
    elif cls == "LateDefector":
        params = {"k": int(getattr(bot, "k", 20))}
    elif cls == "RandomThenTFT":
        params = {"k": int(getattr(bot, "k", 3)),
                  "p": float(getattr(bot, "p", 0.5))}
    elif cls == "Periodic":
        params = {"pattern": tuple(getattr(bot, "pattern", (C, D, C))),
                  "start": int(getattr(bot, "start", 0))}
    elif cls == "MemoryOne":
        params = {"p": tuple(getattr(bot, "p", (1, 1, 1, 0))),
                  "first": int(getattr(bot, "first", C))}
    elif cls == "Reactive":
        params = {"first": int(getattr(bot, "first", C)),
                  "p_after_c": float(getattr(bot, "p_after_c", 0.8)),
                  "p_after_d": float(getattr(bot, "p_after_d", 0.1))}
    elif cls == "HarvestFactor":
        params = {"start": int(getattr(bot, "start", 50)),
                  "repairable": bool(getattr(bot, "repairable", True)),
                  "p_defect": float(getattr(bot, "p_defect", 1.0)),
                  "last_d": bool(getattr(bot, "last_d", True))}
    elif cls == "DynamicWindow":
        params = {"window": int(getattr(bot, "window", 5)),
                  "p_defect": float(getattr(bot, "p_defect", 0.2)),
                  "retaliate": float(getattr(bot, "retaliate", 0.85)),
                  "forgiveness": float(getattr(bot, "forgiveness", 0.25)),
                  "last_d": bool(getattr(bot, "last_d", True))}
    elif cls == "Kamikaze":
        params = {"period": int(getattr(bot, "period", 15)),
                  "p_steal": float(getattr(bot, "p_steal", 1.0))}
    elif cls == "ProbePhase":
        params = {"probe": int(getattr(bot, "probe", 3)),
                  "p_probe": float(getattr(bot, "p_probe", 0.7)),
                  "phase": str(getattr(bot, "phase", "harvest")),
                  "harvest_start": int(getattr(bot, "harvest_start", 50))}
    elif cls == "WindowTFT":
        params = {"window": int(getattr(bot, "window", 5)),
                  "retaliate": float(getattr(bot, "retaliate", 0.8)),
                  "forgive": float(getattr(bot, "forgive", 0.25))}
    elif cls in ("NoisyTFT", "WeakTFT"):
        params = {"p": float(getattr(bot, "p", 0.05))}
    elif cls == "Joss":
        params = {"p": float(getattr(bot, "p", 0.1))}
    elif cls == "Responsive3":
        params = {
            "g_unprov": float(getattr(bot, "g_unprov", 0.55)),
            "g_mutual": float(getattr(bot, "g_mutual", 0.4)),
            "g_resp": float(getattr(bot, "g_resp", 0.95)),
            "min_c": int(getattr(bot, "min_c", 5)),
            "min_d": int(getattr(bot, "min_d", 3)),
            "resp": float(getattr(bot, "resp", 0.38)),
            "rate": float(getattr(bot, "rate", 0.18)),
            "last_d": bool(getattr(bot, "last_d", True)),
            "max_tests": int(getattr(bot, "max_tests", 2)),
            "cooldown": int(getattr(bot, "cooldown", 8)),
        }
    elif cls == "Responsive":
        params = {
            "g": float(getattr(bot, "g", 0.2)),
            "min_c": int(getattr(bot, "min_c", 4)),
            "min_d": int(getattr(bot, "min_d", 3)),
            "resp": float(getattr(bot, "resp", 0.3)),
            "rate": float(getattr(bot, "rate", 0.18)),
            "last_d": bool(getattr(bot, "last_d", True)),
            "max_tests": int(getattr(bot, "max_tests", 2)),
        }
    elif cls == "Responsive2":
        params = {
            "g_unprov": float(getattr(bot, "g_unprov", 1.0)),
            "g_mutual": float(getattr(bot, "g_mutual", 0.1)),
            "strike": int(getattr(bot, "strike", 2)),
            "min_c": int(getattr(bot, "min_c", 4)),
            "min_d": int(getattr(bot, "min_d", 2)),
            "resp": float(getattr(bot, "resp", 0.35)),
            "last_d": bool(getattr(bot, "last_d", True)),
            "max_tests": int(getattr(bot, "max_tests", 3)),
        }
    elif cls == "TFTProbe":
        params = {"probe_round": int(getattr(bot, "probe_round", 60)),
                  "recover": int(getattr(bot, "recover", 2)),
                  "last_d": bool(getattr(bot, "last_d", True))}
    elif cls == "AdaptiveProbe":
        params = {"probe_round": int(getattr(bot, "probe_round", 60)),
                  "recover": int(getattr(bot, "recover", 2)),
                  "last_d": bool(getattr(bot, "last_d", True))}
    elif cls == "PhaseSwitcher":
        params = {"first": str(getattr(bot, "first", "tft")),
                  "switch": int(getattr(bot, "switch", 40)),
                  "second": str(getattr(bot, "second", "extort")),
                  "chi": float(getattr(bot, "chi", 3.0))}
    elif cls == "AdaptiveMix":
        params = {"window": int(getattr(bot, "window", 10)),
                  "threshold": float(getattr(bot, "threshold", 0.35)),
                  "exploit": str(getattr(bot, "exploit", "extort")),
                  "chi": float(getattr(bot, "chi", 3.0))}
    elif cls == "ProbeExploit":
        params = {"probe": int(getattr(bot, "probe", 4)),
                  "mode": str(getattr(bot, "mode", "extort")),
                  "chi": float(getattr(bot, "chi", 3.0))}
    elif cls == "DelayedExtort":
        params = {"delay": int(getattr(bot, "delay", 30)),
                  "chi": float(getattr(bot, "chi", 3.0))}
    elif cls == "MemoryTwo":
        params = {"punish": int(getattr(bot, "punish", 8)),
                  "forgive": bool(getattr(bot, "forgive", True))}
    elif cls == "MetaMix":
        params = {"mode": str(getattr(bot, "mode", "tft"))}
    elif cls == "NoiseAdaptive":
        params = {"window": int(getattr(bot, "window", 10)),
                  "threshold": float(getattr(bot, "threshold", 0.25)),
                  "forgive": float(getattr(bot, "forgive", 0.7)),
                  "punish": int(getattr(bot, "punish", 6))}
    elif cls == "ReputationGrim":
        params = {"threshold": float(getattr(bot, "threshold", 0.55)),
                  "forgive_after": int(getattr(bot, "forgive_after", 0))}
    elif cls == "ContritionCycle":
        params = {"repair": int(getattr(bot, "repair", 2)),
                  "patience": int(getattr(bot, "patience", 3))}
    elif cls == "EndgameHunter":
        params = {"start": int(getattr(bot, "start", 60)),
                  "preempt": int(getattr(bot, "preempt", 8))}
    elif cls == "PatternBreaker":
        params = {"window": int(getattr(bot, "window", 8)),
                  "threshold": float(getattr(bot, "threshold", 0.75))}
    elif cls == "ConditionalExtort":
        params = {"window": int(getattr(bot, "window", 12)),
                  "coop_threshold": float(getattr(bot, "coop_threshold", 0.65)),
                  "chi": float(getattr(bot, "chi", 3.0))}
    elif cls == "LookaheadOpponent":
        params = {"window": int(getattr(bot, "window", 12)),
                  "gamma": float(getattr(bot, "gamma", 0.9)),
                  "horizon": int(getattr(bot, "horizon", 2)),
                  "retaliation": float(getattr(bot, "retaliation", 0.35))}
    elif cls == "MirrorAdaptive":
        params = {"window": int(getattr(bot, "window", 4)),
                  "forgive": float(getattr(bot, "forgive", 0.35))}
    elif cls == "DelayedGrim":
        params = {"patience": int(getattr(bot, "patience", 1))}
    elif cls == "VotingPortfolio":
        params = {"members": tuple(getattr(bot, "members", ())),
                  "tie": str(getattr(bot, "tie", "D"))}
    elif type_name not in TYPE_IDS:
        return None

    return FastRuleSpec(type_name=type_name, params=params,
                        name=str(getattr(bot, "name", type_name)),
                        family=str(getattr(bot, "family", "")))


def _zd_probabilities(kind: str, value: float) -> tuple[float, float, float, float]:
    """Mirror ExtortZD/GenerousZD constructor probabilities."""
    if kind == "extort":
        chi = float(value)
        c = max(chi - 1.0, 1e-9)
        limits = (1.0 / (c * 5.0),
                  1.0 / (chi * 9.0 + 1.0),
                  1.0 / (10.0 + c * 1.0))
        lam = 0.95 * min(limits)
        p = (1.0 - lam * c * 5.0,
             1.0 - lam * (chi * 9.0 + 1.0),
             lam * (10.0 + c * 1.0),
             0.0)
    else:
        kappa = float(value)
        k = max(kappa, 1e-9)
        limits = (1.0 / (k * 10.0 - (k - 1.0) * 6.0),
                  1.0 / (10.0 + (k - 1.0) * 6.0),
                  1.0 / ((k - 1.0) * 5.0))
        valid = [x for x in limits if x > 0]
        lam = 0.95 * min(valid)
        p = (1.0,
             1.0 - lam * (k * 10.0 - (k - 1.0) * 6.0),
             lam * (10.0 + (k - 1.0) * 6.0),
             lam * (k - 1.0) * 5.0)
    return tuple(max(0.0, min(1.0, x)) for x in p)  # type: ignore[return-value]


def encode_spec(spec: FastRuleSpec) -> dict[str, Any]:
    """Encode ``spec`` into the flat arrays accepted by the C ABI.

    The returned dictionary has keys ``type_id``, ``ip`` (16 ints), ``dp``
    (16 doubles), ``seq`` (8 ints) and ``seq_len``.  Unknown parameters are
    ignored so adding a new rule parameter does not break old callers.
    """
    p = dict(spec.params)
    ip = [0] * 16
    dp = [0.0] * 16
    seq = [0] * 8
    seq_len = 0
    name = spec.type_name
    if name in ALIASES:
        name = ALIASES[name]
    if name not in TYPE_IDS:
        raise ValueError(f"unsupported fast rule type: {spec.type_name!r}")

    def as_int(key: str, default: int = 0) -> int:
        return int(p.get(key, default))

    def as_bool(key: str, default: bool = False) -> int:
        return int(bool(p.get(key, default)))

    def as_float(key: str, default: float = 0.0) -> float:
        return float(p.get(key, default))

    if name in ("AllC", "AllD", "TFT", "Grim", "TFT2", "TFTLastD",
                "Pavlov", "ContriteTFT", "WSLS2", "TFTForgiveOnce",
                "TFT2LastD", "TwoTitsForTat", "SuspiciousTFT", "Grudger2",
                "FirmButFair", "Mistrust", "Gradual", "Bully", "Prober",
                "Tester", "RemorsefulProber", "OmegaTFTLite",
                "AllCWithLastD"):
        pass
    elif name == "RandomMoves":
        dp[0] = as_float("p", 0.5)
    elif name == "Majority":
        ip[0] = as_int("k", 3)
    elif name == "GenerousTFT":
        dp[0] = as_float("g", 0.1)
        ip[0] = as_bool("last_d", True)
    elif name == "TwoStrike":
        ip[0] = as_int("strikes", 2)
        ip[1] = as_bool("last_d", True)
    elif name == "ForgiveTFT":
        ip[0] = as_int("missed_limit", 1)
        ip[1] = as_bool("last_d", True)
    elif name == "TFTNoisyForgive":
        dp[0] = as_float("g", 1.0 / 3.0)
    elif name == "GenerousZD":
        for i, value in enumerate(_zd_probabilities("generous",
                                                    as_float("chi", 2.0))):
            dp[i] = value
    elif name == "ExtortZD":
        for i, value in enumerate(_zd_probabilities("extort",
                                                    as_float("chi", 2.0))):
            dp[i] = value
    elif name == "GrudgerK":
        ip[0] = as_int("k", 3)
    elif name == "GrimForgive":
        dp[0] = as_float("g", 0.05)
    elif name == "ProbabilisticGrim":
        ip[0] = as_int("patience", 2)
        dp[0] = as_float("p_punish", 0.85)
    elif name == "ResetGrim":
        ip[0] = as_int("patience", 2)
        ip[1] = as_int("cooldown", 12)
    elif name == "SpitefulGrim":
        ip[0] = as_int("patience", 2)
    elif name == "DelayedPunisher":
        ip[0] = as_int("patience", 3)
        ip[1] = as_int("punish_rounds", 10)
    elif name == "EarlyDefector":
        ip[0] = as_int("k", 20)
    elif name == "LateDefector":
        ip[0] = as_int("k", 20)
    elif name == "RandomThenTFT":
        ip[0] = as_int("k", 3)
        dp[0] = as_float("p", 0.5)
    elif name == "Periodic":
        pattern = tuple(p.get("pattern", (C, D, C)))
        seq_len = min(8, len(pattern))
        for i in range(seq_len):
            seq[i] = int(pattern[i])
        ip[0] = as_int("start", 0)
    elif name == "MemoryOne":
        values = tuple(p.get("p", (1.0, 1.0, 1.0, 0.0)))
        for i in range(4):
            dp[i] = float(values[i]) if i < len(values) else 0.0
        ip[0] = as_int("first", C)
    elif name == "Reactive":
        ip[0] = as_int("first", C)
        dp[0] = as_float("p_after_c", 0.8)
        dp[1] = as_float("p_after_d", 0.1)
    elif name == "HarvestFactor":
        ip[0] = as_int("start", 50)
        ip[1] = as_bool("repairable", True)
        dp[0] = as_float("p_defect", 1.0)
        ip[2] = as_bool("last_d", True)
    elif name == "DynamicWindow":
        ip[0] = as_int("window", 5)
        dp[0] = as_float("p_defect", 0.2)
        dp[1] = as_float("retaliate", 0.85)
        dp[2] = as_float("forgiveness", 0.25)
        ip[1] = as_bool("last_d", True)
    elif name == "Kamikaze":
        ip[0] = as_int("period", 15)
        dp[0] = as_float("p_steal", 1.0)
    elif name == "ProbePhase":
        ip[0] = as_int("probe", 3)
        dp[0] = as_float("p_probe", 0.7)
        ip[1] = 1 if str(p.get("phase", "harvest")) == "harvest" else 0
        ip[2] = as_int("harvest_start", 50)
    elif name == "WindowTFT":
        ip[0] = as_int("window", 5)
        dp[0] = as_float("retaliate", 0.8)
        dp[1] = as_float("forgive", 0.25)
    elif name == "NoisyTFT":
        dp[0] = as_float("p", 0.05)
    elif name == "Joss":
        dp[0] = as_float("p", 0.1)
    elif name == "Responsive3":
        dp[0] = as_float("g_unprov", 0.55)
        dp[1] = as_float("g_mutual", 0.4)
        dp[2] = as_float("g_resp", 0.95)
        ip[0] = as_int("min_c", 5)
        ip[1] = as_int("min_d", 3)
        dp[3] = as_float("resp", 0.38)
        dp[4] = as_float("rate", 0.18)
        ip[2] = as_bool("last_d", True)
        ip[3] = as_int("max_tests", 2)
        ip[4] = as_int("cooldown", 8)
    elif name == "Responsive":
        dp[0] = as_float("g", 0.2)
        ip[0] = as_int("min_c", 4)
        ip[1] = as_int("min_d", 3)
        dp[1] = as_float("resp", 0.3)
        dp[2] = as_float("rate", 0.18)
        ip[2] = as_bool("last_d", True)
        ip[3] = as_int("max_tests", 2)
    elif name == "Responsive2":
        dp[0] = as_float("g_unprov", 1.0)
        dp[1] = as_float("g_mutual", 0.1)
        ip[0] = as_int("strike", 2)
        ip[1] = as_int("min_c", 4)
        ip[2] = as_int("min_d", 2)
        dp[2] = as_float("resp", 0.35)
        ip[3] = as_bool("last_d", True)
        ip[4] = as_int("max_tests", 3)
    elif name == "TFTProbe":
        ip[0] = as_int("probe_round", 60)
        ip[1] = as_int("recover", 2)
        ip[2] = as_bool("last_d", True)
    elif name == "AdaptiveProbe":
        ip[0] = as_int("probe_round", 60)
        ip[1] = as_int("recover", 2)
        ip[2] = as_bool("last_d", True)
    elif name == "PhaseSwitcher":
        first = str(p.get("first", "tft"))
        second = str(p.get("second", "extort"))
        ip[0] = FIRST_CODES.get(first, 0)
        ip[1] = as_int("switch", 40)
        ip[2] = SECOND_CODES.get(second, 0)
        dp[0] = as_float("chi", 3.0)
    elif name == "AdaptiveMix":
        ip[0] = as_int("window", 10)
        dp[0] = as_float("threshold", 0.35)
        ip[1] = EXPLOIT_CODES.get(str(p.get("exploit", "extort")), 0)
        dp[1] = as_float("chi", 3.0)
    elif name == "ProbeExploit":
        ip[0] = as_int("probe", 4)
        ip[1] = EXPLOIT_CODES.get(str(p.get("mode", "extort")), 0)
        dp[0] = as_float("chi", 3.0)
    elif name == "DelayedExtort":
        ip[0] = as_int("delay", 30)
        dp[0] = as_float("chi", 3.0)
    elif name == "MemoryTwo":
        ip[0] = as_int("punish", 8)
        ip[1] = as_bool("forgive", True)
    elif name == "MetaMix":
        ip[0] = META_CODES.get(str(p.get("mode", "tft")), 0)
        dp[0] = 3.0
        ip[1] = 25
    elif name == "NoiseAdaptive":
        ip[0] = as_int("window", 10)
        dp[0] = as_float("threshold", 0.25)
        dp[1] = as_float("forgive", 0.7)
        ip[1] = as_int("punish", 6)
    elif name == "ReputationGrim":
        dp[0] = as_float("threshold", 0.55)
        ip[0] = as_int("forgive_after", 0)
    elif name == "ContritionCycle":
        ip[0] = as_int("repair", 2)
        ip[1] = as_int("patience", 3)
    elif name == "EndgameHunter":
        ip[0] = as_int("start", 60)
        ip[1] = as_int("preempt", 8)
    elif name == "PatternBreaker":
        ip[0] = as_int("window", 8)
        dp[0] = as_float("threshold", 0.75)
    elif name == "ConditionalExtort":
        ip[0] = as_int("window", 12)
        dp[0] = as_float("coop_threshold", 0.65)
        dp[1] = as_float("chi", 3.0)
    elif name == "LookaheadOpponent":
        ip[0] = as_int("window", 12)
        dp[0] = as_float("gamma", 0.9)
        ip[1] = as_int("horizon", 2)
        dp[1] = as_float("retaliation", 0.35)
    elif name == "MirrorAdaptive":
        ip[0] = as_int("window", 4)
        dp[0] = as_float("forgive", 0.35)
    elif name == "DelayedGrim":
        ip[0] = as_int("patience", 1)
    elif name == "VotingPortfolio":
        members = tuple(p.get("members", ("tft", "grim", "extort")))
        seq_len = min(8, len(members))
        for i in range(seq_len):
            seq[i] = PORTFOLIO_CODES.get(str(members[i]), 0)
        ip[0] = 0 if str(p.get("tie", "D")) == "C" else 1
    else:  # pragma: no cover - guarded by TYPE_IDS
        raise ValueError(f"no encoder for {name!r}")

    return {"type_id": TYPE_IDS[name], "ip": ip, "dp": dp,
            "seq": seq, "seq_len": seq_len}


__all__ = [
    "TYPE_NAMES", "TYPE_IDS", "ALIASES", "FastRuleSpec", "UNSUPPORTED_TYPES",
    "attach_fast_spec", "get_fast_spec", "infer_fast_spec", "encode_spec",
]
