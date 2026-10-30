"""Optional adapter for the Axelrod strategy library.

The 243 ordinary Axelrod strategies implement ``axelrod.Player``, not the
small :class:`ipd.core.primitives.Strategy` protocol.  This module bridges the
two interfaces lazily: importing :mod:`ipd` never imports Axelrod, and the
optional dependency is only required when an Axelrod opponent bank is used.

The adapter passes pri_dile's 6/0/10/1 matrix to every Axelrod player through
``match_attributes["game"]``.  Most strategies use it directly; the
zero-determinant family needs its four-vector's enforcement strength
recalibrated because Axelrod's shipped defaults assume 3/0/5/1 (see
``_install_lr_adaptation``).  Match scores are still computed by
:mod:`ipd.core.simulator` with the same 6/0/10/1 matrix.
"""
from __future__ import annotations

import functools
import random
from typing import Any

from ..primitives import C, D, R, S, T, P, Strategy
from .axelrod_legacy import install_threshold_adaptation
from .spec import TypeSpec

_AXL: Any = None


def _install_lr_adaptation(axl: Any) -> None:
    """Recalibrate Axelrod's linear-relation players for pri_dile's matrix.

    Axelrod's ZD strategies derive a memory-one four-vector from the game
    matrix and a strength parameter ``phi``.  Their shipped ``phi`` values are
    tuned for the default 3/0/5/1 game; with 6/0/10/1 several of them produce a
    negative (or >1) probability and ``MemoryOnePlayer.set_four_vector``
    rejects the player.  The slope ``s`` and baseline ``l`` still define the
    intended ZD relation, so only ``phi`` is capped at the largest value that
    keeps the four-vector in [0, 1].  For the default game this is a no-op.
    """
    from axelrod.strategies.zero_determinant import LRPlayer

    if getattr(LRPlayer, "_pri_dile_adapted", False):
        return

    upstream = LRPlayer.receive_match_attributes

    def receive_match_attributes(self):
        if self.l is None or self.phi is None:
            return upstream(self)
        R, P, S, T = self.match_attributes["game"].RPST()
        l = float(self.l)
        phi = float(self.phi)
        s = float(self.s)
        s_min = -min((T - l) / (l - S), (l - S) / (T - l))
        if (l < P) or (l > R) or (s > 1) or (s < s_min):
            raise ValueError("invalid linear-relation parameters")
        # p_i = c_i + d_i * phi.  Find the largest phi that keeps every
        # probability in [0, 1], then use the original phi if it fits.
        c = (1.0, 1.0, 0.0, 0.0)
        d = (
            -(1.0 - s) * (R - l),
            -(s * (l - S) + (T - l)),
            (l - S) + s * (T - l),
            (1.0 - s) * (l - P),
        )
        max_phi = float("inf")
        for ci, di in zip(c, d):
            if di > 0.0:
                max_phi = min(max_phi, (1.0 - ci) / di)
            elif di < 0.0:
                max_phi = min(max_phi, -ci / di)
            elif not (0.0 <= ci <= 1.0):
                raise ValueError("infeasible four-vector")
        effective_phi = min(phi, max_phi) if max_phi >= 0.0 else 0.0
        self.set_four_vector([
            min(1.0, max(0.0, ci + di * effective_phi))
            for ci, di in zip(c, d)
        ])

    LRPlayer.receive_match_attributes = receive_match_attributes
    LRPlayer._pri_dile_adapted = True


def _install_interaction_adaptation(axl: Any) -> None:
    """Make Axelrod's default score helpers use pri_dile's matrix.

    A few historical strategies call ``compute_final_score`` without passing
    the match game explicitly; that helper falls back to ``Game()`` (3/0/5/1).
    Route the fallback through the same 6/0/10/1 game object used by the
    adapter so those strategies do not silently read the old matrix.
    """
    from axelrod import interaction_utils

    if getattr(interaction_utils, "_pri_dile_adapted", False):
        return

    upstream = interaction_utils.compute_scores

    def compute_scores(interactions, game=None):
        if game is None:
            game = _pri_dile_game(axl)
        return upstream(interactions, game)

    interaction_utils.compute_scores = compute_scores
    interaction_utils._pri_dile_adapted = True


def _pri_dile_game(axl: Any):
    """Return the Axelrod game object for pri_dile's canonical matrix."""
    return axl.Game(r=R, s=S, t=T, p=P)


def _load_axelrod():
    """Import Axelrod on first use and cache the module handle."""
    global _AXL
    if _AXL is not None:
        return _AXL
    try:
        import axelrod as axl
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Axelrod opponents require the optional 'axelrod' package; "
            "install with `pip install -e \".[axelrod]\"` or "
            "`pip install -r requirements-axelrod.txt`."
        ) from exc
    _install_lr_adaptation(axl)
    _install_interaction_adaptation(axl)
    install_threshold_adaptation(axl)
    _AXL = axl
    return axl


def require_axelrod() -> None:
    """Raise a clear error if the optional Axelrod dependency is missing."""
    _load_axelrod()


def _to_action(axl, value: int):
    if isinstance(value, axl.Action):
        return value
    value = int(value)
    if value == C:
        return axl.Action.C
    if value == D:
        return axl.Action.D
    raise ValueError(f"invalid action {value!r}")


def _to_int(axl, value: Any) -> int:
    if isinstance(value, axl.Action):
        return int(value.value)
    if value in (C, D):
        return int(value)
    raise ValueError(f"Axelrod strategy returned invalid action {value!r}")


class _HistoryProxy:
    """Minimal opponent object exposing the history API Axelrod uses."""

    name = "PriDileCandidate"
    classifier: dict[str, Any] = {}

    def __init__(self, game: Any, rounds: int):
        axl = _load_axelrod()
        self._history = axl.History()
        self.match_attributes = {
            "length": int(rounds),
            "game": game,
            "noise": 0,
        }

    def record(self, play: Any, coplay: Any) -> None:
        """Append one completed round instead of rebuilding the history."""
        self._history.append(play, coplay)

    @property
    def history(self):
        return self._history

    @property
    def cooperations(self) -> int:
        return self._history.cooperations

    @property
    def defections(self) -> int:
        return self._history.defections

    @property
    def state_distribution(self):
        return self._history.state_distribution


class AxelrodOpponent(Strategy):
    """Adapt one Axelrod ``Player`` class to the pri_dile opponent protocol.

    The adapter maintains the Axelrod player's history incrementally and calls
    its native ``strategy`` method each turn.  This keeps multi-turn state and
    Meta-player team histories correct without depending on Axelrod's Match
    class.
    """

    def __init__(self, player_cls, rng: random.Random | None = None,
                 rounds: int = 100):
        self.player_cls = player_cls
        self._seed = (rng or random.Random()).randrange(1 << 30)
        self._rounds = int(rounds)
        self._player = None
        self._opponent = None
        self.name = player_cls.__name__
        super().__init__(rng)

    def reset(self) -> None:
        axl = _load_axelrod()
        # Every strategy sees the project's 6/0/10/1 matrix.  ZD players are
        # recalibrated by _install_lr_adaptation so their four-vectors remain
        # valid; the match score is computed by the same matrix in the
        # simulator.
        self._game = _pri_dile_game(axl)
        self._player = self.player_cls()
        self._player.set_seed(self._seed)
        self._player.set_match_attributes(
            length=self._rounds, game=self._game, noise=0)
        self._opponent = _HistoryProxy(self._game, self._rounds)

    def set_rounds(self, rounds: int) -> None:
        """Let the evaluation harness communicate the match length."""
        self._rounds = int(rounds)
        if self._player is not None:
            self._player.set_match_attributes(
                length=self._rounds, game=self._game, noise=0)
            self._opponent.match_attributes = self._player.match_attributes

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        axl = _load_axelrod()
        if self._player is None:
            self.reset()
        if t > 0:
            # Append the previous round now that the opponent's reply is known.
            # MetaPlayer overrides update_history to propagate its team's
            # proposed actions as well.
            mine_prev = _to_action(axl, mine[-1])
            theirs_prev = _to_action(axl, theirs[-1])
            self._player.update_history(mine_prev, theirs_prev)
            # The proxy mirrors ``theirs`` as plays and ``mine`` as coplays.
            # Appending is equivalent to the old full reconstruction but is
            # O(1) per round instead of O(t).
            self._opponent.record(theirs_prev, mine_prev)
        history_len = len(self._player.history)
        if t != history_len:
            raise RuntimeError(
                f"{self.name}: match history out of sync at turn {t} "
                f"(expected {history_len}); call reset() before each match")
        self._opponent.match_attributes = self._player.match_attributes
        return _to_int(axl, self._player.strategy(self._opponent))

    def __repr__(self) -> str:
        return self.name


@functools.lru_cache(maxsize=None)
def _bank(include_long_run: bool) -> tuple[TypeSpec, ...]:
    axl = _load_axelrod()
    long_run = set(axl.long_run_time_strategies)
    specs: list[TypeSpec] = []
    for player_cls in axl.strategies:
        if not include_long_run and player_cls in long_run:
            continue
        specs.append(TypeSpec(
            name=player_cls.__name__,
            family="axelrod",
            weight=1.0,
            factory=functools.partial(AxelrodOpponent, player_cls),
        ))
    return tuple(specs)


def axelrod_specs(include_long_run: bool = True) -> list[TypeSpec]:
    """Return the ordinary Axelrod strategies as :class:`TypeSpec` objects.

    ``include_long_run=False`` removes the 19 strategies classified by
    Axelrod as long-running (mostly Meta players that simulate the whole bank
    every turn).  The full bank has 243 strategies; the fast bank has 224.
    """
    return list(_bank(bool(include_long_run)))


def axelrod_by_name(include_long_run: bool = True) -> dict[str, TypeSpec]:
    """Return the selected Axelrod bank keyed by strategy class name."""
    return {spec.name: spec for spec in _bank(bool(include_long_run))}


@functools.lru_cache(maxsize=None)
def _cheating_bank() -> tuple[TypeSpec, ...]:
    """Return strategies Axelrod classifies as not obeying tournament rules.

    Axelrod 4.14.0 ships one such strategy (``Darwin``).  The bank is built
    from ``axelrod.cheating_strategies``, so a newer release that adds more
    cheating classes will be picked up.  The legacy Geller/MindReader family
    is not present in 4.14.0 and is not supported here: those classes need
    direct access to the candidate's strategy object rather than a history
    proxy.
    """
    axl = _load_axelrod()
    specs: list[TypeSpec] = []
    for player_cls in axl.cheating_strategies:
        specs.append(TypeSpec(
            name=player_cls.__name__,
            family="axelrod-cheating",
            weight=1.0,
            factory=functools.partial(AxelrodOpponent, player_cls),
        ))
    return tuple(specs)


def axelrod_cheating_specs() -> list[TypeSpec]:
    """Return Axelrod's cheating strategies (4.14.0: ``Darwin`` only)."""
    return list(_cheating_bank())


def axelrod_cheating_by_name() -> dict[str, TypeSpec]:
    """Return the cheating bank keyed by strategy class name."""
    return {spec.name: spec for spec in _cheating_bank()}


def _sample_field(specs: list[TypeSpec], seed: int,
                  n: int) -> list[Strategy]:
    if n <= 0:
        return []
    if not specs:
        raise RuntimeError("Axelrod opponent bank is empty")
    rng = random.Random(int(seed))
    pool = list(specs)
    rng.shuffle(pool)
    field: list[Strategy] = []
    while len(field) < n:
        for spec in pool:
            opponent = spec.factory(rng)
            opponent.name = spec.name
            field.append(opponent)
            if len(field) >= n:
                break
        if len(field) < n:
            rng.shuffle(pool)
    return field


def sample_axelrod_field(seed: int, n: int,
                         include_long_run: bool = True) -> list[Strategy]:
    """Sample ``n`` opponents from the Axelrod bank.

    Opponents are drawn from shuffled cycles over the selected bank, so a
    field of at least 243 opponents contains every ordinary Axelrod strategy
    exactly once per cycle rather than relying on random sampling with
    replacement.
    """
    return _sample_field(axelrod_specs(include_long_run), seed, n)


def sample_axelrod_cheating_field(seed: int, n: int) -> list[Strategy]:
    """Sample ``n`` opponents from the cheating Axelrod bank."""
    return _sample_field(axelrod_cheating_specs(), seed, n)


__all__ = [
    "AxelrodOpponent", "axelrod_specs", "axelrod_by_name",
    "axelrod_cheating_specs", "axelrod_cheating_by_name",
    "sample_axelrod_field", "sample_axelrod_cheating_field",
    "require_axelrod",
]
