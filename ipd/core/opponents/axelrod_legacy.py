"""Matrix-aware overrides for historical Axelrod strategies.

Axelrod 4.14.0 contains a handful of strategies whose decision rules compare
running scores against literals calibrated for its default 3/0/5/1 matrix.
The adapter runs them on pri_dile's 6/0/10/1 matrix, so this module rewrites
only those score thresholds while leaving the strategy logic unchanged.

The overrides mirror ``axelrod==4.14.0`` and are installed lazily by
:mod:`ipd.core.opponents.axelrod_bank`.  Keep them in sync if the Axelrod pin
changes.
"""
from __future__ import annotations

from typing import Any

# Axelrod's default reward value.  Historical thresholds are expressed in
# units calibrated to this value; R scales them to the active game.
AXELROD_DEFAULT_R = 3.0


def _scale_for_r(r: float) -> float:
    """Return the score-unit scale relative to Axelrod's default R=3."""
    return float(r) / AXELROD_DEFAULT_R


def _two_round_swing(t: float, s: float) -> float:
    """Two maximum single-round swings (two defections vs sucker payoffs)."""
    return 2.0 * (t - s)


def _one_round_swing(t: float, s: float) -> float:
    """One maximum single-round swing (defect vs sucker payoff)."""
    return float(t - s)


def _graaskamp_threshold(rounds: int, r: float) -> float:
    """Score needed after ``rounds`` rounds: ``rounds * R`` minus 7 points.

    The original check is ``score < 3 * rounds - 7``.  The seven-point
    tolerance is a fixed fraction of the reward, so scale it with R.
    """
    return rounds * r - 7.0 * _scale_for_r(r)


def _rowsam_thresholds(
    r: float, p: float
) -> tuple[float, float, float, float]:
    """Score-per-turn bands at 0, 1/4, 1/2 and 3/4 of [P, R].

    Axelrod's defaults (P=1, R=3) produce the historical 1.0/1.5/2.0/2.5.
    """
    step = (r - p) / 4.0
    return p, p + step, p + 2.0 * step, p + 3.0 * step


def _tranquilizer_thresholds(r: float, p: float) -> tuple[float, float]:
    """Symmetric average-score bands at 3/8 and 5/8 of [P, R].

    Axelrod's defaults (P=1, R=3) produce the historical 1.75/2.25.
    """
    return (3.0 * r + 5.0 * p) / 8.0, (5.0 * r + 3.0 * p) / 8.0


def _tranquilizer_difference_scale(t: float, s: float) -> float:
    """The historical ``/100`` is 20 maximum single-round swings."""
    return 20.0 * (t - s)


def install_threshold_adaptation(axl: Any) -> None:
    """Install matrix-aware threshold overrides on Axelrod 4.14.0 classes."""
    if getattr(axl, "_pri_dile_thresholds_installed", False):
        return

    from axelrod.action import Action
    from axelrod.interaction_utils import compute_final_score
    from axelrod.strategies.axelrod_first import FirstByTidemanAndChieruzzi
    from axelrod.strategies.axelrod_second import (
        SecondByGraaskampKatzen, SecondByRowsam,
        SecondByTidemanAndChieruzzi, SecondByTranquilizer,
    )
    from axelrod.strategies.stalker import Stalker

    C, D = Action.C, Action.D

    def first_by_tideman_strategy(self, opponent):
        if not opponent.history:
            return C

        if opponent.history[-1] == D:
            self.remembered_number_of_opponent_defectioons += 1

        self._score_last_round(opponent)

        if self.fresh_start:
            self.fresh_start = False
            return C

        current_round = len(self.history) + 1
        if self.last_fresh_start == 0:
            valid_fresh_start = True
        else:
            valid_fresh_start = current_round - self.last_fresh_start >= 20

        if valid_fresh_start:
            _, _, s, t = self.match_attributes["game"].RPST()
            # The original 10-point lead is two maximum single-round swings.
            valid_points = (
                self.current_score - self.opponent_score
                >= _two_round_swing(t, s)
            )
            valid_rounds = (
                self.match_attributes["length"] - current_round >= 10
            )
            opponent_is_cooperating = opponent.history[-1] == C
            if valid_points and valid_rounds and opponent_is_cooperating:
                N = opponent.cooperations + opponent.defections
                std_deviation = (N ** (1 / 2)) / 2
                lower = N / 2 - 3 * std_deviation
                upper = N / 2 + 3 * std_deviation
                if (
                    self.remembered_number_of_opponent_defectioons <= lower
                    or self.remembered_number_of_opponent_defectioons >= upper
                ):
                    self.last_fresh_start = current_round
                    self._fresh_start()
                    self.fresh_start = True
                    return C

        if self.is_retaliating:
            self._decrease_retaliation_counter()
            return D

        if opponent.history[-1] == D:
            self.is_retaliating = True
            self.retaliation_length += 1
            self.retaliation_remaining = self.retaliation_length
            self._decrease_retaliation_counter()
            return D

        return C

    def graaskamp_katzen_strategy(self, opponent):
        if self.mode == "Defect":
            return D

        turn = len(self.history) + 1
        if turn == 1:
            return C

        self.update_score(opponent)
        r = self.match_attributes["game"].RPST()[0]
        checkpoints = {11: 10, 21: 20, 31: 30, 41: 40, 51: 50, 101: 100}

        if (turn in checkpoints
                and self.own_score < _graaskamp_threshold(
                    checkpoints[turn], r)):
            self.mode = "Defect"
            return D

        return opponent.history[-1]

    def tideman_chieruzzi_strategy(self, opponent):
        current_round = len(self.history) + 1

        if current_round == 1:
            return C

        self._score_last_round(opponent)

        if self.fresh_start:
            self._fresh_start()
            self.last_fresh_start = current_round
            self.fresh_start = False
            return C

        opponent_CDd = False

        opponent_two_turns_ago = C
        if len(opponent.history) >= 2:
            opponent_two_turns_ago = opponent.history[-2]
        if opponent_two_turns_ago == C and opponent.history[-1] == D:
            opponent_CDd = True
            _, _, s, t = self.match_attributes["game"].RPST()
            self.score_to_beat += self.score_to_beat_inc
            # One increment is the maximum single-round score advantage.
            self.score_to_beat_inc += _one_round_swing(t, s)

        if self.current_score - self.opponent_score >= self.score_to_beat:
            return C

        if (not opponent_CDd) and current_round - self.last_fresh_start >= 10:
            N = opponent.cooperations + opponent.defections
            std_deviation = (N ** (1 / 2)) / 2
            lower = N / 2 - 3 * std_deviation
            upper = N / 2 + 3 * std_deviation
            if opponent.defections <= lower or opponent.defections >= upper:
                self.fresh_start = True
                return C

        return D

    def rowsam_strategy(self, opponent):
        turn = len(self.history) + 1
        if turn > 1:
            self._score_last_round(opponent)

        if self.mode == "Defect":
            return D

        if self.mode == "Coop Def Cycle 1":
            self.mode = "Coop Def Cycle 2"
            return C

        if self.mode == "Coop Def Cycle 2":
            self.mode = "Normal"
            return D

        if turn % 18 == 0:
            if self.distrust_points >= 3:
                self.distrust_points -= 1

        if turn % 6 != 0:
            return C

        r, p = self.match_attributes["game"].RPST()[:2]
        q0, q1, q2, q3 = _rowsam_thresholds(r, p)
        points_per_turn = self.current_score / turn
        if points_per_turn < q0:
            self.distrust_points += 5
        elif points_per_turn < q1:
            self.distrust_points += 3
        elif points_per_turn < q2:
            self.distrust_points += 2
        elif points_per_turn < q3:
            self.distrust_points += 1
        else:
            return C

        if self.distrust_points >= 7:
            self.mode = "Defect"
        else:
            self.mode = "Coop Def Cycle 1"
        return D

    def tranquilizer_strategy(self, opponent):
        if not self.history:
            return C

        self.update_state(opponent)
        if self.num_turns_after_good_defection in [1, 2]:
            return C

        current_score = compute_final_score(
            zip(self.history, opponent.history),
            game=self.match_attributes["game"],
        )
        r, p, s, t = self.match_attributes["game"].RPST()
        lower, upper = _tranquilizer_thresholds(r, p)
        difference_scale = _tranquilizer_difference_scale(t, s)
        turns = len(self.history) + 1

        if current_score[0] / turns >= upper:
            probability = (
                (
                    0.95
                    - (
                        (
                            self.one_turn_after_good_defection_ratio
                            + self.two_turns_after_good_defection_ratio
                            - 5
                        )
                        / 15
                    )
                )
                + (1 / (turns ** 2))
                - (self.dict[opponent.history[-1]] / 4)
            )
            if self._random.random() <= probability:
                return C
            self.num_turns_after_good_defection = 1
            return D
        if current_score[0] / turns >= lower:
            probability = (
                (
                    0.25
                    + ((opponent.cooperations + 1) / turns)
                )
                - (self.opponent_consecutive_defections * 0.25)
                + ((current_score[0] - current_score[1]) / difference_scale)
                + (4 / turns)
            )
            if self._random.random() <= probability:
                return C
            return D
        return opponent.history[-1]

    def stalker_strategy(self, opponent):
        if len(self.history) == 0:
            return C

        self.score_last_round(opponent)
        average = self.current_score / len(self.history)

        if average > self.very_good_score:
            return D
        if self.wish_score < average < self.very_good_score:
            return C
        if average > self.wish_score:
            return C
        if self.very_bad_score < average < self.wish_score:
            return D
        return self._random.random_choice()

    def install(cls, method):
        target = getattr(cls, "original_class", cls)
        if not hasattr(target, "_pri_dile_original_strategy"):
            target._pri_dile_original_strategy = target.strategy
        target.strategy = method

    install(FirstByTidemanAndChieruzzi, first_by_tideman_strategy)
    install(SecondByGraaskampKatzen, graaskamp_katzen_strategy)
    install(SecondByTidemanAndChieruzzi, tideman_chieruzzi_strategy)
    install(SecondByRowsam, rowsam_strategy)
    install(SecondByTranquilizer, tranquilizer_strategy)
    install(Stalker, stalker_strategy)
    axl._pri_dile_thresholds_installed = True


__all__ = ["AXELROD_DEFAULT_R", "install_threshold_adaptation"]
