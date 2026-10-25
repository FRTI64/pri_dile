#!/usr/bin/env python3
"""Experimental endgame-harvest variant of the final strategy.

If the opponent has never defected by day E, defect once.  If it ignores that
defection, keep exploiting; if it retaliates, make two cooperative repair
moves and then either return to normal reciprocity or punish.
"""
from __future__ import annotations

from ipd.core.machine import Machine, C, D


class HarvestMachine(Machine):
    """Late-harvest-aware variant.

    Recommended when a substantial fraction of the field is expected to
    cooperate for most of the game and then defect near the end.  The default
    start day is 45; move it later (e.g. 60) if the field contains many
    Grim/Mistrust-style permanent punishers.
    """

    name = "Harvest45"
    HARVEST_START = 45

    def reset(self) -> None:
        super().reset()
        self.harvest = "none"     # none | probe | hunt | repair
        self.harvest_done = False
        self.repair_left = 0

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        """Adapter for the zero-based simulator."""
        return self.move(t + 1, mine, theirs)

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        if day >= 100:
            return D
        if day == 1:
            return C

        if self.harvest == "none":
            if (not self.harvest_done and day >= self.HARVEST_START
                    and self.mode == "nice" and D not in theirs):
                self.harvest_done = True
                self.harvest = "probe"
                return D
            return super().move(day, mine, theirs)

        if self.harvest == "probe":
            # Their response to the first harvest defection is now known.
            if theirs[-1] == C:
                self.harvest = "hunt"
                return D
            self.harvest = "repair"
            self.repair_left = 2
            return C

        if self.harvest == "hunt":
            if theirs[-1] == C:
                return D
            self.harvest = "repair"
            self.repair_left = 2
            return C

        # repair: two cooperative moves, then decide.
        self.repair_left -= 1
        if self.repair_left > 0:
            return C
        self.harvest = "none"
        if theirs[-1] == C:
            self.mode = "nice"
            return C
        self.mode = "punish"
        return D
