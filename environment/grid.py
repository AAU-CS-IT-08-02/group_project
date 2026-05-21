"""
environment/grid.py
-------------------
Defines the 10x10 grid world.

Responsibilities:
  - Stores and validates agent positions
  - Handles movement with boundary clamping
  - Knows where the goal tile is
  - Provides distance utilities used by reward and metrics
"""

import random
from dataclasses import dataclass, field
from typing import Tuple


# ── Constants ──────────────────────────────────────────────────────────
GRID_SIZE   = 10
NUM_ACTIONS = 4   # 0=up, 1=down, 2=left, 3=right

# Action deltas: (row_delta, col_delta)
ACTION_DELTAS = {
    0: (-1,  0),   # up
    1: ( 1,  0),   # down
    2: ( 0, -1),   # left
    3: ( 0,  1),   # right
}


# ── Position type alias ────────────────────────────────────────────────
Position = Tuple[int, int]   # (row, col)


@dataclass
class GridConfig:
    """
    All configurable properties of the grid.
    Passed in from the UI settings panel.
    """
    grid_size      : int      = GRID_SIZE
    pursuer_start  : Position = (0, 0)
    evader_start   : Position = (9, 9)
    goal_pos       : Position = (9, 0)
    random_spawns  : bool     = False   # if True, randomise positions each episode
    random_goal    : bool     = False   # if True, randomise goal tile each episode

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for name, pos in [
            ("pursuer_start", self.pursuer_start),
            ("evader_start",  self.evader_start),
            ("goal_pos",      self.goal_pos),
        ]:
            r, c = pos
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                raise ValueError(
                    f"{name} {pos} is outside the grid "
                    f"(size {self.grid_size}x{self.grid_size})"
                )
        if self.pursuer_start == self.evader_start:
            raise ValueError("pursuer_start and evader_start must be different tiles")


class Grid:
    """
    The 10x10 grid world.

    Tracks current positions of both agents and exposes
    movement, distance, and observation-building utilities.

    Does NOT own episode logic (that lives in game.py).
    """

    def __init__(self, config: GridConfig):
        self.config       = config
        self.grid_size    = config.grid_size
        self.goal_pos     = config.goal_pos

        # Current positions — reset at the start of each episode
        self.pursuer_pos : Position = config.pursuer_start
        self.evader_pos  : Position = config.evader_start

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset both agents to their starting positions.

        If random_spawns is enabled, agents are placed on random distinct
        tiles every episode (neither on the goal tile).
        If random_goal is enabled, the goal tile is also randomised.
        """
        size = self.grid_size

        if self.config.random_goal:
            self.goal_pos = (
                random.randint(0, size - 1),
                random.randint(0, size - 1),
            )
        else:
            self.goal_pos = self.config.goal_pos

        if self.config.random_spawns:
            # Build a pool of all tiles except the goal, then sample 2
            pool = [
                (r, c)
                for r in range(size)
                for c in range(size)
                if (r, c) != self.goal_pos
            ]
            self.pursuer_pos, self.evader_pos = random.sample(pool, 2)
        else:
            self.pursuer_pos = self.config.pursuer_start
            self.evader_pos  = self.config.evader_start

    # ── Movement ───────────────────────────────────────────────────────

    def move(self, pos: Position, action: int) -> Position:
        """
        Apply an action to a position.
        Clamps to grid boundaries — moving into a wall leaves the
        agent in place.

        Parameters
        ----------
        pos    : current (row, col)
        action : int in {0, 1, 2, 3}

        Returns
        -------
        New (row, col) after the move.
        """
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Must be 0–3.")

        dr, dc = ACTION_DELTAS[action]
        new_row = pos[0] + dr
        new_col = pos[1] + dc

        # Clamp to grid
        new_row = max(0, min(self.grid_size - 1, new_row))
        new_col = max(0, min(self.grid_size - 1, new_col))

        return (new_row, new_col)

    def move_pursuer(self, action: int) -> Position:
        """Move the pursuer and update its stored position."""
        self.pursuer_pos = self.move(self.pursuer_pos, action)
        return self.pursuer_pos

    def move_evader(self, action: int) -> Position:
        """Move the evader and update its stored position."""
        self.evader_pos = self.move(self.evader_pos, action)
        return self.evader_pos

    # ── Distance ───────────────────────────────────────────────────────

    def manhattan_distance(self, a: Position, b: Position) -> int:
        """
        Manhattan distance between two grid positions.
        Used for rewards and metrics.

        Manhattan (not Euclidean) is standard for grid worlds
        because diagonal moves aren't possible.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @property
    def agent_distance(self) -> int:
        """Current Manhattan distance between pursuer and evader."""
        return self.manhattan_distance(self.pursuer_pos, self.evader_pos)

    @property
    def evader_to_goal_distance(self) -> int:
        """Current Manhattan distance from evader to goal tile."""
        return self.manhattan_distance(self.evader_pos, self.goal_pos)

    @property
    def pursuer_to_evader_distance(self) -> int:
        """Alias for agent_distance — used in reward calculations."""
        return self.agent_distance

    # ── Win conditions ─────────────────────────────────────────────────

    @property
    def pursuer_wins(self) -> bool:
        """Pursuer wins by occupying the same tile as the evader."""
        return self.pursuer_pos == self.evader_pos

    @property
    def evader_wins(self) -> bool:
        """Evader wins by reaching the goal tile."""
        return self.evader_pos == self.goal_pos

    # ── Observations ───────────────────────────────────────────────────

    def get_observation(self, role: str, steps_remaining: int) -> dict:
        """
        Build the observation dict passed to each agent's
        select_action and update methods.

        Parameters
        ----------
        role            : "pursuer" or "evader"
        steps_remaining : steps left in the current episode

        Returns
        -------
        dict with keys: self_pos, opponent_pos, goal_pos,
                        grid_size, steps_remaining
        """
        if role == "pursuer":
            self_pos     = self.pursuer_pos
            opponent_pos = self.evader_pos
        elif role == "evader":
            self_pos     = self.evader_pos
            opponent_pos = self.pursuer_pos
        else:
            raise ValueError(f"role must be 'pursuer' or 'evader', got '{role}'")

        return {
            "self_pos"        : self_pos,
            "opponent_pos"    : opponent_pos,
            "goal_pos"        : self.goal_pos,
            "grid_size"       : self.grid_size,
            "steps_remaining" : steps_remaining,
        }

    # ── State snapshot (for metrics and replay recording) ─────────────

    def snapshot(self) -> dict:
        """
        Return a lightweight snapshot of the current grid state.
        Stored every step by the metrics recorder for replay.
        """
        return {
            "pursuer_pos" : self.pursuer_pos,
            "evader_pos"  : self.evader_pos,
            "goal_pos"    : self.goal_pos,
        }

    # ── String representation (for debugging) ─────────────────────────

    def __repr__(self) -> str:
        rows = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == self.pursuer_pos and pos == self.evader_pos:
                    row.append("X")    # overlap (capture imminent)
                elif pos == self.pursuer_pos:
                    row.append("P")
                elif pos == self.evader_pos:
                    row.append("E")
                elif pos == self.goal_pos:
                    row.append("G")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)