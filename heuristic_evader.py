# -*- coding: utf-8 -*-

import random

try:
    from agents.base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


class HeuristicEvader(BaseAgent):
    """
    It tries to move toward the goal while also keeping distance
    from the pursuer.
    """

    def __init__(self, role: str, config: dict):
        super().__init__(role, config)

        self.random_prob = float(config.get("random_prob", 0.10))
        self.goal_weight = float(config.get("goal_weight", 0.75))
        self.escape_weight = float(config.get("escape_weight", 0.25))

        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

    def select_action(self, observation: dict) -> int:
        if random.random() < self.random_prob:
            return random.randint(0, 3)

        self_pos = observation["self_pos"]
        pursuer_pos = observation["opponent_pos"]
        goal_pos = observation["goal_pos"]
        grid_size = observation["grid_size"]

        best_action = 0
        best_score = -1e9

        for action, delta in self.actions.items():
            new_pos = self._move(self_pos, delta, grid_size)

            dist_to_goal = self._manhattan(new_pos, goal_pos)
            dist_from_pursuer = self._manhattan(new_pos, pursuer_pos)

            score = (
                -self.goal_weight * dist_to_goal
                + self.escape_weight * dist_from_pursuer
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def update(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
    ) -> None:
        pass

    def reset(self) -> None:
        pass

    def _move(self, pos, delta, grid_size):
        r, c = pos
        dr, dc = delta

        nr = max(0, min(grid_size - 1, r + dr))
        nc = max(0, min(grid_size - 1, c + dc))

        return (nr, nc)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


AGENT_CLASS = HeuristicEvader