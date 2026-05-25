"""
agents/random_pursuer.py
-------------------------
A pursuer that picks actions completely at random.
Useful as a baseline and for testing the environment.
"""

import random
from agents.base_agent import BaseAgent


class RandomPursuer(BaseAgent):
    """Pursuer that selects actions uniformly at random."""

    def __init__(self, role: str = "pursuer", config: dict = {}):
        super().__init__(role, config)

    def select_action(self, observation: dict) -> int:
        return random.randint(0, 3)

    def update(self, observation, action, reward, next_observation, done):
        pass   # no learning

    def reset(self):
        pass   # no per-episode state


AGENT_CLASS = RandomPursuer
