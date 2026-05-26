import numpy as np
import random
from agents.base_agent import BaseAgent


class MyEvader(BaseAgent):

    ROLE = "evader"
    NAME = "Random Evader"

    def __init__(self):

        super().__init__()

        self.reset()

    def reset(self):

        self.epsilon = 1.0

    def select_action(
        self,
        obs
    ):

        return random.randint(
            0,
            3
        )

    def on_step(
        self,
        obs,
        action,
        reward,
        next_obs,
        done
    ):
        pass

    def on_episode_end(
        self,
        episode,
        won
    ):
        pass


AGENT_CLASS = MyEvader