import numpy as np
import random
from collections import defaultdict
from agents.base_agent import BaseAgent


class MyPursuer(BaseAgent):

    ROLE = "pursuer"
    NAME = "SARSA Pursuer"

    def __init__(self):

        super().__init__()

        self.reset()

    def reset(self):

        self.Q = defaultdict(
            lambda: np.zeros(4)
        )

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, obs):

        if random.random() < self.epsilon:
            return random.randint(0, 3)

        return int(
            np.argmax(
                self.Q[str(obs)]
            )
        )

    def on_step(
        self,
        obs,
        action,
        reward,
        next_obs,
        done
    ):

        current = self.Q[str(obs)][action]

        next_best = np.max(
            self.Q[str(next_obs)]
        )

        self.Q[str(obs)][action] += (

            self.alpha

            *

            (
                reward
                +
                self.gamma
                *
                next_best
                -
                current
            )
        )

    def on_episode_end(
        self,
        episode,
        won
    ):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


AGENT_CLASS = MyPursuer