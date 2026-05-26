import numpy as np
import random
from collections import defaultdict
from agents.base_agent import BaseAgent


# Self-contained helper logic

GRID_SIZE = 10

ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT"
]

NUM_ACTIONS = len(ACTIONS)


def move(pos, action):

    x, y = pos

    if ACTIONS[action] == "UP":
        x -= 1

    elif ACTIONS[action] == "DOWN":
        x += 1

    elif ACTIONS[action] == "LEFT":
        y -= 1

    elif ACTIONS[action] == "RIGHT":
        y += 1

    x = max(0, min(GRID_SIZE - 1, x))
    y = max(0, min(GRID_SIZE - 1, y))

    return (x, y)


def get_distance(a, b):

    return abs(a[0] - b[0]) + abs(
        a[1] - b[1]
    )


# Agent Implementation

class MyPursuer(BaseAgent):

    ROLE = "pursuer"
    NAME = "SARSA Pursuer"

    def __init__(self):

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.Q = defaultdict(
            lambda:
            np.zeros(NUM_ACTIONS)
        )

    # REQUIRED by BaseAgent
    def reset(self):

        self.Q = defaultdict(
            lambda:
            np.zeros(NUM_ACTIONS)
        )

        self.epsilon = 1.0

    def select_action(self, obs):

        if random.random() < self.epsilon:

            return random.randint(
                0,
                NUM_ACTIONS - 1
            )

        return int(
            np.argmax(
                self.Q[obs]
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

        current = (
            self.Q[obs][action]
        )

        next_best = np.max(
            self.Q[next_obs]
        )

        target = (

            reward

            +

            self.gamma

            *

            next_best
        )

        self.Q[obs][action] += (

            self.alpha

            *

            (
                target
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

            self.epsilon

            *

            self.epsilon_decay
        )


AGENT_CLASS = MyPursuer