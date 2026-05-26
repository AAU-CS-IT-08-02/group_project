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


# Evader Agent

class MyEvader(BaseAgent):

    ROLE = "evader"
    NAME = "Random Evader"

    def __init__(self):

        self.epsilon = 1.0

        self.Q = defaultdict(
            lambda:
            np.zeros(NUM_ACTIONS)
        )

    # REQUIRED by BaseAgent
    def reset(self):

        self.epsilon = 1.0

        self.Q = defaultdict(
            lambda:
            np.zeros(NUM_ACTIONS)
        )

    def select_action(
        self,
        obs
    ):

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

        pass

    def on_episode_end(
        self,
        episode,
        won
    ):

        self.epsilon = max(
            0.01,
            self.epsilon * 0.995
        )


AGENT_CLASS = MyEvader