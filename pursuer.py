import numpy as np
import random
from collections import defaultdict

from env import (
    NUM_ACTIONS
)


class Pursuer:

    def __init__(

        self,

        alpha=0.1,

        gamma=0.95,

        epsilon=1.0,

        epsilon_decay=0.995,

        epsilon_min=0.01

    ):

        self.Q = defaultdict(
            lambda:
            np.zeros(
                NUM_ACTIONS
            )
        )

        self.alpha = alpha

        self.gamma = gamma

        self.epsilon = epsilon

        self.epsilon_decay = (
            epsilon_decay
        )

        self.epsilon_min = (
            epsilon_min
        )

    def choose_action(
        self,
        state
    ):

        if (
            random.random()
            <
            self.epsilon
        ):

            return random.randint(
                0,
                NUM_ACTIONS - 1
            )

        return np.argmax(
            self.Q[state]
        )

    def update(

        self,

        state,

        action,

        reward,

        next_state,

        next_action

    ):

        current = (
            self.Q[state][action]
        )

        target = (

            reward

            +

            self.gamma

            *

            self.Q[
                next_state
            ][
                next_action
            ]
        )

        self.Q[state][action] += (

            self.alpha

            *

            (

                target

                -

                current

            )
        )

    def decay(self):

        self.epsilon = max(

            self.epsilon_min,

            self.epsilon

            *

            self.epsilon_decay
        )