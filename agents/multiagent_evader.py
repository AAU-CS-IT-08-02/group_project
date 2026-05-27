import numpy as np
import random
from collections import defaultdict
from agents.base_agent import BaseAgent


class MyEvader(BaseAgent):

    def __init__(self, role="evader", config=None):
        super().__init__(role, config)

        self.Q = defaultdict(lambda: np.zeros(4))

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def reset(self):
        pass

    def select_action(self, observation: dict):

        state = self._encode(observation)

        if random.random() < self.epsilon:
            return random.randint(0, 3)

        return int(np.argmax(self.Q[state]))

    def update(self, observation, action, reward, next_observation, done):

        s = self._encode(observation)
        ns = self._encode(next_observation)

        best_next = np.max(self.Q[ns])

        self.Q[s][action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[s][action]
        )

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _encode(self, obs: dict):

        return (
            obs["self_pos"][0],
            obs["self_pos"][1],
            obs["opponent_pos"][0],
            obs["opponent_pos"][1],
            obs["goal_pos"][0],
            obs["goal_pos"][1],
            obs["steps_remaining"]
        )


AGENT_CLASS = MyEvader