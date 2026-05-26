import numpy as np
import random
from collections import defaultdict
from agents.base_agent import BaseAgent


class MyPursuer(BaseAgent):

    def __init__(self, role="pursuer", config=None):

        super().__init__(role, config)

        self.Q = defaultdict(lambda: np.zeros(4))

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def reset(self): #Removed the reset of epsilon, if epsilon is reset, then epsion decay does not work
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

    def _encode(self, obs: dict): #This part was changed to use keywords from template

        
        # convert the template observation dict → tuple state (IMPORTANT) 
        self_row, self_col = obs["self_pos"]
        opp_row, opp_col = obs["opponent_pos"]
        goal_row, goal_col = obs["goal_pos"]
        steps_remaining = obs["steps_remaining"]

        # Keep the state compact but compatible with the base agent contract.
        return (
            self_row,
            self_col,
            opp_row,
            opp_col,
            goal_row,
            goal_col,
            steps_remaining,
        )


AGENT_CLASS = MyPursuer