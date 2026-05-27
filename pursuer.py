import random
from agents.base_agent import BaseAgent


class MyPursuer(BaseAgent):

    def __init__(self, role="pursuer", config=None):
        super().__init__(role, config)

    def reset(self):
        pass

    def select_action(self, observation: dict) -> int:
        return random.randint(0, 3)

    def update(self, observation, action, reward, next_observation, done):
        pass


AGENT_CLASS = MyPursuer