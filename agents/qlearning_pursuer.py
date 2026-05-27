import random
import json
from collections import defaultdict
from base_agent import BaseAgent


class QLearningPursuer(BaseAgent):
    """Tabular Q-learning pursuer for the 10x10 grid."""

    def __init__(self, role: str, config: dict):
        super().__init__(role, config)
        # Hyperparameters
        # `alpha`: learning rate — how strongly new samples update Q-values (0..1)
        self.alpha = config.get("alpha", 0.1)
        # `gamma`: discount factor — importance of future rewards (0..1)
        self.gamma = config.get("gamma", 0.99)
        # `epsilon`: exploration rate — probability of taking a random action
        self.epsilon = config.get("epsilon", 0.2)
        # `epsilon_decay`: multiplicative decay applied to epsilon at episode end
        self.epsilon_decay = config.get("epsilon_decay", 0.9995)
        # `epsilon_min`: minimum exploration rate (epsilon will not go below this)
        self.epsilon_min = config.get("epsilon_min", 0.01)

        # Q-table: maps state -> [q_up,q_down,q_left,q_right]
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def select_action(self, observation: dict) -> int:
        state = self._obs_to_state(observation)
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        q_values = self.q_table[state]
        max_q = max(q_values)
        best = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best)

    def update(self, observation, action, reward, next_observation, done):
        state = self._obs_to_state(observation)
        next_state = self._obs_to_state(next_observation)

        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done else max(self.q_table[next_state])

        self.q_table[state][action] = (
            current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        )

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # no transient state stored by this agent

    def reset(self) -> None:
        # no transient state to reset
        pass

    def save(self, path: str) -> None:
        """Save Q-table in a JSON-readable format: list of [key_list, q_values]."""
        serial = [[list(k), v] for k, v in self.q_table.items()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2)

    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                d = {tuple(k): v for k, v in data}
                self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0], d)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            # file exists but isn't valid JSON; ignore
            pass

    def _obs_to_state(self, observation: dict) -> tuple:
        """State: (self_r,self_c,opp_r,opp_c)
        This keeps the state compact for the pursuer.
        """
        sr, sc = observation["self_pos"]
        or_, oc = observation["opponent_pos"]
        steps = observation.get("steps_remaining", 0)
        return (sr, sc, or_, oc, steps)

    def get_reward(
        self,
        role: str,
        prev_agent_dist: int,
        curr_agent_dist: int,
        prev_evader_goal_dist: int,
        curr_evader_goal_dist: int,
        pursuer_wins: bool,
        evader_wins: bool,
    ) -> float:
        """Pursuer-only reward (role parameter is ignored).

        Returns positive reward for capture, negative if evader reaches
        the goal, and shaping reward for getting closer to the evader.
        """
        if pursuer_wins:
            return 10.0
        if evader_wins:
            return -10.0
        # pursuer rewarded for reducing distance to the evader
        return float(prev_agent_dist - curr_agent_dist)


AGENT_CLASS = QLearningPursuer
