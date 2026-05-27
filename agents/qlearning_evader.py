import random
import json
from collections import defaultdict #to fill in missing keys
from base_agent import BaseAgent


class QLearningEvader(BaseAgent):
    """Tabular Q-learning evader for the 10x10 grid."""

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
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        q_values = self.q_table[state]
        # choose the action with highest Q (tie-breaker: random among best)
        max_q = max(q_values)
        best = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best)

    def update(self, observation, action, reward, next_observation, done):
        state = self._obs_to_state(observation)
        next_state = self._obs_to_state(next_observation)

        # ensure entries exist (defaultdict handles this)
        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done else max(self.q_table[next_state])

        # Q-learning update
        self.q_table[state][action] = (
            current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        )

        # decay epsilon at episode end
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def reset(self) -> None:
        # no transient state to reset
        pass

    def save(self, path: str) -> None:
        """Save Q-table as labeled JSON records that are easy to inspect."""
        serial = []
        for state, q_values in self.q_table.items():
            serial.append({
                "state": self._state_to_record(state),
                "q_values": {
                    "up": q_values[0],
                    "down": q_values[1],
                    "left": q_values[2],
                    "right": q_values[3],
                },
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2)

    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                d = {}
                if isinstance(data, list):
                    for entry in data:
                        # new readable format
                        if isinstance(entry, dict) and "state" in entry and "q_values" in entry:
                            state = self._record_to_state(entry["state"])
                            qv = entry["q_values"]
                            d[state] = [
                                float(qv.get("up", 0.0)),
                                float(qv.get("down", 0.0)),
                                float(qv.get("left", 0.0)),
                                float(qv.get("right", 0.0)),
                            ]
                        # older list format: [state_list, q_values_list]
                        elif isinstance(entry, list) and len(entry) == 2:
                            d[tuple(entry[0])] = list(entry[1])
                self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0], d)
        except FileNotFoundError:
            # nothing to load
            pass
        except json.JSONDecodeError:
            # file exists but isn't JSON (keep previous behavior of ignoring)
            pass

    def _obs_to_state(self, observation: dict) -> tuple:
        """Simple state: (self_r,self_c,opp_r,opp_c, goal_r,goal_c)
        """
        sr, sc = observation["self_pos"]
        or_, oc = observation["opponent_pos"]
        gr, gc = observation.get("goal_pos", (None, None))
        # include steps remaining so reward functions can access time info
        steps = observation.get("steps_remaining", 0)
        return (sr, sc, or_, oc, gr, gc, steps)

    def _state_to_record(self, state: tuple) -> dict:
        sr, sc, or_, oc, gr, gc, steps = state
        return {
            "self_pos": [sr, sc],
            "opponent_pos": [or_, oc],
            "goal_pos": [gr, gc],
            "steps_remaining": steps,
        }

    def _record_to_state(self, record: dict) -> tuple:
        self_pos = record.get("self_pos", [None, None])
        opponent_pos = record.get("opponent_pos", [None, None])
        goal_pos = record.get("goal_pos", [None, None])
        steps = record.get("steps_remaining", 0)
        return (
            self_pos[0],
            self_pos[1],
            opponent_pos[0],
            opponent_pos[1],
            goal_pos[0],
            goal_pos[1],
            steps,
        )

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
        # Evader-only reward (runner still passes role but it's ignored here)
        if evader_wins:
            return 10.0
        if pursuer_wins:
            return -10.0
        # reward for increasing distance from pursuer and closing on goal
        escape = float(curr_agent_dist - prev_agent_dist)
        goal = float(prev_evader_goal_dist - curr_evader_goal_dist)
        return escape + 0.3 * goal


AGENT_CLASS = QLearningEvader
