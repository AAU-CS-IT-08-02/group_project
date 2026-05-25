import random
import pickle
from collections import defaultdict

import numpy as np

try:
    from agents.base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


class MCPursuer(BaseAgent):

    INTENDED_ROLE = "pursuer"
    NAME = "Monte Carlo Pursuer"

    def __init__(self, role: str = "pursuer", config: dict = None):
        super().__init__(role, config or {})

        c = self.config
        self.gamma         = c.get("gamma",         0.95)
        self.epsilon       = c.get("epsilon",       1.0)
        self.epsilon_min   = c.get("epsilon_min",   0.01)
        self.epsilon_decay = c.get("epsilon_decay", 0.9995)
        self.num_actions   = 4

        self.capture_reward = c.get("capture_reward",  100.0)
        self.escape_penalty = c.get("escape_penalty", -100.0)
        self.step_penalty   = c.get("step_penalty",    -0.5)

        self.Q = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float64))
        self.N = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.int64))

        self._trajectory = []

    # ── Interface ─────────────────────────────────────────────

    def select_action(self, observation: dict) -> int:
        state = self._obs_to_state(observation)
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        q = self.Q[state]
        return int(np.random.choice(np.flatnonzero(q == q.max())))

    def update(self, observation, action, reward, next_observation, done):
        state = self._obs_to_state(observation)
        self._trajectory.append((state, action, reward))

        if done:
            self._mc_backup()
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

    def reset(self):
        self._trajectory = []

    # ── Custom reward function ─────────────────────────────────────────

    def get_reward(self, role, prev_agent_dist, curr_agent_dist,
                   prev_evader_goal_dist, curr_evader_goal_dist,
                   pursuer_wins, evader_wins):
        if pursuer_wins:
            return self.capture_reward
        if evader_wins:
            return self.escape_penalty
        return self.step_penalty

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "Q": dict(self.Q),
                "N": dict(self.N),
                "epsilon": self.epsilon,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self.Q = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64),
            blob["Q"],
        )
        self.N = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.int64),
            blob["N"],
        )
        self.epsilon = blob.get("epsilon", self.epsilon_min)

    # ── Helpers ────────────────────────────────────────────────────────

    def _obs_to_state(self, obs: dict) -> tuple:
        sr, sc = obs["self_pos"]
        or_, oc = obs["opponent_pos"]
        return (sr, sc, or_, oc)

    def _mc_backup(self):
        first_visit = {}
        for t, (s, a, _) in enumerate(self._trajectory):
            first_visit.setdefault((s, a), t)

        G = 0.0
        for t in reversed(range(len(self._trajectory))):
            s, a, r = self._trajectory[t]
            G = self.gamma * G + r
            if first_visit[(s, a)] == t:
                self.N[s][a] += 1
                self.Q[s][a] += (G - self.Q[s][a]) / self.N[s][a]


AGENT_CLASS = MCPursuer