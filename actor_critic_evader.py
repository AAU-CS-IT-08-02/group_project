# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

try:
    from agents.base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        x = self.shared(obs)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


class ActorCriticEvader(BaseAgent):
    """
    The evader learns to reach the goal while avoiding capture.
    """

    def __init__(self, role: str, config: dict):
        super().__init__(role, config)

        self.gamma = float(config.get("gamma", 0.99))
        self.lr = float(config.get("lr", 0.0003))
        self.entropy_coef = float(config.get("entropy_coef", 0.003))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu"
        )

        self.obs_dim = 11
        self.action_dim = 4

        self.model = ActorCriticNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.training = bool(config.get("training", True))

    def _obs_to_vector(self, observation: dict) -> np.ndarray:
        sr, sc = observation["self_pos"]
        orow, ocol = observation["opponent_pos"]
        gr, gc = observation["goal_pos"]
        grid_size = observation["grid_size"]
        steps_remaining = observation["steps_remaining"]

        denom = max(grid_size - 1, 1)

        agent_dist = abs(sr - orow) + abs(sc - ocol)
        goal_dist = abs(sr - gr) + abs(sc - gc)

        vec = np.array([
            sr / denom,
            sc / denom,

            orow / denom,
            ocol / denom,

            gr / denom,
            gc / denom,

            (orow - sr) / denom,
            (ocol - sc) / denom,

            (gr - sr) / denom,
            (gc - sc) / denom,

            steps_remaining / 200.0,
        ], dtype=np.float32)

        return vec

    def select_action(self, observation: dict) -> int:
        obs_vec = self._obs_to_vector(observation)
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)

            if self.training:
                explore_prob = max(0.005, 0.05 * (0.9995 ** getattr(self, "step_counter", 0)))
                self.step_counter = getattr(self, "step_counter", 0) + 1

                if random.random() < explore_prob:
                    action = dist.sample()
                else:
                    action = torch.argmax(logits, dim=-1)
            else:
                action = torch.argmax(logits, dim=-1)

        return int(action.item())

    def update(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
    ) -> None:
        if not self.training:
            return

        obs_vec = self._obs_to_vector(observation)
        next_obs_vec = self._obs_to_vector(next_observation)

        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.long, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

        logits, value = self.model(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()

        with torch.no_grad():
            _, next_value = self.model(next_obs_t)
            target = reward_t + self.gamma * next_value.squeeze(-1) * (1.0 - done_t)

        value = value.squeeze(-1)
        advantage = target - value

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy.mean()
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def reset(self) -> None:
        pass

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
        """
        Custom reward for the evader.
        """
        if role != "evader":
            raise NotImplementedError

        if evader_wins:
            return 35.0

        if pursuer_wins:
            return -25.0

        reward = -0.01

        # Good for evader: increase distance from pursuer
        reward += 1.0 * float(curr_agent_dist - prev_agent_dist)

        # Good for evader: move closer to goal
        reward += 1.2 * float(prev_evader_goal_dist - curr_evader_goal_dist)

        return reward

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])


AGENT_CLASS = ActorCriticEvader