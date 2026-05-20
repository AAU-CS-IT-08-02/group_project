# -*- coding: utf-8 -*-

import os
import random
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    grid_size: int = 10
    max_steps: int = 100

    total_episodes: int = 3000
    rollout_episodes: int = 20
    ppo_epochs: int = 6
    mini_batch_size: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.025
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    capture_reward: float = 35.0
    evader_goal_penalty: float = 18.0
    distance_reward_scale: float = 1.8
    goal_progress_penalty_scale: float = 0.8
    step_penalty: float = 0.01

    evader_random_prob: float = 0.25

    print_every: int = 20
    checkpoint_every: int = 50
    save_dir: str = "checkpoints_grid_ac"
    eval_episodes: int = 50


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GridPursuitEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n = cfg.grid_size

        self.actions = [
            (0, 0),    # stay
            (-1, 0),   # up
            (1, 0),    # down
            (0, -1),   # left
            (0, 1),    # right
        ]

        self.pursuer = None
        self.evader = None
        self.goal = None
        self.step_count = 0

    def reset(self):
        self.step_count = 0

        self.pursuer = [
            random.randint(3, 5),
            random.randint(3, 5)
        ]

        self.evader = [
            random.randint(5, 7),
            random.randint(4, 6)
        ]

        self.goal = [
            random.randint(8, 9),
            random.randint(8, 9)
        ]

        while self.goal == self.evader or self.goal == self.pursuer:
            self.goal = [
                random.randint(8, 9),
                random.randint(8, 9)
            ]

        return self._get_obs()

    def _get_obs(self):
        px, py = self.pursuer
        ex, ey = self.evader
        gx, gy = self.goal
        denom = self.n - 1

        obs = np.array([
            px / denom,
            py / denom,

            ex / denom,
            ey / denom,

            gx / denom,
            gy / denom,

            (ex - px) / denom,
            (ey - py) / denom,
            manhattan(self.pursuer, self.evader) / (2 * denom),

            (gx - px) / denom,
            (gy - py) / denom,

            (gx - ex) / denom,
            (gy - ey) / denom,
            manhattan(self.evader, self.goal) / (2 * denom),
        ], dtype=np.float32)

        return obs

    def _move(self, pos, action):
        dx, dy = self.actions[action]
        nx = min(max(pos[0] + dx, 0), self.n - 1)
        ny = min(max(pos[1] + dy, 0), self.n - 1)
        return [nx, ny]

    def _evader_policy(self):
        if random.random() < self.cfg.evader_random_prob:
            return random.randint(0, len(self.actions) - 1)

        best_action = 0
        best_score = -1e9

        for a in range(len(self.actions)):
            new_pos = self._move(self.evader, a)

            dist_goal = manhattan(new_pos, self.goal)
            dist_pursuer = manhattan(new_pos, self.pursuer)

            # goal seeking + avoidance
            score = -0.80 * dist_goal + 0.20 * dist_pursuer

            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def step(self, pursuer_action):
        self.step_count += 1

        old_dist_evader = manhattan(self.pursuer, self.evader)
        old_dist_goal = manhattan(self.evader, self.goal)

        self.pursuer = self._move(self.pursuer, pursuer_action)

        if self.pursuer == self.evader:
            return self._get_obs(), self.cfg.capture_reward, True, {
                "captured": True,
                "evader_reached_goal": False,
                "timeout": False,
            }

        evader_action = self._evader_policy()
        self.evader = self._move(self.evader, evader_action)

        new_dist_evader = manhattan(self.pursuer, self.evader)
        new_dist_goal = manhattan(self.evader, self.goal)

        captured = self.pursuer == self.evader
        evader_reached_goal = self.evader == self.goal
        timeout = self.step_count >= self.cfg.max_steps

        reward = -self.cfg.step_penalty

        reward += self.cfg.distance_reward_scale * (
            old_dist_evader - new_dist_evader
        )

        reward -= self.cfg.goal_progress_penalty_scale * (
            old_dist_goal - new_dist_goal
        )

        if captured:
            reward += self.cfg.capture_reward

        if evader_reached_goal:
            reward -= self.cfg.evader_goal_penalty

        done = captured or evader_reached_goal or timeout

        return self._get_obs(), reward, done, {
            "captured": captured,
            "evader_reached_goal": evader_reached_goal,
            "timeout": timeout,
        }

    def render(self):
        grid = [["." for _ in range(self.n)] for _ in range(self.n)]

        gx, gy = self.goal
        ex, ey = self.evader
        px, py = self.pursuer

        grid[gx][gy] = "G"
        grid[ex][ey] = "E"
        grid[px][py] = "P"

        print()
        for row in grid:
            print(" ".join(row))
        print()


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class Buffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []


class Agent:
    def __init__(self, obs_dim, action_dim, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(obs_t)
            value = self.critic(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def greedy_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(obs_t)
            action = torch.argmax(logits, dim=-1)

        return int(action.item())

    def compute_gae(self, rewards, dones, values):
        advantages = []
        gae = 0.0
        values = values + [0.0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, buffer):
        obs = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(buffer.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=self.device)

        advantages, returns = self.compute_gae(
            buffer.rewards,
            buffer.dones,
            buffer.values
        )

        advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=self.device)
        returns = torch.tensor(np.array(returns), dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.size(0)
        idxs = np.arange(n)

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idxs)

            for start in range(0, n, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb = idxs[start:end]

                logits = self.actor(obs[mb])
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()
                values_pred = self.critic(obs[mb]).squeeze(-1)

                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_eps,
                    1.0 + self.cfg.clip_eps
                ) * advantages[mb]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns[mb] - values_pred).pow(2).mean()

                loss = (
                    actor_loss
                    + self.cfg.value_coef * critic_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)

                self.actor_optim.step()
                self.critic_optim.step()


def save_checkpoint(agent, cfg, episode, filename):
    os.makedirs(cfg.save_dir, exist_ok=True)
    path = os.path.join(cfg.save_dir, filename)

    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "actor_optim": agent.actor_optim.state_dict(),
        "critic_optim": agent.critic_optim.state_dict(),
        "episode": episode,
        "config": asdict(cfg),
    }, path)


def load_checkpoint(agent, path, device):
    checkpoint = torch.load(path, map_location=device)

    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])

    if "actor_optim" in checkpoint:
        agent.actor_optim.load_state_dict(checkpoint["actor_optim"])

    if "critic_optim" in checkpoint:
        agent.critic_optim.load_state_dict(checkpoint["critic_optim"])

    return int(checkpoint.get("episode", 0))


def train(resume_path=None):
    cfg = Config()
    set_seed(cfg.seed)

    env = GridPursuitEnv(cfg)
    obs_dim = len(env.reset())
    action_dim = len(env.actions)

    agent = Agent(obs_dim, action_dim, cfg)

    start_episode = 1
    if resume_path is not None and os.path.exists(resume_path):
        ep = load_checkpoint(agent, resume_path, cfg.device)
        start_episode = ep + 1
        print(f"Resumed from episode {ep}")

    reward_history = []
    capture_history = []
    goal_history = []
    timeout_history = []

    print("Device:", cfg.device)

    for episode in range(start_episode, cfg.total_episodes + 1):
        buffer = Buffer()

        for _ in range(cfg.rollout_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            final_info = None

            while not done:
                action, logp, value = agent.select_action(obs)

                next_obs, reward, done, info = env.step(action)

                buffer.obs.append(obs)
                buffer.actions.append(action)
                buffer.log_probs.append(logp)
                buffer.rewards.append(reward)
                buffer.dones.append(float(done))
                buffer.values.append(value)

                ep_reward += reward
                obs = next_obs
                final_info = info

            reward_history.append(ep_reward)
            capture_history.append(1.0 if final_info["captured"] else 0.0)
            goal_history.append(1.0 if final_info["evader_reached_goal"] else 0.0)
            timeout_history.append(1.0 if final_info["timeout"] else 0.0)

        agent.update(buffer)

        if episode % cfg.checkpoint_every == 0:
            save_checkpoint(agent, cfg, episode, "checkpoint_latest.pt")

        if episode % cfg.print_every == 0:
            window = cfg.print_every * cfg.rollout_episodes
            print(
                f"Episode {episode:4d} | "
                f"avg_reward={np.mean(reward_history[-window:]):8.3f} | "
                f"capture_rate={np.mean(capture_history[-window:]):6.3f} | "
                f"evader_goal_rate={np.mean(goal_history[-window:]):6.3f} | "
                f"timeout_rate={np.mean(timeout_history[-window:]):6.3f}"
            )

    save_checkpoint(agent, cfg, cfg.total_episodes, "final_model.pt")
    print("Training finished.")


def evaluate(checkpoint_path, render=False):
    cfg = Config()
    env = GridPursuitEnv(cfg)

    obs_dim = len(env.reset())
    action_dim = len(env.actions)

    agent = Agent(obs_dim, action_dim, cfg)
    load_checkpoint(agent, checkpoint_path, cfg.device)

    captures = 0
    goals = 0
    timeouts = 0

    for ep in range(cfg.eval_episodes):
        obs = env.reset()
        done = False
        info = None

        if render:
            print(f"\nEpisode {ep + 1}")
            env.render()

        while not done:
            action = agent.greedy_action(obs)
            obs, reward, done, info = env.step(action)

            if render:
                env.render()

        if info["captured"]:
            captures += 1
        elif info["evader_reached_goal"]:
            goals += 1
        else:
            timeouts += 1

    print("Evaluation")
    print(f"Capture rate     : {captures / cfg.eval_episodes:.3f}")
    print(f"Evader goal rate : {goals / cfg.eval_episodes:.3f}")
    print(f"Timeout rate     : {timeouts / cfg.eval_episodes:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints_grid_ac/checkpoint_latest.pt")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.resume)
    else:
        evaluate(args.checkpoint, render=args.render)


if __name__ == "__main__":
    main()