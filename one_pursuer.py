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

    world_size: float = 10.0
    max_steps: int = 220

    capture_radius: float = 0.60
    goal_radius: float = 0.55

    pursuer_speed: float = 0.38
    evader_speed: float = 0.25

    total_episodes: int = 3000
    rollout_episodes: int = 20
    ppo_epochs: int = 6
    mini_batch_size: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    capture_reward: float = 25.0
    evader_goal_penalty: float = 12.0
    distance_reward_scale: float = 1.2
    goal_progress_penalty_scale: float = 0.5
    step_penalty: float = 0.01

    print_every: int = 20
    checkpoint_every: int = 50
    save_dir: str = "checkpoints_single_balanced"
    eval_episodes: int = 30


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def clip_pos(p, world_size):
    return np.clip(p, 0.0, world_size)


class SinglePursuitGoalEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.world_size = cfg.world_size

        self.action_vectors = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
        ], dtype=np.float32)

        for i in range(len(self.action_vectors)):
            n = np.linalg.norm(self.action_vectors[i])
            if n > 1e-8:
                self.action_vectors[i] /= n

        self.pursuer = None
        self.evader = None
        self.goal = None
        self.step_count = 0

    def reset(self):
        self.step_count = 0

        self.pursuer = np.array([
            np.random.uniform(2.5, 4.0),
            np.random.uniform(2.5, 4.0)
        ], dtype=np.float32)

        self.evader = np.array([
            np.random.uniform(5.5, 7.0),
            np.random.uniform(4.5, 6.5)
        ], dtype=np.float32)

        self.goal = np.array([
            np.random.uniform(8.5, 9.5),
            np.random.uniform(8.5, 9.5)
        ], dtype=np.float32)

        return self._get_obs()

    def _get_obs(self):
        rel_evader = self.evader - self.pursuer
        rel_goal = self.goal - self.pursuer
        evader_to_goal = self.goal - self.evader

        obs = np.array([
            self.pursuer[0] / self.world_size,
            self.pursuer[1] / self.world_size,

            self.evader[0] / self.world_size,
            self.evader[1] / self.world_size,

            self.goal[0] / self.world_size,
            self.goal[1] / self.world_size,

            rel_evader[0] / self.world_size,
            rel_evader[1] / self.world_size,
            dist(self.pursuer, self.evader) / self.world_size,

            rel_goal[0] / self.world_size,
            rel_goal[1] / self.world_size,

            evader_to_goal[0] / self.world_size,
            evader_to_goal[1] / self.world_size,
            dist(self.evader, self.goal) / self.world_size,

            self.pursuer[0] / self.world_size,
            (self.world_size - self.pursuer[0]) / self.world_size,
            self.pursuer[1] / self.world_size,
            (self.world_size - self.pursuer[1]) / self.world_size,
        ], dtype=np.float32)

        return obs

    def _evader_policy(self):
        to_goal = unit(self.goal - self.evader)
        away_pursuer = unit(self.evader - self.pursuer)

        d = dist(self.evader, self.pursuer)

        if d < 1.5:
            move = 0.40 * to_goal + 0.60 * away_pursuer
        else:
            move = 0.70 * to_goal + 0.30 * away_pursuer

        move += np.random.uniform(-0.15, 0.15, size=(2,)).astype(np.float32)
        return unit(move)

    def step(self, action):
        self.step_count += 1

        old_pursuer_evader_dist = dist(self.pursuer, self.evader)
        old_evader_goal_dist = dist(self.evader, self.goal)

        move = self.action_vectors[action] * self.cfg.pursuer_speed
        self.pursuer = clip_pos(self.pursuer + move, self.world_size)

        if dist(self.pursuer, self.evader) <= self.cfg.capture_radius:
            return self._get_obs(), self.cfg.capture_reward, True, {
                "captured": True,
                "evader_reached_goal": False,
                "timeout": False,
            }

        evader_move = self._evader_policy() * self.cfg.evader_speed
        self.evader = clip_pos(self.evader + evader_move, self.world_size)

        new_pursuer_evader_dist = dist(self.pursuer, self.evader)
        new_evader_goal_dist = dist(self.evader, self.goal)

        captured = new_pursuer_evader_dist <= self.cfg.capture_radius
        evader_reached_goal = new_evader_goal_dist <= self.cfg.goal_radius
        timeout = self.step_count >= self.cfg.max_steps

        reward = -self.cfg.step_penalty

        reward += self.cfg.distance_reward_scale * (
            old_pursuer_evader_dist - new_pursuer_evader_dist
        )

        reward -= self.cfg.goal_progress_penalty_scale * (
            old_evader_goal_dist - new_evader_goal_dist
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


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def greedy_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t)
            return int(torch.argmax(logits, dim=-1).item())

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

        advantages, returns = self.compute_gae(buffer.rewards, buffer.dones, buffer.values)

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
                distribution = Categorical(logits=logits)

                new_log_probs = distribution.log_prob(actions[mb])
                entropy = distribution.entropy().mean()
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

    env = SinglePursuitGoalEnv(cfg)
    obs_dim = len(env.reset())
    action_dim = len(env.action_vectors)

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


def evaluate(checkpoint_path):
    cfg = Config()
    env = SinglePursuitGoalEnv(cfg)

    obs_dim = len(env.reset())
    action_dim = len(env.action_vectors)

    agent = Agent(obs_dim, action_dim, cfg)
    load_checkpoint(agent, checkpoint_path, cfg.device)

    captures = 0
    goals = 0
    timeouts = 0

    for ep in range(cfg.eval_episodes):
        obs = env.reset()
        done = False
        info = None

        while not done:
            action = agent.greedy_action(obs)
            obs, reward, done, info = env.step(action)

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
    parser.add_argument("--checkpoint", type=str, default="checkpoints_single_balanced/checkpoint_latest.pt")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.resume)
    else:
        evaluate(args.checkpoint)


if __name__ == "__main__":
    main()