# -*- coding: utf-8 -*-
import os
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment
    num_pursuers: int = 3
    world_size: float = 10.0
    max_steps: int = 180

    capture_radius: float = 0.55
    goal_radius: float = 0.50

    pursuer_speed: float = 0.32
    evader_speed: float = 0.30

    # PPO
    total_episodes: int = 6000
    rollout_episodes: int = 20
    ppo_epochs: int = 8
    mini_batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Rewards for pursuers
    capture_reward: float = 20.0
    evader_goal_penalty: float = 18.0
    distance_reward_scale: float = 0.8
    goal_progress_penalty_scale: float = 1.2
    surround_reward_scale: float = 0.15
    intercept_reward_scale: float = 0.12
    collision_penalty: float = 0.25
    step_penalty: float = 0.01

    # Logging / saving
    print_every: int = 20
    checkpoint_every: int = 50
    save_dir: str = "checkpoints"

    # Evaluation
    eval_episodes: int = 50


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clip_position(pos: np.ndarray, world_size: float) -> np.ndarray:
    return np.clip(pos, 0.0, world_size)


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


# ============================================================
# ENVIRONMENT
# ============================================================

class PursuitEvasionGoalEnv:
    """
    2D continuous world.
    - 3 pursuera
    - 1 evader
    - evader ima GOAL
    - pursueri moraju uhvatit evadera prije goal-a
    - pursuer akcije su diskretne
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_pursuers = cfg.num_pursuers
        self.world_size = cfg.world_size
        self.max_steps = cfg.max_steps
        self.capture_radius = cfg.capture_radius
        self.goal_radius = cfg.goal_radius
        self.pursuer_speed = cfg.pursuer_speed
        self.evader_speed = cfg.evader_speed

        self.action_vectors = np.array(
            [
                [0.0, 0.0],    # stay
                [0.0, 1.0],    # up
                [0.0, -1.0],   # down
                [-1.0, 0.0],   # left
                [1.0, 0.0],    # right
                [-1.0, 1.0],   # up-left
                [1.0, 1.0],    # up-right
                [-1.0, -1.0],  # down-left
                [1.0, -1.0],   # down-right
            ],
            dtype=np.float32
        )

        for i in range(len(self.action_vectors)):
            n = np.linalg.norm(self.action_vectors[i])
            if n > 1e-8:
                self.action_vectors[i] /= n

        self.pursuers: List[np.ndarray] = []
        self.evader: np.ndarray = None
        self.goal: np.ndarray = None
        self.step_count: int = 0

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        self.step_count = 0

        # Pursueri spawnaju više lijevo/dolje
        self.pursuers = []
        for _ in range(self.num_pursuers):
            pos = np.array(
                [
                    np.random.uniform(0.5, 3.0),
                    np.random.uniform(0.5, 3.0)
                ],
                dtype=np.float32
            )
            self.pursuers.append(pos)

        # Evader negdi u srednje-desnom dijelu
        self.evader = np.array(
            [
                np.random.uniform(6.5, 8.5),
                np.random.uniform(4.0, 6.5)
            ],
            dtype=np.float32
        )

        # Goal gore-livo ili gore-desno, da ima smisla putanja
        if np.random.rand() < 0.5:
            self.goal = np.array(
                [
                    np.random.uniform(0.8, 2.0),
                    np.random.uniform(8.0, 9.2)
                ],
                dtype=np.float32
            )
        else:
            self.goal = np.array(
                [
                    np.random.uniform(8.0, 9.2),
                    np.random.uniform(8.0, 9.2)
                ],
                dtype=np.float32
            )

        return self._get_obs()

    def _get_obs(self) -> Tuple[List[np.ndarray], np.ndarray]:
        pursuer_obs = []

        for i in range(self.num_pursuers):
            self_pos = self.pursuers[i]
            rel_evader = self.evader - self_pos
            rel_goal = self.goal - self_pos
            evader_to_goal = self.goal - self.evader

            teammates = []
            for j in range(self.num_pursuers):
                if i == j:
                    continue
                rel_tm = self.pursuers[j] - self_pos
                teammates.extend(rel_tm.tolist())

            agent_id = [0.0] * self.num_pursuers
            agent_id[i] = 1.0

            obs = np.array(
                [
                    self_pos[0] / self.world_size,
                    self_pos[1] / self.world_size,

                    rel_evader[0] / self.world_size,
                    rel_evader[1] / self.world_size,
                    pairwise_dist(self_pos, self.evader) / self.world_size,

                    rel_goal[0] / self.world_size,
                    rel_goal[1] / self.world_size,
                    pairwise_dist(self_pos, self.goal) / self.world_size,

                    evader_to_goal[0] / self.world_size,
                    evader_to_goal[1] / self.world_size,
                    pairwise_dist(self.evader, self.goal) / self.world_size,

                    *[x / self.world_size for x in teammates],

                    self_pos[0] / self.world_size,                       # left wall
                    (self.world_size - self_pos[0]) / self.world_size,  # right wall
                    self_pos[1] / self.world_size,                       # bottom
                    (self.world_size - self_pos[1]) / self.world_size,  # top

                    *agent_id
                ],
                dtype=np.float32
            )
            pursuer_obs.append(obs)

        global_state = []
        for p in self.pursuers:
            global_state.extend((p / self.world_size).tolist())

        global_state.extend((self.evader / self.world_size).tolist())
        global_state.extend((self.goal / self.world_size).tolist())
        global_state = np.array(global_state, dtype=np.float32)

        return pursuer_obs, global_state

    def _evader_policy(self) -> np.ndarray:
        """
        Jači heuristic evader:
        - ide prema goal-u
        - bježi od najbližeg pursuera
        - malo bježi od centroida pursuera
        - malo zig-zag šuma
        """
        pursuer_arr = np.stack(self.pursuers, axis=0)
        dists = np.linalg.norm(pursuer_arr - self.evader, axis=1)
        nearest_idx = np.argmin(dists)
        nearest = pursuer_arr[nearest_idx]
        centroid = np.mean(pursuer_arr, axis=0)

        to_goal = unit_vector(self.goal - self.evader)
        away_nearest = unit_vector(self.evader - nearest)
        away_centroid = unit_vector(self.evader - centroid)

        # Ako je pursuer vrlo blizu, više prioriteta na izbjegavanje
        nearest_dist = float(dists[nearest_idx])

        if nearest_dist < 1.5:
            w_goal = 0.40
            w_nearest = 0.45
            w_centroid = 0.15
        else:
            w_goal = 0.70
            w_nearest = 0.20
            w_centroid = 0.10

        noise = np.random.uniform(-0.15, 0.15, size=(2,)).astype(np.float32)

        move = (
            w_goal * to_goal
            + w_nearest * away_nearest
            + w_centroid * away_centroid
            + noise
        )

        move = unit_vector(move)
        if np.linalg.norm(move) < 1e-8:
            move = unit_vector(np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32))
        return move.astype(np.float32)

    def _compute_surround_score(self) -> float:
        angles = []
        for p in self.pursuers:
            vec = p - self.evader
            angle = math.atan2(vec[1], vec[0])
            angles.append(angle)

        angles = sorted(angles)
        if len(angles) < 3:
            return 0.0

        gaps = []
        for i in range(len(angles)):
            cur = angles[i]
            nxt = angles[(i + 1) % len(angles)]
            gap = nxt - cur
            if gap < 0:
                gap += 2 * math.pi
            gaps.append(gap)

        max_gap = max(gaps)
        surround_score = 1.0 - (max_gap / (2 * math.pi))
        return float(max(0.0, surround_score))

    def _compute_intercept_score(self) -> float:
        """
        Nagrada ako su pursueri blizu linije evader-goal.
        """
        ev = self.evader
        gl = self.goal
        line_vec = gl - ev
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-8:
            return 0.0

        score = 0.0
        for p in self.pursuers:
            pvec = p - ev
            # projekcija na line
            t = np.dot(pvec, line_vec) / (line_len ** 2)
            # zanima nas segment između evadera i goal-a
            if 0.0 <= t <= 1.0:
                proj = ev + t * line_vec
                dist_to_line = np.linalg.norm(p - proj)
                score += math.exp(-dist_to_line)

        return float(score / self.num_pursuers)

    def step(self, pursuer_actions: List[int]) -> Tuple[List[np.ndarray], np.ndarray, List[float], bool, Dict]:
        self.step_count += 1

        old_dist_sum = sum(pairwise_dist(p, self.evader) for p in self.pursuers)
        old_evader_goal_dist = pairwise_dist(self.evader, self.goal)

        # move pursuers
        for i, a in enumerate(pursuer_actions):
            move = self.action_vectors[a] * self.pursuer_speed
            self.pursuers[i] = clip_position(self.pursuers[i] + move, self.world_size)

        # move evader
        evader_move = self._evader_policy() * self.evader_speed
        self.evader = clip_position(self.evader + evader_move, self.world_size)

        new_dist_sum = sum(pairwise_dist(p, self.evader) for p in self.pursuers)
        new_evader_goal_dist = pairwise_dist(self.evader, self.goal)

        captured = any(pairwise_dist(p, self.evader) <= self.capture_radius for p in self.pursuers)
        evader_reached_goal = new_evader_goal_dist <= self.goal_radius

        rewards = [-self.cfg.step_penalty for _ in range(self.num_pursuers)]

        # pursuerima dobro kad smanje total dist do evadera
        dist_progress = old_dist_sum - new_dist_sum
        for i in range(self.num_pursuers):
            rewards[i] += self.cfg.distance_reward_scale * dist_progress

        # pursuerima loše kad evader napreduje prema goal-u
        goal_progress = old_evader_goal_dist - new_evader_goal_dist
        for i in range(self.num_pursuers):
            rewards[i] -= self.cfg.goal_progress_penalty_scale * goal_progress

        # surround bonus
        surround_score = self._compute_surround_score()
        for i in range(self.num_pursuers):
            rewards[i] += self.cfg.surround_reward_scale * surround_score

        # intercept bonus
        intercept_score = self._compute_intercept_score()
        for i in range(self.num_pursuers):
            rewards[i] += self.cfg.intercept_reward_scale * intercept_score

        # collision penalty
        for i in range(self.num_pursuers):
            for j in range(i + 1, self.num_pursuers):
                if pairwise_dist(self.pursuers[i], self.pursuers[j]) < 0.35:
                    rewards[i] -= self.cfg.collision_penalty
                    rewards[j] -= self.cfg.collision_penalty

        done = False

        if captured:
            rewards = [r + self.cfg.capture_reward for r in rewards]
            done = True
        elif evader_reached_goal:
            rewards = [r - self.cfg.evader_goal_penalty for r in rewards]
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        next_obs, global_state = self._get_obs()
        info = {
            "captured": captured,
            "evader_reached_goal": evader_reached_goal,
            "evader_pos": self.evader.copy(),
            "goal_pos": self.goal.copy(),
            "pursuer_positions": [p.copy() for p in self.pursuers],
            "surround_score": surround_score,
            "intercept_score": intercept_score,
        }
        return next_obs, global_state, rewards, done, info


# ============================================================
# NETWORKS
# ============================================================

class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ============================================================
# ROLLOUT BUFFER
# ============================================================

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.obs.clear()
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


# ============================================================
# AGENT
# ============================================================

class MAPPOAgent:
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = SharedActor(obs_dim, action_dim).to(self.device)
        self.critic = CentralCritic(state_dim).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def select_action(self, obs: np.ndarray, state: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_t)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def act_greedy(self, obs: np.ndarray) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t)
            action = torch.argmax(logits, dim=-1)
        return int(action.item())

    def evaluate_actions(self, obs_b, state_b, action_b):
        logits = self.actor(obs_b)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_b)
        entropy = dist.entropy()
        values = self.critic(state_b).squeeze(-1)
        return log_probs, entropy, values

    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0.0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, buffer: RolloutBuffer):
        obs = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=self.device)
        states = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(buffer.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=self.device)

        rewards = buffer.rewards
        dones = buffer.dones
        values = buffer.values

        next_value = 0.0
        advantages, returns = self.compute_gae(rewards, dones, values, next_value)

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

                mb_obs = obs[mb]
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_advantages = advantages[mb]
                mb_returns = returns[mb]

                new_log_probs, entropy, values_pred = self.evaluate_actions(
                    mb_obs, mb_states, mb_actions
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = ((mb_returns - values_pred) ** 2).mean()

                loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy.mean()

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()


# ============================================================
# CHECKPOINTS
# ============================================================

def save_checkpoint(agent: MAPPOAgent, cfg: Config, episode: int, filename: str) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    path = os.path.join(cfg.save_dir, filename)
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "actor_optim": agent.actor_optim.state_dict(),
            "critic_optim": agent.critic_optim.state_dict(),
            "episode": episode,
            "config": asdict(cfg),
        },
        path
    )


def load_checkpoint(agent: MAPPOAgent, path: str, device: str) -> int:
    checkpoint = torch.load(path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])

    if "actor_optim" in checkpoint:
        agent.actor_optim.load_state_dict(checkpoint["actor_optim"])
    if "critic_optim" in checkpoint:
        agent.critic_optim.load_state_dict(checkpoint["critic_optim"])

    return int(checkpoint.get("episode", 0))


# ============================================================
# TRAIN
# ============================================================

def train(resume_path: str = None):
    cfg = Config()
    set_seed(cfg.seed)

    env = PursuitEvasionGoalEnv(cfg)
    sample_obs, sample_state = env.reset()
    obs_dim = len(sample_obs[0])
    state_dim = len(sample_state)
    action_dim = len(env.action_vectors)

    agent = MAPPOAgent(obs_dim, state_dim, action_dim, cfg)

    start_episode = 1
    if resume_path is not None and os.path.exists(resume_path):
        loaded_ep = load_checkpoint(agent, resume_path, cfg.device)
        start_episode = loaded_ep + 1
        print(f"Resumed from checkpoint: {resume_path}, starting at episode {start_episode}")

    episode_rewards = []
    capture_history = []
    evader_goal_history = []

    for episode in range(start_episode, cfg.total_episodes + 1):
        rollout = RolloutBuffer()

        for _ in range(cfg.rollout_episodes):
            pursuer_obs, global_state = env.reset()
            done = False
            ep_reward_sum = 0.0
            captured = False
            evader_reached_goal = False

            while not done:
                actions = []
                action_log_probs = []
                state_values = []

                for i in range(cfg.num_pursuers):
                    a, logp, val = agent.select_action(pursuer_obs[i], global_state)
                    actions.append(a)
                    action_log_probs.append(logp)
                    state_values.append(val)

                next_obs, next_state, rewards, done, info = env.step(actions)

                # Team reward
                team_reward = float(np.mean(rewards))
                team_value = float(np.mean(state_values))

                for i in range(cfg.num_pursuers):
                    rollout.obs.append(pursuer_obs[i])
                    rollout.states.append(global_state)
                    rollout.actions.append(actions[i])
                    rollout.log_probs.append(action_log_probs[i])
                    rollout.rewards.append(team_reward)
                    rollout.dones.append(float(done))
                    rollout.values.append(team_value)

                ep_reward_sum += team_reward
                pursuer_obs = next_obs
                global_state = next_state
                captured = info["captured"]
                evader_reached_goal = info["evader_reached_goal"]

            episode_rewards.append(ep_reward_sum)
            capture_history.append(1.0 if captured else 0.0)
            evader_goal_history.append(1.0 if evader_reached_goal else 0.0)

        agent.update(rollout)

        if episode % cfg.checkpoint_every == 0:
            save_checkpoint(agent, cfg, episode, "checkpoint_latest.pt")
            save_checkpoint(agent, cfg, episode, f"checkpoint_ep{episode}.pt")

        if episode % cfg.print_every == 0:
            avg_reward = np.mean(episode_rewards[-cfg.print_every * cfg.rollout_episodes:])
            avg_capture = np.mean(capture_history[-cfg.print_every * cfg.rollout_episodes:])
            avg_goal = np.mean(evader_goal_history[-cfg.print_every * cfg.rollout_episodes:])
            print(
                f"Episode {episode:4d} | "
                f"avg_reward={avg_reward:8.3f} | "
                f"capture_rate={avg_capture:6.3f} | "
                f"evader_goal_rate={avg_goal:6.3f}"
            )

    save_checkpoint(agent, cfg, cfg.total_episodes, "final_model.pt")
    print("Training finished. Final model saved.")


# ============================================================
# EVALUATE
# ============================================================

def evaluate(checkpoint_path: str):
    cfg = Config()
    set_seed(cfg.seed)

    env = PursuitEvasionGoalEnv(cfg)
    sample_obs, sample_state = env.reset()
    obs_dim = len(sample_obs[0])
    state_dim = len(sample_state)
    action_dim = len(env.action_vectors)

    agent = MAPPOAgent(obs_dim, state_dim, action_dim, cfg)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint(agent, checkpoint_path, cfg.device)

    captures = 0
    evader_goals = 0
    timeouts = 0
    lengths = []

    for ep in range(cfg.eval_episodes):
        pursuer_obs, global_state = env.reset()
        done = False
        steps = 0
        info = None

        while not done:
            actions = [agent.act_greedy(pursuer_obs[i]) for i in range(cfg.num_pursuers)]
            next_obs, next_state, rewards, done, info = env.step(actions)
            pursuer_obs = next_obs
            global_state = next_state
            steps += 1

        lengths.append(steps)

        if info["captured"]:
            captures += 1
        elif info["evader_reached_goal"]:
            evader_goals += 1
        else:
            timeouts += 1

    print(f"Evaluation over {cfg.eval_episodes} episodes")
    print(f"Capture rate      : {captures / cfg.eval_episodes:.3f}")
    print(f"Evader goal rate  : {evader_goals / cfg.eval_episodes:.3f}")
    print(f"Timeout rate      : {timeouts / cfg.eval_episodes:.3f}")
    print(f"Avg episode length: {np.mean(lengths):.2f}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resume training")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt", help="Checkpoint path for eval")
    args = parser.parse_args()

    if args.mode == "train":
        train(resume_path=args.resume)
    elif args.mode == "eval":
        evaluate(args.checkpoint)


if __name__ == "__main__":
    main()