"""
training/metrics.py
--------------------
Records and computes all metrics discussed in the design.

Categories tracked:
  1. Game outcome       — win rate, episode length, time to first contact
  2. Training progress  — cumulative reward, convergence, sample efficiency,
                          reward accumulation heatmap
  3. Policy quality     — policy variance, state coverage, path efficiency
  4. Behavioural        — agent distance, trajectory, goal proximity
  5. Reliability        — policy variance (post-convergence), PAC, CI across seeds
"""

import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from environment.game import StepResult


# ── Per-episode record ─────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    """All raw data captured during one episode."""
    episode_number      : int
    winner              : Optional[str]        # "pursuer", "evader", None
    episode_length      : int                  # total steps taken
    time_to_contact     : Optional[int]        # step when agents first adjacent
    pursuer_total_reward: float
    evader_total_reward : float

    # Per-step lists (one entry per step)
    agent_distances     : List[int]    = field(default_factory=list)
    goal_distances      : List[int]    = field(default_factory=list)   # evader→goal
    pursuer_rewards     : List[float]  = field(default_factory=list)
    evader_rewards      : List[float]  = field(default_factory=list)

    # Grid positions visited (for heatmaps)
    pursuer_positions   : List[Tuple]  = field(default_factory=list)
    evader_positions    : List[Tuple]  = field(default_factory=list)

    # Full grid snapshots (for replay)
    snapshots           : List[dict]   = field(default_factory=list)


# ── Main metrics class ─────────────────────────────────────────────────

class MetricsRecorder:
    """
    Records metrics during training and computes aggregates
    for the dashboard.

    Usage
    -----
        recorder = MetricsRecorder(grid_size=10, smoothing_window=100)

        # during an episode:
        recorder.begin_episode(episode_number)
        recorder.record_step(result)         # call each step
        recorder.end_episode(winner, length)

        # after training:
        summary = recorder.compute_summary()
    """

    def __init__(
        self,
        grid_size        : int = 10,
        smoothing_window : int = 100,
        convergence_threshold : float = 0.02,
        convergence_window    : int   = 200,
    ):
        self.grid_size             = grid_size
        self.smoothing_window      = smoothing_window
        self.convergence_threshold = convergence_threshold
        self.convergence_window    = convergence_window

        self.episodes : List[EpisodeRecord] = []
        self._current : Optional[EpisodeRecord] = None

    # ── Episode lifecycle ──────────────────────────────────────────────

    def begin_episode(self, episode_number: int) -> None:
        """Call at the start of each episode."""
        self._current = EpisodeRecord(
            episode_number       = episode_number,
            winner               = None,
            episode_length       = 0,
            time_to_contact      = None,
            pursuer_total_reward = 0.0,
            evader_total_reward  = 0.0,
        )

    def record_step(self, result: StepResult) -> None:
        """Call after every game step with the StepResult."""
        ep = self._current
        if ep is None:
            raise RuntimeError("Call begin_episode() before record_step()")

        snapshot = result.grid_snapshot
        p_pos    = snapshot["pursuer_pos"]
        e_pos    = snapshot["evader_pos"]
        g_pos    = snapshot["goal_pos"]

        # Distances
        agent_dist = abs(p_pos[0]-e_pos[0]) + abs(p_pos[1]-e_pos[1])
        goal_dist  = abs(e_pos[0]-g_pos[0]) + abs(e_pos[1]-g_pos[1])

        ep.agent_distances.append(agent_dist)
        ep.goal_distances.append(goal_dist)

        # Time to first contact (distance <= 1 means adjacent or on same tile)
        if ep.time_to_contact is None and agent_dist <= 1:
            ep.time_to_contact = result.step_number

        # Rewards
        ep.pursuer_rewards.append(result.pursuer_reward)
        ep.evader_rewards.append(result.evader_reward)
        ep.pursuer_total_reward += result.pursuer_reward
        ep.evader_total_reward  += result.evader_reward

        # Positions for heatmaps
        ep.pursuer_positions.append(p_pos)
        ep.evader_positions.append(e_pos)

        # Snapshots for replay
        ep.snapshots.append(snapshot)

    def end_episode(self, winner: Optional[str], length: int) -> None:
        """Call at the end of each episode."""
        ep = self._current
        if ep is None:
            raise RuntimeError("Call begin_episode() before end_episode()")
        ep.winner         = winner
        ep.episode_length = length
        self.episodes.append(ep)
        self._current = None

    # ── ─────────────────────────────────────────────────────────────────
    # Computed metrics
    # ── ─────────────────────────────────────────────────────────────────

    # 1. GAME OUTCOME ──────────────────────────────────────────────────

    def win_rates(self) -> Dict[str, List[float]]:
        """
        Smoothed win rate over training for pursuer and evader.
        Returns dict with keys 'pursuer', 'evader', 'episodes'.
        Each value is a list aligned by episode.
        """
        w   = self.smoothing_window
        eps = [e.episode_number for e in self.episodes]

        p_wins = [1.0 if e.winner == "pursuer" else 0.0 for e in self.episodes]
        e_wins = [1.0 if e.winner == "evader"  else 0.0 for e in self.episodes]

        return {
            "episodes"  : eps,
            "pursuer"   : self._rolling_average(p_wins, w),
            "evader"    : self._rolling_average(e_wins, w),
        }

    def episode_lengths(self) -> List[int]:
        return [e.episode_length for e in self.episodes]

    def time_to_contact_series(self) -> List[Optional[int]]:
        return [e.time_to_contact for e in self.episodes]

    # 2. TRAINING PROGRESS ─────────────────────────────────────────────

    def cumulative_rewards(self) -> Dict[str, List[float]]:
        """Smoothed cumulative reward per episode for each agent."""
        w = self.smoothing_window
        p = [e.pursuer_total_reward for e in self.episodes]
        ev = [e.evader_total_reward  for e in self.episodes]
        return {
            "pursuer" : self._rolling_average(p,  w),
            "evader"  : self._rolling_average(ev, w),
            "episodes": [e.episode_number for e in self.episodes],
        }

    def convergence_episode(self) -> Dict[str, Optional[int]]:
        """
        Find the episode where each agent's win rate converged.
        Returns {'pursuer': int|None, 'evader': int|None}
        """
        wr    = self.win_rates()
        p_wr  = wr["pursuer"]
        e_wr  = wr["evader"]
        eps   = wr["episodes"]

        return {
            "pursuer" : self._find_convergence(p_wr, eps),
            "evader"  : self._find_convergence(e_wr, eps),
        }

    def reward_accumulation_heatmap(
        self, n_episode_bins: int = 50, n_step_bins: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Reward accumulation heatmap.
        X axis: episode bins, Y axis: step bins
        Value: average reward earned in that (episode_bin, step_bin)

        Returns
        -------
        {
            'pursuer': 2D array (step_bins x episode_bins),
            'evader':  2D array (step_bins x episode_bins),
        }
        """
        n_eps    = len(self.episodes)
        p_matrix = np.zeros((n_step_bins, n_episode_bins))
        e_matrix = np.zeros((n_step_bins, n_episode_bins))
        counts   = np.zeros((n_step_bins, n_episode_bins))

        for i, ep in enumerate(self.episodes):
            ep_bin = min(int(i / n_eps * n_episode_bins), n_episode_bins - 1)
            n_steps = len(ep.pursuer_rewards)

            for s, (pr, er) in enumerate(zip(ep.pursuer_rewards, ep.evader_rewards)):
                step_bin = min(int(s / max(n_steps, 1) * n_step_bins), n_step_bins - 1)
                p_matrix[step_bin, ep_bin] += pr
                e_matrix[step_bin, ep_bin] += er
                counts[step_bin, ep_bin]   += 1

        # Avoid division by zero
        safe_counts = np.where(counts == 0, 1, counts)
        return {
            "pursuer" : p_matrix / safe_counts,
            "evader"  : e_matrix / safe_counts,
        }

    # 3. POLICY QUALITY ────────────────────────────────────────────────

    def policy_variance(self, last_n: int = 500) -> Dict[str, float]:
        """
        Standard deviation of win rate over the last N episodes.
        Low = stable policy. High = erratic.
        """
        recent = self.episodes[-last_n:] if len(self.episodes) >= last_n else self.episodes

        p_wins = [1.0 if e.winner == "pursuer" else 0.0 for e in recent]
        e_wins = [1.0 if e.winner == "evader"  else 0.0 for e in recent]

        return {
            "pursuer" : float(np.std(p_wins)) if p_wins else 0.0,
            "evader"  : float(np.std(e_wins)) if e_wins else 0.0,
        }

    def state_coverage_heatmap(self) -> Dict[str, np.ndarray]:
        """
        How often each grid tile was visited across all episodes.
        Returns two (grid_size x grid_size) arrays.
        """
        p_map = np.zeros((self.grid_size, self.grid_size))
        e_map = np.zeros((self.grid_size, self.grid_size))

        for ep in self.episodes:
            for r, c in ep.pursuer_positions:
                p_map[r, c] += 1
            for r, c in ep.evader_positions:
                e_map[r, c] += 1

        # Normalise to 0–1
        if p_map.max() > 0: p_map /= p_map.max()
        if e_map.max() > 0: e_map /= e_map.max()

        return {"pursuer": p_map, "evader": e_map}

    def path_efficiency(self) -> Dict[str, List[float]]:
        """
        Ratio of optimal path length to actual path length per episode.
        1.0 = perfectly efficient. Lower = more wandering.

        For pursuer: optimal = manhattan distance from start to evader's
                               final position
        For evader:  optimal = manhattan distance from start to goal
        """
        results_p, results_e = [], []

        p_start = None
        e_start = None

        for ep in self.episodes:
            if not ep.snapshots:
                continue

            # Infer starts from first snapshot
            first = ep.snapshots[0]
            p_start = first["pursuer_pos"]
            e_start = first["evader_pos"]
            g_pos   = first["goal_pos"]
            last    = ep.snapshots[-1]

            # Optimal paths
            p_optimal = abs(p_start[0]-last["evader_pos"][0]) + abs(p_start[1]-last["evader_pos"][1])
            e_optimal = abs(e_start[0]-g_pos[0])              + abs(e_start[1]-g_pos[1])

            actual_length = ep.episode_length

            results_p.append(p_optimal / max(actual_length, 1))
            results_e.append(e_optimal / max(actual_length, 1))

        return {
            "pursuer" : results_p,
            "evader"  : results_e,
            "episodes": [e.episode_number for e in self.episodes if e.snapshots],
        }

    # 4. BEHAVIOURAL ───────────────────────────────────────────────────

    def agent_distance_series(self, episode_index: int) -> List[int]:
        """Per-step agent distance for a specific episode (for replay view)."""
        return self.episodes[episode_index].agent_distances

    def goal_proximity_series(self, episode_index: int) -> List[int]:
        """Per-step evader-to-goal distance for a specific episode."""
        return self.episodes[episode_index].goal_distances

    def trajectory(self, episode_index: int) -> Dict[str, List[Tuple]]:
        """Full position trajectory for both agents in one episode."""
        ep = self.episodes[episode_index]
        return {
            "pursuer" : ep.pursuer_positions,
            "evader"  : ep.evader_positions,
        }

    # 5. RELIABILITY ───────────────────────────────────────────────────

    def performance_after_convergence(self) -> Dict[str, Dict[str, float]]:
        """
        Split post-convergence episodes into early / mid / late windows
        and compute win rate in each. Detects drift.

        Returns
        -------
        {
            'pursuer': {'early': float, 'mid': float, 'late': float},
            'evader':  {'early': float, 'mid': float, 'late': float},
        }
        """
        conv = self.convergence_episode()

        results = {}
        for role in ("pursuer", "evader"):
            conv_ep = conv[role]
            if conv_ep is None:
                results[role] = {"early": None, "mid": None, "late": None}
                continue

            post = [e for e in self.episodes if e.episode_number >= conv_ep]
            n    = len(post)
            if n < 3:
                results[role] = {"early": None, "mid": None, "late": None}
                continue

            third = n // 3
            windows = {
                "early": post[:third],
                "mid"  : post[third:2*third],
                "late" : post[2*third:],
            }

            results[role] = {
                k: float(np.mean([1.0 if e.winner == role else 0.0 for e in v]))
                for k, v in windows.items()
            }

        return results

    def win_rate_confidence_interval(
        self, seeds_data: List["MetricsRecorder"]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean ± std of final win rate across multiple seeds.

        Parameters
        ----------
        seeds_data : list of MetricsRecorder instances, one per seed

        Returns
        -------
        {
            'pursuer': {'mean': float, 'std': float, 'n_seeds': int},
            'evader':  {'mean': float, 'std': float, 'n_seeds': int},
        }
        """
        p_rates, e_rates = [], []

        for recorder in seeds_data:
            if not recorder.episodes:
                continue
            final = recorder.episodes[-200:]   # last 200 episodes
            p_rates.append(np.mean([1.0 if e.winner == "pursuer" else 0.0 for e in final]))
            e_rates.append(np.mean([1.0 if e.winner == "evader"  else 0.0 for e in final]))

        return {
            "pursuer": {
                "mean"    : float(np.mean(p_rates)) if p_rates else 0.0,
                "std"     : float(np.std(p_rates))  if p_rates else 0.0,
                "n_seeds" : len(p_rates),
            },
            "evader": {
                "mean"    : float(np.mean(e_rates)) if e_rates else 0.0,
                "std"     : float(np.std(e_rates))  if e_rates else 0.0,
                "n_seeds" : len(e_rates),
            },
        }

    # ── Full summary (for dashboard) ───────────────────────────────────

    def compute_summary(self) -> dict:
        """
        Compute all metrics and return as a single dict.
        Called by the dashboard after training completes.
        """
        if not self.episodes:
            return {}

        return {
            # Game outcome
            "win_rates"             : self.win_rates(),
            "episode_lengths"       : self.episode_lengths(),
            "time_to_contact"       : self.time_to_contact_series(),

            # Training progress
            "cumulative_rewards"    : self.cumulative_rewards(),
            "convergence_episode"   : self.convergence_episode(),
            "reward_heatmap"        : self.reward_accumulation_heatmap(),

            # Policy quality
            "policy_variance"       : self.policy_variance(),
            "state_coverage"        : self.state_coverage_heatmap(),
            "path_efficiency"       : self.path_efficiency(),

            # Reliability
            "performance_pac"       : self.performance_after_convergence(),

            # Raw (for replay)
            "total_episodes"        : len(self.episodes),
            "episodes"              : self.episodes,
        }

    # ── Utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _rolling_average(values: List[float], window: int) -> List[float]:
        """Compute a rolling average over a list of values."""
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(sum(values[start:i+1]) / (i - start + 1))
        return result

    def _find_convergence(
        self, smoothed_values: List[float], episodes: List[int]
    ) -> Optional[int]:
        """
        Find the first episode where the smoothed value stays within
        convergence_threshold of its final value for convergence_window
        consecutive episodes.
        """
        if len(smoothed_values) < self.convergence_window:
            return None

        final_value = smoothed_values[-1]
        threshold   = self.convergence_threshold
        window      = self.convergence_window

        for i in range(len(smoothed_values) - window):
            window_slice = smoothed_values[i: i + window]
            if all(abs(v - final_value) <= threshold for v in window_slice):
                return episodes[i]

        return None
