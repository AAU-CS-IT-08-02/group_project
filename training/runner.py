"""
training/runner.py
-------------------
Runs training episodes and coordinates the game, agents,
and metrics recorder.

The runner is intentionally thin — it just orchestrates the
other modules. All game logic lives in game.py, all learning
lives in the agent files, all recording lives in metrics.py.

Supports:
  - Normal training (visualisation off)
  - Callback hook for the UI to read live stats without blocking
  - Checkpoint saving at configurable intervals
  - Replay-only mode (no update() calls, frozen policies)
"""

import time
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List

from environment.game import Game, GameConfig
from training.metrics import MetricsRecorder
from agents.base_agent import BaseAgent


# ── Runner config ──────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    """Configuration for a training run."""
    n_episodes          : int   = 5_000
    checkpoint_interval : int   = 500      # save every N episodes (0 = off)
    checkpoint_dir      : str   = "checkpoints"
    smoothing_window    : int   = 100      # for live reward curve
    convergence_threshold: float = 0.02
    convergence_window  : int   = 200
    live_update_interval: int   = 50       # callback every N episodes


# ── Live stats (passed to UI callback) ────────────────────────────────

@dataclass
class LiveStats:
    """Snapshot of training progress passed to the UI each interval."""
    episode           : int
    total_episodes    : int
    pursuer_win_rate  : float       # rolling average
    evader_win_rate   : float       # rolling average
    pursuer_reward    : float       # rolling average
    evader_reward     : float       # rolling average
    avg_episode_length: float
    elapsed_seconds   : float
    eta_seconds       : float
    converged         : bool


# ── Agent loader ───────────────────────────────────────────────────────

def load_agent_from_file(
    filepath: str,
    role    : str,
    config  : dict,
) -> BaseAgent:
    """
    Dynamically load an agent class from a .py file.

    The file must define AGENT_CLASS at module level pointing
    to a class that subclasses BaseAgent.

    Parameters
    ----------
    filepath : path to the agent .py file
    role     : "pursuer" or "evader"
    config   : hyperparameter dict passed to the agent constructor

    Returns
    -------
    Instantiated BaseAgent subclass
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {filepath}")
    if path.suffix != ".py":
        raise ValueError(f"Agent file must be a .py file, got: {filepath}")

    # Load the module dynamically
    spec   = importlib.util.spec_from_file_location("agent_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to load agent file '{filepath}': {e}") from e

    # Find AGENT_CLASS
    if not hasattr(module, "AGENT_CLASS"):
        raise AttributeError(
            f"Agent file '{filepath}' must define AGENT_CLASS = YourAgentClass"
        )

    agent_class = module.AGENT_CLASS

    # Validate it's a proper subclass
    if not (isinstance(agent_class, type) and issubclass(agent_class, BaseAgent)):
        raise TypeError(
            f"AGENT_CLASS in '{filepath}' must be a subclass of BaseAgent"
        )

    return agent_class(role=role, config=config)


# ── Runner ─────────────────────────────────────────────────────────────

class Runner:
    """
    Orchestrates training between the game, two agents,
    and the metrics recorder.

    Usage
    -----
        runner = Runner(
            pursuer      = pursuer_agent,
            evader       = evader_agent,
            game_config  = game_config,
            runner_config= runner_config,
        )
        summary = runner.train(on_update=my_ui_callback)
    """

    def __init__(
        self,
        pursuer       : BaseAgent,
        evader        : BaseAgent,
        game_config   : GameConfig,
        runner_config : RunnerConfig,
    ):
        self.pursuer  = pursuer
        self.evader   = evader
        self.game     = Game(game_config)
        self.config   = runner_config
        self.recorder = MetricsRecorder(
            smoothing_window      = runner_config.smoothing_window,
            convergence_threshold = runner_config.convergence_threshold,
            convergence_window    = runner_config.convergence_window,
        )

        self._stop_requested = False

    # ── Training ───────────────────────────────────────────────────────

    def train(
        self,
        on_update : Optional[Callable[[LiveStats], None]] = None,
    ) -> dict:
        """
        Run the full training loop.

        Parameters
        ----------
        on_update : optional callback called every live_update_interval
                    episodes with a LiveStats snapshot. Used by the UI
                    to update live stats without blocking.

        Returns
        -------
        Full metrics summary dict from MetricsRecorder.compute_summary()
        """
        self._stop_requested = False
        start_time           = time.time()
        n                    = self.config.n_episodes

        # Rolling accumulators for live stats
        recent_p_wins    = []
        recent_e_wins    = []
        recent_p_rewards = []
        recent_e_rewards = []
        recent_lengths   = []
        w                = self.config.smoothing_window

        for ep in range(1, n + 1):

            if self._stop_requested:
                break

            # ── Run one episode ────────────────────────────────────────
            ep_length, winner, p_reward, e_reward = self._run_episode(ep)

            # ── Update rolling accumulators ────────────────────────────
            recent_p_wins.append(1.0 if winner == "pursuer" else 0.0)
            recent_e_wins.append(1.0 if winner == "evader"  else 0.0)
            recent_p_rewards.append(p_reward)
            recent_e_rewards.append(e_reward)
            recent_lengths.append(ep_length)

            # Keep only last window
            if len(recent_p_wins) > w:
                recent_p_wins    = recent_p_wins[-w:]
                recent_e_wins    = recent_e_wins[-w:]
                recent_p_rewards = recent_p_rewards[-w:]
                recent_e_rewards = recent_e_rewards[-w:]
                recent_lengths   = recent_lengths[-w:]

            # ── Checkpointing ──────────────────────────────────────────
            if (
                self.config.checkpoint_interval > 0
                and ep % self.config.checkpoint_interval == 0
            ):
                self._save_checkpoints(ep)

            # ── UI callback ────────────────────────────────────────────
            if on_update and (ep % self.config.live_update_interval == 0 or ep == n):
                elapsed = time.time() - start_time
                eta     = (elapsed / ep) * (n - ep) if ep > 0 else 0.0

                on_update(LiveStats(
                    episode            = ep,
                    total_episodes     = n,
                    pursuer_win_rate   = sum(recent_p_wins)    / max(len(recent_p_wins), 1),
                    evader_win_rate    = sum(recent_e_wins)    / max(len(recent_e_wins), 1),
                    pursuer_reward     = sum(recent_p_rewards) / max(len(recent_p_rewards), 1),
                    evader_reward      = sum(recent_e_rewards) / max(len(recent_e_rewards), 1),
                    avg_episode_length = sum(recent_lengths)   / max(len(recent_lengths), 1),
                    elapsed_seconds    = elapsed,
                    eta_seconds        = eta,
                    converged          = self._check_converged(),
                ))

        return self.recorder.compute_summary()

    def stop(self) -> None:
        """Request training to stop after the current episode."""
        self._stop_requested = True

    # ── Replay (frozen policy, no learning) ───────────────────────────

    def replay_episode(self, episode_index: int) -> List[dict]:
        """
        Re-run a specific episode with frozen policies (no update calls).
        Returns the list of grid snapshots for the visualiser.

        Note: this re-plays using the CURRENT (trained) policy,
        not the policy at the time of the original episode.
        For exact replay, use the recorded snapshots in
        recorder.episodes[episode_index].snapshots.
        """
        obs_p, obs_e = self.game.reset()
        self.pursuer.reset()
        self.evader.reset()

        snapshots = [self.game.grid.snapshot()]
        done      = False

        while not done:
            action_p = self.pursuer.select_action(obs_p)
            action_e = self.evader.select_action(obs_e)
            result   = self.game.step(action_p, action_e, pursuer=self.pursuer, evader=self.evader)

            snapshots.append(result.grid_snapshot)
            obs_p = result.next_pursuer_obs
            obs_e = result.next_evader_obs
            done  = result.done

        return snapshots

    # ── Private helpers ────────────────────────────────────────────────

    def _run_episode(self, episode_number: int):
        """Run one full episode. Returns (length, winner, p_reward, e_reward)."""
        obs_p, obs_e = self.game.reset()
        self.pursuer.reset()
        self.evader.reset()
        self.recorder.begin_episode(episode_number)

        done     = False
        p_reward = 0.0
        e_reward = 0.0

        while not done:
            # Select actions
            action_p = self.pursuer.select_action(obs_p)
            action_e = self.evader.select_action(obs_e)

            # Step the game
            result = self.game.step(action_p, action_e, pursuer=self.pursuer, evader=self.evader)

            # Record metrics
            self.recorder.record_step(result)

            # Update agents (learning step)
            self.pursuer.update(
                obs_p, action_p, result.pursuer_reward,
                result.next_pursuer_obs, result.done
            )
            self.evader.update(
                obs_e, action_e, result.evader_reward,
                result.next_evader_obs, result.done
            )

            # Advance observations
            obs_p     = result.next_pursuer_obs
            obs_e     = result.next_evader_obs
            done      = result.done
            p_reward += result.pursuer_reward
            e_reward += result.evader_reward

        self.recorder.end_episode(result.winner, result.step_number)
        return result.step_number, result.winner, p_reward, e_reward

    def _save_checkpoints(self, episode: int) -> None:
        """Save both agents' weights at a checkpoint."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.pursuer.save(f"{self.config.checkpoint_dir}/pursuer_ep{episode}.ckpt")
        self.evader.save( f"{self.config.checkpoint_dir}/evader_ep{episode}.ckpt")

    def _check_converged(self) -> bool:
        """Quick check if both agents appear to have converged."""
        conv = self.recorder.convergence_episode()
        return conv["pursuer"] is not None and conv["evader"] is not None