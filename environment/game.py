"""
environment/game.py
--------------------
Manages a single episode of the pursuit-evasion game.

Responsibilities:
  - Stepping both agents simultaneously each turn
  - Calculating rewards for pursuer and evader
  - Detecting win conditions and episode termination
  - Returning step data to the runner

Reward scheme (all values configurable via GameConfig):
  Pursuer
    +10.0   on capture
    +0.5    each step closer to evader
    -0.5    each step further from evader
    -0.1    per step (time penalty)

  Evader
    +10.0   on reaching goal
    +0.5    each step further from pursuer
    -0.5    each step closer to pursuer
    +0.3    each step closer to goal
    -0.1    per step (time penalty)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from environment.grid import Grid, GridConfig


# ── Reward config ──────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """All reward values — configurable from the UI."""
    capture_reward      : float =  10.0   # pursuer wins
    goal_reward         : float =  10.0   # evader wins
    closer_to_target    : float =   0.5   # moving toward objective
    further_from_target : float =  -0.5   # moving away from objective
    evader_closer_goal  : float =   0.3   # evader approaching goal
    time_penalty        : float =  -0.1   # per step for both agents


# ── Step result ────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """
    Everything the runner needs after one game step.
    Passed to metrics recorder and agent update calls.
    """
    pursuer_obs      : dict
    evader_obs       : dict
    pursuer_action   : int
    evader_action    : int
    pursuer_reward   : float
    evader_reward    : float
    next_pursuer_obs : dict
    next_evader_obs  : dict
    done             : bool
    winner           : Optional[str]   # "pursuer", "evader", or None (timeout)
    step_number      : int
    grid_snapshot    : dict            # for replay recording


# ── Game config ────────────────────────────────────────────────────────

@dataclass
class GameConfig:
    """Top-level config combining grid and reward settings."""
    grid_config   : GridConfig   = None
    reward_config : RewardConfig = None
    max_steps     : int          = 200

    def __post_init__(self):
        if self.grid_config is None:
            self.grid_config = GridConfig()
        if self.reward_config is None:
            self.reward_config = RewardConfig()


# ── Game ───────────────────────────────────────────────────────────────

class Game:
    """
    Manages one episode of the pursuit-evasion game.

    Usage
    -----
        game = Game(config)
        obs_p, obs_e = game.reset()

        while not done:
            action_p = pursuer.select_action(obs_p)
            action_e = evader.select_action(obs_e)
            result   = game.step(action_p, action_e)
            done     = result.done
    """

    def __init__(self, config: GameConfig):
        self.config  = config
        self.grid    = Grid(config.grid_config)
        self.rewards = config.reward_config

        self._step_count : int = 0

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self) -> Tuple[dict, dict]:
        """
        Reset the grid and step counter for a new episode.

        Returns
        -------
        (pursuer_obs, evader_obs) — initial observations
        """
        self.grid.reset()
        self._step_count = 0
        return self._get_observations()

    # ── Step ───────────────────────────────────────────────────────────

    def step(self, pursuer_action: int, evader_action: int, pursuer=None, evader=None) -> StepResult:
        """
        Advance the game by one step.

        Both agents move simultaneously. Win conditions are checked
        after both moves are applied.

        Parameters
        ----------
        pursuer_action : int in {0, 1, 2, 3}
        evader_action  : int in {0, 1, 2, 3}

        Returns
        -------
        StepResult with observations, rewards, done flag, and winner
        """
        # ── Capture state before moves (for reward calculation) ────────
        prev_agent_dist       = self.grid.agent_distance
        prev_evader_goal_dist = self.grid.evader_to_goal_distance

        # ── Store pre-move observations ────────────────────────────────
        steps_remaining      = self.config.max_steps - self._step_count
        pursuer_obs          = self.grid.get_observation("pursuer", steps_remaining)
        evader_obs           = self.grid.get_observation("evader",  steps_remaining)

        # ── Apply moves simultaneously ─────────────────────────────────
        self.grid.move_pursuer(pursuer_action)
        self.grid.move_evader(evader_action)
        self._step_count += 1

        # ── Check win conditions ───────────────────────────────────────
        pursuer_wins = self.grid.pursuer_wins
        evader_wins  = self.grid.evader_wins
        timed_out    = self._step_count >= self.config.max_steps

        done   = pursuer_wins or evader_wins or timed_out
        winner = (
            "pursuer" if pursuer_wins else
            "evader"  if evader_wins  else
            None                            # timeout = no winner
        )

        # ── Calculate rewards ──────────────────────────────────────────
        pursuer_reward, evader_reward = self._calculate_rewards(
            prev_agent_dist,
            prev_evader_goal_dist,
            pursuer_wins,
            evader_wins,
            pursuer_agent=pursuer,
            evader_agent=evader,
        )

        # ── Build post-move observations ───────────────────────────────
        steps_remaining_next      = self.config.max_steps - self._step_count
        next_pursuer_obs          = self.grid.get_observation("pursuer", steps_remaining_next)
        next_evader_obs           = self.grid.get_observation("evader",  steps_remaining_next)

        return StepResult(
            pursuer_obs      = pursuer_obs,
            evader_obs       = evader_obs,
            pursuer_action   = pursuer_action,
            evader_action    = evader_action,
            pursuer_reward   = pursuer_reward,
            evader_reward    = evader_reward,
            next_pursuer_obs = next_pursuer_obs,
            next_evader_obs  = next_evader_obs,
            done             = done,
            winner           = winner,
            step_number      = self._step_count,
            grid_snapshot    = self.grid.snapshot(),
        )

    # ── Reward calculation ─────────────────────────────────────────────

    def _calculate_rewards(
        self,
        prev_agent_dist      : int,
        prev_evader_goal_dist: int,
        pursuer_wins         : bool,
        evader_wins          : bool,
        pursuer_agent        = None,
        evader_agent         = None,
    ) -> Tuple[float, float]:
        """
        Calculate rewards for both agents based on what happened this step.

        If an agent defines get_reward(), that method is called instead of
        the default RewardConfig values. The two agents are checked
        independently — one can use a custom function while the other uses
        the default.

        Returns
        -------
        (pursuer_reward, evader_reward)
        """
        curr_agent_dist       = self.grid.agent_distance
        curr_evader_goal_dist = self.grid.evader_to_goal_distance

        # Shared kwargs passed to every get_reward() call
        reward_kwargs = dict(
            prev_agent_dist       = prev_agent_dist,
            curr_agent_dist       = curr_agent_dist,
            prev_evader_goal_dist = prev_evader_goal_dist,
            curr_evader_goal_dist = curr_evader_goal_dist,
            pursuer_wins          = pursuer_wins,
            evader_wins           = evader_wins,
        )

        pursuer_reward = self._agent_reward("pursuer", pursuer_agent, **reward_kwargs)
        evader_reward  = self._agent_reward("evader",  evader_agent,  **reward_kwargs)

        return pursuer_reward, evader_reward

    def _agent_reward(
        self,
        role         : str,
        agent        ,
        prev_agent_dist      : int,
        curr_agent_dist      : int,
        prev_evader_goal_dist: int,
        curr_evader_goal_dist: int,
        pursuer_wins         : bool,
        evader_wins          : bool,
    ) -> float:
        """
        Return the reward for one agent.

        If the agent has a get_reward() method (and it does not raise
        NotImplementedError), use that. Otherwise fall back to default.
        """
        if agent is not None:
            try:
                return agent.get_reward(
                    role                  = role,
                    prev_agent_dist       = prev_agent_dist,
                    curr_agent_dist       = curr_agent_dist,
                    prev_evader_goal_dist = prev_evader_goal_dist,
                    curr_evader_goal_dist = curr_evader_goal_dist,
                    pursuer_wins          = pursuer_wins,
                    evader_wins           = evader_wins,
                )
            except (NotImplementedError, AttributeError):
                pass   # agent chose not to override — use default

        # ── Default reward (RewardConfig) ─────────────────────────────
        r = self.rewards

        if role == "pursuer":
            if pursuer_wins: return  r.capture_reward
            if evader_wins:  return -r.goal_reward
            delta = prev_agent_dist - curr_agent_dist
            shaping = (r.closer_to_target    if delta > 0 else
                       r.further_from_target if delta < 0 else 0.0)
            return shaping + r.time_penalty

        else:  # evader
            if evader_wins:  return  r.goal_reward
            if pursuer_wins: return -r.capture_reward
            # Escape shaping (opposite of pursuer distance change)
            delta  = prev_agent_dist - curr_agent_dist
            escape = (-r.closer_to_target    if delta > 0 else
                       -r.further_from_target if delta < 0 else 0.0)
            # Goal shaping
            gdelta = prev_evader_goal_dist - curr_evader_goal_dist
            goal   = (r.evader_closer_goal  if gdelta > 0 else
                     -r.evader_closer_goal  if gdelta < 0 else 0.0)
            return escape + goal + r.time_penalty

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_observations(self) -> Tuple[dict, dict]:
        """Return current observations for both agents."""
        steps_remaining = self.config.max_steps - self._step_count
        return (
            self.grid.get_observation("pursuer", steps_remaining),
            self.grid.get_observation("evader",  steps_remaining),
        )

    @property
    def step_count(self) -> int:
        return self._step_count