"""
=====================================================================
  AGENT TEMPLATE — Pursuit-Evasion RL Tool
=====================================================================

HOW TO USE THIS FILE
--------------------
1. Copy this file and rename it, e.g. "my_q_learning_pursuer.py"
2. Rename the class (e.g. MyQLearningPursuer)
3. Fill in the three required methods:
       select_action  —  pick a move given the current state
       update         —  learn from what just happened
       reset          —  clear per-episode state
4. Set AGENT_CLASS at the bottom of this file
5. Load the file in the UI — the tool will auto-detect your class

ROLES
-----
  pursuer : wins by stepping onto the evader's tile
  evader  : wins by stepping onto the goal tile (before being caught)

Pass your role as a string: "pursuer" or "evader"
The runner enforces that you only load one of each.

OBSERVATION (what you receive every step)
-----------------------------------------
{
    "self_pos"        : (row, col),   # YOUR current position
    "opponent_pos"    : (row, col),   # the other agent's position
    "goal_pos"        : (row, col),   # the evader's goal tile
    "grid_size"       : int,          # always 10
    "steps_remaining" : int,          # steps left in this episode
}

  - Positions are (row, col) tuples, zero-indexed from top-left
  - (0, 0) is top-left, (9, 9) is bottom-right
  - Both agents receive the full observation (no partial observability
    by default — you can ignore fields you don't want)

ACTIONS
-------
  0 = up      row - 1
  1 = down    row + 1
  2 = left    col - 1
  3 = right   col + 1

  Moving outside the grid boundary leaves the agent in place.
  Both agents move simultaneously every step.

REWARDS (default reward scheme)
--------------------------------------------------------------
  Pursuer
    +10   on capture (same tile as evader)
    +0.5  each step the distance to evader decreases
    -0.5  each step the distance to evader increases
    -0.1  per step (time penalty)

  Evader
    +10   on reaching the goal tile
    +0.5  each step the distance to pursuer increases
    -0.5  each step the distance to pursuer decreases
    +0.1  each step closer to goal
    -0.1  per step (time penalty)

CONFIG
------
Pass any hyperparameters through the config dict.
Example:
    config = {
        "alpha":   0.1,    # learning rate
        "gamma":   0.99,   # discount factor
        "epsilon": 0.1,    # exploration rate
    }

The config dict is passed in by the runner using the values
set in the UI's "Agent config" panel (JSON format).
=====================================================================
"""

import random
from base_agent import BaseAgent      # adjust import path if needed


class MyAgentTemplate(BaseAgent):
    """
    Template agent — replace this docstring with your algorithm name
    and a one-line description.

    Example: QLearningPursuer — tabular Q-learning with ε-greedy policy.
    """

    # ------------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------------

    def __init__(self, role: str, config: dict):
        """
        Set up your agent.

        Call super().__init__ first — it validates the role and stores
        self.role and self.config for you.

        Then initialise your algorithm's data structures here:
        Q-tables, weights, etc.

        Parameters
        ----------
        role   : "pursuer" or "evader"
        config : dict of hyperparameters from the UI
        """
        super().__init__(role, config)

        # ── Pull hyperparameters from config (with safe defaults) ──────
        self.alpha   = config.get("alpha",   0.1)    # learning rate
        self.gamma   = config.get("gamma",   0.99)   # discount factor
        self.epsilon = config.get("epsilon", 0.1)    # exploration rate

        # ── Initialise your data structures here ───────────────────────
        #
        # Example: a Q-table for a 10x10 grid with 4 actions
        #   Keys are (self_row, self_col, opp_row, opp_col)
        #   You may want a more compact state representation.
        #
        self.q_table = {}    # replace with your structure

        # ── Per-episode state (also reset in reset()) ──────────────────
        self.last_obs    = None
        self.last_action = None

    # ------------------------------------------------------------------
    # REQUIRED METHOD 1 — select_action
    # ------------------------------------------------------------------

    def select_action(self, observation: dict) -> int:
        """
        Choose an action for this step.

        This is called every step during both training and replay.
        It must return an int in {0, 1, 2, 3}.

        Parameters
        ----------
        observation : dict  (see file header for keys)

        Returns
        -------
        int : 0=up, 1=down, 2=left, 3=right
        """

        # ── Build a state key from the observation ─────────────────────
        state = self._obs_to_state(observation)

        # ── ε-greedy example ───────────────────────────────────────────
        if random.random() < self.epsilon:
            return random.randint(0, 3)          # explore

        q_values = self.q_table.get(state, [0.0, 0.0, 0.0, 0.0])
        return int(q_values.index(max(q_values)))  # exploit

    # ------------------------------------------------------------------
    # REQUIRED METHOD 2 — update
    # ------------------------------------------------------------------

    def update(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
    ) -> None:
        """
        Learn from the transition (observation → action → reward → next).

        This is called every step during training.
        During replay (after training), this is NOT called — your
        policy is frozen and only select_action is used.

        Parameters
        ----------
        observation      : state the action was taken in
        action           : action that was taken (0–3)
        reward           : scalar reward received this step
        next_observation : state reached after the action
        done             : True if the episode ended on this step
        """

        # ── Example: Q-learning update ─────────────────────────────────
        state      = self._obs_to_state(observation)
        next_state = self._obs_to_state(next_observation)

        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0, 0.0, 0.0]

        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done else max(self.q_table[next_state])

        # Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
        self.q_table[state][action] = (
            current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        )

    # ------------------------------------------------------------------
    # REQUIRED METHOD 3 — reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset per-episode transient state.

        Called at the start of every episode.
        Do NOT reset your Q-table, weights, or any learned parameters here.
        Only reset things that should start fresh each episode, such as:
          - eligibility traces
          - hidden/recurrent states
          - episode-level counters
          - memory of the last observation/action
        """

        self.last_obs    = None
        self.last_action = None

        # ── Add any other per-episode resets below ─────────────────────

    # ------------------------------------------------------------------
    # OPTIONAL — custom reward function
    # ------------------------------------------------------------------

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
        Define your own reward function.

        If you implement this method, the game will call it every step
        instead of using the built-in reward values. If you leave it
        out (or do not override it), the default rewards are used.

        Both agents are checked independently — your pursuer can have
        a custom reward while the evader uses the default, or vice versa.

        Parameters
        ----------
        role
            Which agent is being rewarded right now — "pursuer" or
            "evader". Useful if both agents share one file.

        prev_agent_dist
            How far apart the two agents were BEFORE this step.
            Measured in Manhattan distance (no diagonals).

        curr_agent_dist
            How far apart the two agents are AFTER this step.
            If this is less than prev_agent_dist, the pursuer moved
            closer. If more, the evader escaped further.

        prev_evader_goal_dist
            How far the evader was from the goal tile BEFORE this step.

        curr_evader_goal_dist
            How far the evader is from the goal tile AFTER this step.
            If this is less than prev_evader_goal_dist, the evader
            moved toward the goal.

        pursuer_wins
            True if the pursuer just captured the evader (same tile).
            This is a terminal event — the episode ends after this step.

        evader_wins
            True if the evader just reached the goal tile.
            Also terminal.

        Returns
        -------
        float
            The reward to give this agent for this step.

        ── Examples ──────────────────────────────────────────────────

        1. Sparse reward only (no shaping — harder to learn but clean)
        ---------------------------------------------------------------
        def get_reward(self, role, prev_agent_dist, curr_agent_dist,
                       prev_evader_goal_dist, curr_evader_goal_dist,
                       pursuer_wins, evader_wins):
            if role == "pursuer":
                if pursuer_wins: return 1.0
                if evader_wins:  return -1.0
                return 0.0
            else:
                if evader_wins:  return 1.0
                if pursuer_wins: return -1.0
                return 0.0

        2. Distance-only shaping (no time penalty)
        ---------------------------------------------------------------
        def get_reward(self, role, prev_agent_dist, curr_agent_dist,
                       prev_evader_goal_dist, curr_evader_goal_dist,
                       pursuer_wins, evader_wins):
            if role == "pursuer":
                if pursuer_wins: return 10.0
                if evader_wins:  return -10.0
                # +1 for each tile closer, -1 for each tile further
                return float(prev_agent_dist - curr_agent_dist)
            else:
                if evader_wins:  return 10.0
                if pursuer_wins: return -10.0
                # reward for increasing distance from pursuer AND closing on goal
                escape = float(curr_agent_dist - prev_agent_dist)
                goal   = float(prev_evader_goal_dist - curr_evader_goal_dist)
                return escape + goal

        3. Aggresive time pressure (large time penalty to force quick wins)
        ---------------------------------------------------------------
        def get_reward(self, role, prev_agent_dist, curr_agent_dist,
                       prev_evader_goal_dist, curr_evader_goal_dist,
                       pursuer_wins, evader_wins):
            if role == "pursuer":
                if pursuer_wins: return 10.0
                if evader_wins:  return -10.0
                return -1.0   # heavy penalty every step
            else:
                if evader_wins:  return 10.0
                if pursuer_wins: return -10.0
                return -1.0

        ── Default reward (for reference) ────────────────────────────
        The built-in reward used when get_reward() is NOT defined:

        Pursuer:
            +10.0  on capture
            +0.5   each step closer to evader
            -0.5   each step further from evader
            -0.1   per step (time penalty)

        Evader:
            +10.0  on reaching goal
            +0.5   each step further from pursuer
            -0.5   each step closer to pursuer
            +0.3   each step closer to goal
            -0.3   each step further from goal
            -0.1   per step (time penalty)
        """
        raise NotImplementedError   # remove this line when you implement it

    # ------------------------------------------------------------------
    # OPTIONAL — save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save learned parameters to disk.

        Override this if your algorithm has persistent state worth saving.
        The tool calls this at checkpoints (configurable in the UI).

        Example (pickle):
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.q_table, f)
        """
        pass   # remove and implement if needed

    def load(self, path: str) -> None:
        """
        Load learned parameters from disk.

        Override this alongside save().

        Example (pickle):
            import pickle
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
        """
        pass   # remove and implement if needed

    # ------------------------------------------------------------------
    # PRIVATE HELPERS — add as many as you like below here
    # ------------------------------------------------------------------

    def _obs_to_state(self, observation: dict) -> tuple:
        """
        Convert an observation dict into a hashable state key.

        This is just a helper for the Q-table example above.
        Replace or remove depending on your algorithm.

        A simple state: (my_row, my_col, opp_row, opp_col)
        A richer state could include steps_remaining or distance to goal.
        """
        sr, sc = observation["self_pos"]
        or_, oc = observation["opponent_pos"]
        return (sr, sc, or_, oc)


# ======================================================================
#  REQUIRED — tell the tool which class to load from this file
# ======================================================================
#
#  Set this to your class name.
#  The runner imports this file and looks for AGENT_CLASS.
#
AGENT_CLASS = MyAgentTemplate