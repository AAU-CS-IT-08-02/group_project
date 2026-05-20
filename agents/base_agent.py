from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Abstract base class for all pursuit-evasion agents.

    Every agent file loaded into the tool must subclass this.
    The runner calls these methods — nothing else is required.

    Roles
    -----
    "pursuer" : wins by occupying the same tile as the evader
    "evader"  : wins by reaching the goal tile before being caught

    Observation dict (passed to select_action and update)
    ------------------------------------------------------
    {
        "self_pos"        : (row, col),   # this agent's position
        "opponent_pos"    : (row, col),   # the other agent's position
        "goal_pos"        : (row, col),   # evader's goal tile
        "grid_size"       : int,          # always 10
        "steps_remaining" : int,          # steps left in this episode
    }

    Actions
    -------
    0 = up    (row - 1)
    1 = down  (row + 1)
    2 = left  (col - 1)
    3 = right (col + 1)

    Attempting to move outside the grid leaves the agent in place.
    """

    def __init__(self, role: str, config: dict):
        """
        Parameters
        ----------
        role   : "pursuer" or "evader"
        config : arbitrary dict — use it for hyperparameters, e.g.
                 {"alpha": 0.1, "gamma": 0.99, "epsilon": 0.1}
        """
        if role not in ("pursuer", "evader"):
            raise ValueError(f"role must be 'pursuer' or 'evader', got '{role}'")
        self.role = role
        self.config = config

    @abstractmethod
    def select_action(self, observation: dict) -> int:
        """
        Choose an action given the current observation.

        Called once per step for this agent.

        Parameters
        ----------
        observation : dict  (see class docstring for keys)

        Returns
        -------
        int : action in {0, 1, 2, 3}
        """

    @abstractmethod
    def update(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
    ) -> None:
        """
        Receive the outcome of the last step and update internal state.

        Called immediately after select_action, before the next step.

        Parameters
        ----------
        observation      : state the action was taken in
        action           : action that was taken (0–3)
        reward           : scalar reward received
        next_observation : state reached after the action
        done             : True if the episode ended on this step
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset any per-episode state (e.g. eligibility traces, hidden states).

        Called at the start of every episode, before the first select_action.
        Do NOT reset learned parameters (weights, Q-table) here — only
        transient episode state.
        """

    def save(self, path: str) -> None:
        """
        Save learned parameters to disk.

        Optional — override if your algorithm has persistent weights.
        Called automatically at checkpoints if enabled in the UI.

        Parameters
        ----------
        path : file path to save to, e.g. "checkpoints/pursuer_ep500.pkl"
        """

    def load(self, path: str) -> None:
        """
        Load learned parameters from disk.

        Optional — override alongside save().

        Parameters
        ----------
        path : file path to load from
        """
