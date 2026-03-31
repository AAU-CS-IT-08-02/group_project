# =============================================================================
# MyAlgorithmTemplate.py  —  Starting point for your own algorithm
# =============================================================================
#
# HOW TO USE THIS FILE
# --------------------
# 1. Copy (or rename) this file inside the  algorithms/  folder.
# 2. Open the copy and look for the section labelled  "YOUR ALGORITHM GOES HERE".
# 3. Replace the example movement code with your own logic.
# 4. Open the GUI, hit  ↻  to refresh, then pick your file from the dropdown.
# 5. Click  Load  and watch it run.
#
# THE TWO THINGS EVERY ALGORITHM FILE MUST HAVE
# ----------------------------------------------
# PEG_Gui.py will only accept a file that exposes these two names:
#
#   env        — an adapter object (created at the bottom of this file)
#   step(env)  — a function that advances the simulation one step
#                and returns "pursuers_win", "evader_win", or None
#
# Everything else in this file is support code that makes the GUI work.
# You only need to change the  TemplateSimulation  class in the middle.
#
# WHAT THE GUI EXPECTS BACK FROM step()
# --------------------------------------
#   "pursuers_win"  — a pursuer caught the evader  → GUI resets episode + counts win
#   "evader_win"    — evader reached the goal       → GUI resets episode + counts win
#   None            — nobody won yet                → GUI keeps the episode running
#
# =============================================================================

import numpy as np   # numpy gives us fast math and arrays (positions, distances)


# =============================================================================
# Agent  —  one character on the grid
# =============================================================================
class Agent:
    """
    A simple data object for one agent (pursuer or evader).

    Attributes:
        name  — label shown on the GUI canvas, e.g. "evader" or "pursuer_0"
        pos   — numpy [x, y] position on the grid; float so movement is smooth
        speed — how many grid units the agent moves per step
    """
    def __init__(self, name, pos, speed):
        self.name  = name
        self.pos   = np.array(pos, dtype=float)
        self.speed = float(speed)


# =============================================================================
# TemplateSimulation  —  YOUR CORE ALGORITHM LIVES HERE
# =============================================================================
class TemplateSimulation:
    """
    This class controls the game rules and how agents move.

    To make your own algorithm:
      - Edit the  step()  method below.
      - The rest of the class (setup, boundaries, win checking) can stay as-is.

    The GUI needs these methods to exist:
        step()  — advance one time step, return winner string or None
        reset() — randomize positions for a new episode
    """

    def __init__(
        self,
        world_width=10,
        world_height=10,
        num_pursuers=3,
        pursuer_speed=0.4,
        evader_speed=0.5,
        capture_distance=0.5,
        goal=None,
    ):
        """
        Set up the simulation with the given settings.

        These parameters come from the GUI sliders when "Start New Session" is pressed.
        You normally don't need to change this method.
        """
        self.world_width      = float(world_width)
        self.world_height     = float(world_height)
        self.capture_distance = float(capture_distance)

        # The goal is where the evader is trying to go.
        # If not given, place it at the top-right corner of the grid.
        self.goal = np.array(
            goal if goal is not None else (self.world_width - 1, self.world_height - 1),
            dtype=float,
        )

        # Create the evader at a random position
        self.evader = Agent("evader", self._random_pos(), evader_speed)

        # Create all pursuers at random positions
        self.pursuers = [
            Agent(f"pursuer_{i}", self._random_pos(), pursuer_speed)
            for i in range(int(num_pursuers))
        ]

        self.step_count = 0   # How many steps have run this episode

        # ------------------------------------------------------------------
        # Q-learning configuration (for evader)
        # ------------------------------------------------------------------
        self.q_table = {}

        # 9 discrete evader moves: stay + 8 directions
        self.action_vectors = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, -1.0]),
            np.array([-1.0, 1.0]),
            np.array([-1.0, -1.0]),
        ]

        self.alpha = 0.15
        self.gamma = 0.95
        self.epsilon = 0.45
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

        self.episode_count = 0
        self.evader_win_count = 0
        self.total_terminal_episodes = 0
        self._max_world_dist = float(np.linalg.norm([self.world_width, self.world_height]))

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS  (you probably won't need to change these)
    # -------------------------------------------------------------------------

    def _random_pos(self):
        """Return a random (x, y) position anywhere inside the grid."""
        return np.random.uniform([0, 0], [self.world_width, self.world_height])

    def _move_toward(self, agent, target):
        """
        Move an agent one step directly toward a target position.

        Steps:
          1. Calculate the direction vector (target minus agent position).
          2. Normalize it to length 1 so speed stays constant.
          3. Add  speed × direction  to the agent position.
          4. Clamp to keep the agent inside the grid boundary.
        """
        direction = target - agent.pos
        dist      = np.linalg.norm(direction)   # Euclidean distance to target
        if dist > 0:
            direction = direction / dist         # Normalize: make length = 1
        agent.pos += agent.speed * direction

        # Keep the agent inside the grid — np.clip(v, min, max) clamps every value
        agent.pos = np.clip(agent.pos, [0, 0], [self.world_width, self.world_height])

    # -------------------------------------------------------------------------
    # Q-learning helpers (evader policy)
    # -------------------------------------------------------------------------

    def _discrete_xy(self, point):
        """Convert float [x, y] to clipped integer grid coordinates."""
        x = int(np.clip(np.round(point[0]), 0, self.world_width))
        y = int(np.clip(np.round(point[1]), 0, self.world_height))
        return x, y

    def _nearest_pursuer(self):
        """Return the nearest pursuer to the evader and its distance."""
        best = None
        best_dist = float("inf")
        for pursuer in self.pursuers:
            d = float(np.linalg.norm(pursuer.pos - self.evader.pos))
            if d < best_dist:
                best_dist = d
                best = pursuer
        return best, best_dist

    def _pursuer_distances_from_evader(self):
        """Return pursuers sorted by distance to evader: [(pursuer, dist), ...]."""
        pairs = [
            (pursuer, float(np.linalg.norm(pursuer.pos - self.evader.pos)))
            for pursuer in self.pursuers
        ]
        pairs.sort(key=lambda item: item[1])
        return pairs

    def _threat_density_count(self, distances):
        """Count how many pursuers are inside a danger radius around the evader."""
        danger_radius = self.capture_distance * 2.4
        return int(sum(1 for d in distances if d < danger_radius))

    def _direction_sign(self, value, deadzone=0.20):
        """Map a float delta to {-1, 0, 1} with a small deadzone around 0."""
        if value > deadzone:
            return 1
        if value < -deadzone:
            return -1
        return 0

    def _distance_bin(self, distance):
        """Bucket a distance into one of 5 coarse bins for compact Q-table states."""
        ratio = float(np.clip(distance / max(self._max_world_dist, 1e-6), 0.0, 0.999999))
        return int(ratio * 5)

    def _state_for_evader(self):
        """
        Build a compact relative state tuple for evader learning.

        State contains:
          - goal direction (x, y) relative to evader
          - nearest pursuer direction + binned distance
          - second-nearest pursuer direction + binned distance
          - count of nearby pursuers (threat density)
          - binned distance to goal

        This representation is much smaller than absolute coordinates and
        generalizes better across positions.
        """
        sorted_threats = self._pursuer_distances_from_evader()
        nearest, nearest_dist = sorted_threats[0]
        second, second_dist = sorted_threats[1] if len(sorted_threats) > 1 else sorted_threats[0]

        goal_delta = self.goal - self.evader.pos
        threat1_delta = nearest.pos - self.evader.pos
        threat2_delta = second.pos - self.evader.pos

        goal_dist = float(np.linalg.norm(goal_delta))
        near_count = self._threat_density_count([d for _, d in sorted_threats])

        return (
            self._direction_sign(goal_delta[0]),
            self._direction_sign(goal_delta[1]),
            self._direction_sign(threat1_delta[0]),
            self._direction_sign(threat1_delta[1]),
            self._distance_bin(nearest_dist),
            self._direction_sign(threat2_delta[0]),
            self._direction_sign(threat2_delta[1]),
            self._distance_bin(second_dist),
            min(near_count, 4),
            self._distance_bin(goal_dist),
        )

    def _q_values(self, state):
        """Return Q-values for state, lazily initializing missing keys."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_vectors), dtype=float)
        return self.q_table[state]

    def _choose_action(self, state):
        """Epsilon-greedy action with a safety-aware heuristic tie-breaker."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(len(self.action_vectors)))

        qvals = self._q_values(state).copy()
        # Add a small heuristic prior so early training is less random and
        # the evader avoids obviously dangerous moves.
        for action_index in range(len(self.action_vectors)):
            qvals[action_index] += 0.15 * self._heuristic_action_score(action_index)
        return int(np.argmax(qvals))

    def _move_by_action(self, agent, action_index):
        """Move agent according to one discrete action vector."""
        direction = self.action_vectors[action_index].copy()
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        agent.pos += agent.speed * direction
        agent.pos = np.clip(agent.pos, [0, 0], [self.world_width, self.world_height])

    def _simulated_evader_pos_for_action(self, action_index):
        """Return evader next position for an action without mutating simulation state."""
        direction = self.action_vectors[action_index].copy()
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        next_pos = self.evader.pos + self.evader.speed * direction
        return np.clip(next_pos, [0, 0], [self.world_width, self.world_height])

    def _heuristic_action_score(self, action_index):
        """Heuristic value using all pursuers (not just the nearest one)."""
        next_pos = self._simulated_evader_pos_for_action(action_index)
        goal_dist = float(np.linalg.norm(next_pos - self.goal))

        all_dists = [float(np.linalg.norm(next_pos - p.pos)) for p in self.pursuers]
        nearest_dist = min(all_dists)
        # Harmonic-style crowd distance: penalizes being close to many pursuers.
        crowd_pressure = float(sum(1.0 / max(d, 1e-6) for d in all_dists))
        crowded_count = self._threat_density_count(all_dists)

        score = (-0.7 * goal_dist) + (1.7 * nearest_dist) - (1.3 * crowd_pressure) - (0.6 * crowded_count)
        if nearest_dist < self.capture_distance * 1.50:
            score -= 2.0
        if nearest_dist < self.capture_distance * 1.10:
            score -= 4.0
        return score

    def _reward(
        self,
        old_goal_dist,
        new_goal_dist,
        old_near_dist,
        new_near_dist,
        old_crowd_pressure,
        new_crowd_pressure,
        old_near_count,
        new_near_count,
        terminal,
    ):
        """
        Reward for evader behavior.

        - Large positive if evader reaches goal.
        - Large negative if captured.
        - Shape toward goal.
        - Strongly reward reducing crowd pressure from multiple pursuers.
        - Small positive living reward to encourage survival.
        """
        if terminal == "evader_win":
            return 220.0
        if terminal == "pursuers_win":
            return -260.0

        toward_goal = (old_goal_dist - new_goal_dist) * 8.0
        away_from_nearest = (new_near_dist - old_near_dist) * 9.0
        reduce_crowding = (old_crowd_pressure - new_crowd_pressure) * 7.0
        reduce_nearby_count = (old_near_count - new_near_count) * 2.5
        danger_penalty = -3.0 if new_near_dist < self.capture_distance * 1.30 else 0.0
        return toward_goal + away_from_nearest + reduce_crowding + reduce_nearby_count + danger_penalty - 0.02

    def _update_q(self, state, action, reward, next_state, terminal):
        """Standard Q-learning update."""
        qvals = self._q_values(state)
        current_q = qvals[action]
        if terminal is None:
            target = reward + self.gamma * float(np.max(self._q_values(next_state)))
        else:
            target = reward
        qvals[action] = current_q + self.alpha * (target - current_q)

    def check_terminal(self):
        """
        Check whether someone has won.

        Returns:
            "pursuers_win" — if any pursuer is within capture_distance of the evader
            "evader_win"   — if the evader is within 0.5 units of the goal
            None           — nobody won yet, keep going
        """
        for pursuer in self.pursuers:
            if np.linalg.norm(pursuer.pos - self.evader.pos) < self.capture_distance:
                return "pursuers_win"
        if np.linalg.norm(self.evader.pos - self.goal) < 0.5:
            return "evader_win"
        return None

    def reset(self):
        """Randomize all positions while keeping learned Q-values."""
        self.evader.pos = self._random_pos()
        self.goal       = self._random_pos()   # Give the evader a new goal each episode
        for pursuer in self.pursuers:
            pursuer.pos = self._random_pos()
        self.step_count = 0
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # =========================================================================
    # YOUR ALGORITHM GOES HERE
    # =========================================================================
    def step(self):
        """One step: evader uses Q-learning; pursuers move greedily toward evader."""
        state = self._state_for_evader()
        action = self._choose_action(state)

        old_goal_dist = float(np.linalg.norm(self.evader.pos - self.goal))
        old_all_dists = [float(np.linalg.norm(self.evader.pos - p.pos)) for p in self.pursuers]
        old_near_dist = min(old_all_dists)
        old_crowd_pressure = float(sum(1.0 / max(d, 1e-6) for d in old_all_dists))
        old_near_count = self._threat_density_count(old_all_dists)

        # Evader acts from learned policy.
        self._move_by_action(self.evader, action)

        # Pursuers are greedy opponents.
        for pursuer in self.pursuers:
            self._move_toward(pursuer, self.evader.pos)

        terminal = self.check_terminal()
        if terminal is not None:
            self.total_terminal_episodes += 1
            if terminal == "evader_win":
                self.evader_win_count += 1

        new_goal_dist = float(np.linalg.norm(self.evader.pos - self.goal))
        new_all_dists = [float(np.linalg.norm(self.evader.pos - p.pos)) for p in self.pursuers]
        new_near_dist = min(new_all_dists)
        new_crowd_pressure = float(sum(1.0 / max(d, 1e-6) for d in new_all_dists))
        new_near_count = self._threat_density_count(new_all_dists)
        next_state = self._state_for_evader()

        reward = self._reward(
            old_goal_dist,
            new_goal_dist,
            old_near_dist,
            new_near_dist,
            old_crowd_pressure,
            new_crowd_pressure,
            old_near_count,
            new_near_count,
            terminal,
        )
        self._update_q(state, action, reward, next_state, terminal)

        self.step_count += 1
        return terminal


# =============================================================================
# GUI COMPATIBILITY LAYER  —  no need to change anything below this line
# =============================================================================
# The GUI (PEG_Gui.py) was designed around a specific data structure.
# These classes act as a bridge between our simulation and that structure.
#
# Think of it like an adapter plug: our simulation produces data in one
# "shape", and these classes reshape it so the GUI can read it without
# needing to know anything about how our simulation works internally.
#
# The GUI reads:
#   env.unwrapped.world.agents
# Each agent needs:
#   .name, .adversary, .state.p_pos, .state.p_vel

class _CompatState:
    """Stores position and velocity in the exact attributes the GUI reads."""
    def __init__(self, pos, vel):
        self.p_pos = np.array(pos, dtype=float)
        self.p_vel = np.array(vel, dtype=float)


class _CompatAgent:
    """An agent wrapper in the format the GUI expects."""
    def __init__(self, name, adversary, pos):
        self.name      = name
        self.adversary = adversary                              # True = pursuer, False = evader
        self.state     = _CompatState(pos=pos, vel=np.zeros(2))  # Velocity starts at zero


class _CompatWorld:
    """
    Keeps a list of _CompatAgent objects in sync with the real simulation.

    After every call to step(), _sync_from_sim() copies new positions
    from TemplateSimulation's agents into these GUI-readable wrappers.
    It also computes velocity = new_pos - old_pos for the info panel display.
    """
    def __init__(self, sim):
        self.sim    = sim
        self.agents = []
        self._sync_from_sim()   # Sync once immediately so the GUI has initial data

    def _sync_from_sim(self):
        """Copy latest positions from the simulation into the compat agent list."""
        old_pos = {a.name: a.state.p_pos.copy() for a in self.agents}  # Save old positions
        updated = []

        for pursuer in self.sim.pursuers:
            agent = _CompatAgent(pursuer.name, adversary=True, pos=pursuer.pos)
            if pursuer.name in old_pos:
                agent.state.p_vel = agent.state.p_pos - old_pos[pursuer.name]  # velocity = displacement
            updated.append(agent)

        evader = _CompatAgent(self.sim.evader.name, adversary=False, pos=self.sim.evader.pos)
        if self.sim.evader.name in old_pos:
            evader.state.p_vel = evader.state.p_pos - old_pos[self.sim.evader.name]
        updated.append(evader)

        self.agents = updated


class _CompatUnwrapped:
    """
    A minimal holder so  env.unwrapped.world  resolves correctly.
    (The GUI expects this chain of attributes to exist.)
    """
    def __init__(self, world):
        self.world = world


class TemplateEnvAdapter:
    """
    The adapter object the GUI talks to.

    This sits between the GUI and your simulation:
        GUI  →  adapter (env)  →  TemplateSimulation

    The GUI calls:
        env.configure(...)     — rebuild simulation with new slider values
        env.reset()            — randomize positions for a new episode
        env.unwrapped.world    — read agent positions for drawing

    You should not need to change this class.
    """

    def __init__(self):
        self.sim       = TemplateSimulation()
        self.unwrapped = _CompatUnwrapped(_CompatWorld(self.sim))

    def configure(
        self,
        num_pursuers=None,
        pursuer_speed=None,
        evader_speed=None,
        world_width=None,
        world_height=None,
    ):
        """
        Rebuild the simulation with updated settings from the GUI sliders.
        Called when the user presses "Start New Session".
        Any parameter left as None keeps its current value.
        """
        self.sim = TemplateSimulation(
            world_width      = self.sim.world_width           if world_width    is None else world_width,
            world_height     = self.sim.world_height          if world_height   is None else world_height,
            num_pursuers     = len(self.sim.pursuers)         if num_pursuers   is None else int(num_pursuers),
            pursuer_speed    = self.sim.pursuers[0].speed     if pursuer_speed  is None else pursuer_speed,
            evader_speed     = self.sim.evader.speed          if evader_speed   is None else evader_speed,
            capture_distance = self.sim.capture_distance,
            goal             = self.sim.goal,
        )
        self.unwrapped.world.sim = self.sim
        self.unwrapped.world._sync_from_sim()

    def reset(self):
        """Randomize positions for a new episode and sync the compat layer."""
        self.sim.reset()
        self.unwrapped.world._sync_from_sim()

    def render(self):
        """Required by the plugin interface but unused — the GUI does all drawing."""
        return None


# =============================================================================
# REQUIRED EXPORTS  —  the GUI loader looks for exactly these two names
# =============================================================================

# `env` is the shared adapter instance that the GUI stores and reuses.
env = TemplateEnvAdapter()   # REQUIRED — do not rename or remove


def get_info_sections(simulation, context=None):
    """Optional GUI hook: display Q-learning stats in the info panel."""
    if simulation is None:
        return []

    q_table_size = len(getattr(simulation, "q_table", {}))
    epsilon_value = getattr(simulation, "epsilon", None)
    episode_value = getattr(simulation, "episode_count", None)
    evader_wins_value = getattr(simulation, "evader_win_count", None)
    terminal_episodes = getattr(simulation, "total_terminal_episodes", None)

    lines = [f"Q-table states : {q_table_size}"]
    if epsilon_value is not None:
        lines += [f"Epsilon        : {float(epsilon_value):.4f}"]
    if episode_value is not None:
        lines += [f"Episodes       : {int(episode_value)}"]
    if evader_wins_value is not None:
        lines += [f"Evader wins    : {int(evader_wins_value)}"]
    if terminal_episodes:
        win_rate = float(evader_wins_value) / float(terminal_episodes)
        lines += [f"Evader winrate : {100.0 * win_rate:.1f}%"]

    return [("Q-LEARNING EVADER", lines)]


def step(env):
    """
    Run one simulation step and update the compat layer so the GUI can draw.

    This is called by the GUI's main loop every frame (or when "+1 Step" is clicked).

    Returns:
        "pursuers_win"  — a pursuer caught the evader
        "evader_win"    — the evader reached the goal
        None            — episode still running
    """
    result = env.sim.step()                 # Run one step of your algorithm
    env.unwrapped.world._sync_from_sim()    # Sync new positions to the GUI-readable layer
    return result
