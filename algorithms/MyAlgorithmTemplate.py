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
        """Randomize all positions to start a fresh episode."""
        self.evader.pos = self._random_pos()
        self.goal       = self._random_pos()   # Give the evader a new goal each episode
        for pursuer in self.pursuers:
            pursuer.pos = self._random_pos()
        self.step_count = 0

    # =========================================================================
    # YOUR ALGORITHM GOES HERE
    # =========================================================================
    def step(self):
        """
        Advance the simulation one step.  THIS IS WHERE YOU WRITE YOUR LOGIC.

        The example below is a simple default:
          - Evader moves toward the goal with a small random wobble (jitter).
          - Pursuers greedily chase the evader in a straight line.

        To write your own algorithm, replace the movement code inside this method.
        Keep the structure:  move agents → increment step_count → return check_terminal().

        You have access to:
            self.evader           — the evader Agent (use self.evader.pos, .speed)
            self.pursuers         — list of pursuer Agents
            self.goal             — numpy [x, y] goal position
            self.world_width      — grid width
            self.world_height     — grid height
            self._move_toward(agent, target)  — helper to move an agent toward a point
            self._random_pos()    — helper to get a random grid position
            np.linalg.norm(v)     — length of a vector (straight-line distance)
            np.clip(v, min, max)  — clamp values within a range (for boundaries)

        Returns:
            "pursuers_win", "evader_win", or None  (see check_terminal above)
        """

        # --- EXAMPLE: evader moves toward goal with a small random wobble ---
        # Adding jitter makes the evader slightly unpredictable.
        # np.random.uniform(-0.25, 0.25, size=2) returns a random [dx, dy] offset.
        jitter = np.random.uniform(-0.25, 0.25, size=2)

        # Clip the jittered target so it stays inside the grid
        wobbled_goal = np.clip(
            self.goal + jitter,
            [0, 0],
            [self.world_width, self.world_height],
        )
        self._move_toward(self.evader, wobbled_goal)

        # --- EXAMPLE: pursuers greedily chase the evader ---
        for pursuer in self.pursuers:
            self._move_toward(pursuer, self.evader.pos)

        # Always increment and check at the end
        self.step_count += 1
        return self.check_terminal()


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
