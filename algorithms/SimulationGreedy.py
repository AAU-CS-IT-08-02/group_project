# =============================================================================
# SimulationGreedy.py  —  A "greedy" pursuer-evader algorithm
# =============================================================================
#
# WHAT THIS FILE DOES
# -------------------
# Simulates a game on a rectangular grid where:
#   - PURSUERS  always move in a straight line toward the evader.
#               ("Greedy" = always take the most direct path to your target,
#               no planning ahead, no tricks.)
#   - EVADER    always moves in a straight line toward a goal position.
#   - Agents    cannot leave the grid boundary (clipped to stay inside).
#
# HOW THIS FILE CONNECTS TO THE GUI  (the plugin contract)
# --------------------------------------------------------
# PEG_Gui.py loads algorithm files dynamically.  It looks for two things:
#   env       — an object that manages the simulation state
#   step(env) — a function that advances the simulation by one step
#
# As long as this file is in the  algorithms/  folder, it will appear
# in the GUI's dropdown menu automatically.
#
# =============================================================================

import numpy as np   # numpy gives us fast math and array operations


# =============================================================================
# Agent  —  a single character on the grid (one pursuer or the evader)
# =============================================================================
class Agent:
    """
    Stores the data for one agent.

    Attributes:
        name  — a readable label, e.g. "evader" or "pursuer_0"
        pos   — the agent's current position as a numpy array [x, y]
                Using numpy (instead of a plain list) lets us do math like
                subtraction and normalization directly on it.
        speed — how many grid units the agent moves per step
    """
    def __init__(self, name, pos, speed):
        self.name  = name
        self.pos   = np.array(pos, dtype=float)   # Always stored as floats for smooth movement
        self.speed = speed


# =============================================================================
# PursuitSimulation  —  the core game logic
# =============================================================================
class PursuitSimulation:
    """
    Manages everything about one episode of the pursuit-evasion game.

    Responsibilities:
      - Create and position agents at the start.
      - Move them each step according to the greedy strategy.
      - Keep agents inside the grid boundaries.
      - Decide when someone has won.
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
        Set up the simulation.

        Parameters explained:
            world_width, world_height — size of the rectangular playing field in grid units
            num_pursuers              — how many pursuer agents to create
            pursuer_speed             — how far (in grid units) each pursuer moves per step
            evader_speed              — how far the evader moves per step
            capture_distance          — how close a pursuer must get to "catch" the evader
            goal                      — the (x, y) position the evader is trying to reach;
                                        defaults to the top-right corner of the grid
        """
        self.world_width      = world_width
        self.world_height     = world_height
        self.capture_distance = capture_distance

        # Goal position — evader wins by reaching this spot.
        # If no goal was given, place it at the top-right corner.
        self.goal = np.array(
            goal if goal is not None else (world_width - 1, world_height - 1),
            dtype=float,
        )

        # Create the evader at a random position anywhere on the grid.
        # np.random.uniform([0,0], [w, h]) picks a random (x, y) inside the grid.
        self.evader = Agent(
            "evader",
            np.random.uniform([0, 0], [world_width, world_height]),
            evader_speed,
        )

        # Create all pursuers, each at a random position.
        self.pursuers = []
        for i in range(num_pursuers):
            pos = np.random.uniform([0, 0], [world_width, world_height])
            self.pursuers.append(Agent(f"pursuer_{i}", pos, pursuer_speed))

        self.step_count = 0   # Tracks how many steps have been taken this episode

    # -------------------------------------------------------------------------

    def greedy_move(self, agent, target):
        """
        Move an agent one step directly toward a target position.

        How it works (step by step):
          1. Compute the DIRECTION vector: target minus agent position.
             This points from the agent toward the target.
          2. Find the LENGTH (distance) of that vector.
          3. NORMALIZE: divide the vector by its length to get length = 1.
             This ensures speed doesn't depend on how far away the target is.
          4. Move: add  (speed × direction)  to the agent's position.
          5. CLAMP: if the new position is outside the grid, push it back to
             the nearest edge (so agents can never leave the boundary).

        "Greedy" refers to the strategy: always move straight toward your target,
        no matter what.  Simple, fast, but easy to outsmart.
        """
        direction = target - agent.pos          # Vector pointing from agent to target
        dist      = np.linalg.norm(direction)   # Straight-line distance to target

        if dist > 0:
            direction = direction / dist        # Normalize so length = 1

        agent.pos += agent.speed * direction    # Move by (speed) units toward target

        # Keep the agent inside the grid.
        # np.clip(value, min, max) replaces anything below min with min,
        # and anything above max with max.
        agent.pos = np.clip(agent.pos, [0, 0], [self.world_width, self.world_height])

    # -------------------------------------------------------------------------

    def step(self):
        """
        Advance the game by one time step.

        What happens each step:
          1. The evader moves toward its goal.
          2. Every pursuer moves toward the evader's current position.
          3. We check whether anyone has won.

        Returns:
            "pursuers_win"  — if any pursuer is now close enough to catch the evader
            "evader_win"    — if the evader reached the goal
            None            — nobody won yet; the episode continues
        """
        # Evader moves toward the goal
        self.greedy_move(self.evader, self.goal)

        # All pursuers move toward the evader (using its current position)
        for pursuer in self.pursuers:
            self.greedy_move(pursuer, self.evader.pos)

        self.step_count += 1
        return self.check_terminal()   # Check if the episode is now over

    # -------------------------------------------------------------------------

    def check_terminal(self):
        """
        Check whether the game has ended.

        Win conditions:
          Pursuers win: any pursuer is within 'capture_distance' of the evader.
          Evader wins:  the evader is within 0.5 units of the goal.

        Returns one of:  "pursuers_win"  |  "evader_win"  |  None
        """
        # Check every pursuer for a capture
        for pursuer in self.pursuers:
            dist = np.linalg.norm(pursuer.pos - self.evader.pos)
            if dist < self.capture_distance:
                return "pursuers_win"

        # Check if evader reached the goal
        if np.linalg.norm(self.evader.pos - self.goal) < 0.5:
            return "evader_win"

        return None   # Game still going

    # -------------------------------------------------------------------------

    def reset(self):
        """
        Randomize all positions to start a fresh episode.

        The goal also moves to a new random position so each episode
        is different.  step_count resets to 0.
        """
        self.evader.pos = np.random.uniform([0, 0], [self.world_width, self.world_height])
        self.goal       = np.random.uniform([0, 0], [self.world_width, self.world_height])

        for pursuer in self.pursuers:
            pursuer.pos = np.random.uniform([0, 0], [self.world_width, self.world_height])

        self.step_count = 0

    # -------------------------------------------------------------------------

    def get_state(self):
        """
        Return a snapshot dictionary of current positions (useful for logging).
        Not used by the GUI directly, but handy if you want to print or record data.
        """
        return {
            "evader":   self.evader.pos,
            "pursuers": [p.pos for p in self.pursuers],
            "goal":     self.goal,
            "step":     self.step_count,
        }


# =============================================================================
# GUI COMPATIBILITY LAYER
# =============================================================================
# PEG_Gui.py was built to read agent data in a specific format.
# The three classes below act as a "bridge" (called an adapter pattern):
# they wrap our simulation data into the exact structure the GUI expects.
#
# The GUI reads:
#   env.unwrapped.world.agents   — list of agent objects
#
# Each agent object must have:
#   .name             — string identifier
#   .adversary        — True for pursuers, False for the evader
#   .state.p_pos      — numpy [x, y] current position
#   .state.p_vel      — numpy [vx, vy] velocity (displacement from last step)

class _CompatState:
    """Holds position and velocity in the format the GUI expects to read."""
    def __init__(self, pos, vel):
        self.p_pos = np.array(pos, dtype=float)
        self.p_vel = np.array(vel, dtype=float)


class _CompatAgent:
    """Wraps a simulation agent in the GUI-readable format."""
    def __init__(self, name, adversary, pos):
        self.name      = name
        self.adversary = adversary                              # True = pursuer, False = evader
        self.state     = _CompatState(pos=pos, vel=np.zeros(2))  # Start velocity at zero


class _CompatWorld:
    """
    Maintains a list of _CompatAgent objects that mirrors the simulation.

    After every simulation step, _sync_from_sim() copies new positions
    from PursuitSimulation into these compat agents.
    It also computes velocity by comparing the old and new positions.
    """
    def __init__(self, sim):
        self.sim    = sim
        self.agents = []
        self._sync_from_sim()   # Do an initial sync so the GUI has data immediately

    def _sync_from_sim(self):
        """
        Copy the latest agent positions from the simulation into compat agents.
        Also calculate velocity = (new position) - (old position).
        """
        # Save the previous positions so we can compute velocity below
        old_pos = {agent.name: agent.state.p_pos.copy() for agent in self.agents}

        updated = []

        # Create a compat agent for each pursuer
        for pursuer in self.sim.pursuers:
            agent = _CompatAgent(name=pursuer.name, adversary=True, pos=pursuer.pos)
            # velocity = how much the agent moved this step
            if pursuer.name in old_pos:
                agent.state.p_vel = agent.state.p_pos - old_pos[pursuer.name]
            updated.append(agent)

        # Create a compat agent for the evader
        evader = _CompatAgent(name=self.sim.evader.name, adversary=False, pos=self.sim.evader.pos)
        if self.sim.evader.name in old_pos:
            evader.state.p_vel = evader.state.p_pos - old_pos[self.sim.evader.name]
        updated.append(evader)

        self.agents = updated


class _CompatUnwrapped:
    """
    A tiny holder so  env.unwrapped.world  works exactly as the GUI expects.
    The GUI was designed for a more complex environment library, and this
    recreates just the part of that structure we need.
    """
    def __init__(self, world):
        self.world = world


# =============================================================================
# GreedyEnvAdapter  —  the main adapter object (exposed to the GUI as `env`)
# =============================================================================
class GreedyEnvAdapter:
    """
    Wraps PursuitSimulation and exposes the interface PEG_Gui.py expects.

    The GUI calls:
        env.configure(...)        — rebuild simulation with new slider settings
        env.reset()               — randomize positions for a new episode
        env.unwrapped.world       — the _CompatWorld the GUI reads for drawing
    """

    def __init__(self):
        self.sim       = PursuitSimulation()
        self.unwrapped = _CompatUnwrapped(_CompatWorld(self.sim))

    def configure(self, num_pursuers=None, pursuer_speed=None, evader_speed=None,
                  world_width=None, world_height=None):
        """
        Rebuild the simulation with updated parameters from the GUI sliders.

        Any parameter left as None keeps its current value.
        This is called when the user presses "Start New Session" in the GUI.
        """
        # Build a new simulation, keeping old values for any parameter not provided
        self.sim = PursuitSimulation(
            world_width      = world_width    if world_width    is not None else self.sim.world_width,
            world_height     = world_height   if world_height   is not None else self.sim.world_height,
            num_pursuers     = num_pursuers   if num_pursuers   is not None else len(self.sim.pursuers),
            pursuer_speed    = pursuer_speed  if pursuer_speed  is not None else self.sim.pursuers[0].speed,
            evader_speed     = evader_speed   if evader_speed   is not None else self.sim.evader.speed,
            capture_distance = self.sim.capture_distance,
            goal             = self.sim.goal,
        )
        # Point the compat world at the new simulation and sync it
        self.unwrapped.world.sim = self.sim
        self.unwrapped.world._sync_from_sim()

    def reset(self):
        """Randomize positions to start a fresh episode."""
        self.sim.reset()
        self.unwrapped.world._sync_from_sim()   # Sync new positions to the GUI's compat layer

    def render(self):
        """Required by the plugin interface but unused — the GUI handles all drawing."""
        return None


# =============================================================================
# REQUIRED PLUGIN EXPORTS
# =============================================================================
# PEG_Gui.py looks for exactly these two names when it loads this file.
# If either is missing, the GUI will refuse to load the module.

# `env` — the single shared adapter instance.
# The GUI will call env.reset() right after loading to get a fresh start.
env = GreedyEnvAdapter()


def step(env):
    """
    Run one step of the simulation and sync the compat layer for the GUI.

    Called by the GUI's main loop (or "+1 Step" button) every frame.

    Returns:
        "pursuers_win"  — a pursuer caught the evader this step
        "evader_win"    — the evader reached the goal this step
        None            — the episode is still running
    """
    result = env.sim.step()                  # Advance the simulation physics by one step
    env.unwrapped.world._sync_from_sim()     # Update the compat layer so the GUI can read new positions
    return result
