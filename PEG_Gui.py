# =============================================================================
# PEG_Gui.py  —  Pursuer-Evader Game  |  Main window & visual interface
# =============================================================================
#
# WHAT THIS FILE DOES
# -------------------
# Opens a window with three columns:
#   LEFT   — info panel showing live numbers (positions, speeds, win counts)
#   CENTER — the simulation canvas where you can watch agents move
#   RIGHT  — controls (buttons and sliders to change settings)
#
# The actual simulation logic (how agents move) lives in separate files
# inside the  algorithms/  folder.  This file just draws what those
# algorithms produce and sends user commands to them.
#
# HOW TO RUN
# ----------
#   python PEG_Gui.py
#
# =============================================================================


# --- Standard library: load modules by name while the program is running -----
# importlib lets us load a Python file by its filename (a string) at runtime.
# We need this so the user can switch algorithm files from the dropdown
# without restarting the whole program.
import importlib

# pathlib.Path gives us a clean way to work with folder/file paths on any OS.
from pathlib import Path

# --- GUI toolkit (tkinter) ---------------------------------------------------
# tkinter is the built-in Python library for creating windows, buttons, etc.
# We import it as "tk" so we can write  tk.Button(...)  instead of
# tkinter.Button(...)  — shorter and easier to read.
import tkinter as tk

# ttk ("themed tkinter") gives us nicer-looking widgets like the Combobox
# (the dropdown menu for choosing an algorithm).
from tkinter import ttk

# --- Math library (numpy) ----------------------------------------------------
# numpy is a popular library for working with numbers and arrays efficiently.
# Here we mainly use it to calculate distances between agent positions.
import numpy as np


# =============================================================================
# COLOR THEME
# =============================================================================
# Colors are written as hex codes — the same format used in HTML/CSS.
# "#RRGGBB" where RR=red, GG=green, BB=blue each between 00 (dark) and FF (bright).
# Storing them here means you can change the whole look by editing just this section.

BG            = "#121417"   # Very dark gray — the main window background
SIDE_BG       = "#1B1F24"   # Slightly lighter — used for the left and right panels
PANEL_BG      = "#242A31"   # Darker box color — used inside panels (info box, controls box)
TEXT          = "#E6EDF3"   # Near white — the main text color
MUTED         = "#9BA7B4"   # Gray-blue — less important labels and captions
ACCENT        = "#5AB0FF"   # Bright blue — used for highlights and slider knobs
BUTTON_BG     = "#2D333B"   # Dark button background
BUTTON_ACTIVE = "#3A424D"   # Slightly lighter — shown when the mouse hovers over a button
CANVAS_BG     = "#0F1115"   # Almost black — background of the simulation drawing area
GRID_COLOR    = "#2A3138"   # Subtle dark blue-gray — color of the grid lines

CANVAS_PADDING = 20         # How many pixels of empty space to leave around the grid drawing


# =============================================================================
# WHERE TO FIND ALGORITHM FILES
# =============================================================================
# Path(__file__) is the path of this script (PEG_Gui.py).
# .parent goes up one level to the folder this file lives in.
# / "algorithms" then points to the algorithms sub-folder.
ALGO_DIR = Path(__file__).parent / "algorithms"


def discover_algorithms():
    """
    Scan the algorithms/ folder and return a sorted list of module names.

    Example: if the folder contains  SimulationGreedy.py  and  MyTemplate.py ,
    this function returns  ["MyTemplate", "SimulationGreedy"] .

    Files whose names start with an underscore (like  __init__.py ) are
    skipped — those are Python internal/private files, not algorithm files.
    """
    if not ALGO_DIR.exists():
        return []   # Folder doesn't exist yet — return empty list, nothing to show

    return sorted(
        path.stem                               # .stem = filename without the ".py" part
        for path in ALGO_DIR.glob("*.py")       # Find every .py file in algorithms/
        if path.name != "__init__.py"           # Skip the package marker file
        and not path.name.startswith("_")       # Skip any other private/internal files
    )


# =============================================================================
# CREATE THE MAIN WINDOW
# =============================================================================
# tk.Tk() builds the application window.  Everything we create after this
# will appear inside this window.
root = tk.Tk()
root.title("Pursuer Evader GUI")
root.geometry("1280x780")   # Default size when the window first opens (pixels)
root.configure(bg=BG)       # Paint the window background with our dark color


# =============================================================================
# THREE-COLUMN LAYOUT  —  left | center | right
# =============================================================================
# A "Frame" is an invisible rectangular container.  We use three frames placed
# side by side to divide the window into three columns.
left_panel   = tk.Frame(root, bg=SIDE_BG, width=320)   # Info column  (fixed 320 px wide)
center_panel = tk.Frame(root, bg=BG)                    # Canvas area  (stretches with window)
right_panel  = tk.Frame(root, bg=SIDE_BG, width=320)   # Controls col (fixed 320 px wide)

# .pack() places each frame inside the window, one after another.
# The center panel uses  expand=True  so it grows when the window is resized.
for panel, side, expandable in (
    (left_panel,   "left",  False),
    (right_panel,  "right", False),
    (center_panel, "left",  True),    # Placed last so it fills the remaining space
):
    panel.pack(
        side=side,
        fill="both" if expandable else "y",   # Center fills width+height; sides only height
        expand=expandable,
        padx=16 if expandable else 0,         # Small space around the center canvas
        pady=16 if expandable else 0,
    )
    if not expandable:
        panel.pack_propagate(False)           # Lock fixed panels at their set width (320 px)


# =============================================================================
# SIMULATION CANVAS  (the drawing area in the center)
# =============================================================================
# A Canvas widget is a blank area you can draw shapes, lines, and text on.
# Every frame, we clear it and redraw the grid, agents, and goal from scratch.
canvas = tk.Canvas(
    center_panel,
    width=860, height=740,
    bg=CANVAS_BG,
    highlightthickness=1,
    highlightbackground="#30363D",   # Thin border outline around the drawing area
)
canvas.pack(fill="both", expand=True)   # Canvas fills the entire center panel


# =============================================================================
# PANEL TITLE LABELS
# =============================================================================
# Add a bold heading at the top of each side panel.
for panel, title in ((left_panel, "Simulation Info"), (right_panel, "Configuration")):
    tk.Label(
        panel,
        text=title,
        font=("Segoe UI", 16, "bold"),
        bg=SIDE_BG,
        fg=TEXT,
        pady=10,
    ).pack(fill="x", padx=12, pady=(12, 6))


# =============================================================================
# INFO TEXT BOX  (left panel)
# =============================================================================
# This scrollable text area shows live numbers during the simulation:
# agent positions, velocities, win counts, distances, etc.
info_frame = tk.LabelFrame(
    left_panel, text=" Info ",
    bg=PANEL_BG, fg=MUTED,
    font=("Segoe UI", 10, "bold"), bd=1, relief="solid",
)
info_frame.pack(fill="both", expand=True, padx=12, pady=(6, 12))

# A Scrollbar lets the user scroll through longer info text.
# It is connected to the Text widget below.
scrollbar = tk.Scrollbar(info_frame)
scrollbar.pack(side="right", fill="y", pady=10)

# The Text widget stores and displays the info text.
# state="disabled" makes it read-only so the user cannot type in it.
info_text = tk.Text(
    info_frame,
    font=("Consolas", 10),
    bg=PANEL_BG, fg=TEXT,
    insertbackground=TEXT,
    relief="flat", bd=0,
    yscrollcommand=scrollbar.set,   # Link the text widget's scroll to the scrollbar
    wrap="word",
)
info_text.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
scrollbar.config(command=info_text.yview)   # Link the scrollbar's position back to the text
info_text.config(state="disabled")          # Read-only


# =============================================================================
# CONTROLS BOX  (right panel)
# =============================================================================
# This box holds the algorithm selector dropdown, buttons, and sliders.
# It is placed inside a scrollable container so controls remain reachable
# even when the window height is too small to show everything at once.
controls_scroll_container = tk.Frame(right_panel, bg=SIDE_BG)
controls_scroll_container.pack(fill="both", expand=True, padx=12, pady=(6, 12))

controls_canvas = tk.Canvas(
    controls_scroll_container,
    bg=SIDE_BG,
    highlightthickness=0,
    bd=0,
)
controls_canvas.pack(side="left", fill="both", expand=True)

controls_scrollbar = tk.Scrollbar(
    controls_scroll_container,
    orient="vertical",
    command=controls_canvas.yview,
)
controls_scrollbar.pack(side="right", fill="y")
controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

controls_inner = tk.Frame(controls_canvas, bg=SIDE_BG)
controls_window_id = controls_canvas.create_window((0, 0), window=controls_inner, anchor="nw")


def _update_controls_scroll_region(_event=None):
    controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))


def _resize_controls_inner_width(event):
    controls_canvas.itemconfigure(controls_window_id, width=event.width)


def _on_controls_mousewheel(event):
    """Scroll the right controls panel with the mouse wheel."""
    # Windows/macOS: event.delta is positive/negative in wheel increments.
    step = -1 if event.delta > 0 else 1
    controls_canvas.yview_scroll(step, "units")


def _bind_controls_mousewheel(_event=None):
    controls_canvas.bind_all("<MouseWheel>", _on_controls_mousewheel)


def _unbind_controls_mousewheel(_event=None):
    controls_canvas.unbind_all("<MouseWheel>")


controls_inner.bind("<Configure>", _update_controls_scroll_region)
controls_canvas.bind("<Configure>", _resize_controls_inner_width)
controls_canvas.bind("<Enter>", _bind_controls_mousewheel)
controls_canvas.bind("<Leave>", _unbind_controls_mousewheel)
controls_inner.bind("<Enter>", _bind_controls_mousewheel)
controls_inner.bind("<Leave>", _unbind_controls_mousewheel)

controls_frame = tk.LabelFrame(
    controls_inner, text=" Controls ",
    bg=PANEL_BG, fg=MUTED,
    font=("Segoe UI", 10, "bold"), bd=1, relief="solid",
)
controls_frame.pack(fill="x")


# =============================================================================
# RUNTIME STATE  —  variables that track what's happening during the simulation
# =============================================================================

# tk.StringVar is a special tkinter string.  When you assign a new value to it,
# any widget connected to it (label, dropdown) automatically updates on screen.
selected_algorithm = tk.StringVar(value="")      # Name of the algorithm currently chosen
selected_compare_algorithm = tk.StringVar(value="")  # Optional second algorithm for graph comparison
status_text        = tk.StringVar(value="Ready") # Short message shown below the Load button

running            = False   # True while the simulation plays automatically (Play pressed)
step_delay_seconds = 0.25    # Pause between automatic steps, in seconds (lower = faster)
step_count         = 0       # How many steps have been taken in this session
simulation_step_limit = 300  # Maximum number of steps to run before auto-stopping

# history stores past states so the "-1 Step" button can rewind
# Each entry is a snapshot: (list of positions, list of velocities)
history            = []

pursuer_wins       = 0   # How many times pursuers caught the evader
evader_wins        = 0   # How many times the evader reached the goal
wins_over_time     = [(0, 0)]  # (step_count, pursuer_wins) points used for graphing
compare_pursuer_wins = 0
compare_wins_over_time = [(0, 0)]

# These values come from the sliders.
# They are applied to the simulation when "Start New Session" is clicked.
# (Changing a slider does NOT instantly change the running simulation except speed sliders.)
pursuer_count = 3
pursuer_speed = 0.4
evader_speed  = 0.5
grid_width    = 10
grid_height   = 10

# References to the objects loaded from the algorithm module.
# They are None until the user picks an algorithm and clicks "Load".
active_env      = None   # The environment adapter — manages the simulation internally
active_world    = None   # The "world" the GUI reads to get agent positions for drawing
active_step_fn  = None   # The step() function from the algorithm — advances simulation 1 step
active_module   = None   # Loaded algorithm module (used for optional extra info sections)
compare_env     = None   # Comparison environment (not shown on canvas)
compare_step_fn = None   # Comparison step function


# =============================================================================
# HELPER: build a consistently styled button
# =============================================================================
def make_button(parent, text, command, pady=(6, 6)):
    """
    Create a dark-themed button and add it to 'parent'.

    Parameters:
        parent  — the container widget the button will live inside
        text    — the label displayed on the button
        command — the function to call when the button is clicked
        pady    — vertical padding (space above and below the button in pixels)
    """
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=BUTTON_BG,
        fg=TEXT,
        activebackground=BUTTON_ACTIVE,
        activeforeground=TEXT,
        relief="flat",      # No raised/sunken 3D effect — flat modern look
        bd=0,               # No border around the button
        padx=10,
        pady=6,
        highlightthickness=0,
        cursor="hand2",     # Show a hand-pointer cursor when hovering
    )
    btn.pack(fill="x", padx=10, pady=pady)
    return btn


# =============================================================================
# HELPER: build a consistently styled slider (Scale widget)
# =============================================================================
def make_slider(parent, label, start, end, step, value, callback):
    """
    Create a horizontal slider and add it to 'parent'.

    A slider (also called a Scale widget) is a draggable knob on a track.
    The user slides it to pick a numeric value.

    Parameters:
        parent   — the container widget the slider will live inside
        label    — text shown above the slider
        start    — the minimum value at the left end
        end      — the maximum value at the right end
        step     — how much the value changes per tick (e.g. 0.05 for fine control)
        value    — the starting position of the knob
        callback — function called every time the knob moves; receives the new value as a string
    """
    slider = tk.Scale(
        parent,
        from_=start, to=end,
        resolution=step,
        orient="horizontal",
        label=label,
        command=callback,
    )
    slider.configure(
        bg=PANEL_BG, fg=TEXT,
        troughcolor=BUTTON_ACTIVE,   # Color of the track the knob slides along
        activebackground=ACCENT,
        highlightthickness=0,
        relief="flat",
    )
    slider.set(value)   # Set the knob to the initial position
    slider.pack(fill="x", padx=10, pady=(0, 10))
    return slider


# =============================================================================
# HELPER: read the current grid size from the loaded simulation
# =============================================================================
def current_world_size():
    """
    Return the grid dimensions as (width, height).

    Tries to read them from the active simulation first.
    Falls back to the slider values if no simulation is loaded.

    This function also handles older algorithm files that stored a single
    'world_size' number instead of separate width/height values.
    """
    simulation = getattr(active_env, "sim", None)   # Safely get env.sim; returns None if not set

    if simulation is None:
        return grid_width, grid_height   # No algorithm loaded yet — use slider defaults

    # Modern rectangular grid format
    if hasattr(simulation, "world_width") and hasattr(simulation, "world_height"):
        return float(simulation.world_width), float(simulation.world_height)

    # Older square-only format (single world_size attribute)
    if hasattr(simulation, "world_size"):
        s = float(simulation.world_size)
        return s, s

    return grid_width, grid_height   # Fallback if the attribute names are unexpected


def current_goal_position():
    """
    Return the goal's (x, y) position from the active simulation.
    Falls back to the center of the grid if no goal is defined.
    """
    simulation = getattr(active_env, "sim", None)
    if simulation is not None and hasattr(simulation, "goal"):
        return np.array(simulation.goal, dtype=float)
    # Default: center of the grid
    w, h = current_world_size()
    return np.array([w * 0.5, h * 0.5], dtype=float)


# =============================================================================
# HELPERS: filter agents by role
# =============================================================================
def pursuer_agents():
    """Return only the agents that are marked as adversaries (the pursuers)."""
    # agent.adversary is True for pursuers and False for the evader
    return [a for a in (active_world.agents if active_world else []) if getattr(a, "adversary", False)]

def evader_agents():
    """Return only the agents that are NOT adversaries (the evader)."""
    return [a for a in (active_world.agents if active_world else []) if not getattr(a, "adversary", False)]


# =============================================================================
# STATE SAVING / RESTORING  —  powers the "-1 Step" (rewind) feature
# =============================================================================
def snapshot_state():
    """
    Capture a copy of every agent's position and velocity right now.

    Returns a tuple of two lists:
        (list_of_positions, list_of_velocities)
    Each list has one entry per agent, in the same order as active_world.agents.

    We use .copy() to make sure the saved data won't change when agents move.
    """
    return (
        [a.state.p_pos.copy() for a in active_world.agents],
        [a.state.p_vel.copy() for a in active_world.agents],
    )

def save_state():
    """Push the current state onto the history stack (if a world is loaded)."""
    if active_world is not None:
        history.append(snapshot_state())

def restore_state():
    """
    Go back one step by restoring the previous state from the history stack.

    The history always stores the current frame too, so we need at least
    2 entries before we can step back (current + one earlier frame).
    """
    global step_count
    if active_world is None or len(history) < 2:
        return                                      # Nothing to go back to
    history.pop()                                   # Remove the most recent (current) state
    positions, velocities = history[-1]             # Read the frame just before it
    for agent, pos, vel in zip(active_world.agents, positions, velocities):
        agent.state.p_pos = pos.copy()
        agent.state.p_vel = vel.copy()
    step_count -= 1
    # Remove graph points that are ahead of the restored step.
    while len(wins_over_time) > 1 and wins_over_time[-1][0] > step_count:
        wins_over_time.pop()
    while len(compare_wins_over_time) > 1 and compare_wins_over_time[-1][0] > step_count:
        compare_wins_over_time.pop()


# =============================================================================
# WINNER HANDLING  —  count wins and auto-reset the episode
# =============================================================================
def apply_winner_and_reset(winner):
    """
    If someone won this step, record the win and start a fresh episode.

    'winner' is the value returned by the algorithm's step() function:
        "pursuers_win"  — a pursuer caught the evader
        "evader_win"    — the evader reached the goal
        None            — no winner yet, episode still running
    """
    global pursuer_wins, evader_wins
    if winner == "pursuers_win":
        pursuer_wins += 1
        active_env.reset()   # Randomize positions for the next episode
    elif winner == "evader_win":
        evader_wins += 1
        active_env.reset()


def apply_compare_winner_and_reset(winner):
    """Update win count for comparison simulation and reset it on terminal states."""
    global compare_pursuer_wins
    if compare_env is None:
        return
    if winner == "pursuers_win":
        compare_pursuer_wins += 1
        compare_env.reset()
    elif winner == "evader_win":
        compare_env.reset()


def record_wins_point():
    """Record the current (step, pursuer_wins) point for graph plotting."""
    if wins_over_time and wins_over_time[-1][0] == step_count:
        wins_over_time[-1] = (step_count, pursuer_wins)
    else:
        wins_over_time.append((step_count, pursuer_wins))


def record_compare_wins_point():
    """Record the current (step, compare_pursuer_wins) graph point."""
    if compare_step_fn is None or compare_env is None:
        return
    if compare_wins_over_time and compare_wins_over_time[-1][0] == step_count:
        compare_wins_over_time[-1] = (step_count, compare_pursuer_wins)
    else:
        compare_wins_over_time.append((step_count, compare_pursuer_wins))


# =============================================================================
# DISTANCE CALCULATION  —  used by the info panel
# =============================================================================
def pairwise_distances():
    """
    Calculate the distance between every pursuer and the evader.

    Returns a list of  (pursuer_name, evader_name, distance)  tuples.
    If there are 3 pursuers and 1 evader, the list has 3 entries.

    np.linalg.norm computes the straight-line (Euclidean) distance
    between two 2D points.
    """
    return [
        (pursuer.name, evader.name,
         np.linalg.norm(pursuer.state.p_pos - evader.state.p_pos))
        for pursuer in pursuer_agents()
        for evader  in evader_agents()
    ]


# =============================================================================
# DRAW THE SIMULATION  —  called every step to refresh the canvas
# =============================================================================
def draw_simulation():
    """
    Erase the canvas completely and redraw everything from scratch:
    grid lines, border, corner labels, goal (green circle), and agents.

    The grid is automatically scaled to fill the available canvas space,
    so resizing the window makes the simulation area bigger or smaller.
    """
    canvas.delete("all")   # Remove everything drawn in the previous frame

    if active_world is None:
        return   # No algorithm loaded yet — nothing to draw

    # --- Work out a scale factor so the grid fits neatly on the canvas -------
    # winfo_width/height ask tkinter how big the canvas widget currently is.
    canvas_w = max(canvas.winfo_width(),  120)   # Canvas width  in pixels
    canvas_h = max(canvas.winfo_height(), 120)   # Canvas height in pixels

    world_w, world_h = current_world_size()      # Grid size in simulation units

    # Pick the smaller of the two possible scales so both dimensions fit.
    # We subtract 2*CANVAS_PADDING to leave an empty border around the grid.
    scale = min(
        max(canvas_w - 2 * CANVAS_PADDING, 40) / world_w,
        max(canvas_h - 2 * CANVAS_PADDING, 40) / world_h,
    )

    # Pixel size of the drawn grid
    draw_w = world_w * scale
    draw_h = world_h * scale

    # Top-left corner pixel of the grid, centered on the canvas
    x0 = (canvas_w - draw_w) / 2
    y0 = (canvas_h - draw_h) / 2
    x1 = x0 + draw_w
    y1 = y0 + draw_h

    def to_canvas(point):
        """
        Convert simulation (x, y) to canvas pixel coordinates.
        Note: the canvas Y axis points downward but simulation Y points upward,
        so we flip Y by subtracting from y1 (the bottom of the grid).
        """
        return x0 + point[0] * scale, y1 - point[1] * scale

    # --- Draw vertical grid lines (one per column boundary) ------------------
    for col in range(int(world_w) + 1):
        x = x0 + col * scale
        canvas.create_line(x, y0, x, y1, fill=GRID_COLOR)

    # --- Draw horizontal grid lines (one per row boundary) -------------------
    for row in range(int(world_h) + 1):
        y = y1 - row * scale
        canvas.create_line(x0, y, x1, y, fill=GRID_COLOR)

    # --- Thick border rectangle around the whole grid ------------------------
    canvas.create_rectangle(x0, y0, x1, y1, outline="#4D5763", width=2)

    # --- Corner labels so you can read the coordinate range ------------------
    canvas.create_text(x0, y1 + 14, text="(0, 0)",
                       anchor="w", font=("Segoe UI", 8), fill="#7D8590")
    canvas.create_text(x1, y0 - 10, text=f"({int(world_w)}, {int(world_h)})",
                       anchor="e", font=("Segoe UI", 8), fill="#7D8590")

    # --- Goal: drawn as a filled green circle --------------------------------
    gx, gy = to_canvas(current_goal_position())
    canvas.create_oval(gx - 8, gy - 8, gx + 8, gy + 8, fill="#00C853", outline="")

    # --- Agents: red circles for pursuers, blue for the evader ---------------
    for agent in active_world.agents:
        x, y  = to_canvas(agent.state.p_pos)
        color = "#FF6B6B" if agent.adversary else "#5AB0FF"   # Red = pursuer, Blue = evader
        canvas.create_oval(x - 9, y - 9, x + 9, y + 9, fill=color, outline="")
        canvas.create_text(x + 14, y, text=agent.name, anchor="w", font=("Segoe UI", 9), fill=TEXT)


# =============================================================================
# REFRESH THE INFO PANEL  —  update the left text box and redraw the canvas
# =============================================================================
def refresh_info_panel():
    """
    Rebuild the text content of the left info panel with the latest numbers,
    then redraw the simulation canvas so both update together every step.
    """
    simulation = getattr(active_env, "sim", None)

    # Read current speeds; fall back to slider values if no simulation is loaded
    ps = simulation.pursuers[0].speed if simulation and getattr(simulation, "pursuers", None) else pursuer_speed
    es = simulation.evader.speed      if simulation and hasattr(simulation, "evader")         else evader_speed

    w, h = current_world_size()

    # Build a list of text lines; they will be joined with newlines
    # Group related values into sections so the panel is easier to scan.
    lines = [
        "=== SESSION ===",
        f"Algorithm      : {selected_algorithm.get()}",
        f"Compare algo   : {selected_compare_algorithm.get() or 'None'}",
        f"Status         : {status_text.get()}",
        f"Step           : {step_count}",
        f"Step limit     : {simulation_step_limit}",
        f"Pursuers wins  : {pursuer_wins}",
        f"Compare wins   : {compare_pursuer_wins}",
        f"Evader wins    : {evader_wins}",
        "",
        "=== CONFIGURATION ===",
        f"Grid           : {w:.0f} x {h:.0f}",
        f"Pursuers       : {len(pursuer_agents())}",
        f"Pursuer speed  : {ps:.2f}",
        f"Evader speed   : {es:.2f}",
        "",
    ]

    # Optional external info sections from the loaded algorithm module.
    # Contract (optional): module.get_info_sections(simulation, context_dict)
    # should return:
    #   [
    #     ("SECTION TITLE", ["line 1", "line 2", ...]),
    #     ...
    #   ]
    if active_module is not None and hasattr(active_module, "get_info_sections"):
        try:
            extra_sections = active_module.get_info_sections(
                simulation,
                {
                    "step_count": step_count,
                    "pursuer_wins": pursuer_wins,
                    "evader_wins": evader_wins,
                    "compare_pursuer_wins": compare_pursuer_wins,
                    "simulation_step_limit": simulation_step_limit,
                },
            )
            for section in extra_sections or []:
                if not isinstance(section, (tuple, list)) or len(section) != 2:
                    continue
                title, section_lines = section
                lines += [f"=== {title} ==="]
                lines += [str(item) for item in (section_lines or [])]
                lines += [""]
        except Exception as ex:
            lines += ["=== EXTRA INFO ERROR ===", str(ex), ""]

    # Per-agent position and velocity details
    if active_world is not None:
        lines += ["=== AGENTS ==="]
        for agent in active_world.agents:
            pos = agent.state.p_pos
            vel = agent.state.p_vel
            lines += [
                f"- {agent.name}",
                f"    pos : {pos[0]:.3f}, {pos[1]:.3f}",
                f"    vel : {vel[0]:.3f}, {vel[1]:.3f}",
                "",
            ]
        lines += ["=== DISTANCES ==="]
        lines += [f"{p} → {e} : {dist:.3f}" for p, e, dist in pairwise_distances()]

    # Write lines to the read-only Text widget
    info_text.config(state="normal")       # Temporarily allow writing
    info_text.delete("1.0", "end")         # Clear the old content
    info_text.insert("1.0", "\n".join(lines))
    info_text.config(state="disabled")     # Lock it as read-only again

    draw_simulation()   # Redraw the canvas at the same time


# =============================================================================
# SYNC CONTROLS  —  update sliders to show the loaded algorithm's settings
# =============================================================================
def sync_controls_from_env():
    """
    After loading an algorithm, read its current settings and push them
    to the sliders so the GUI matches what the algorithm is actually using.

    Without this, sliders might show stale values from a previous session.
    """
    global pursuer_count, pursuer_speed, evader_speed, grid_width, grid_height
    simulation = getattr(active_env, "sim", None)
    if simulation is None:
        return

    # Read each setting, keeping the current value as fallback
    pursuer_count           = len(simulation.pursuers)          if hasattr(simulation, "pursuers") else pursuer_count
    pursuer_speed           = simulation.pursuers[0].speed      if getattr(simulation, "pursuers", None) else pursuer_speed
    evader_speed            = simulation.evader.speed           if hasattr(simulation, "evader")   else evader_speed
    grid_width, grid_height = current_world_size()

    # Push the new values to the slider widgets so they reflect the loaded state
    pursuer_count_slider.set(pursuer_count)
    pursuer_speed_slider.set(pursuer_speed)
    evader_speed_slider.set(evader_speed)
    grid_width_slider.set(grid_width)
    grid_height_slider.set(grid_height)


# =============================================================================
# LOAD AN ALGORITHM  —  import a module from algorithms/ and connect it
# =============================================================================
def load_backend():
    """
    Load the algorithm selected in the dropdown and wire it up to the GUI.

    Steps:
      1. Read the chosen name from the dropdown.
      2. Import (or re-import) the .py file from algorithms/<name>.py
         using importlib.  Re-importing means code changes to the file take
         effect immediately — no need to restart the program.
      3. Check that the module has the required 'env' and 'step' objects.
      4. Store those objects, reset the simulation, and update the controls.
      5. Save the initial state so the "-1 Step" button works from frame 0.
    """
    global active_env, active_world, active_step_fn, active_module
    global running, step_count, pursuer_wins, evader_wins, history, wins_over_time
    global compare_pursuer_wins, compare_wins_over_time

    name = selected_algorithm.get().strip()
    if not name:
        status_text.set("No algorithm selected")
        refresh_info_panel()
        return

    try:
        # importlib.import_module("algorithms.SimulationGreedy") loads the file.
        # importlib.reload() forces a fresh load even if it was imported before.
        module = importlib.reload(importlib.import_module(f"algorithms.{name}"))

        # Every algorithm file must expose these two things:
        if not hasattr(module, "env") or not hasattr(module, "step"):
            raise ValueError("Module must expose 'env' and 'step(env)'")

        # Store references to the algorithm's objects
        active_env     = module.env
        active_step_fn = module.step
        active_module  = module

        # Reset the simulation to a fresh starting state
        active_env.reset()
        active_world = active_env.unwrapped.world

        # Reset all session-level counters
        running      = False
        step_count   = 0
        pursuer_wins = 0
        evader_wins  = 0
        history      = []
        wins_over_time = [(0, 0)]
        compare_pursuer_wins = 0
        compare_wins_over_time = [(0, 0)]
        if compare_env is not None:
            if hasattr(compare_env, "configure"):
                compare_env.configure(
                    num_pursuers=pursuer_count,
                    pursuer_speed=pursuer_speed,
                    evader_speed=evader_speed,
                    world_width=grid_width,
                    world_height=grid_height,
                )
            compare_env.reset()

        sync_controls_from_env()   # Update sliders to match the loaded algorithm
        save_state()               # Record frame 0 so "-1 Step" works immediately
        status_text.set("Loaded")

    except Exception as ex:
        # If anything went wrong (file not found, missing 'env', etc.) show the error
        active_module = None
        status_text.set(f"Load failed: {ex}")

    refresh_info_panel()   # Update the info panel and canvas


def load_compare_backend():
    """Load second algorithm used only for graph comparison metrics."""
    global compare_env, compare_step_fn, compare_pursuer_wins, compare_wins_over_time
    name = selected_compare_algorithm.get().strip()
    if not name:
        compare_env = None
        compare_step_fn = None
        compare_pursuer_wins = 0
        compare_wins_over_time = [(0, 0)]
        status_text.set("Comparison disabled")
        refresh_info_panel()
        return
    try:
        module = importlib.reload(importlib.import_module(f"algorithms.{name}"))
        if not hasattr(module, "env") or not hasattr(module, "step"):
            raise ValueError("Compare module must expose 'env' and 'step(env)'")
        compare_env = module.env
        compare_step_fn = module.step
        if hasattr(compare_env, "configure"):
            compare_env.configure(
                num_pursuers=pursuer_count,
                pursuer_speed=pursuer_speed,
                evader_speed=evader_speed,
                world_width=grid_width,
                world_height=grid_height,
            )
        compare_env.reset()
        compare_pursuer_wins = 0
        compare_wins_over_time = [(0, 0)]
        status_text.set("Loaded + compare ready")
    except Exception as ex:
        compare_env = None
        compare_step_fn = None
        status_text.set(f"Compare load failed: {ex}")
    refresh_info_panel()


# =============================================================================
# PLAYBACK CONTROLS
# =============================================================================
def play():
    """Start running the simulation automatically (steps repeat on a timer)."""
    global running
    running = True

def pause():
    """Stop automatic playback.  The simulation keeps its current state."""
    global running
    running = False

def new_session():
    """
    Apply the current slider settings and start a completely fresh episode.

    Use this button whenever you change the grid size, number of pursuers,
    or other settings that need a restart to take effect.

    Order of operations:
      1. Stop automatic play.
      2. Call env.configure() with the current slider values so the algorithm
         rebuilds its simulation with the new settings.
      3. Reset positions to random starting points.
      4. Clear the history and win counters.
    """
    global running, step_count, pursuer_wins, evader_wins, wins_over_time
    global compare_pursuer_wins, compare_wins_over_time
    if active_env is None:
        return   # No algorithm loaded — nothing to reset

    running = False   # Stop auto-play while we prepare the new session

    # configure() is optional — if the algorithm supports it, pass slider values
    if hasattr(active_env, "configure"):
        active_env.configure(
            num_pursuers  = pursuer_count,
            pursuer_speed = pursuer_speed,
            evader_speed  = evader_speed,
            world_width   = grid_width,
            world_height  = grid_height,
        )

    active_env.reset()
    step_count = 0
    pursuer_wins = 0
    evader_wins = 0
    wins_over_time = [(0, 0)]
    compare_pursuer_wins = 0
    compare_wins_over_time = [(0, 0)]
    if compare_env is not None:
        if hasattr(compare_env, "configure"):
            compare_env.configure(
                num_pursuers=pursuer_count,
                pursuer_speed=pursuer_speed,
                evader_speed=evader_speed,
                world_width=grid_width,
                world_height=grid_height,
            )
        compare_env.reset()
    history.clear()
    save_state()
    status_text.set("Session reset")
    refresh_info_panel()

def step_forward():
    """
    Advance the simulation by exactly one step (even when auto-play is off).

    Useful for watching what happens one move at a time.
    Pauses auto-play if it was running.
    """
    global running, step_count
    if active_step_fn is None:
        return
    if step_count >= simulation_step_limit:
        status_text.set("Step limit reached")
        refresh_info_panel()
        return
    running = False
    apply_winner_and_reset(active_step_fn(active_env))   # Run one algorithm step
    if compare_step_fn is not None and compare_env is not None:
        apply_compare_winner_and_reset(compare_step_fn(compare_env))
    step_count += 1
    record_wins_point()
    record_compare_wins_point()
    save_state()
    refresh_info_panel()

def step_backward():
    """
    Rewind the simulation by one step, restoring the previous saved state.

    Has no effect if there is no earlier state in the history.
    Pauses auto-play if it was running.
    """
    global running
    running = False
    if compare_step_fn is not None and compare_env is not None:
        status_text.set("Step backward rewinds visible simulation only")
    restore_state()
    refresh_info_panel()


# =============================================================================
# SLIDER CALLBACKS  —  called every time a slider is moved
# =============================================================================
# Most sliders just store the new value.  The value is applied to the
# simulation when "Start New Session" is clicked.
#
# Speed sliders are special: they update the running simulation immediately
# so you can watch agents speed up or slow down in real time.

def set_step_delay(value):
    """Update how long the program waits between automatic steps."""
    global step_delay_seconds
    step_delay_seconds = float(value)

def set_step_limit(value):
    """Update the maximum number of steps allowed in the current run."""
    global simulation_step_limit
    simulation_step_limit = int(float(value))
    refresh_info_panel()

def set_pursuer_count(value):
    """Update the number of pursuers (takes effect on next new session)."""
    global pursuer_count
    pursuer_count = int(float(value))

def set_grid_width(value):
    """Update the grid width (takes effect on next new session)."""
    global grid_width
    grid_width = float(value)

def set_grid_height(value):
    """Update the grid height (takes effect on next new session)."""
    global grid_height
    grid_height = float(value)

def set_pursuer_speed(value):
    """
    Update pursuer speed — takes effect immediately for all active pursuers.

    We write directly to each pursuer's .speed attribute so the change
    is visible right away without needing a new session.
    """
    global pursuer_speed
    pursuer_speed = float(value)
    simulation = getattr(active_env, "sim", None)
    if simulation and getattr(simulation, "pursuers", None):
        for pursuer in simulation.pursuers:
            pursuer.speed = pursuer_speed   # Apply to every pursuer in the current episode
    refresh_info_panel()

def set_evader_speed(value):
    """
    Update evader speed — takes effect immediately for the active evader.

    Same idea as set_pursuer_speed: writes directly to the evader's
    .speed attribute so the change is instant.
    """
    global evader_speed
    evader_speed = float(value)
    simulation = getattr(active_env, "sim", None)
    if simulation and hasattr(simulation, "evader"):
        simulation.evader.speed = evader_speed
    refresh_info_panel()

def refresh_algorithms_dropdown():
    """
    Re-scan the algorithms/ folder and update the dropdown list.

    Use this after adding or renaming a file in  algorithms/  while
    the GUI is already open — no need to restart.
    """
    names = discover_algorithms()
    algorithm_combo["values"] = names
    compare_algorithm_combo["values"] = names
    if names and selected_algorithm.get() not in names:
        selected_algorithm.set(names[0])   # Auto-select first if prior choice disappeared
    if names and selected_compare_algorithm.get() and selected_compare_algorithm.get() not in names:
        selected_compare_algorithm.set("")
    if not names:
        selected_algorithm.set("")
        selected_compare_algorithm.set("")
        status_text.set("No algorithms found in algorithms/")
    refresh_info_panel()


def show_pursuer_wins_graph():
    """Open a new window showing pursuer wins as a function of step count."""
    graph_window = tk.Toplevel(root)
    graph_window.title("Pursuer Wins Over Steps (Comparison)")
    graph_window.geometry("760x460")
    graph_window.configure(bg=BG)

    graph_canvas = tk.Canvas(
        graph_window,
        width=740,
        height=420,
        bg=CANVAS_BG,
        highlightthickness=1,
        highlightbackground="#30363D",
    )
    graph_canvas.pack(fill="both", expand=True, padx=10, pady=10)

    width = max(graph_canvas.winfo_reqwidth(), 740)
    height = max(graph_canvas.winfo_reqheight(), 420)
    margin_left, margin_right, margin_top, margin_bottom = 60, 20, 20, 45
    plot_x0, plot_y0 = margin_left, margin_top
    plot_x1, plot_y1 = width - margin_right, height - margin_bottom

    graph_canvas.create_text(
        width / 2,
        12,
        text="Pursuer Wins vs Step Count (Primary vs Compare)",
        fill=TEXT,
        font=("Segoe UI", 11, "bold"),
        anchor="n",
    )

    # Axes
    graph_canvas.create_line(plot_x0, plot_y1, plot_x1, plot_y1, fill=TEXT, width=2)  # X axis
    graph_canvas.create_line(plot_x0, plot_y1, plot_x0, plot_y0, fill=TEXT, width=2)  # Y axis

    max_step = max(
        simulation_step_limit,
        max((point[0] for point in wins_over_time), default=0),
        max((point[0] for point in compare_wins_over_time), default=0),
        1,
    )
    max_wins = max(
        max((point[1] for point in wins_over_time), default=0),
        max((point[1] for point in compare_wins_over_time), default=0),
        1,
    )

    # Axis labels and ticks
    for tick in range(6):
        ratio = tick / 5
        x = plot_x0 + ratio * (plot_x1 - plot_x0)
        step_label = int(round(ratio * max_step))
        graph_canvas.create_line(x, plot_y1, x, plot_y1 + 4, fill=MUTED)
        graph_canvas.create_text(x, plot_y1 + 16, text=str(step_label), fill=MUTED, font=("Segoe UI", 8))

    for tick in range(6):
        ratio = tick / 5
        y = plot_y1 - ratio * (plot_y1 - plot_y0)
        win_label = int(round(ratio * max_wins))
        graph_canvas.create_line(plot_x0 - 4, y, plot_x0, y, fill=MUTED)
        graph_canvas.create_text(plot_x0 - 10, y, text=str(win_label), fill=MUTED, font=("Segoe UI", 8), anchor="e")

    graph_canvas.create_text((plot_x0 + plot_x1) / 2, height - 12, text="Step", fill=TEXT, font=("Segoe UI", 9))
    graph_canvas.create_text(14, (plot_y0 + plot_y1) / 2, text="Pursuer Wins", fill=TEXT, font=("Segoe UI", 9), angle=90)

    # Plot primary algorithm line
    if wins_over_time:
        points = []
        for step_value, wins_value in wins_over_time:
            x = plot_x0 + (step_value / max_step) * (plot_x1 - plot_x0)
            y = plot_y1 - (wins_value / max_wins) * (plot_y1 - plot_y0)
            points.extend([x, y])
        if len(points) >= 4:
            graph_canvas.create_line(*points, fill=ACCENT, width=2, smooth=True)
        for i in range(0, len(points), 2):
            graph_canvas.create_oval(points[i] - 2, points[i + 1] - 2, points[i] + 2, points[i + 1] + 2, fill=ACCENT, outline="")

    # Plot comparison algorithm line
    if compare_step_fn is not None and compare_env is not None and compare_wins_over_time:
        compare_points = []
        for step_value, wins_value in compare_wins_over_time:
            x = plot_x0 + (step_value / max_step) * (plot_x1 - plot_x0)
            y = plot_y1 - (wins_value / max_wins) * (plot_y1 - plot_y0)
            compare_points.extend([x, y])
        if len(compare_points) >= 4:
            graph_canvas.create_line(*compare_points, fill="#FF6B6B", width=2, smooth=True)
        for i in range(0, len(compare_points), 2):
            graph_canvas.create_oval(
                compare_points[i] - 3,
                compare_points[i + 1] - 3,
                compare_points[i] + 3,
                compare_points[i + 1] + 3,
                fill="#FF6B6B",
                outline="",
            )
    else:
        graph_canvas.create_text(
            (plot_x0 + plot_x1) / 2,
            plot_y0 + 18,
            text="No compare data yet — choose a compare algorithm and click 'Load Compare'.",
            fill="#FF6B6B",
            font=("Segoe UI", 9),
        )

    # Legend
    legend_y = plot_y0 + 10
    graph_canvas.create_line(plot_x1 - 170, legend_y, plot_x1 - 140, legend_y, fill=ACCENT, width=2)
    graph_canvas.create_text(plot_x1 - 135, legend_y, text=f"Primary: {selected_algorithm.get()}", fill=TEXT, anchor="w", font=("Segoe UI", 8))
    graph_canvas.create_line(plot_x1 - 170, legend_y + 16, plot_x1 - 140, legend_y + 16, fill="#FF6B6B", width=2)
    compare_name = selected_compare_algorithm.get() or "None"
    graph_canvas.create_text(plot_x1 - 135, legend_y + 16, text=f"Compare: {compare_name}", fill=TEXT, anchor="w", font=("Segoe UI", 8))


# =============================================================================
# BUILD THE RIGHT-PANEL CONTROLS
# =============================================================================

# --- Algorithm selector row: [dropdown ▾]  [↻]  [Load] ----------------------
module_row = tk.Frame(controls_frame, bg=PANEL_BG)
module_row.pack(fill="x", padx=10, pady=(10, 6))

# Style the ttk Combobox (dropdown) to fit the dark color theme
style = ttk.Style()
style.theme_use("clam")
style.configure("Algo.TCombobox",
                fieldbackground="#1F242B",
                background=BUTTON_BG,
                foreground=TEXT)

# The dropdown is populated with file names found in algorithms/
algorithm_combo = ttk.Combobox(
    module_row,
    textvariable=selected_algorithm,
    values=discover_algorithms(),
    state="readonly",           # User can only pick from the list, not type freely
    style="Algo.TCombobox",
)
algorithm_combo.pack(side="left", fill="x", expand=True)

# Optional second algorithm used for graph comparison only
compare_row = tk.Frame(controls_frame, bg=PANEL_BG)
compare_row.pack(fill="x", padx=10, pady=(0, 6))

tk.Label(compare_row, text="Compare:", bg=PANEL_BG, fg=MUTED).pack(side="left", padx=(0, 8))
compare_algorithm_combo = ttk.Combobox(
    compare_row,
    textvariable=selected_compare_algorithm,
    values=discover_algorithms(),
    state="readonly",
    style="Algo.TCombobox",
)
compare_algorithm_combo.pack(side="left", fill="x", expand=True)
compare_algorithm_combo.bind("<<ComboboxSelected>>", lambda _event: load_compare_backend())

load_compare_button = tk.Button(
    compare_row,
    text="Load Compare",
    command=load_compare_backend,
    bg=BUTTON_BG,
    fg=TEXT,
    activebackground=BUTTON_ACTIVE,
    activeforeground=TEXT,
    relief="flat",
    bd=0,
    padx=8,
)
load_compare_button.pack(side="left", padx=(8, 0))

# ↻  button: re-scan the algorithms/ folder without restarting the program
refresh_button = tk.Button(
    module_row, text="↻",
    command=refresh_algorithms_dropdown,
    bg=BUTTON_BG, fg=TEXT,
    activebackground=BUTTON_ACTIVE, activeforeground=TEXT,
    relief="flat", bd=0, padx=10,
)
refresh_button.pack(side="left", padx=(8, 0))

# Load button: import the selected algorithm file and connect it to the GUI
load_button = tk.Button(
    module_row, text="Load",
    command=load_backend,
    bg=BUTTON_BG, fg=TEXT,
    activebackground=BUTTON_ACTIVE, activeforeground=TEXT,
    relief="flat", bd=0, padx=10,
)
load_button.pack(side="left", padx=(8, 0))

# Small status label just below the Load button row
tk.Label(controls_frame, textvariable=status_text,
         bg=PANEL_BG, fg=MUTED, anchor="w").pack(fill="x", padx=10, pady=(0, 8))

# --- Playback buttons --------------------------------------------------------
make_button(controls_frame, "Play",              play)
make_button(controls_frame, "Pause",             pause)
make_button(controls_frame, "Start New Session", new_session)

# --- Sliders -----------------------------------------------------------------
# Step Delay: how long between automatic steps (drag left = faster, right = slower)
make_slider(controls_frame, "Step Delay (s)",    0.5,  0.01, 0.01, step_delay_seconds, set_step_delay)
make_slider(controls_frame, "Max Steps",            10, 20000, 10,   simulation_step_limit, set_step_limit)

# Agent configuration sliders (speed sliders apply live; others need new session)
pursuer_count_slider = make_slider(controls_frame, "Number of Pursuers", 1, 12,  1,    pursuer_count, set_pursuer_count)
pursuer_speed_slider = make_slider(controls_frame, "Pursuer Speed",      0.1, 1.5, 0.05, pursuer_speed, set_pursuer_speed)
evader_speed_slider  = make_slider(controls_frame, "Evader Speed",       0.1, 1.5, 0.05, evader_speed,  set_evader_speed)

# Grid size sliders (only applied on "Start New Session")
grid_width_slider  = make_slider(controls_frame, "Grid Width",  1, 40, 1, grid_width,  set_grid_width)
grid_height_slider = make_slider(controls_frame, "Grid Height", 1, 40, 1, grid_height, set_grid_height)

# --- Manual step buttons -----------------------------------------------------
make_button(controls_frame, "+1 Step", step_forward)
make_button(controls_frame, "-1 Step", step_backward)
make_button(controls_frame, "Show Pursuer Wins Graph", show_pursuer_wins_graph)


# =============================================================================
# MAIN LOOP  —  the heartbeat that keeps the simulation running
# =============================================================================
def loop():
    """
    This function is called repeatedly on a timer to drive the simulation.

    How it works:
      - If  running == True  (Play was pressed) and an algorithm is loaded:
          1. Call the algorithm's step() function to advance one step.
          2. Check for a winner and auto-reset the episode if needed.
          3. Increment the step counter.
          4. Save the state to history (so "-1 Step" can rewind).
          5. Refresh the info panel and canvas.
          6. Schedule the next call after  step_delay_seconds  have passed.

      - If paused (running == False):
          Wait 100 ms then check again.  The window stays fully interactive
          (buttons, sliders, resize) because we never block the main thread.

    This pattern (root.after instead of a while loop) is standard for
    tkinter animations.  A regular while loop would freeze the window.
    """
    global step_count
    if running and active_step_fn is not None:
        if step_count >= simulation_step_limit:
            pause()
            status_text.set("Step limit reached")
            refresh_info_panel()
            root.after(100, loop)
            return
        apply_winner_and_reset(active_step_fn(active_env))
        if compare_step_fn is not None and compare_env is not None:
            apply_compare_winner_and_reset(compare_step_fn(compare_env))
        step_count += 1
        record_wins_point()
        record_compare_wins_point()
        save_state()
        refresh_info_panel()
        root.after(int(step_delay_seconds * 1000), loop)   # Schedule next step
    else:
        root.after(100, loop)   # Idle: check again in 100 ms


# =============================================================================
# STARTUP  —  auto-discover and load the first algorithm on launch
# =============================================================================
names = discover_algorithms()
if names and not selected_algorithm.get():
    selected_algorithm.set(names[0])   # Pre-select the first file in the dropdown

if names:
    load_backend()   # Auto-load so the GUI shows something immediately on start
else:
    status_text.set("No algorithms found in algorithms/")
    refresh_info_panel()

# --- Start everything --------------------------------------------------------
# loop() drives the simulation timer.
# root.mainloop() starts the tkinter event loop that handles window events
# (mouse clicks, keyboard, resize) until the window is closed.
loop()
root.mainloop()
