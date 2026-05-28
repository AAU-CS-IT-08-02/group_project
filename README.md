# RL Pursuit-Evasion — Analysis & Comparison Tool

A desktop application for training, analysing, and comparing reinforcement learning agents in a pursuit-evasion game. One agent (the **pursuer**) tries to catch another (the **evader**), which tries to reach a goal tile before being caught. You can plug in your own algorithms, train them against each other, watch a replay, inspect detailed metrics, and compare two runs side by side.


## Requirements
- **Python 3.9 or later**
- **Tkinter** — included with most Python installations (see notes below)
- **numpy** and **matplotlib** — installed via `requirements.txt`

### A note on Tkinter

Tkinter is part of the Python standard library and is bundled with most installations. However, on some Linux distributions it must be installed separately:

```bash
# Debian / Ubuntu
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

On **Windows** and **macOS**, Tkinter is included with the official Python installer from [python.org](https://www.python.org/downloads/) — no extra step needed. (If you use Homebrew Python on macOS, you may need `brew install python-tk`.)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/rl-pursuit-evasion.git
   cd rl-pursuit-evasion
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Program

From the project root, run:

```bash
python main.py
```

The application window will open. It is sized automatically to fit your screen.

---

## Quick Start

1. **Load Agents tab** — Click *Browse* under **Pursuer** and select `agents/random_pursuer.py`. Click *Browse* under **Evader** and select `agents/q_learning_evader.py`. Both cards should show a green tick. Click **Continue to Training**.

2. **Train & Run tab** — Set *Total Episodes* to `2000` for a quick first run. Leave *Live grid during training* **unchecked** (this keeps training fast). Click **▶ Start**. The left panel updates live as training progresses.

3. **Watch the replay** — When training finishes, the last episode replays automatically in the middle panel. Use the playback controls to step through it. Green = pursuer (P), purple = evader (E), bright green tile = goal (G).

4. **Results tab** — Browse the five metric sub-tabs to see charts of how the agents performed. Click **Export stats to .txt** to save a report.

5. **Compare tab** *(optional)* — Save the current run, train a second run with different agents, then load the saved run to compare them side by side.


## Project Structure

```
rl-pursuit-evasion/
├── main.py                     # Entry point — run this
├── requirements.txt
│
├── environment/                # The game
│   ├── grid.py                 # The 10x10 board and movement
│   └── game.py                 # Episode logic and rewards
│
├── agents/                     # Agent definitions
│   ├── base_agent.py           # The interface every agent must follow
│   ├── agent_template.py       # Copy this to write your own
│   ├── random_pursuer.py       # Example: random pursuer
│   └── q_learning_evader.py    # Example: Q-learning evader
│
├── training/                   # Training machinery
│   ├── runner.py               # Runs episodes and coordinates everything
│   └── metrics.py              # Records and computes all statistics
│
└── ui/                         # The desktop application
    ├── app.py                  # Main window and tabs
    ├── upload_frame.py         # Tab 1: Load agents
    ├── training_frame.py       # Tab 2: Train & watch
    ├── results_frame.py        # Tab 3: Metric charts
    └── comparison_frame.py     # Tab 4: Compare runs
---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'tkinter'` | Install Tkinter for your OS (see [Requirements](#requirements)). |
| `ModuleNotFoundError: No module named 'matplotlib'` | Run `pip install -r requirements.txt`. |
| Charts don't appear / look broken | Make sure you ran `python main.py` from the **project root**, not from inside a subfolder. |
| "AGENT_CLASS missing" when loading an agent | Add `AGENT_CLASS = YourClassName` to the bottom of the agent file. |
| "This is an evader agent, not a pursuer" | The agent file's default `role` doesn't match the slot. Load it in the correct slot. |
| Training is very slow | Turn off *Live grid during training* in the right-hand settings panel. |
| The window is too big / small | It auto-sizes to your screen; drag the edges to resize, or maximise it. |

If you run the program from a terminal, any errors during training are printed there — check the terminal output first if something isn't working.

---
