# Pursuer–Evader GUI (Python)

A visual simulator for a pursuer–evader game with a plugin-style algorithm system.

- **Main app:** `PEG_Gui.py`
- **Algorithms folder:** `algorithms/`
- **Included algorithms:**
  - `SimulationGreedy.py`
  - `MyAlgorithmTemplate.py`

---

## What this project does

This app opens a GUI where you can:

- choose an algorithm from a dropdown,
- configure grid width/height and speeds,
- run/pause/step the simulation,
- view live state info (positions, velocities, distances, win counts).

The GUI auto-discovers algorithms from the `algorithms/` folder.

---

## Requirements

- Python **3.10+** (recommended: 3.11 or 3.12)
- `numpy`
- `tkinter` (usually bundled with Python on Windows/macOS)

> `tkinter` note:
> - Windows/macOS Python installers usually include it.
> - On some Linux distros, install it separately (example: `python3-tk`).

---

## Project structure

```text
P8/
├─ PEG_Gui.py
├─ README.md
├─ requirements.txt
└─ algorithms/
   ├─ __init__.py
   ├─ SimulationGreedy.py
   └─ MyAlgorithmTemplate.py
```

---

## Quick start (new device)

### 1) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd P8
```

### 2) Create and activate a virtual environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the GUI

```bash
python PEG_Gui.py
```

---

## How algorithms are loaded

The GUI imports modules from `algorithms/` and expects each algorithm file to export:

1. `env` (adapter/environment object)
2. `step(env)` function

`step(env)` must return one of:

- `"pursuers_win"`
- `"evader_win"`
- `None` (episode still running)

Use `algorithms/MyAlgorithmTemplate.py` as your starting point for new algorithms.

---

## Adding your own algorithm

1. Create a new file in `algorithms/`, e.g. `MyCoolAlgo.py`.
2. Follow the same export contract (`env` and `step(env)`).
3. Launch GUI and click refresh (`↻`) in the algorithm row.
4. Select your algorithm from the dropdown and click **Load**.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

Your environment is missing dependencies. Activate your venv and run:

```bash
pip install numpy
```

### `No algorithms found in algorithms/`

- Make sure files are inside `algorithms/`
- Make sure they are `.py` files
- Ensure they export `env` and `step`
- Click the GUI refresh button (`↻`)

### Tkinter errors on Linux

Install tkinter package from your distro, for example:

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

---

## For GitHub

Add a `.gitignore` with at least:

```gitignore
.venv/
__pycache__/
*.pyc
```
