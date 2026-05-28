"""
training_frame.py — Screen 2: three-panel training view.

Left   : live stats updating every N episodes (win rate, reward, ETA, etc.)
Middle : grid visualisation (disabled during training for speed; shows replay after)
Right  : training configuration sliders, toggles, and Start/Stop buttons
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
from pathlib import Path
from typing import Optional

# ── Optional matplotlib for embedded charts ───────────────────────────────────
HAS_MPL = False  # mini chart uses pure tkinter now


class TrainingFrame(ttk.Frame):
    def __init__(self, parent, state: dict, on_done, palette: dict):
        super().__init__(parent)
        self.state   = state
        self.on_done = on_done
        self.P       = palette

        self._training_thread: Optional[threading.Thread] = None
        self._stop_flag  = threading.Event()
        self._stats_q: queue.Queue = queue.Queue()

        # Live stat vars
        self._episode_var    = tk.StringVar(value="—")
        self._pursuer_wr_var = tk.StringVar(value="—")
        self._evader_wr_var  = tk.StringVar(value="—")
        self._avg_len_var    = tk.StringVar(value="—")
        self._p_reward_var   = tk.StringVar(value="—")
        self._e_reward_var   = tk.StringVar(value="—")
        self._eta_var        = tk.StringVar(value="—")
        self._status_var     = tk.StringVar(value="Ready")

        # Config vars
        self._total_episodes = tk.IntVar(value=2000)
        self._max_steps      = tk.IntVar(value=200)
        self._vis_enabled    = tk.BooleanVar(value=False)  # off during training
        self._vis_replay     = tk.BooleanVar(value=True)
        self._random_spawns  = tk.BooleanVar(value=False)
        self._random_goal    = tk.BooleanVar(value=False)
        self._update_freq    = tk.IntVar(value=50)
        self._replay_speed   = tk.IntVar(value=300)   # ms per step in replay

        # Replay state
        self._replay_episode = []   # list of (pursuer_pos, evader_pos, goal_pos)
        self._replay_idx     = 0
        self._replay_after   = None

        # Reward history for mini-chart
        self._reward_history: list = []   # [(episode, p_reward, e_reward)]

        self._build()

    # ──────────────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=2, minsize=200)
        self.columnconfigure(1, weight=5)
        self.columnconfigure(2, weight=2, minsize=200)
        self.rowconfigure(0, weight=1)

        self._build_left()
        self._build_middle()
        self._build_right()

    # ── Left panel: live stats ────────────────────────────────────────────────

    def _build_left(self):
        # Outer frame holds the scrollbar + canvas together
        outer = tk.Frame(self, bg=self.P["surface"],
                         highlightbackground=self.P["border"],
                         highlightthickness=1)
        outer.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        _canvas = tk.Canvas(outer, bg=self.P["surface"],
                            highlightthickness=0, bd=0)
        _scrollbar = ttk.Scrollbar(outer, orient="vertical",
                                   command=_canvas.yview)
        _canvas.configure(yscrollcommand=_scrollbar.set)

        _canvas.grid(row=0, column=0, sticky="nsew")
        _scrollbar.grid(row=0, column=1, sticky="ns")

        left = tk.Frame(_canvas, bg=self.P["surface"])
        _win_id = _canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_frame_configure(e):
            _canvas.configure(scrollregion=_canvas.bbox("all"))
        def _on_canvas_configure(e):
            _canvas.itemconfig(_win_id, width=e.width)
        def _on_mousewheel(e):
            _canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        left.bind("<Configure>", _on_frame_configure)
        _canvas.bind("<Configure>", _on_canvas_configure)
        _canvas.bind("<MouseWheel>", _on_mousewheel)
        left.bind("<MouseWheel>", _on_mousewheel)

        left.columnconfigure(0, weight=1)

        _lbl(left, "Live Stats", style="header", P=self.P).pack(
            fill="x", padx=14, pady=(14, 2))
        _divider(left, self.P).pack(fill="x", padx=14, pady=4)

        # Episode progress
        _lbl(left, "Episode", style="muted", P=self.P).pack(
            fill="x", padx=14, pady=(6, 0))
        _lbl(left, self._episode_var, style="stat_blue", P=self.P).pack(
            fill="x", padx=14)

        # Progress bar
        self._progress_bar = ttk.Progressbar(left, maximum=100, length=200)
        self._progress_bar.pack(fill="x", padx=14, pady=(4, 10))

        _divider(left, self.P).pack(fill="x", padx=14, pady=4)

        # Win rates
        wr_frame = tk.Frame(left, bg=self.P["surface"])
        wr_frame.pack(fill="x", padx=14, pady=4)
        wr_frame.columnconfigure(0, weight=1)
        wr_frame.columnconfigure(1, weight=1)

        _stat_cell(wr_frame, "Pursuer Win %", self._pursuer_wr_var,
                   self.P["accent3"], self.P, col=0)
        _stat_cell(wr_frame, "Evader Win %", self._evader_wr_var,
                   self.P["accent2"], self.P, col=1)

        _divider(left, self.P).pack(fill="x", padx=14, pady=4)

        _lbl(left, "Avg Episode Length", style="muted", P=self.P).pack(
            fill="x", padx=14, pady=(6, 0))
        _lbl(left, self._avg_len_var, style="stat_blue", P=self.P).pack(
            fill="x", padx=14)

        _divider(left, self.P).pack(fill="x", padx=14, pady=4)

        # Rewards
        rew_frame = tk.Frame(left, bg=self.P["surface"])
        rew_frame.pack(fill="x", padx=14, pady=4)
        rew_frame.columnconfigure(0, weight=1)
        rew_frame.columnconfigure(1, weight=1)

        _stat_cell(rew_frame, "Pursuer Reward", self._p_reward_var,
                   self.P["accent3"], self.P, col=0)
        _stat_cell(rew_frame, "Evader Reward", self._e_reward_var,
                   self.P["accent2"], self.P, col=1)

        _divider(left, self.P).pack(fill="x", padx=14, pady=4)

        _lbl(left, "ETA", style="muted", P=self.P).pack(fill="x", padx=14, pady=(6,0))
        _lbl(left, self._eta_var, style="stat_blue", P=self.P).pack(fill="x", padx=14)

        _divider(left, self.P).pack(fill="x", padx=14, pady=4)

        # Reward mini-chart — pure tkinter, no matplotlib needed
        self._reward_tk_canvas = tk.Canvas(
            left, bg=self.P["surface"],
            height=100, highlightthickness=0, bd=0)
        self._reward_tk_canvas.pack(fill="x", padx=14, pady=(4, 14))

        # Status
        _divider(left, self.P).pack(fill="x", padx=14, pady=4)
        status_row = tk.Frame(left, bg=self.P["surface"])
        status_row.pack(fill="x", padx=14, pady=(4, 14))
        self._status_dot = tk.Label(status_row, text="●",
                                    bg=self.P["surface"], fg=self.P["muted"],
                                    font=("Segoe UI", 10))
        self._status_dot.pack(side="left")
        tk.Label(status_row, textvariable=self._status_var,
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=4)

    # ── Middle panel: visualisation ───────────────────────────────────────────

    def _build_middle(self):
        mid = tk.Frame(self, bg=self.P["bg"])
        mid.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        mid.columnconfigure(0, weight=1)
        mid.rowconfigure(1, weight=1)

        # Header row
        hdr = tk.Frame(mid, bg=self.P["surface"],
                       highlightbackground=self.P["border"],
                       highlightthickness=1)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        _lbl(hdr, "Grid Visualisation", style="header", P=self.P).grid(
            row=0, column=0, padx=14, pady=10, sticky="w")

        # Vis status badge
        self._vis_badge_var = tk.StringVar(value="OFF — faster training")
        badge = tk.Label(hdr, textvariable=self._vis_badge_var,
                         bg=self.P["border"], fg=self.P["muted"],
                         font=("Segoe UI", 8), padx=8, pady=2)
        badge.grid(row=0, column=1, padx=14, pady=10, sticky="e")
        self._vis_badge = badge

        # Canvas
        canvas_frame = tk.Frame(mid, bg=self.P["bg"])
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self._grid_canvas = tk.Canvas(
            canvas_frame,
            bg=self.P["bg"], highlightthickness=0)
        self._grid_canvas.grid(row=0, column=0, sticky="nsew")
        self._grid_canvas.bind("<Configure>", self._on_canvas_resize)

        # Replay controls (hidden until training done)
        self._replay_frame = tk.Frame(mid, bg=self.P["surface"],
                                      highlightbackground=self.P["border"],
                                      highlightthickness=1)
        self._replay_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=(4, 0))
        self._replay_frame.columnconfigure(2, weight=1)
        self._replay_frame.grid_remove()

        ttk.Button(self._replay_frame, text="◀◀", width=3,
                   command=self._replay_start).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(self._replay_frame, text="▶", width=3,
                   command=self._replay_play).grid(row=0, column=1, padx=2, pady=6)
        ttk.Button(self._replay_frame, text="▶▶", width=3,
                   command=self._replay_end).grid(row=0, column=2, padx=2, pady=6)

        self._replay_step_var = tk.StringVar(value="Step: —")
        tk.Label(self._replay_frame, textvariable=self._replay_step_var,
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 9)).grid(row=0, column=3, padx=10, pady=6)

        ttk.Scale(self._replay_frame, from_=0, to=100,
                  variable=tk.DoubleVar(),
                  command=self._on_replay_scrub,
                  orient="horizontal").grid(
            row=0, column=4, sticky="ew", padx=10, pady=6)
        self._replay_frame.columnconfigure(4, weight=1)

        # Draw placeholder
        self._draw_placeholder("Load agents and start training to begin.")

    # ── Right panel: configuration ────────────────────────────────────────────

    def _build_right(self):
        # Outer frame holds scrollbar + canvas
        outer_r = tk.Frame(self, bg=self.P["surface"],
                           highlightbackground=self.P["border"],
                           highlightthickness=1)
        outer_r.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        outer_r.columnconfigure(0, weight=1)
        outer_r.rowconfigure(0, weight=1)

        _rcanvas = tk.Canvas(outer_r, bg=self.P["surface"],
                             highlightthickness=0, bd=0)
        _rscroll = ttk.Scrollbar(outer_r, orient="vertical",
                                 command=_rcanvas.yview)
        _rcanvas.configure(yscrollcommand=_rscroll.set)

        _rcanvas.grid(row=0, column=0, sticky="nsew")
        _rscroll.grid(row=0, column=1, sticky="ns")

        right = tk.Frame(_rcanvas, bg=self.P["surface"])
        _rwin = _rcanvas.create_window((0, 0), window=right, anchor="nw")

        def _r_frame_cfg(e):
            _rcanvas.configure(scrollregion=_rcanvas.bbox("all"))
        def _r_canvas_cfg(e):
            _rcanvas.itemconfig(_rwin, width=e.width)
        def _r_mwheel(e):
            _rcanvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        right.bind("<Configure>", _r_frame_cfg)
        _rcanvas.bind("<Configure>", _r_canvas_cfg)
        _rcanvas.bind("<MouseWheel>", _r_mwheel)
        right.bind("<MouseWheel>", _r_mwheel)

        right.columnconfigure(0, weight=1)

        _lbl(right, "Training Settings", style="header", P=self.P).pack(
            fill="x", padx=14, pady=(14, 2))
        _divider(right, self.P).pack(fill="x", padx=14, pady=6)

        # Sliders
        _lbl(right, "Total Episodes", style="muted", P=self.P).pack(
            fill="x", padx=14, pady=(6, 0))
        ep_row = tk.Frame(right, bg=self.P["surface"])
        ep_row.pack(fill="x", padx=14)
        self._ep_spinbox = ttk.Spinbox(
            ep_row, from_=100, to=100000, increment=100,
            textvariable=self._total_episodes, width=10)
        self._ep_spinbox.pack(side="left")
        tk.Label(ep_row, text=" episodes", bg=self.P["surface"],
                 fg=self.P["muted"], font=("Segoe UI", 9)).pack(side="left")

        _lbl(right, "Max Steps / Episode", style="muted", P=self.P).pack(
            fill="x", padx=14, pady=(10, 0))
        step_row = tk.Frame(right, bg=self.P["surface"])
        step_row.pack(fill="x", padx=14)
        self._steps_spinbox = ttk.Spinbox(
            step_row, from_=10, to=2000, increment=10,
            textvariable=self._max_steps, width=10)
        self._steps_spinbox.pack(side="left")
        tk.Label(step_row, text=" steps", bg=self.P["surface"],
                 fg=self.P["muted"], font=("Segoe UI", 9)).pack(side="left")

        _lbl(right, "Stats Update Every", style="muted", P=self.P).pack(
            fill="x", padx=14, pady=(10, 0))
        freq_row = tk.Frame(right, bg=self.P["surface"])
        freq_row.pack(fill="x", padx=14)
        self._freq_spinbox = ttk.Spinbox(
            freq_row, from_=1, to=500, increment=10,
            textvariable=self._update_freq, width=10)
        self._freq_spinbox.pack(side="left")
        tk.Label(freq_row, text=" episodes", bg=self.P["surface"],
                 fg=self.P["muted"], font=("Segoe UI", 9)).pack(side="left")

        _divider(right, self.P).pack(fill="x", padx=14, pady=12)

        # Visualisation toggle
        vis_row = tk.Frame(right, bg=self.P["surface"])
        vis_row.pack(fill="x", padx=14, pady=4)
        tk.Label(vis_row, text="Live grid during training",
                 bg=self.P["surface"], fg=self.P["text"],
                 font=("Segoe UI", 10)).pack(side="left")
        ttk.Checkbutton(vis_row, variable=self._vis_enabled,
                        command=self._on_vis_toggle).pack(side="right")

        tk.Label(right,
                 text="Turning off speeds up training significantly.",
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 8), wraplength=210, justify="left").pack(
            fill="x", padx=14, pady=(0, 8))

        replay_row = tk.Frame(right, bg=self.P["surface"])
        replay_row.pack(fill="x", padx=14, pady=4)
        tk.Label(replay_row, text="Show replay after training",
                 bg=self.P["surface"], fg=self.P["text"],
                 font=("Segoe UI", 10)).pack(side="left")
        ttk.Checkbutton(replay_row, variable=self._vis_replay).pack(side="right")

        _divider(right, self.P).pack(fill="x", padx=14, pady=12)

        # Replay speed
        _lbl(right, "Replay Speed", style="muted", P=self.P).pack(
            fill="x", padx=14, pady=(0, 0))
        speed_row = tk.Frame(right, bg=self.P["surface"])
        speed_row.pack(fill="x", padx=14)
        ttk.Spinbox(
            speed_row, from_=50, to=2000, increment=50,
            textvariable=self._replay_speed, width=10).pack(side="left")
        tk.Label(speed_row, text=" ms/step", bg=self.P["surface"],
                 fg=self.P["muted"], font=("Segoe UI", 9)).pack(side="left")

        _divider(right, self.P).pack(fill="x", padx=14, pady=16)

        # ── Spawn settings ────────────────────────────────────────────────────
        _lbl(right, "Spawn Settings", style="header", P=self.P).pack(
            fill="x", padx=14, pady=(0, 2))

        spawn_row = tk.Frame(right, bg=self.P["surface"])
        spawn_row.pack(fill="x", padx=14, pady=4)
        tk.Label(spawn_row, text="Random agent spawns",
                 bg=self.P["surface"], fg=self.P["text"],
                 font=("Segoe UI", 10)).pack(side="left")
        ttk.Checkbutton(spawn_row, variable=self._random_spawns).pack(side="right")

        tk.Label(right,
                 text="Agents start at random tiles each episode.",
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 8), wraplength=0, justify="left").pack(
            fill="x", padx=14, pady=(0, 6))

        goal_row = tk.Frame(right, bg=self.P["surface"])
        goal_row.pack(fill="x", padx=14, pady=4)
        tk.Label(goal_row, text="Random goal tile",
                 bg=self.P["surface"], fg=self.P["text"],
                 font=("Segoe UI", 10)).pack(side="left")
        ttk.Checkbutton(goal_row, variable=self._random_goal).pack(side="right")

        tk.Label(right,
                 text="Goal tile moves to a random position each episode.",
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 8), wraplength=0, justify="left").pack(
            fill="x", padx=14, pady=(0, 8))

        _divider(right, self.P).pack(fill="x", padx=14, pady=(0, 16))

        # ── Memory (Q-table / weights) ────────────────────────────────────────
        _lbl(right, "Agent Memory", style="header", P=self.P).pack(
            fill="x", padx=14, pady=(0, 2))
        tk.Label(right,
                 text="Save or load each agent's learned parameters\n"
                      "(Q-table, weights, etc.) independently.",
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 8), justify="left").pack(
            fill="x", padx=14, pady=(0, 8))

        # Pursuer memory row
        p_mem = tk.Frame(right, bg=self.P["surface"])
        p_mem.pack(fill="x", padx=14, pady=(0, 4))
        p_mem.columnconfigure(0, weight=1)
        p_mem.columnconfigure(1, weight=1)

        tk.Frame(p_mem, bg=self.P["accent3"], width=8, height=8).grid(
            row=0, column=0, sticky="w")
        tk.Label(p_mem, text=" Pursuer", bg=self.P["surface"],
                 fg=self.P["text"], font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=(10, 0))

        ttk.Button(p_mem, text="💾 Save",
                   command=lambda: self._save_memory("pursuer")).grid(
            row=1, column=0, sticky="ew", padx=(0, 2), pady=2)
        ttk.Button(p_mem, text="📂 Load",
                   command=lambda: self._load_memory("pursuer")).grid(
            row=1, column=1, sticky="ew", padx=(2, 0), pady=2)

        self._p_mem_status = tk.StringVar(value="")
        tk.Label(p_mem, textvariable=self._p_mem_status,
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 8)).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # Evader memory row
        e_mem = tk.Frame(right, bg=self.P["surface"])
        e_mem.pack(fill="x", padx=14, pady=(0, 8))
        e_mem.columnconfigure(0, weight=1)
        e_mem.columnconfigure(1, weight=1)

        tk.Frame(e_mem, bg=self.P["accent2"], width=8, height=8).grid(
            row=0, column=0, sticky="w")
        tk.Label(e_mem, text=" Evader", bg=self.P["surface"],
                 fg=self.P["text"], font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=(10, 0))

        ttk.Button(e_mem, text="💾 Save",
                   command=lambda: self._save_memory("evader")).grid(
            row=1, column=0, sticky="ew", padx=(0, 2), pady=2)
        ttk.Button(e_mem, text="📂 Load",
                   command=lambda: self._load_memory("evader")).grid(
            row=1, column=1, sticky="ew", padx=(2, 0), pady=2)

        self._e_mem_status = tk.StringVar(value="")
        tk.Label(e_mem, textvariable=self._e_mem_status,
                 bg=self.P["surface"], fg=self.P["muted"],
                 font=("Segoe UI", 8)).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))

        _divider(right, self.P).pack(fill="x", padx=14, pady=(0, 16))

        # Start / Stop buttons
        btn_frame = tk.Frame(right, bg=self.P["surface"])
        btn_frame.pack(fill="x", padx=14, pady=(0, 14))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self._start_btn = ttk.Button(
            btn_frame, text="▶  Start",
            style="Accent.TButton",
            command=self._start_training)
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._stop_btn = ttk.Button(
            btn_frame, text="■  Stop",
            style="Danger.TButton",
            state="disabled",
            command=self._stop_training)
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    # ──────────────────────────────────────────────────────────────────────────
    # Training thread
    # ──────────────────────────────────────────────────────────────────────────

    def _start_training(self):
        pursuer_cls = self.state.get("pursuer_class")
        evader_cls  = self.state.get("evader_class")
        if not pursuer_cls or not evader_cls:
            return

        # Reset state
        self._stop_flag.clear()
        self._reward_history.clear()
        self._set_controls_during_training(True)
        self._update_status("Training…", self.P["accent"])
        self._draw_placeholder("Training in progress — visualisation disabled for speed.")

        self._training_thread = threading.Thread(
            target=self._train_worker,
            args=(pursuer_cls, evader_cls,
                  self._total_episodes.get(),
                  self._max_steps.get(),
                  self._update_freq.get(),
                  self._random_spawns.get(),
                  self._random_goal.get()),
            daemon=True)
        self._training_thread.start()

        # Poll queue for updates
        self.after(50, self._poll_stats)

    def _train_worker(self, pursuer_cls, evader_cls,
                      total_eps, max_steps, update_freq,
                      random_spawns=False, random_goal=False):
        """Runs in background thread. Posts stat dicts to self._stats_q."""
        try:
            from training.runner  import Runner, RunnerConfig
            from environment.game import Game, GameConfig
            from environment.grid import GridConfig

            pursuer = pursuer_cls(role="pursuer", config={})
            evader  = evader_cls(role="evader",   config={})

            grid_config = GridConfig(
                random_spawns = random_spawns,
                random_goal   = random_goal,
            )
            game_config = GameConfig(
                grid_config = grid_config,
                max_steps   = max_steps,
            )
            runner_config = RunnerConfig(
                n_episodes           = total_eps,
                live_update_interval = update_freq,
                smoothing_window     = update_freq,
            )

            runner = Runner(
                pursuer       = pursuer,
                evader        = evader,
                game_config   = game_config,
                runner_config = runner_config,
            )

            # Keep references so the memory save/load buttons can reach them
            self._live_pursuer = pursuer
            self._live_evader  = evader

            # Wire the stop flag: runner.stop() is called when the flag is set.
            # We poll it inside the on_update callback (called every update_freq eps).
            last_live  = {}   # holds the most recent LiveStats snapshot
            last_traj  = []   # holds the last recorded episode trajectory

            def on_update(live):
                nonlocal last_live
                last_live = live

                # Pull the last episode's trajectory from the recorder
                traj = []
                if runner.recorder.episodes:
                    snap = runner.recorder.episodes[-1].snapshots
                    traj = [(s["pursuer_pos"], s["evader_pos"], s["goal_pos"])
                            for s in snap]

                self._stats_q.put({
                    "episode":    live.episode,
                    "total":      live.total_episodes,
                    "pursuer_wr": round(live.pursuer_win_rate * 100, 1),
                    "evader_wr":  round(live.evader_win_rate  * 100, 1),
                    "avg_len":    round(live.avg_episode_length, 1),
                    "p_reward":   round(live.pursuer_reward, 3),
                    "e_reward":   round(live.evader_reward,  3),
                    "eta":        live.eta_seconds,
                    "done":       False,
                    "trajectory": traj,
                    "metrics_obj": None,   # not done yet
                })

                # Yield to main thread so it can process the update
                time.sleep(0.05)

                # Honour the stop button
                if self._stop_flag.is_set():
                    runner.stop()

            summary = runner.train(on_update=on_update)

            # Final push with the complete summary + last trajectory
            traj = []
            if runner.recorder.episodes:
                snap = runner.recorder.episodes[-1].snapshots
                traj = [(s["pursuer_pos"], s["evader_pos"], s["goal_pos"])
                        for s in snap]

            # Win rates come back as 0–1 fractions; convert to % for display
            wr = summary.get("win_rates", {})
            p_wr_series = [v * 100 for v in wr.get("pursuer", [])]
            e_wr_series = [v * 100 for v in wr.get("evader",  [])]
            last_p_wr   = p_wr_series[-1] if p_wr_series else 0.0
            last_e_wr   = e_wr_series[-1] if e_wr_series else 0.0

            cr = summary.get("cumulative_rewards", {})

            self._stats_q.put({
                "episode":    total_eps,
                "total":      total_eps,
                "pursuer_wr": round(last_p_wr, 1),
                "evader_wr":  round(last_e_wr, 1),
                "avg_len":    round(
                    sum(summary.get("episode_lengths", [0])) /
                    max(len(summary.get("episode_lengths", [1])), 1), 1),
                "p_reward":   round(
                    (cr.get("pursuer", [0])[-1] if cr.get("pursuer") else 0), 3),
                "e_reward":   round(
                    (cr.get("evader",  [0])[-1] if cr.get("evader")  else 0), 3),
                "eta":        0,
                "done":       True,
                "trajectory": traj,
                "metrics_obj": runner.recorder,
            })

        except ImportError:
            # Backend not on sys.path — run simulation for UI testing
            self._simulate_training(total_eps, max_steps, update_freq)

        except Exception as e:
            import traceback
            msg = f"Training error: {e}"
            print(f"[_train_worker] {msg}")
            traceback.print_exc()
            # Push a done signal so the UI unlocks — never leave it frozen
            self._stats_q.put({
                "episode":    total_eps,
                "total":      total_eps,
                "pursuer_wr": 0.0,
                "evader_wr":  0.0,
                "avg_len":    0.0,
                "p_reward":   0.0,
                "e_reward":   0.0,
                "eta":        0,
                "done":       True,
                "trajectory": [],
                "metrics_obj": None,
                "error":       msg,
            })

    def _simulate_training(self, total_eps, max_steps, update_freq):
        """Fake training loop so the UI can be tested without the backend."""
        import random
        import math

        t_start = time.time()
        p_wr = 0.45
        e_wr = 0.55

        for ep in range(1, total_eps + 1):
            if self._stop_flag.is_set():
                break

            # Drift win rates as "learning" progresses
            p_wr = min(0.85, p_wr + random.gauss(0.001, 0.01))
            e_wr = 1.0 - p_wr

            time.sleep(0.002)  # simulate episode time

            if ep % update_freq == 0 or ep == total_eps:
                elapsed = time.time() - t_start
                rate = ep / elapsed if elapsed > 0 else 1
                eta  = (total_eps - ep) / rate

                # Fake trajectory (10x10 grid)
                traj = []
                px, py = 9, 9
                ex, ey = 0, 0
                gx, gy = 5, 5
                for _ in range(random.randint(10, 50)):
                    traj.append(((px, py), (ex, ey), (gx, gy)))
                    px = max(0, min(9, px + random.randint(-1, 1)))
                    py = max(0, min(9, py + random.randint(-1, 1)))
                    ex = max(0, min(9, ex + random.randint(-1, 1)))
                    ey = max(0, min(9, ey + random.randint(-1, 1)))

                self._stats_q.put({
                    "episode":    ep,
                    "total":      total_eps,
                    "pursuer_wr": round(p_wr * 100, 1),
                    "evader_wr":  round(e_wr * 100, 1),
                    "avg_len":    round(random.gauss(40, 5), 1),
                    "p_reward":   round(random.gauss(-0.2, 0.3), 3),
                    "e_reward":   round(random.gauss(0.1, 0.3), 3),
                    "eta":        eta,
                    "done":       ep == total_eps,
                    "trajectory": traj,
                    "metrics_obj": None,
                })

    def _poll_stats(self):
        try:
            while True:
                try:
                    data = self._stats_q.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._apply_stats(data)
                except Exception as e:
                    import traceback
                    print(f"[training_frame] _apply_stats error: {e}")
                    traceback.print_exc()
        except Exception as e:
            print(f"[training_frame] _poll_stats error: {e}")

        if self._training_thread and self._training_thread.is_alive():
            self.after(50, self._poll_stats)
        else:
            # Thread finished — final drain
            while True:
                try:
                    data = self._stats_q.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._apply_stats(data)
                except Exception as e:
                    import traceback
                    print(f"[training_frame] final drain error: {e}")
                    traceback.print_exc()

    def _apply_stats(self, data: dict):
        try:
            ep    = data["episode"]
            total = data["total"]

            self._episode_var.set(f"{ep:,} / {total:,}")
            self._progress_bar["value"] = int(ep / total * 100)
            self._pursuer_wr_var.set(f"{data['pursuer_wr']:.1f}%")
            self._evader_wr_var.set(f"{data['evader_wr']:.1f}%")
            self._avg_len_var.set(f"{data['avg_len']:.1f} steps")
            self._p_reward_var.set(f"{data['p_reward']:.3f}")
            self._e_reward_var.set(f"{data['e_reward']:.3f}")

            eta = data["eta"]
            if eta < 60:
                self._eta_var.set(f"{eta:.0f}s")
            elif eta < 3600:
                self._eta_var.set(f"{eta/60:.1f}m")
            else:
                self._eta_var.set(f"{eta/3600:.1f}h")

            self._reward_history.append((ep, data["p_reward"], data["e_reward"]))
            if len(self._reward_history) >= 2:
                try:
                    self._update_reward_chart()
                except Exception as chart_err:
                    print(f"[chart] {chart_err}")

            if data.get("done"):
                self._on_training_complete(data)

        except Exception as e:
            import traceback
            print(f"[_apply_stats] {e}")
            traceback.print_exc()

    def _update_reward_chart(self):
        """Draw reward curves directly on tk.Canvas — no matplotlib, no blocking."""
        c = self._reward_tk_canvas
        c.delete("all")

        if len(self._reward_history) < 2:
            return

        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            w, h = 200, 100

        pad = 4
        eps = [r[0] for r in self._reward_history]
        p_r = [r[1] for r in self._reward_history]
        e_r = [r[2] for r in self._reward_history]

        all_vals = p_r + e_r
        mn, mx = min(all_vals), max(all_vals)
        if mx == mn:
            mx = mn + 1

        def to_x(ep):
            return pad + (ep - eps[0]) / max(eps[-1] - eps[0], 1) * (w - 2*pad)

        def to_y(val):
            return (h - pad) - (val - mn) / (mx - mn) * (h - 2*pad)

        # Zero line
        y0 = to_y(0)
        if pad < y0 < h - pad:
            c.create_line(pad, y0, w - pad, y0,
                          fill=self.P["border"], width=1, dash=(2, 3))

        # Pursuer line (green)
        pts_p = []
        for ep, val in zip(eps, p_r):
            pts_p += [to_x(ep), to_y(val)]
        if len(pts_p) >= 4:
            c.create_line(*pts_p, fill=self.P["accent3"], width=1.5, smooth=True)

        # Evader line (purple)
        pts_e = []
        for ep, val in zip(eps, e_r):
            pts_e += [to_x(ep), to_y(val)]
        if len(pts_e) >= 4:
            c.create_line(*pts_e, fill=self.P["accent2"], width=1.5, smooth=True)

        # Axis labels
        c.create_text(pad + 2, pad + 2, anchor="nw",
                      text=f"{mx:.1f}", fill=self.P["muted"],
                      font=("Segoe UI", 7))
        c.create_text(pad + 2, h - pad - 2, anchor="sw",
                      text=f"{mn:.1f}", fill=self.P["muted"],
                      font=("Segoe UI", 7))

    def _on_training_complete(self, data: dict):
        self.state["metrics"] = data.get("metrics_obj")
        self._set_controls_during_training(False)
        if data.get("error"):
            self._update_status(f"Error — check terminal", self.P["danger"])
        else:
            self._update_status("Done", self.P["accent3"])
        self._progress_bar["value"] = 100
        self._eta_var.set("complete")

        if self._vis_replay.get() and data.get("trajectory"):
            self._setup_replay(data["trajectory"])

        self.on_done()

    # ──────────────────────────────────────────────────────────────────────────
    # Grid canvas
    # ──────────────────────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event):
        self._canvas_w = event.width
        self._canvas_h = event.height
        if hasattr(self, "_last_grid_state"):
            self._draw_grid(*self._last_grid_state)

    def _draw_placeholder(self, msg: str):
        c = self._grid_canvas
        c.delete("all")
        w = c.winfo_width()  or 400
        h = c.winfo_height() or 400
        c.create_text(w//2, h//2, text=msg,
                      fill=self.P["muted"],
                      font=("Segoe UI", 10),
                      width=w - 40)

    def _draw_grid(self, pursuer_pos, evader_pos, goal_pos):
        """Draw the 10x10 grid with agents and goal."""
        self._last_grid_state = (pursuer_pos, evader_pos, goal_pos)
        c = self._grid_canvas
        c.delete("all")

        cw = c.winfo_width()  or 400
        ch = c.winfo_height() or 400

        GRID = 10
        margin = 20
        cell = min((cw - 2*margin) // GRID, (ch - 2*margin) // GRID)
        ox = (cw - GRID * cell) // 2
        oy = (ch - GRID * cell) // 2

        for r in range(GRID):
            for col in range(GRID):
                x0 = ox + col * cell
                y0 = oy + r   * cell
                x1 = x0 + cell
                y1 = y0 + cell
                fill = self.P["surface"]
                if (col, r) == goal_pos:
                    fill = "#1f3a2b"
                c.create_rectangle(x0, y0, x1, y1,
                                   fill=fill, outline=self.P["border"],
                                   width=1)

        # Goal marker
        if goal_pos:
            gx = ox + goal_pos[0] * cell
            gy = oy + goal_pos[1] * cell
            pad = cell // 4
            c.create_rectangle(gx+pad, gy+pad,
                                gx+cell-pad, gy+cell-pad,
                                fill=self.P["accent3"], outline="",
                                tags="goal")
            c.create_text(gx+cell//2, gy+cell//2, text="G",
                          fill="#ffffff", font=("Segoe UI", max(8, cell//3), "bold"))

        # Pursuer
        if pursuer_pos:
            px = ox + pursuer_pos[0] * cell
            py = oy + pursuer_pos[1] * cell
            r  = cell // 3
            cx = px + cell // 2
            cy = py + cell // 2
            c.create_oval(cx-r, cy-r, cx+r, cy+r,
                          fill=self.P["accent3"], outline="")
            c.create_text(cx, cy, text="P",
                          fill="#ffffff", font=("Segoe UI", max(7, r-2), "bold"))

        # Evader
        if evader_pos:
            ex2 = ox + evader_pos[0] * cell
            ey2 = oy + evader_pos[1] * cell
            r   = cell // 3
            cx  = ex2 + cell // 2
            cy  = ey2 + cell // 2
            c.create_oval(cx-r, cy-r, cx+r, cy+r,
                          fill=self.P["accent2"], outline="")
            c.create_text(cx, cy, text="E",
                          fill="#ffffff", font=("Segoe UI", max(7, r-2), "bold"))

        # Legend
        lx = ox
        ly = oy + GRID * cell + 8
        for colour, label in [
            (self.P["accent3"], "Pursuer (P)"),
            (self.P["accent2"], "Evader (E)"),
            (self.P["accent3"], "Goal (G)"),
        ]:
            c.create_oval(lx, ly+3, lx+8, ly+11, fill=colour, outline="")
            c.create_text(lx+12, ly+7, anchor="w", text=label,
                          fill=self.P["muted"], font=("Segoe UI", 8))
            lx += 100

    # ──────────────────────────────────────────────────────────────────────────
    # Replay
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_replay(self, trajectory: list):
        self._replay_episode = trajectory
        self._replay_idx     = 0
        self._replay_frame.grid()
        self._vis_badge_var.set("REPLAY — last episode")
        self._vis_badge.config(bg=self.P["accent"], fg="#ffffff")
        self._draw_replay_step(0)

    def _draw_replay_step(self, idx: int):
        if not self._replay_episode:
            return
        idx = max(0, min(idx, len(self._replay_episode) - 1))
        self._replay_idx = idx
        step = self._replay_episode[idx]
        self._draw_grid(*step)
        self._replay_step_var.set(
            f"Step {idx + 1} / {len(self._replay_episode)}")

    def _replay_start(self):
        if self._replay_after:
            self.after_cancel(self._replay_after)
            self._replay_after = None
        self._draw_replay_step(0)

    def _replay_end(self):
        self._draw_replay_step(len(self._replay_episode) - 1)

    def _replay_play(self):
        if self._replay_idx >= len(self._replay_episode) - 1:
            self._replay_idx = 0
        self._step_replay()

    def _step_replay(self):
        if self._replay_idx < len(self._replay_episode) - 1:
            self._replay_idx += 1
            self._draw_replay_step(self._replay_idx)
            self._replay_after = self.after(
                self._replay_speed.get(), self._step_replay)

    def _on_replay_scrub(self, val):
        if not self._replay_episode:
            return
        pct = float(val) / 100
        idx = int(pct * (len(self._replay_episode) - 1))
        self._draw_replay_step(idx)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _set_controls_during_training(self, training: bool):
        state = "disabled" if training else "normal"
        for w in (self._ep_spinbox, self._steps_spinbox, self._freq_spinbox,
                  self._start_btn):
            w.configure(state=state)
        self._stop_btn.configure(state="normal" if training else "disabled")

    def _stop_training(self):
        self._stop_flag.set()
        self._update_status("Stopping…", self.P["warning"])

    # ──────────────────────────────────────────────────────────────────────────
    # Memory save / load
    # ──────────────────────────────────────────────────────────────────────────

    def _save_memory(self, role: str):
        """Save the trained agent's memory (Q-table, weights, etc.) to a file."""
        from tkinter import filedialog, messagebox

        cls = self.state.get(f"{role}_class")
        if cls is None:
            messagebox.showwarning("No agent", f"Load a {role} agent file first.")
            return

        # The live agent instance lives on the runner; fall back to a fresh
        # instance if training hasn't run yet — the user gets a warning.
        agent = getattr(self, f"_live_{role}", None)
        if agent is None:
            messagebox.showwarning(
                "Not trained",
                f"Train at least once before saving the {role}'s memory.\n"
                "There is nothing to save yet.")
            return

        path = filedialog.asksaveasfilename(
            title=f"Save {role} memory",
            defaultextension=".pkl",
            filetypes=[
                ("Pickle files", "*.pkl"),
                ("All files", "*.*"),
            ],
            initialfile=f"{role}_memory.pkl",
        )
        if not path:
            return

        try:
            agent.save(path)
            status_var = self._p_mem_status if role == "pursuer" else self._e_mem_status
            status_var.set(f"✓ Saved: {Path(path).name}")
        except NotImplementedError:
            messagebox.showinfo(
                "Not implemented",
                f"This {role} agent doesn't implement save().\n"
                "Add a save() method to your agent class.")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _load_memory(self, role: str):
        """Load previously saved memory into the current agent instance."""
        from tkinter import filedialog, messagebox

        agent = getattr(self, f"_live_{role}", None)
        if agent is None:
            messagebox.showwarning(
                "No agent instance",
                f"Train at least one episode first so an agent instance exists,\n"
                f"then load memory into it.")
            return

        path = filedialog.askopenfilename(
            title=f"Load {role} memory",
            filetypes=[
                ("Pickle files", "*.pkl"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            agent.load(path)
            status_var = self._p_mem_status if role == "pursuer" else self._e_mem_status
            status_var.set(f"✓ Loaded: {Path(path).name}")
        except NotImplementedError:
            messagebox.showinfo(
                "Not implemented",
                f"This {role} agent doesn't implement load().\n"
                "Add a load() method to your agent class.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _update_status(self, msg: str, colour: str):
        self._status_var.set(msg)
        self._status_dot.config(fg=colour)

    def _on_vis_toggle(self):
        if self._vis_enabled.get():
            self._vis_badge_var.set("ON — slower training")
            self._vis_badge.config(bg=self.P["warning"], fg="#000000")
        else:
            self._vis_badge_var.set("OFF — faster training")
            self._vis_badge.config(bg=self.P["border"], fg=self.P["muted"])


# ── Small helpers ─────────────────────────────────────────────────────────────

def _lbl(parent, text_or_var, style: str, P: dict) -> tk.Label:
    cfg = {"bg": P["surface"]}
    if style == "header":
        cfg.update(fg=P["text"],  font=("Segoe UI", 11, "bold"))
    elif style == "muted":
        cfg.update(fg=P["muted"], font=("Segoe UI", 9))
    elif style == "stat_blue":
        cfg.update(fg=P["accent"], font=("Segoe UI", 16, "bold"))
    else:
        cfg.update(fg=P["text"],  font=("Segoe UI", 10))

    if isinstance(text_or_var, tk.StringVar):
        return tk.Label(parent, textvariable=text_or_var, **cfg)
    else:
        return tk.Label(parent, text=text_or_var, **cfg)


def _divider(parent, P: dict) -> tk.Frame:
    return tk.Frame(parent, bg=P["border"], height=1)


def _stat_cell(parent, label: str, var: tk.StringVar,
               colour: str, P: dict, col: int):
    cell = tk.Frame(parent, bg=P["surface"])
    cell.grid(row=0, column=col, sticky="ew", padx=(0 if col else 0, 8))
    tk.Label(cell, textvariable=var, bg=P["surface"], fg=colour,
             font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(cell, text=label, bg=P["surface"], fg=P["muted"],
             font=("Segoe UI", 8)).pack(anchor="w")