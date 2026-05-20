"""
upload_frame.py — Screen 1: load and validate the two agent files.

The user picks a .py file for the pursuer and one for the evader.
Each file is validated to confirm it exports an AGENT_CLASS that
inherits from BaseAgent and has the expected role attribute.
Only once both are valid does the Continue button become clickable.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import importlib.util
import sys
import traceback


class UploadFrame(ttk.Frame):
    def __init__(self, parent, state: dict, on_continue, palette: dict):
        super().__init__(parent)
        self.state      = state
        self.on_continue = on_continue
        self.P          = palette
        self.configure(style="TFrame")

        self._pursuer_path = None
        self._evader_path  = None
        self._pursuer_ok   = False
        self._evader_ok    = False

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Centre content vertically
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        container = ttk.Frame(self, style="TFrame")
        container.grid(row=1, column=0, sticky="nsew", padx=60, pady=40)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        # ── Header ────────────────────────────────────────────────────────────
        ttk.Label(container, text="Load Agent Files",
                  style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(container,
                  text="Select a .py file for each role. Each file must export "
                       "an AGENT_CLASS that inherits BaseAgent.",
                  style="Sub.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 28))

        # ── Two agent cards side by side ──────────────────────────────────────
        self._pursuer_card = self._make_agent_card(
            container, row=2, col=0,
            role="Pursuer",
            color=self.P["accent3"],
            desc="Catches the evader. Wins when it occupies the same cell.",
        )
        self._evader_card = self._make_agent_card(
            container, row=2, col=1,
            role="Evader",
            color=self.P["accent2"],
            desc="Reaches the goal tile. Wins when it steps on it.",
        )

        # ── Divider + continue ────────────────────────────────────────────────
        ttk.Separator(container, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=28)

        btn_row = ttk.Frame(container, style="TFrame")
        btn_row.grid(row=4, column=0, columnspan=2, sticky="e")

        self._continue_btn = ttk.Button(
            btn_row, text="Continue to Training  →",
            style="Accent.TButton",
            state="disabled",
            command=self._continue)
        self._continue_btn.pack(side="right")

        # ── Help accordion ────────────────────────────────────────────────────
        help_frame = ttk.LabelFrame(container, text="What does a valid agent file look like?",
                                    padding=12)
        help_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(20, 0))

        code_text = (
            "from agents.base_agent import BaseAgent\n"
            "import numpy as np\n\n"
            "class MyPursuer(BaseAgent):\n"
            "    ROLE = 'pursuer'          # 'pursuer' or 'evader'\n"
            "    NAME = 'My Q-Learning'    # displayed in the UI\n\n"
            "    def select_action(self, obs):\n"
            "        return np.random.randint(4)  # 0=up 1=down 2=left 3=right\n\n"
            "    def on_step(self, obs, action, reward, next_obs, done): pass\n"
            "    def on_episode_end(self, episode, won): pass\n\n"
            "AGENT_CLASS = MyPursuer      # required export"
        )

        code_widget = tk.Text(
            help_frame, height=14, wrap="none",
            bg=self.P["bg"], fg=self.P["text"],
            font=("Consolas", 9), relief="flat",
            insertbackground=self.P["text"],
            state="normal", cursor="arrow",
            padx=10, pady=8
        )
        code_widget.insert("1.0", code_text)
        code_widget.configure(state="disabled")
        code_widget.pack(fill="x")

    def _make_agent_card(self, parent, row, col, role, color, desc):
        """Build one agent-picker card and return a reference dict."""
        # Add padding between the two cards
        padx = (0, 12) if col == 0 else (12, 0)

        card = tk.Frame(parent, bg=self.P["surface"],
                        highlightbackground=self.P["border"],
                        highlightthickness=1)
        card.grid(row=row, column=col, sticky="nsew", padx=padx)
        card.columnconfigure(0, weight=1)

        # Role label with colour dot
        header = tk.Frame(card, bg=self.P["surface"])
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 0))

        tk.Frame(header, bg=color, width=10, height=10).pack(side="left")
        tk.Label(header, text=f"  {role}", bg=self.P["surface"],
                 fg=self.P["text"], font=("Segoe UI", 13, "bold")).pack(side="left")

        tk.Label(card, text=desc, bg=self.P["surface"],
                 fg=self.P["muted"], font=("Segoe UI", 9),
                 wraplength=300, justify="left").grid(
            row=1, column=0, sticky="w", padx=16, pady=(4, 12))

        ttk.Separator(card, orient="horizontal").grid(
            row=2, column=0, sticky="ew")

        # File picker area
        pick_row = tk.Frame(card, bg=self.P["surface"])
        pick_row.grid(row=3, column=0, sticky="ew", padx=16, pady=12)
        pick_row.columnconfigure(0, weight=1)

        path_var = tk.StringVar(value="No file selected")
        path_label = tk.Label(pick_row, textvariable=path_var,
                              bg=self.P["surface"], fg=self.P["muted"],
                              font=("Segoe UI", 9), anchor="w",
                              wraplength=260, justify="left")
        path_label.grid(row=0, column=0, sticky="ew")

        pick_btn = ttk.Button(
            pick_row, text="Browse…",
            command=lambda r=role, pv=path_var: self._pick_file(r, pv))
        pick_btn.grid(row=1, column=0, sticky="w", pady=(8, 0))

        # Validation badge area
        badge_frame = tk.Frame(card, bg=self.P["surface"])
        badge_frame.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 16))

        badge_var    = tk.StringVar(value="")
        badge_color  = tk.StringVar(value=self.P["muted"])
        badge_label  = tk.Label(badge_frame, textvariable=badge_var,
                                bg=self.P["surface"], fg=self.P["muted"],
                                font=("Segoe UI", 9))
        badge_label.pack(side="left")

        ref = {
            "path_var":    path_var,
            "badge_var":   badge_var,
            "badge_label": badge_label,
            "card":        card,
        }

        if role == "Pursuer":
            self._pursuer_ref = ref
        else:
            self._evader_ref = ref

        return ref

    # ── File picking & validation ─────────────────────────────────────────────

    def _pick_file(self, role: str, path_var: tk.StringVar):
        path = filedialog.askopenfilename(
            title=f"Select {role} agent file",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if not path:
            return

        p = Path(path)
        path_var.set(str(p))

        ok, msg, agent_class = self._validate_agent(p, role.lower())
        ref = self._pursuer_ref if role == "Pursuer" else self._evader_ref

        if ok:
            ref["badge_var"].set(f"✓  {msg}")
            ref["badge_label"].config(fg=self.P["accent3"])
            if role == "Pursuer":
                self._pursuer_path = p
                self._pursuer_ok   = True
                self.state["pursuer_path"]  = p
                self.state["pursuer_class"] = agent_class
            else:
                self._evader_path = p
                self._evader_ok   = True
                self.state["evader_path"]  = p
                self.state["evader_class"] = agent_class
        else:
            ref["badge_var"].set(f"✗  {msg}")
            ref["badge_label"].config(fg=self.P["danger"])
            if role == "Pursuer":
                self._pursuer_ok = False
            else:
                self._evader_ok = False

        self._update_continue()

    def _validate_agent(self, path: Path, expected_role: str):
        """
        Try to import the file and check:
        1. It defines AGENT_CLASS
        2. AGENT_CLASS has a ROLE attribute matching expected_role
        3. AGENT_CLASS has select_action, on_step, on_episode_end methods
        Returns (ok: bool, message: str, agent_class | None)
        """
        # Temporarily add the file's directory to sys.path so relative
        # imports inside the agent file work correctly.
        parent_dir = str(path.parent)
        inserted = False
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            inserted = True

        module_name = f"_agent_validation_{path.stem}"
        try:
            spec   = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            if inserted:
                sys.path.remove(parent_dir)
            return False, f"Import error: {e}", None
        finally:
            pass

        if inserted:
            try:
                sys.path.remove(parent_dir)
            except ValueError:
                pass

        # Check AGENT_CLASS export
        if not hasattr(module, "AGENT_CLASS"):
            return False, "Missing AGENT_CLASS export", None

        cls = module.AGENT_CLASS

        # Check required methods (matches BaseAgent abstract interface)
        for method in ("select_action", "update", "reset"):
            if not callable(getattr(cls, method, None)):
                return False, f"Missing method: {method}()", None

        # Detect intended role from __init__ default, or explicit INTENDED_ROLE
        import inspect
        intended = getattr(cls, "INTENDED_ROLE", None)
        if intended is None:
            try:
                sig = inspect.signature(cls.__init__)
                role_param = sig.parameters.get("role")
                if role_param and role_param.default is not inspect.Parameter.empty:
                    intended = role_param.default
            except (ValueError, TypeError):
                pass

        if intended is not None and intended != expected_role:
            return False, f"This is a {intended} agent, not a {expected_role}", None

        # Try to instantiate
        try:
            instance = cls(role=expected_role, config={})
            _ = instance
        except Exception as e:
            return False, f"Cannot instantiate: {e}", None

        name = getattr(cls, "NAME", cls.__name__)
        return True, f"Valid — {name}", cls

    def _update_continue(self):
        if self._pursuer_ok and self._evader_ok:
            self._continue_btn.configure(state="normal")
        else:
            self._continue_btn.configure(state="disabled")

    def _continue(self):
        self.on_continue()