"""
app.py — Main window entry point for the RL Pursuit-Evasion Tool.

Run with:
    python -m ui.app
or from project root:
    python main.py
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Make sure the project root is on sys.path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.upload_frame import UploadFrame
from ui.training_frame import TrainingFrame
from ui.results_frame import ResultsFrame
from ui.comparison_frame import ComparisonFrame


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#1a1d23",   # near-black background
    "surface":   "#22262f",   # card / panel surface
    "border":    "#2e3340",   # subtle borders
    "accent":    "#4f8ef7",   # blue accent
    "accent2":   "#a78bfa",   # purple accent (evader)
    "accent3":   "#34d399",   # green accent (pursuer)
    "text":      "#e2e8f0",   # primary text
    "muted":     "#8b95a8",   # secondary / muted text
    "danger":    "#f87171",   # error / stop
    "warning":   "#fbbf24",   # warning / paused
}


def apply_theme(root: tk.Tk) -> ttk.Style:
    """Configure a dark theme across all standard ttk widgets."""
    style = ttk.Style(root)
    style.theme_use("clam")

    bg      = PALETTE["bg"]
    surface = PALETTE["surface"]
    border  = PALETTE["border"]
    accent  = PALETTE["accent"]
    text    = PALETTE["text"]
    muted   = PALETTE["muted"]

    # ── Notebook (tabs) ───────────────────────────────────────────────────────
    style.configure("TNotebook",
        background=bg, borderwidth=0, tabmargins=[0, 0, 0, 0])
    style.configure("TNotebook.Tab",
        background=surface, foreground=muted,
        padding=[18, 8], font=("Segoe UI", 10),
        borderwidth=0)
    style.map("TNotebook.Tab",
        background=[("selected", bg), ("active", border)],
        foreground=[("selected", text), ("active", text)])

    # ── Frames ────────────────────────────────────────────────────────────────
    style.configure("TFrame",       background=bg)
    style.configure("Card.TFrame",  background=surface, relief="flat")
    style.configure("TLabelframe",  background=surface, foreground=text,
                    bordercolor=border, relief="flat", padding=10)
    style.configure("TLabelframe.Label",
                    background=surface, foreground=muted,
                    font=("Segoe UI", 9, "bold"))

    # ── Labels ────────────────────────────────────────────────────────────────
    style.configure("TLabel",       background=bg,      foreground=text,
                    font=("Segoe UI", 10))
    style.configure("Card.TLabel",  background=surface, foreground=text,
                    font=("Segoe UI", 10))
    style.configure("Title.TLabel", background=bg,      foreground=text,
                    font=("Segoe UI", 18, "bold"))
    style.configure("Sub.TLabel",   background=bg,      foreground=muted,
                    font=("Segoe UI", 10))
    style.configure("Stat.TLabel",  background=surface, foreground=accent,
                    font=("Segoe UI", 22, "bold"))
    style.configure("StatLabel.TLabel", background=surface, foreground=muted,
                    font=("Segoe UI", 9))
    style.configure("Muted.TLabel", background=surface, foreground=muted,
                    font=("Segoe UI", 9))

    # ── Buttons ───────────────────────────────────────────────────────────────
    style.configure("TButton",
        background=surface, foreground=text,
        font=("Segoe UI", 10), padding=[12, 6],
        borderwidth=1, relief="flat")
    style.map("TButton",
        background=[("active", border), ("disabled", bg)],
        foreground=[("disabled", muted)])

    style.configure("Accent.TButton",
        background=accent, foreground="#ffffff",
        font=("Segoe UI", 10, "bold"), padding=[14, 7],
        borderwidth=0, relief="flat")
    style.map("Accent.TButton",
        background=[("active", "#3a7be0"), ("disabled", border)],
        foreground=[("disabled", muted)])

    style.configure("Danger.TButton",
        background=PALETTE["danger"], foreground="#ffffff",
        font=("Segoe UI", 10, "bold"), padding=[14, 7],
        borderwidth=0, relief="flat")
    style.map("Danger.TButton",
        background=[("active", "#ef4444"), ("disabled", border)])

    # ── Entry / Spinbox ───────────────────────────────────────────────────────
    style.configure("TEntry",
        fieldbackground=surface, background=surface,
        foreground=text, insertcolor=text,
        bordercolor=border, lightcolor=border,
        darkcolor=border, padding=6,
        font=("Segoe UI", 10))
    style.configure("TSpinbox",
        fieldbackground=surface, background=surface,
        foreground=text, arrowcolor=muted,
        bordercolor=border, padding=6,
        font=("Segoe UI", 10))

    # ── Combobox ──────────────────────────────────────────────────────────────
    style.configure("TCombobox",
        fieldbackground=surface, background=surface,
        foreground=text, arrowcolor=muted,
        bordercolor=border, selectbackground=accent,
        font=("Segoe UI", 10))
    root.option_add("*TCombobox*Listbox.background",   surface)
    root.option_add("*TCombobox*Listbox.foreground",   text)
    root.option_add("*TCombobox*Listbox.selectBackground", accent)

    # ── Progressbar ───────────────────────────────────────────────────────────
    style.configure("TProgressbar",
        background=accent, troughcolor=border,
        borderwidth=0, thickness=6)

    # ── Scrollbar ─────────────────────────────────────────────────────────────
    style.configure("TScrollbar",
        background=border, troughcolor=bg,
        arrowcolor=muted, borderwidth=0)

    # ── Separator ─────────────────────────────────────────────────────────────
    style.configure("TSeparator", background=border)

    # ── Checkbutton / Radiobutton ─────────────────────────────────────────────
    style.configure("TCheckbutton",
        background=surface, foreground=text,
        font=("Segoe UI", 10))
    style.map("TCheckbutton",
        background=[("active", surface)],
        indicatorcolor=[("selected", accent), ("!selected", border)])
    style.configure("TRadiobutton",
        background=surface, foreground=text,
        font=("Segoe UI", 10))
    style.map("TRadiobutton",
        background=[("active", surface)],
        indicatorcolor=[("selected", accent), ("!selected", border)])

    # ── Scale ─────────────────────────────────────────────────────────────────
    style.configure("TScale",
        background=bg, troughcolor=border,
        sliderrelief="flat", sliderlength=18)

    return style


class App(tk.Tk):
    """Root application window."""

    def __init__(self):
        super().__init__()
        self.title("RL Pursuit-Evasion — Analysis & Comparison Tool")
        self.geometry("1280x800")
        self.minsize(960, 640)
        self.configure(bg=PALETTE["bg"])

        self.style = apply_theme(self)

        # Shared state passed between frames
        self.state = {
            "pursuer_path":  None,
            "evader_path":   None,
            "pursuer_class": None,
            "evader_class":  None,
            "metrics":       None,   # filled after training
            "run_label":     None,   # human-readable name for saved runs
        }

        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar ──────────────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=PALETTE["surface"], height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(
            topbar, text="RL Pursuit-Evasion",
            bg=PALETTE["surface"], fg=PALETTE["text"],
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=20, pady=14)

        tk.Label(
            topbar, text="Analysis & Comparison Tool",
            bg=PALETTE["surface"], fg=PALETTE["muted"],
            font=("Segoe UI", 10)
        ).pack(side="left", padx=0, pady=14)

        # Version badge
        tk.Label(
            topbar, text="v1.0",
            bg=PALETTE["border"], fg=PALETTE["muted"],
            font=("Segoe UI", 8), padx=6, pady=2
        ).pack(side="right", padx=20, pady=16)

        # Thin accent line under topbar
        tk.Frame(self, bg=PALETTE["accent"], height=2).pack(fill="x")

        # Notebook (tabs) ──────────────────────────────────────────────────────
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=0, pady=0)

        self.upload_tab    = UploadFrame(self.notebook,    self.state, self._on_agents_loaded, PALETTE)
        self.training_tab  = TrainingFrame(self.notebook,  self.state, self._on_training_done, PALETTE)
        self.results_tab   = ResultsFrame(self.notebook,   self.state, PALETTE)
        self.comparison_tab = ComparisonFrame(self.notebook, self.state, PALETTE)

        self.notebook.add(self.upload_tab,     text="  ① Load Agents  ")
        self.notebook.add(self.training_tab,   text="  ② Train & Run  ")
        self.notebook.add(self.results_tab,    text="  ③ Results  ")
        self.notebook.add(self.comparison_tab, text="  ④ Compare  ")

        # Tabs 2-4 start disabled until agents are loaded
        self.notebook.tab(1, state="disabled")
        self.notebook.tab(2, state="disabled")
        self.notebook.tab(3, state="disabled")

        # Status bar ───────────────────────────────────────────────────────────
        statusbar = tk.Frame(self, bg=PALETTE["surface"], height=28)
        statusbar.pack(fill="x", side="bottom")
        statusbar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready — load two agent files to begin.")
        tk.Label(
            statusbar, textvariable=self._status_var,
            bg=PALETTE["surface"], fg=PALETTE["muted"],
            font=("Segoe UI", 9), anchor="w"
        ).pack(side="left", padx=12)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_agents_loaded(self):
        """Called by UploadFrame once both agent files pass validation."""
        self.notebook.tab(1, state="normal")
        self.notebook.select(1)
        self.set_status(
            f"Agents loaded — Pursuer: {self.state['pursuer_path'].name}  ·  "
            f"Evader: {self.state['evader_path'].name}"
        )

    def _on_training_done(self):
        """Called by TrainingFrame once a training run completes."""
        self.notebook.tab(2, state="normal")
        self.notebook.tab(3, state="normal")
        self.notebook.select(2)
        self.results_tab.refresh()
        self.comparison_tab.refresh()
        self.set_status("Training complete — results available.")

    def set_status(self, msg: str):
        self._status_var.set(msg)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()